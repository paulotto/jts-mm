# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Render an :class:`~biomech.anim.layouts.Animation` to MP4 (or WebM/GIF) (an11 §5).

* Frames go straight from the Agg canvas (RGB bytes) into an ``ffmpeg`` rawvideo
  pipe -- no PNG round trip; memory does not grow with the video length.
* The timeline is split into chunks that render in joblib (loky) workers. Each
  writes its own segment with identical encoder settings; the concat demuxer
  joins them with ``-c copy``. ``n_jobs=1`` renders in-process.
* H.264 ``yuv420p``, CRF 18, ``+faststart``, even pixel sizes (``crf=0`` is
  lossless, used by the tests to compare chunked and single-process renders).
* ``on_progress(done, total)`` and ``cancel() -> bool`` make it drivable from a
  GUI thread (the viewer's RunManager). A cancelled render raises
  :class:`Cancelled` and leaves **no** partial file: everything is written in a
  temporary folder next to the output and moved into place only when complete.
* A JSON manifest next to the video records the recipe (layout, window, speed,
  pseudonym, whether camera frames are included).
* Clip files ``<stem>.clip.npz/.json`` next to the video hold everything the renderer
  needs (no camera frames, no study id; :mod:`biomech.anim.clipfile`), so the video can be
  re-rendered without the store (``biomech animate --from-clip``).
* Layouts in :data:`~biomech.anim.layouts.STILLS` (the D5 supplementary videos) also get a
  static filmstrip ``<stem>_filmstrip.png`` (:func:`filmstrip_png`), the fallback where a
  video cannot be embedded.
* Layouts with burned-in captions also get them as a subtitle track ``<stem>.vtt``
  (WebVTT, the same cues; :func:`write_vtt`), for players and accessibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from biomech.anim.layouts import STILL_WIDTH, STILLS, Animation, frame_size
from biomech.anim.timecode import read_timecode

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
CHUNK_FRAMES = 150                   # frames per worker task (progress granularity)


class Cancelled(RuntimeError):
    """The render was cancelled; no output file was written."""


@dataclass
class VideoResult:
    path: Path
    n_frames: int
    duration_s: float
    size: tuple[int, int]
    fps: float
    elapsed_s: float
    n_chunks: int
    manifest: Path | None = None
    frame_hashes: list[str] = field(default_factory=list)
    clip_files: tuple[Path, Path] | None = None
    filmstrip: Path | None = None
    subtitles: Path | None = None


def caption_cues(anim: Animation) -> list[tuple[float, float, str]]:
    """The burned-in caption cues (video seconds) of the animation's layout, if any."""
    from biomech.anim.layers import Caption

    for ly in anim.compose().layers:
        if isinstance(ly, Caption):
            return [(float(a), float(b), str(t)) for a, b, t in ly.cues if t]
    return []


def write_vtt(cues: list[tuple[float, float, str]], path: str | Path) -> Path:
    """A WebVTT subtitle file of ``cues`` (start s, end s, text)."""

    def ts(v: float) -> str:
        ms = int(round(max(v, 0.0) * 1000))
        return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}." \
               f"{ms % 1000:03d}"

    lines = ["WEBVTT", ""]
    for k, (a, b, t) in enumerate(cues, start=1):
        lines += [str(k), f"{ts(a)} --> {ts(b)}", t, ""]
    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def encoder_args(suffix: str, crf: int = 18) -> list[str]:
    """Encoder settings, identical for every chunk of one video."""
    if suffix == ".webm":
        return ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0",
                "-crf", str(max(crf + 12, 0)), "-row-mt", "1", "-deadline", "good"]
    rate = ["-qp", "0"] if crf == 0 else ["-crf", str(crf)]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", *rate,
            "-threads", "2", "-movflags", "+faststart"]


def _render_chunk(anim: Animation, size, frames: range, seg: str, enc: list[str],
                  cancel_file: str | None, stamp: bool, want_hashes: bool,
                  progress: Callable[[int], None] | None = None,
                  cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Render ``frames`` into the segment file ``seg`` (runs in a worker or in-process)."""
    W, H = frame_size(size)
    scene = anim.scene((W, H), stamp=stamp)
    hashes: list[str] = []
    proc = None
    try:
        scene.prepare(frames)
        proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-framerate", f"{anim.fps:g}", "-i", "-", *enc, seg],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for i in range(len(frames)):
            if (cancel is not None and cancel()) or (
                    cancel_file and i % 5 == 0 and os.path.exists(cancel_file)):
                proc.stdin.close()
                proc.kill()
                proc.wait()
                return {"cancelled": True, "n": i}
            buf = scene.frame(i)
            if want_hashes:
                hashes.append(hashlib.sha1(buf.tobytes()).hexdigest())
            proc.stdin.write(buf.tobytes())
            if progress is not None:
                progress(1)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {Path(seg).name}: {err[-800:]}")
    finally:
        scene.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
    return {"cancelled": False, "n": len(frames), "hashes": hashes, "start": frames.start}


def _concat(segs: list[str], out: str, tmp: Path) -> None:
    lst = tmp / "segments.txt"
    lst.write_text("".join(f"file '{Path(s).name}'\n" for s in segs), encoding="utf-8")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-c", "copy"]
    if out.endswith(".mp4"):
        cmd += ["-movflags", "+faststart"]
    r = subprocess.run(cmd + [out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {r.stderr[-800:]}")


def _to_gif(src: str, out: str, fps: float, width: int) -> None:
    flt = (f"fps={min(fps, 15):g},scale={min(width, 960)}:-2:flags=lanczos,split[a][b];"
           "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-vf", flt, out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg GIF conversion failed: {r.stderr[-800:]}")


def render_video(anim: Animation, out: str | Path, *, size: str | int | tuple = "1080p",
                 n_jobs: int | None = None, crf: int = 18, chunk_frames: int = CHUNK_FRAMES,
                 on_progress: Callable[[int, int], None] | None = None,
                 cancel: Callable[[], bool] | None = None, stamp: bool = True,
                 frame_hashes: bool = False, manifest: bool = True, clip_files: bool = True,
                 stills: bool | None = None) -> VideoResult:
    """Render ``anim`` to ``out`` (.mp4, .webm or .gif).

    After the video, ``clip_files`` writes ``<stem>.clip.npz/.json`` (G2) and ``stills``
    (default: the layout is in ``STILLS``) a PNG filmstrip ``<stem>_filmstrip.png``.

    ``n_jobs``: worker processes (default: up to 8; 1 renders in this process, which
    is what a GUI thread with fine-grained progress wants). ``on_progress(done,
    total)`` is called with frame counts; ``cancel()`` is polled about every 0.2 s
    (and every frame in-process); when it returns True the render stops, the
    temporary files are removed and :class:`Cancelled` is raised.
    """
    out = Path(out)
    suffix = out.suffix.lower()
    if suffix not in (".mp4", ".webm", ".gif"):
        raise ValueError(f"unsupported output {out.name!r} (mp4, webm, gif)")
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = frame_size(size)
    tl = anim.timeline
    total = tl.n_frames
    n_jobs = n_jobs or min(8, os.cpu_count() or 1)
    n_chunks = max(n_jobs if n_jobs > 1 else 1, int(np.ceil(total / max(chunk_frames, 1))))
    chunks = tl.chunks(n_chunks)
    enc = encoder_args(".webm" if suffix == ".webm" else ".mp4", crf)
    seg_ext = ".webm" if suffix == ".webm" else ".mp4"
    t_start = time.time()
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.stem}.", dir=out.parent))
    cancel_file = str(tmp / "CANCEL")
    segs = [str(tmp / f"seg_{k:04d}{seg_ext}") for k in range(len(chunks))]
    done = 0
    hashes: dict[int, list[str]] = {}
    stop = threading.Event()

    def report(n: int) -> None:
        nonlocal done
        done += n
        if on_progress is not None:
            on_progress(done, total)

    try:
        if n_jobs == 1:
            for k, fr in enumerate(chunks):
                r = _render_chunk(anim, (W, H), fr, segs[k], enc, None, stamp, frame_hashes,
                                  progress=report, cancel=cancel)
                if r["cancelled"]:
                    raise Cancelled("render cancelled")
                hashes[fr.start] = r["hashes"]
        else:
            from joblib import Parallel, delayed

            def watch() -> None:          # turn cancel() into a file the workers can see
                while not stop.wait(0.2):
                    if cancel is not None and cancel():
                        Path(cancel_file).touch()
                        return

            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            jobs = (delayed(_render_chunk)(anim, (W, H), fr, segs[k], enc, cancel_file, stamp,
                                           frame_hashes) for k, fr in enumerate(chunks))
            cancelled = False
            for r in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs):
                if r["cancelled"]:
                    cancelled = True
                    continue
                hashes[r["start"]] = r["hashes"]
                report(r["n"])
            if cancelled or os.path.exists(cancel_file):
                raise Cancelled("render cancelled")
        if cancel is not None and cancel():
            raise Cancelled("render cancelled")
        final = str(tmp / f"out{suffix}")
        if suffix == ".gif":
            joined = str(tmp / "joined.mp4")
            _concat(segs, joined, tmp)
            _to_gif(joined, final, anim.fps, W)
        else:
            _concat(segs, final, tmp)
        os.replace(final, out)
    finally:
        stop.set()
        shutil.rmtree(tmp, ignore_errors=True)
    elapsed = time.time() - t_start
    res = VideoResult(out, total, tl.duration_s, (W, H), anim.fps, elapsed, len(chunks),
                      frame_hashes=[h for k in sorted(hashes) for h in hashes[k]])
    if clip_files:
        from biomech.anim.clipfile import save_clip

        res.clip_files = save_clip(anim, out, size=size)
    if stills if stills is not None else anim.layout in STILLS:
        res.filmstrip = filmstrip_png(anim, out.with_name(f"{out.stem}_filmstrip.png"),
                                      width=STILL_WIDTH.get(anim.layout, 640))
    cues = caption_cues(anim)
    if cues:
        res.subtitles = write_vtt(cues, out.with_suffix(".vtt"))
    if manifest:
        # Sampled (frame, master time) pairs: biomech check-outputs reads the hidden
        # timecode of these frames back and compares (biomech.check.outputs).
        k = np.unique(np.linspace(0, total - 1, 12).round().astype(int))
        m = dict(anim.manifest(), file=out.name, size=[W, H], width=W, height=H,
                 expected_duration_s=total / anim.fps, encoder=" ".join(enc),
                 timecode=bool(stamp), timecode_tol_us=0.5 * anim.speed * 1e6 / anim.fps,
                 frames=[[int(i), int(round(t))] for i, t in zip(k, tl.frame_times(k))]
                 if stamp else [],
                 created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 elapsed_s=round(elapsed, 2), n_chunks=len(chunks))
        if res.clip_files:
            m["clip_files"] = [p.name for p in res.clip_files]
        if res.filmstrip:
            m["filmstrip"] = res.filmstrip.name
        if res.subtitles:
            m["subtitles"] = res.subtitles.name
        res.manifest = out.with_suffix(".json")
        res.manifest.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return res


def filmstrip_png(anim: Animation, out: str | Path, *, n: int = 6, cols: int = 3,
                  width: int = 640, gap: int = 8) -> Path:
    """A static filmstrip: ``n`` frames evenly inside the video (first and last excluded),
    each rendered by the video's :class:`~biomech.anim.layouts.Scene` at ``width`` px
    (no timecode strip), tiled ``cols`` per row on the theme's surface colour."""
    from PIL import Image as PILImage

    from biomech.anim.layers import rgb

    out = Path(out)
    tl = anim.timeline
    k = np.unique(np.linspace(0, tl.n_frames - 1, n + 2).round().astype(int)[1:-1])
    if not k.size:
        k = np.array([0])
    W, H = frame_size(width)
    scene = anim.scene((W, H), stamp=False)
    try:
        scene.prepare(k)
        frames = [scene.frame(i) for i in range(len(k))]
    finally:
        scene.close()
    cols = max(1, min(cols, len(frames)))
    rows = int(np.ceil(len(frames) / cols))
    bg = (rgb("surface", anim.theme) * 255).round().astype(np.uint8)
    img = np.empty((rows * H + (rows + 1) * gap, cols * W + (cols + 1) * gap, 3), np.uint8)
    img[:] = bg
    for j, f in enumerate(frames):
        r, c = divmod(j, cols)
        y, x = gap + r * (H + gap), gap + c * (W + gap)
        img[y:y + H, x:x + W] = f
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.png")
    PILImage.fromarray(img).save(tmp, format="PNG", optimize=True)
    os.replace(tmp, out)
    return out


# -- checks ----------------------------------------------------------------------------------------
def probe(path: str | Path) -> dict[str, Any]:
    """Codec, pixel format, size, frame count, fps and duration (ffprobe, frames counted)."""
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames,"
                        "r_frame_rate:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    s = d["streams"][0]
    num, den = (float(v) for v in s["r_frame_rate"].split("/"))
    return {"codec": s["codec_name"], "pix_fmt": s.get("pix_fmt"), "width": int(s["width"]),
            "height": int(s["height"]), "n_frames": int(s["nb_read_frames"]),
            "fps": num / den if den else float("nan"),
            "duration_s": float(d["format"]["duration"])}


def video_frame_size(path: str | Path) -> tuple[int, int]:
    """(width, height) of the first video stream (header only; nothing is decoded)."""
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    return int(st["width"]), int(st["height"])


def decode(path: str | Path, frames: list[int] | None = None) -> list[np.ndarray]:
    """Decoded RGB frames (all, or the given indices) of a video; decoding stops after the
    last wanted frame (no full-file decode, no frame count, for a few early frames)."""
    W, H = video_frame_size(path)
    want = None if frames is None else set(int(f) for f in frames)
    last = max(want) if want else (-1 if want is not None else None)
    out: list[np.ndarray] = []
    if last == -1:                           # nothing wanted
        return out
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    k = 0
    n = W * H * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            if want is None or k in want:
                out.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
            if last is not None and k >= last:   # stop after the last wanted frame
                break
            k += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait()
    return out


def frame_md5(path: str | Path) -> list[str]:
    """Per-frame MD5 of the decoded video (ffmpeg framemd5)."""
    r = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "framemd5", "-"],
                       capture_output=True, text=True, check=True)
    return [ln.split(",")[-1].strip() for ln in r.stdout.splitlines()
            if ln and not ln.startswith("#")]


def verify_sync(path: str | Path, anim: Animation, n_samples: int = 12) -> dict[str, Any]:
    """Read the hidden timecode of sampled frames and compare with the timeline:
    ``max_err_us`` (0 when every sampled frame shows exactly its master-clock time),
    ``unreadable`` (frames without a valid timecode)."""
    tl = anim.timeline
    idx = np.unique(np.linspace(0, tl.n_frames - 1, n_samples).round().astype(int)).tolist()
    frames = decode(path, idx)
    want = tl.frame_times(np.array(idx))
    got = [read_timecode(f) for f in frames]
    errs = [abs(g - round(w)) for g, w in zip(got, want) if g is not None]
    return {"frames": idx, "expected_us": [int(round(w)) for w in want], "read_us": got,
            "unreadable": sum(g is None for g in got),
            "max_err_us": max(errs) if errs else None}
