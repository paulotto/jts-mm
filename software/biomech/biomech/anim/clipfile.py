# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Clip files (G2): everything a render needs, stored next to the video (an11).

Rendering a video writes ``videos/<stem>.clip.npz`` and ``videos/<stem>.clip.json``:

* ``.clip.npz`` -- every array the renderer reads after loading from the store and the
  session: the clip's modalities as cut for the video (kinematics, maxillary landmarks,
  EMG, IMU, labels, decoder decisions, optical poses and quality) and the derived D5
  modalities (``extra/hinge``, ``extra/release``, ``extra/online``); for the cohort layout
  the cycle-normalized paths. No strings, no camera frames.
* ``.clip.json`` -- the structure (a tree whose array leaves name their npz key), the
  playback settings (layout, fps, speed, theme, scrolling window, trail, loops, captions),
  the pseudonym, phases, segments, window, the size of the original render and an array
  index (shape, dtype).

:func:`load_clip_file` rebuilds the :class:`~biomech.anim.layouts.Animation` from the two
files alone -- no result store, no session -- so ``biomech animate --from-clip`` can
re-render with another speed, frame rate, theme or size.

Privacy: the study id is never written (the clip's ``pid`` is dropped; the pseudonym is
the participant's only name). Camera previews are never stored, whatever ``with_camera``
was: a clip of a camera video re-renders without camera views (``camera_frames:
"omitted"``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from biomech.anim.layouts import Animation
from biomech.anim.source import Clip, CohortClip

FORMAT = "biomech.anim.clip"
VERSION = 1
CLIP_FIELDS = ("kin", "maxilla", "emg", "imu", "labels", "posterior", "optical")


def clip_paths(video: str | Path) -> tuple[Path, Path]:
    """(``<stem>.clip.npz``, ``<stem>.clip.json``) next to a video (or for a stem path)."""
    v = Path(video)
    stem = v.name
    for suf in (".clip.npz", ".clip.json", ".mp4", ".webm", ".gif"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return v.with_name(f"{stem}.clip.npz"), v.with_name(f"{stem}.clip.json")


# -- tree encoding ---------------------------------------------------------------------------------
def _encode(obj: Any, key: str, arrays: dict[str, np.ndarray]) -> Any:
    """JSON tree of ``obj``; arrays go to ``arrays`` under ``key`` (leaf ``{"$a": key}``)."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "OUS":
            raise TypeError(f"{key}: string/object arrays are not stored in clip files")
        arrays[key] = obj
        return {"$a": key}
    if isinstance(obj, dict):
        return {str(k): _encode(v, f"{key}/{k}", arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if all(isinstance(v, (str, int, float, bool)) or v is None for v in obj):
            return list(obj)
        return {"$list": [_encode(v, f"{key}/{i}", arrays) for i, v in enumerate(obj)]}
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"{key}: cannot store {type(obj).__name__} in a clip file")


def _decode(node: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(node, dict):
        if set(node) == {"$a"}:
            return arrays[node["$a"]]
        if set(node) == {"$list"}:
            return [_decode(v, arrays) for v in node["$list"]]
        return {k: _decode(v, arrays) for k, v in node.items()}
    return node


def _pseudonym(label: str) -> str:
    m = re.search(r"(\d+)\s*$", label)
    return f"P{m.group(1)}" if m else label


# -- save / load ---------------------------------------------------------------------------------
def save_clip(anim: Animation, video: str | Path, *, size: Any = None
              ) -> tuple[Path, Path]:
    """Write the clip files of ``anim`` next to ``video``; returns (npz, json)."""
    npz, js = clip_paths(video)
    c = anim.clip
    arrays: dict[str, np.ndarray] = {}
    doc: dict[str, Any] = {
        "format": FORMAT, "version": VERSION,
        "animation": {"layout": anim.layout, "fps": anim.fps, "speed": anim.speed,
                      "theme": anim.theme, "window_s": list(anim.window_s),
                      "trail_s": anim.trail_s, "loops": anim.loops,
                      "captions": [list(x) for x in anim.captions]
                      if anim.captions is not None else None},
        "render": {"video": Path(video).name,
                   "size": list(size) if isinstance(size, tuple) else size},
    }
    if isinstance(c, Clip):
        tree = {f: _encode(getattr(c, f), f, arrays) for f in CLIP_FIELDS
                if getattr(c, f) is not None}
        tree["extra"] = {k: _encode(v, f"extra/{k}", arrays) for k, v in c.extra.items()}
        meta = {k: v for k, v in c.meta.items() if k in ("take", "classes", "window_s")}
        doc["clip"] = {"kind": "clip", "label": c.label, "pseudonym": _pseudonym(c.label),
                       "segments": [[ph, float(a), float(b)] for ph, a, b in c.segments],
                       "phases": [ph for ph, _, _ in c.segments], "meta": meta,
                       "camera_frames": "omitted" if c.previews else "none",
                       "tree": tree}
    else:
        tree = {"trials": _encode(list(c.trials), "trials", arrays),
                "means": _encode(c.means, "means", arrays),
                "speed": _encode(c.speed, "speed", arrays)}
        if c.extra:
            tree["extra"] = {k: _encode(v, f"extra/{k}", arrays) for k, v in c.extra.items()}
        doc["clip"] = {"kind": "cohort", "task": c.task, "labels": list(c.labels),
                       "meta": {k: v for k, v in c.meta.items() if k == "n_trials"},
                       "camera_frames": "none", "tree": tree}
    doc["arrays"] = {k: [list(v.shape), str(v.dtype)] for k, v in sorted(arrays.items())}
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(f".{npz.name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz)
    tmpj = js.with_name(f".{js.name}.tmp")
    tmpj.write_text(json.dumps(doc, indent=1, allow_nan=True), encoding="utf-8")
    os.replace(tmpj, js)
    return npz, js


def read_clip_doc(path: str | Path) -> dict[str, Any]:
    return json.loads(clip_paths(path)[1].read_text(encoding="utf-8"))


def load_clip_file(path: str | Path) -> Animation:
    """The :class:`Animation` stored in ``<stem>.clip.npz/.json`` (either file, or the
    video next to them), with the settings it was rendered with."""
    npz, js = clip_paths(path)
    if not npz.is_file() or not js.is_file():
        raise FileNotFoundError(f"clip files missing: {npz.name} and {js.name} are both needed")
    doc = json.loads(js.read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"{js.name} is not a clip file")
    if int(doc.get("version", 0)) > VERSION:
        raise ValueError(f"{js.name}: clip format {doc['version']} is newer than {VERSION}")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    a, c = doc["animation"], doc["clip"]
    tree = _decode(c["tree"], arrays)
    if c["kind"] == "cohort":
        clip: Clip | CohortClip = CohortClip(task=c["task"], labels=list(c["labels"]),
                                             trials=list(tree["trials"]), means=tree["means"],
                                             speed=tree["speed"], meta=dict(c.get("meta", {})),
                                             extra=dict(tree.get("extra", {})))
    else:
        clip = Clip(pid=c["pseudonym"], label=c["label"],
                    segments=[(str(ph), float(t0), float(t1)) for ph, t0, t1 in c["segments"]],
                    **{f: tree.get(f) for f in CLIP_FIELDS}, previews=None, with_camera=False,
                    meta=dict(c.get("meta", {})), extra=dict(tree.get("extra", {})))
    caps = a.get("captions")
    return Animation(clip, a["layout"], float(a["fps"]), float(a["speed"]), a["theme"],
                     tuple(a["window_s"]), float(a["trail_s"]), int(a.get("loops", 2)),
                     tuple((float(x[0]), float(x[1]), str(x[2])) for x in caps)
                     if caps is not None else None)


def clip_stem(anim: Animation, doc: dict[str, Any], *, size: Any = None) -> str:
    """Video stem of a re-render from a clip (:func:`biomech.anim.layouts.video_stem` with
    the clip's pseudonym, phases and window and the new settings)."""
    from biomech.anim.layouts import video_stem

    c = doc["clip"]
    if c["kind"] == "cohort":
        return video_stem(anim.layout, None, [c["task"]], speed=anim.speed, theme=anim.theme,
                          size=size, fps=anim.fps)
    w = c.get("meta", {}).get("window_s") or [None, None]
    n = c["pseudonym"][1:] if c["pseudonym"].startswith("P") else c["pseudonym"]
    return video_stem(anim.layout, None, c["phases"], False, pseudonym=n, t0_s=w[0],
                      t1_s=w[1], speed=anim.speed, theme=anim.theme, size=size,
                      window_s=anim.window_s, trail_s=anim.trail_s, fps=anim.fps)
