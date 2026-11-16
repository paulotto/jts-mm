<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an11 — Animated visualizer (`biomech.anim`)

The animator renders synchronized videos of one take: the jaw, the tracking and every modality.
It can also play the whole cohort, cycle by cycle. Outputs are MP4 (also WebM/GIF), an interactive
Plotly preview for the viewer, a vector filmstrip of key frames for the paper, and PNG frames for
the LaTeX `animate` package. Every render also writes clip files (`<stem>.clip.npz/.json`)
from which the video can be re-rendered without the result store or the session, and the three
supplementary-video layouts (`v1_hinge`, `v2_intent`, `v3_release`) get a PNG filmstrip next
to the MP4.

```bash
biomech animate --config experiments/icorr2027/study.yaml --pid s01 --layout multimodal \
    --phase chewing --speed 0.5 --fps 30 --size 1080p
# -> data/analysis/icorr2027/videos/multimodal_P<n>_chewing.mp4 (+ .json manifest; P<n> = pseudonym)
make -C experiments/icorr2027 video          # s01, multimodal, chewing, 0.5x, 1080p, plus --filmstrip

# Supplementary videos (<pid> = a study id; files carry only the pseudonym P<n>)
biomech animate --config experiments/icorr2027/study.yaml --pid <pid> --layout v1_hinge
biomech animate --config experiments/icorr2027/study.yaml --pid <pid> --layout v2_intent
biomech animate --config experiments/icorr2027/study.yaml --pid <pid> --layout v3_release
# -> videos/v1_hinge_P<n>_open_close.mp4, v2_intent_P<n>_open_close_1x.mp4,
#    v3_release_P<n>_open_close+chewing.mp4, each with .json, .clip.npz/.clip.json, _filmstrip.png

# Re-render from the clip alone (no --config, no store, no session)
biomech animate --from-clip data/analysis/icorr2027/videos/v2_intent_P<n>_open_close_1x.clip.json \
    --theme light --size 720p
```

| Option | Meaning |
|---|---|
| `--layout` | `multimodal` (default), `kinematics`, `tracking`, `emg`, `cohort`, the supplementary-video layouts `v1_hinge`, `v2_intent`, `v3_release` (*Supplementary videos*), `validation` (systems paper, validation study only; *Revision of 2026-10-23*), and `workspace3d`, `emg_spectrogram`, `onset_rules` (*Video layouts of J2*) |
| `--variant`, `--snapshot` | the store to read (a paper set's: e.g. `--variant reporting_population --snapshot r17a`); the manifest records `snapshot` and `variant` |
| `--phase` | phase to play; repeat it to play several phases with cuts between them. Default `chewing`; `v1_hinge` and `v2_intent`: `open_close`; `v3_release`: `open_close` and `chewing` (side by side). For `cohort`, the task (default `open_close`) |
| `--t0`, `--t1` | window in seconds from the start of each selected phase |
| `--speed` | playback speed (default 0.5; `v2_intent` 1, simulated real time): 0.25 is slow motion, 4 is a time-lapse. For `cohort`, cycles per second |
| `--fps`, `--size` | frame rate (default 30); `1080p` (default), `720p`, `4k`, `print` (7.16 in at 300 dpi = 2148 × 1208), a width, or `WxH` |
| `--theme` | `dark` (default for videos) or `light` |
| `--with-camera` | include the camera previews (off by default, see *Privacy*) |
| `--window BEFORE AFTER` | scrolling panels: seconds before and after the cursor (default 3 1; `v2_intent` always uses AFTER = 0) |
| `--trail S` | length of the fading path behind moving landmarks (default 1 s) |
| `--captions FILE` | JSON list of `[start_s, end_s, text]` cues in video seconds; replaces a layout's default burned-in captions |
| `--from-clip FILE` | re-render from `<stem>.clip.npz` + `<stem>.clip.json` (either path); `--config` is then not needed; `--speed`, `--fps`, `--theme`, `--size`, `--window`, `--trail`, `--captions` override the clip's settings (default: the clip's, size: the original render's). Output: `<clip folder>/<stem>.mp4` with the stem of the new settings |
| `--out FILE` | `.mp4`, `.webm` or `.gif`; default `<out_dir>/videos/<stem>.mp4` (*File names* below) |
| `--filmstrip` | also write a pgfplots filmstrip of six key frames (`<video>_tex/`) |
| `--animate-tex` | also write PNG frames and an `\animategraphics` snippet (`<video>_animate/`) |
| `-j` | render workers (default 0 = up to 8) |

## Design

An animation is a static `FigureSpec` background plus dynamic layers sampled on a video
timeline. There is no second plotting system. Backgrounds are rendered with
`viz.render_mpl.build`, which uses the same tokens, units, tick rules and layout engine as the paper
figures, scaled by 1.4 for video. Videos default to the dark theme.

| Module | Role |
|---|---|
| `anim/timeline.py` | `Timeline(segments, fps, speed)`: video time ↔ master clock, phase cuts, chunks |
| `anim/source.py` | `Clip` (one participant: kinematics, EMG, IMU, labels, decoder decisions, optical poses and quality, previews; `extra`: the derived supplementary-video modalities `hinge`, `release`, `online`) and `CohortClip`, both read through `DataContext`; a modality that cannot be loaded is left out with a logged warning (`_skip`), and the layout re-flows |
| `anim/layers.py` | `Cursor`, `Scope` (scrolling series), `Trail`/`Marker`, `Pose`, `Triads`, `Image`, `Ribbon`, `Bars`, `Text`, `Chip`; for the supplementary videos `Outline`, `Events`, `Tally`, `Caption`, `Shifted` |
| `anim/layouts.py` | the five core layouts, the frame compositor `Scene`, `Animation` (a picklable recipe) with `make()`, and the per-layout hooks (`EXTRAS`, `TIMELINES`, `PREVIEWS`, `DEFAULT_PHASES`, `DEFAULT_SPEED`, `BANNERS`, `STILLS`) |
| `anim/supplementary.py` | the supplementary-video layouts `v1_hinge`, `v2_intent`, `v3_release`, registered into `LAYOUTS` on import of `biomech.anim` |
| `anim/clipfile.py` | `save_clip`, `load_clip_file`, `clip_stem`: the clip files |
| `anim/render_video.py` | parallel chunked encoding, progress and cancel, clip files, `filmstrip_png`, `probe`, `decode`, `verify_sync` |
| `anim/render_plotly.py` | `preview_figure`: Plotly frames and a slider |
| `anim/render_latex.py` | `filmstrip` (pgfplots key frames) and `animate_frames` |
| `anim/timecode.py` | the hidden per-frame timecode (`stamp`, `read_timecode`) |

### Timeline

- `Timeline` maps video time to the master clock (`t_master_us`). The segments (phases or
  windows) play back to back at `speed` × real time. The video cuts between segments, so the gap
  between two phases is never shown.
- Frame *k* shows the instant at video time *k*/fps.
- `to_master` and `to_video` are exact inverses inside the segments, which are half-open.
- The on-screen clock shows seconds since the start of the current segment.
- `cohort` plays a pseudo-clock: one cycle (0–100 %) lasts 1/`speed` s and is looped (`loops`).

### Layers

Every layer works in three steps:

1. `prepare(t_us, seg)` computes the state of every frame once, vectorized through
   `biomech.timebase`: linear interpolation, sample-and-hold or nearest sample, and slerp for
   rotations. It never interpolates across a gap longer than `GAP_S` = 0.25 s; scrolling
   lines are broken at such gaps (`layers.break_gaps`), so no line crosses missing data, and
   coloured paths draw samples without a value in a neutral grey (`NAN_RGB`), never as the
   lowest ramp colour (the Plotly preview leaves them out).
2. `series(i)` returns that state as backend-neutral `Series`. The Plotly preview and the
   filmstrip are built from these, so every output shows the same numbers.
3. `mpl_init` and `mpl_update` create animated matplotlib artists once and then only set their
   data. This is the fast path for video.

| Layer | Shows |
|---|---|
| `Trail` | A moving landmark and its trail over exactly [t − τ, t]: the interpolated tail, the samples inside, and the interpolated head. Opacity fades with age. The incisor trail is coloured by incisal speed on the sequential ramp, oriented per theme so that faster means more contrast. `Marker` is a trail with τ = 0 |
| `Pose` | The mandible as a ghost outline (condyle – molar – incisor – molar – condyle) with a translucent dental arch, and the frame triad at the incisor. While \|ω\| ≥ 20 °/s it also shows the instantaneous helical axis q = r + ω×v/\|ω\|², taken through the point closest to the condylar midpoint and clipped to the panel box. In a plane, the axis appears as its pierce point through the midsagittal plane, the instantaneous centre of rotation (×). ω and v are the stored spatial `omega` and `incisor_vel` (an02). Older stores without them use the functions of `kinematics/derivatives.py` |
| `Scope` | Scrolling time series: the window [t + w₀, t + w₁] (default −3 s … +1 s) against time relative to now. Dots mark the current values on the dashed "now" line |
| `Cursor` | A time line on a full-window panel |
| `Ribbon` | Coloured rows (intent label, decoder decisions, tracking quality) scrolling with the window. The decoder rows are the window decisions of `ml_eval` (LOSO, primary model, EMG + IMU; arrays `pred__<pid>__emg_imu__t_us` / `__yhat`) as one-hot rows — decisions, not calibrated probabilities |
| `Image` | The camera preview held at or before t (previews arrive at about 8 Hz). It is blank when the latest frame is older than 0.6 s |
| `Triads` | Optical bodies (CRA, MTA, CT) in the camera frame, with name tags |
| `Bars` | Instantaneous activation per muscle |
| `Text`, `Chip` | Header: pseudonym, phase, clock, speed, and the current intent label as a coloured chip (`Chip(method="hold")`: the latest value at or before t, causal) |
| `Outline` | (supplementary videos) A moving polyline through K points: the fixed-hinge ghost of the mandible outline, or (K = 2) a vector such as the hinge residual, with a head marker |
| `Events` | (supplementary videos) Event markers in the scrolling window at their time relative to now; with a window ending at 0 an event appears only once it has happened |
| `Tally` | (supplementary videos) Live counters: running counts, last and median value of events up to now, counted over the played segments in play order |
| `Caption` | (supplementary videos) Burned-in captions: the cue `(start_s, end_s, text)` playing at the frame's video time, wrapped to two lines |
| `Shifted` | (supplementary videos) Any layer shown at master time t + offset (the right column of `v3_release`) |

Colours follow `viz/style.py`. Muscles and landmarks keep their entity colours, axes use
`axis:x/y/z`, and the speed ramp is `SEQUENTIAL`.

| Item | Colour |
|---|---|
| Intent labels | REST grey, OPEN orange (openers), CLOSE blue (closers), PROTRUSION green, RETRUSION vermillion, LEFT purple, RIGHT sky |
| Tracking quality | quiet when fine: 2 cameras faint grey, 1 camera yellow, lost `ink2` (high contrast) |

### Layouts

Missing modalities drop their panels, and the remaining panels re-flow into the space. For
example, without IMU the angle panel shows only the optical angle; without labels or optical
data there is no ribbon; without EMG the angle panel takes the whole right column. A layout
without any data to show raises `ValueError`.

| Layout | Regions |
|---|---|
| `multimodal` | 3-D mandible (ghost, trails, triad, helical axis); sagittal plane (rotation centre); scrolling closer and opener EMG (% of max recorded, D11), optical vs IMU opening angle, and intent-label and tracking-quality ribbons; optional camera views; key strip |
| `kinematics` | 3-D mandible; sagittal, frontal and horizontal planes on one mm-for-mm scale; opening angle with cursor |
| `tracking` | the optical scene in the camera frame (body triads, faint origin paths), tags detected, reprojection RMS, cameras per body; optional camera views |
| `emg` | activation raster (% of max recorded; sequential ramp, colorbar) and closer/opener envelopes with a sweeping cursor; instantaneous bars |
| `cohort` | every participant's cycle-normalized incisor paths in the sagittal plane (single cycles faint, participant mean, speed-coloured trail), in the grid that gives the largest mm-for-mm panels |
| `v1_hinge` | (supplementary video) "Not a hinge": 3-D mandible with the fixed-hinge ghost, the fixed axis, the residual arrow and the helical axis; sagittal plane (axis pierce point ◆); hinge residual over time; inset opening angle vs incisal displacement; captions |
| `v2_intent` | (supplementary video) "Intent cues in simulated real time": banner, closer envelope with the release threshold and decisions, opener activation, opening angle, label and stream-decision ribbon (past only), live counters; captions |
| `v3_release` | (supplementary video) Open–close vs chewing release behaviour, side by side in step, same axes: closer envelope, threshold, decisions, opening angle, running decision count per column; captions |
| `validation` | (systems paper; validation study only) JTS-MM against Qualisys from the stored `ref_kinematics` / `ref_fusion` arrays: the mandible of both systems as the triangle lower incisor – condyles in 3-D and in the sagittal and frontal planes (JTS-MM blue solid after the frame alignment, Qualisys dashed, incisor trails); scrolling opening rotation of both systems, incisor disagreement, rotation disagreement of JTS-MM optics and of the IMU alone (green), each in full on the frames the stored numbers use (evaluated tasks, reference not gated) and faint elsewhere; rows for gated and tool-in-view frames; captions; key |

Regions are separate specs, each fitted to its box on a 12.8 × 7.2 in design canvas. Stacked
regions share their axis columns (`align`), so the time axes of the right column line up.

## Rendering

- **Frames.** `Scene.frame(i)` restores each region's cached background, draws only the
  animated artists (blitting), and pastes the regions into one RGB array. It then stamps the
  timecode. A 1080p multimodal frame takes about 22 ms.
- **Video.** `render_video(anim, out, size=..., n_jobs=..., on_progress=..., cancel=...)`:
  - Splits the timeline into chunks of about 150 frames.
  - Renders the chunks in joblib (loky) workers. Each worker rebuilds its scene from the
    picklable `Animation` and pipes raw RGB into its own `ffmpeg` process. There is no PNG round
    trip, and memory does not grow with the video length.
  - Encodes every chunk with identical settings: H.264 `yuv420p`, CRF 18, `+faststart`, even
    sizes. The concat demuxer then joins the segments with `-c copy`.
  - Writes everything to a temporary folder next to the output and moves the file into place
    only when it is complete.
  - Writes a manifest `<video>.json` next to the video: recipe (layout, fps, speed, theme,
    window, trail), pseudonym, `with_camera`, modalities, segments, sampled `[frame, t_us]`
    pairs, and encoder settings; for the supplementary videos also `banner` and `captions` (if given), and the
    names of the clip files and the filmstrip.
  - Writes the clip files `<stem>.clip.npz/.json` (`clip_files=True`, default; *Clip files*)
    and, for the layouts in `STILLS` (the supplementary-video layouts; `stills=` overrides), a PNG filmstrip
    `<stem>_filmstrip.png`: six frames evenly inside the video, rendered by the same `Scene`
    at 640 px without the timecode, tiled 3 × 2 (`filmstrip_png`). Nothing is written when
    the render is cancelled.
- **Speed.** The synthetic 18-s multimodal clip (540 frames, 1080p) renders in about 8 s with
  8 workers.
- **Determinism.** Chunked and single-process renders produce identical frames. The tests compare
  raw-frame hashes and the decoded frames of lossless (`crf=0`) encodes.

### Hidden timecode

- Every video frame carries its master-clock time in a strip of 56 blocks in the bottom-right
  corner:
  - 8 sync bits;
  - 40 bits of µs;
  - an 8-bit checksum.
- The blocks are 4 px up to 1080 rows and 8 px at 4K. Bit 1 differs from the background by 80
  grey levels.
- `read_timecode(frame_rgb)` thresholds the block means between the two sync levels. It returns
  `None` on a sync or checksum mismatch.
- The code survives H.264 at CRF 18. `verify_sync(path, anim)` and `biomech check-outputs` (an12)
  read it back at sampled frames and compare it with the timeline.

## Supplementary videos

Three layouts in `anim/supplementary.py`, registered into `LAYOUTS` and rendered like every
other layout (CLI `--layout`, viewer *Videos* page). They reuse the core panels and layers;
the derived data are loaded once by `source.EXTRAS` (only for the layout that needs them,
`layouts.EXTRAS`) and travel in the clip (and its clip files). Each has burned-in captions
(default cues per layout, spread evenly over each segment's video time; `--captions` or
`make(captions=...)` replaces them), a caption band above the key strip, and a PNG filmstrip.
No exoskeleton or closed-loop device exists: decoder decisions are an offline replay of
recorded data (pseudo-online, ml08), never a device's actions.

**V1 `v1_hinge`, "not a hinge"** (default phase `open_close`).

- *Hinge* (`extra["hinge"]`): the participant's best-fit fixed hinge from stage `kinematics`,
  table `hinge`, model `hinge_fit` (C1): axis direction `axis_x/y/z` through its point at
  y = 0 (`axis_pierce_x_mm`, `axis_pierce_z_mm`). The ghost mandible starts from the closed
  reference pose `T_closed` (kinematics arrays) and turns about the fixed axis by the angle that
  best maps the five landmarks onto the measured ones at each sample
  (`kinematics.hinge_fit.best_angles`, the fit's own rule). Residual = measured − ghost at the
  incisor. The fit itself references each opening to its own closed start pose; the video uses
  one closed pose throughout, so its residual can exceed the table's RMS (stated in the
  caption, which also quotes the table's RMS and max).
- *Panels*: 3-D mandible (measured pose, condyle and incisor trails, instantaneous helical
  axis; no frame triad) with the ghost (yellow, dashed), the fixed axis (yellow) and the
  residual arrow (green, from ghost to measured incisor; drawn ×1, 2, 5, … when the residual
  would be invisible, the gain is in the key and caption); sagittal plane with the same
  overlays and the fixed axis's pierce point (◆); the residual magnitude over time; inset:
  opening angle vs incisal displacement (|incisor − closed incisor|), measured (trail) and
  fixed-hinge prediction (yellow). The jaw views widen to include the fixed axis when it lies
  within 80 mm of the jaw box; otherwise the caption gives its pierce point.

**V2 `v2_intent`, "intent cues in simulated real time"** (default phase `open_close`, speed 1).

- *Banner* "simulated real time, no device" under the header (bold, burned in; with
  "(played at 0.5×)" appended when the speed is not 1); also `banner` in the manifest.
- *Past only*: every scrolling panel shows [t − BEFORE, t] (`--window BEFORE AFTER`: AFTER is
  forced to 0); events appear when they happen; decoder decisions are held from their emission
  time on (`RibbonRow(method="hold")`, never the nearest sample).
- *Release* (`extra["release"]`): the causal closer-release detector (X3,
  `biomech.timing.release`) recomputed from the session exactly as the timing stage does:
  causally cleaned EMG (`EmgPrep.causal`), rest level of the same chain, group closer envelope E
  in dB re rest (`causal_db`), threshold = lagged baseline − `drop_db`
  (`source.release_threshold`, the baseline of `detect`), decisions `detect(...)` (▼). The test
  checks that `detect` decides exactly where E ≤ the drawn threshold has held for `hold_s`.
  Envelope and threshold are kept at 200 Hz for display.
- *Stream decoder* (`extra["online"]`): stage `ml_online` cohort arrays
  `online__<pid>__{t_us, t_end_us, y, yhat, yhat_raw, release_us}` (primary feature set, no
  calibration) and table `decisions` (same feature set, mode `none`: hit / false / other,
  latency). Ribbon rows: label (labels stage), stream decision (smoothed), stream raw; the
  stream's own release decisions are purple ticks (|) in the envelope panel.
- *Counters* (`Tally`, right column): stream decision now, label now, onsets detected, last and
  median decision latency so far, false activations, other decision changes, release decisions
  (X3 and stream), counted up to the current instant.
- Opener activation (emg stage, % of max recorded) and the opening angles (optical reference,
  IMU) are the offline signals; only the release chain and the stream decisions are causal.

**V3 `v3_release`, open–close vs chewing** (default phases `open_close`, `chewing`).

- Two phases side by side, played in step from each phase's start (`--t0/--t1` cut both):
  the timeline (`layouts.TIMELINES`) has one segment, the first phase, as long as the shorter
  phase; the right column's layers are `Shifted` by the offset between the phase starts. The
  hidden timecode carries the left (first) phase's master time; the header names both phases.
- Per column: closer envelope with the release threshold and decisions (X3, as in V2), opening
  angle, and the running count of release decisions (with the rate per minute after 5 s). Both
  columns share the y scales (limits from both phases) and the time window.

Previews and the LaTeX filmstrip use the layouts' main panels (`layouts.PREVIEWS`; film panel:
V1 sagittal plane, V2 and V3 the closer envelope).

## Clip files

`render_video` writes, next to the video, `<stem>.clip.npz` and `<stem>.clip.json`
(`anim/clipfile.py`):

- `.clip.npz`: every array the renderer reads after loading: the clip's modalities as cut
  for the video (`kin/…`, `maxilla/…`, `emg/…`, `imu/…`, `labels/…`, `posterior/…`,
  `optical/…`) and the supplementary-video modalities (`extra/hinge/…`, `extra/release/…`, `extra/online/…`);
  for `cohort` the paths, means and speeds. Compressed; no strings.
- `.clip.json`: the structure (a tree whose array leaves name their npz key), the playback
  settings (layout, fps, speed, theme, window, trail, loops, captions), the pseudonym (`P<n>`)
  and label, phases, segments, the cut window, the original render size and an array index
  (shape, dtype).
- `load_clip_file(path)` rebuilds the `Animation` from the two files alone;
  `biomech animate --from-clip` re-renders it with other settings. Re-rendered with the same
  settings it gives the identical frames (test).
- Privacy: no study id (the clip's `pid` is not written; `check-outputs` scans the clip JSON
  like every manifest), and never camera frames: a clip of a `--with-camera` video records
  `camera_frames: "omitted"` and re-renders without camera views.

## Viewer entry points

The viewer's *Videos* page (`/videos`, alias `/animation`; an09) uses two functions; `viewer/videos.py` runs the render.
The layout menu lists every registered layout (the supplementary videos included); choosing a layout sets its default
phases and speed (`default_phases`, `default_speed`); a finished render links the video and, for
the supplementary videos, the filmstrip; the clip files are written as from the CLI:

```python
from biomech.anim import make
from biomech.anim.render_plotly import preview_figure
from biomech.anim.render_video import Cancelled, render_video

anim = make(ctx, "s01", "multimodal", "chewing", t0_s=None, t1_s=None, speed=0.5, fps=30,
            theme="dark", with_camera=False)          # ctx: biomech.io.context.DataContext
fig = preview_figure(anim, fps=10, max_frames=150)     # go.Figure: frames, ▶/❚❚, slider

def job(progress, stop):                                # inside viewer/videos.py VideoJob
    try:
        res = render_video(anim, out_dir / "videos" / "x.mp4", size="1080p", n_jobs=None,
                           on_progress=lambda done, total: progress(done / total),
                           cancel=stop.is_set)
    except Cancelled:
        ...                                             # no partial file exists
```

- **`make`.** `make(ctx, pid, layout, phase, t0_s, t1_s, *, speed, fps, theme, with_camera,
  window_s, trail_s, task, loops, captions) -> Animation` loads the data once (`phase=None`,
  `speed=None`: the layout's defaults).
  - `Animation` is a picklable recipe.
  - `anim.timeline` gives the frame count and duration.
  - `anim.manifest()` describes the recipe.
  - For `cohort`, pass `pid=None` and `task=...`.
- **`preview_figure`.** `preview_figure(anim, *, theme=None, fps=10, max_frames=150,
  width_in=11, height_px=None)` builds the compact single-spec form of the layout
  (`layouts.preview`).
  - Scrolling panels become full-window panels with a cursor.
  - One frame per step carries only the changing trace data.
  - The frame title shows the pseudonym, phase and clock.
  - `fig.layout.meta["frame_times_us"]` maps frames to the master clock.
  - It takes under 1 s for 150 frames.
- **`render_video`.** `render_video(anim, out, *, size, n_jobs, crf, chunk_frames, on_progress,
  cancel, stamp, frame_hashes, manifest, clip_files, stills) -> VideoResult` (also
  `.clip_files`, `.filmstrip`).
  - `on_progress(done_frames, total_frames)` is called from the calling thread.
  - `cancel()` is polled every 0.2 s while workers run, and every frame in-process.
  - When `cancel()` returns True, the workers stop within a few frames, the temporary files are
    removed, and `Cancelled` is raised.
  - `n_jobs=1` renders in the calling thread and reports progress per frame.

## LaTeX

- **`filmstrip(anim, out_dir, n=6)`.** Writes `figures/film_<layout>/fig.tex`, the style file,
  and `standalone_<id>.tex`.
  - It shows `n` key frames of the layout's film panel: the sagittal plane for `multimodal` and
    `kinematics`, the optical scene for `tracking`, the closer envelope for `emg`, and a
    participant's path for `cohort`.
  - Each key frame carries its layers' state: the trail up to t, the mandible outline, the
    rotation centre and the cursor.
  - Five or more key frames wrap into two rows.
  - Everything is vector graphics, via `viz.render_pgf`. The paper's size test passes: it fits
    7.16 in, with no overfull boxes or pgfplots warnings.
  - With camera previews in the clip, a row of the held camera frames is added (PNG).
- **`animate_frames(anim, out_dir, fps=10, size="720p", theme="light")`.** Writes
  `frames/frame-<k>.png`, without the timecode strip, and `animate.tex`:
  `\animategraphics[controls,loop,...]{fps}{frames/frame-}{0}{N-1}`, which needs
  `\usepackage{animate}`. `animate_demo(out_dir)` writes a compilable wrapper. The `animate`
  package is not part of every TeX install; install it before compiling.

## Privacy

- **Camera frames.** They show the participant, so camera views are off by default.
  `--with-camera` (`with_camera=True`) is the only way they are loaded. The manifest records
  `with_camera`, and `check-outputs` checks that the flag is present.
- **Identifiers.** Only the pseudonym appears on screen: "Participant 3" in the header, "P3" in
  the cohort grid, from `DataContext.pseudonym`. pid strings, age and sex never appear, neither
  on screen nor in the file name, the manifest, the clip files, the filmstrip or a caption.
- **Output location.** Videos go to `data/analysis/<study>/videos/`, which is git-ignored. Do not
  render pilot data into tracked folders.

## Tests

`software/biomech/tests/test_anim.py` covers:

- **Timeline.** The video and master-clock maps are inverses, including cuts and chunks.
- **Timecode.** Round trip under noise; a flipped bit is rejected.
- **Layer state.** Equals the directly interpolated data:
  - trail head = `resample(path, t)`;
  - the trail covers exactly [t − τ, t];
  - pose outline and slerped rotations;
  - helical axis of a pure hinge;
  - scope window;
  - ribbon values.
- **Marker position.** The head marker's position equals the data.
- **Short render.** 2 s at 10 fps, 320 px. ffprobe confirms 20 frames, 2.0 s, h264 `yuv420p`,
  320 × 180 (even), and the timecode matches the timeline at every sampled frame.
- **Chunked vs single-process.** Identical frame hashes and identical decoded frames
  (`crf=0`).
- **Cancel.** In-process and in workers: `Cancelled`, no output, no temporary leftovers.
- **Missing modalities.** Dropped panels re-flow, and the layout still renders.
- **Other outputs.** The other layouts, the cohort, the Plotly frames, the filmstrip
  (LuaLaTeX, size test), the `animate` frames, and the CLI.

`software/biomech/tests/test_r4_anim_clip.py` (on the `synth_study` fixture):

- a short render writes the clip files; re-rendered from them with the same settings the frame
  hashes are identical; copied to a folder far from the study, `animate --from-clip --theme
  light` (no `--config`) gives the same frame count, master-clock timeline and readable
  timecode; no study id in the clip JSON, the npz keys or any file name;
- camera previews are never stored (`camera_frames: "omitted"`); cohort clips round-trip;
  the supplementary-video clips reproduce their frames exactly; the CLI layout choices equal `LAYOUTS`.

`software/biomech/tests/test_r4_anim_supplementary.py`: each layout renders with changing
captions, a readable timecode and the pseudonym only; V2 carries the banner, plays at 1× and
shows only the past (every scrolling layer ends at 0 s, stream decisions held from emission);
the drawn release threshold reproduces `detect` exactly; the hinge ghost keeps every landmark's
distance to the fixed axis and the residual is measured − ghost; V3 plays the second phase in
step (`Shifted` by the phase offset); `Tally` counts in play order; custom captions; the PNG
filmstrip size; Plotly previews; a missing derived modality is named; the CLI writes video,
manifest (banner), clip files and filmstrip.

## File names and changes

- **File names** (`layouts.video_stem`, CLI and viewer) carry the pseudonym, never the study
  id (`multimodal_P1_open_close+chewing`, `_camera` with camera frames; `cohort_<task>`), and
  every setting that changes the frames when it differs from the default: window
  (`_t<t0>-<t1>`), speed (`_1x`; default 0.5), theme (`_light`), size (`_720p`; default
  1080p), scroll window and trail (`_w<before>-<after>_tr<trail>`, e.g. `_w5-1_tr1`, written
  when either differs from 3 s / 1 s / 1 s). Two renders with different settings never
  overwrite each other. Manifests record the pseudonym only.
- **Scroll window and trail** are settable in the viewer (window before/after the cursor,
  trail length) and in the CLI (`--window BEFORE AFTER`, `--trail S`).
- **Still frame** (viewer): one frame rendered by the same `Scene` as `render_video` at a
  chosen video time, i.e. exactly what the file will show; the Plotly preview stays the fast,
  downsampled scrubber.
- **Gaps:** scrolling lines are broken wherever consecutive samples are more than `GAP_S`
  (0.25 s) apart (`layers.break_gaps`), so no line is drawn across missing data; coloured
  paths draw samples without a value in a neutral grey (`NAN_RGB`), never as the lowest ramp
  colour, and the Plotly preview leaves them out.
- **Decoder overlay:** the decoder's window decisions (LOSO, primary model, EMG + IMU) from
  `ml_eval` (`pred__<pid>__emg_imu__*`) are one row "Decoded" coloured by the decided class
  (formerly seven one-hot rows labelled "P(class)", which read as probabilities). The
  ribbon legend lists the classes of the label and decoder rows.
- **Loading:** a modality that cannot be loaded is left out with a logged warning (the layout
  re-flows); errors are no longer swallowed silently.
- Labels: EMG panels read "% of max recorded" (D11).

## Revision of 2026-10-23

- **Systems-paper layout `validation`** (`anim/validation_video.py`, registered on import;
  `biomech animate --config experiments/jtsmm_validation/study.yaml --layout validation
  --pid s10 --phase open_close`): see the layout table. Stored results only; the loader
  (`source.EXTRAS["validation"]`) cuts the arrays to the clip and travels with the clip files.
- **Reader text:** layout names in the viewer (`vocab.yaml` domain `video_layout`); tracking
  rows named by body ("Head piece", "Mouth piece", "Tool"); units in the panel titles of the
  EMG and angle panels (the rotated y labels overflowed the short panels).
- **IMU angle** dashed in the sky hue of the IMU feature set (`anim:imu`), next to the solid
  optical angle; before, both were blue solid lines.
- **Subtitles:** every video with burned-in captions also gets `<stem>.vtt` (WebVTT, the same
  cues; `render_video.write_vtt`), listed in the manifest.
- **Manifest:** records the store the clip was read from (`snapshot`: the frozen tag or
  "live").
- **Viewer:** the video theme is explicit (dark by default, as the CLI); the page theme no
  longer changes the rendered video.

## Video layouts of J2 (2026-11-02)

Three layouts in `anim/r8_videos.py` (change-log entry J2), registered on import; each has burned-in subtitles
(and the `.vtt`) and a PNG filmstrip at 1280 px per frame (`layouts.STILL_WIDTH`), usable as
a paper figure when rendered with `--theme light`. Every number on screen comes from the
stored results, the configuration or a named display constant.

| Layout | Shows |
|---|---|
| `workspace3d` | All participants (no `--pid`; a cohort layout, `layouts.COHORT_LAYOUTS`): every incisal position of the five tasks as displacement from rest (`kin_r8.incisor_cloud`, thinned to `WS_POINTS` per participant; colour and marker per participant), the convex hull of all positions as a cage (`kin_r8.hull_cage`, volume) and the design box (`kin_r8.design_box`: backward by `posterior_excursion_mm` when stored, else from rest), turning once in `ROT_S` s (`--speed` scales it) on an equal-mm cube; right: participants with their share outside the box, and the stored design values |
| `emg_spectrogram` | One participant and task (default chewing): band-passed EMG of the six channels (one row each, scaled to its P99.5, min/max-decimated to 1 ms bins), closer and opener envelopes, opening angle; right: a scrolling short-time spectrum per muscle group (window and overlap of `emg_r8`, channel PSDs averaged in the group, masked windows grey) and the group spectra at the current instant |
| `onset_rules` | One participant, open–close: up to `ONSET_CYCLES` consecutive cycles (the run with most usable cycles), one segment per cycle (the onset search window ± `ONSET_MARGIN_S`); the digastric (the one with more usable cycles) band-passed EMG and TKEO re its rest threshold (recomputed as in `emg_r8` *chain*), the mechanical onset at 0, the search window shaded, the stored first-onset (▼) and driving-burst (◆) onsets appearing as the cursor passes them; right: per rule this cycle's lead and usability and the running count and median lead; one subtitle line per cycle from the stored onsets |

New layers: `Spin` (a 3-D panel turning; the axes are redrawn every frame), `ScrollImage`
(scrolling image strip), `Profile` (curves over a fixed x at the current instant),
`SegmentTrace` / `SegmentEvents` (one window per segment, revealed by the cursor; events
appear once passed). The header progress bar now follows the frame's segment, so segments may
overlap in master time.

**Videos as paper items.** A paper set lists its videos under `supplementary_videos:`
(`{layout, pid?, phase?, speed?, theme?, label}`; `biomech.papers`). `biomech check-outputs`
matches each entry with its render (stem of `biomech animate`, any size) in the videos folder
of the set's store and of the study: not rendered → warn; rendered from another snapshot or
variant than the set pins, an unknown layout or a participant not in the store → fail. The
gallery lists the entries first, with their render and status.

