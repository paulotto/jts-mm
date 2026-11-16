<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an09 — Analysis viewer

```bash
pip install -e software/biomech[viewer]
biomech view --config experiments/icorr2027/study.yaml        # http://127.0.0.1:8060
biomech view --config ... --verbose                            # keep the per-request log
```

It is a local, single-user tool (Dash + Plotly). It binds to 127.0.0.1 by default. **The data
Signals page shows subject data**, so do not expose it on a network. Participants appear
by pseudonym everywhere (`P3`, `viewer/data.py::pid_label`); only the Overview's operator
toggle adds the study ids (`unit_label`). Figures, titles and exports carry the pseudonym
alone, which the privacy check in `check/outputs.py` enforces. The werkzeug
access log (one line per request) is silenced unless `--verbose` is given
(`viewer/app.py::quiet_logging`).

| Page | What it does |
|---|---|
| **Overview** (`/`) | Study info (config and its hash, frozen-plan tag, cohort size, result folder; paths relative to the repository). Stage × participant status against the current configuration (up-to-date / stale / missing / failed), participants by pseudonym; *show study ids (operator)* adds the session ids, also in the run log (local use only). **Run** starts `biomech.pipeline.run` in the background (stages, participants, force, workers); **Cancel** stops after the current unit; progress and a log tail come from the run events. The page polls only while a run is in progress. |
| **Signals** (`/signals`; `/explorer` is an alias) | **Both systems** tab (the validation study only, its default tab; `viewer/validation_signals.py`): the stored `ref_kinematics` and `ref_fusion` arrays of one task — lower incisor x, y, z and the opening rotation of JTS-MM (blue, solid) and Qualisys (dashed) in their own frames or with the frames aligned (leave-one-task-out alignment, LOTO), the incisor and rotation disagreement of JTS-MM optics and of the IMU alone (green), drawn in full on the frames the stored numbers use (evaluated tasks, reference not gated; the tables' n) and faint elsewhere, and bands where the calibration tool is in view. Phase names are reader names. Built figures are cached per inputs and store state. The phase selector takes several phases and "all motions" (every motion task). The time series, the camera frames and the single-phase atlas views use the first selected phase; the note beside the selector names it. *Signals* tab: one row per quantity, each with its own unit — EMG (raw / cleaned / envelope in % of max recorded; min/max-decimated), **masks** (envelope mode; one line per channel, marks coloured and shaped by the rule: lead-off, rail and envelope are *masked*; the low-frequency rule is *flagged (kept)* since D14; "dilation margin" = masked only by the ±`dilate_s` widening), opening angle (optical and three IMU estimators), incisal xyz relative to the phase median, ω (°/s), relative \|a\| (m/s²) and the intent label. Spike lines follow the cursor on every row; hover shows the value with its unit. *Overlays* (`data.OVERLAYS`): mechanical opening (orange) / closing (blue) onsets (`timing.onsets`) and the causal release decisions (X3, `timing.release`) as dotted full-height lines across all rows; the decoder's window-level output (LOSO, primary model; EMG + IMU and IMU, the `ml_eval` arrays `pred__<pid>__<feature set>__*`) as markers in the label row. *Show unmasked envelope (faint)* adds the envelope without the mask; outside envelope mode the option is disabled and says "envelope mode only". The signal figure sits in a 980-px box that can be dragged taller or shorter. Clicking a time shows the nearest stored camera preview frames and marks the moving landmarks at that time in the 3-D view (the cursor trace is patched in, the 3-D figure is not rebuilt). The participant, phases and signal mode are kept in the URL (`/signals?p=P3&phase=chewing&mode=clean`; the participant as its pseudonym), so a link reopens the same state. A 3D view (its own callback, so a camera change does not rebuild the time series; orthographic camera presets `data.CAMERAS`: *oblique, sagittal, frontal, horizontal*) shows the paths of the lower incisor, molars and skin-palpated condyles, plus the static upper landmarks; several phases are overlaid in task colours. The divider between the columns can be dragged. A landmark table lists all eight landmarks. *Atlas* tab: every registered `data` view (an10). |
| **Results** (`/results`; the old `/analysis` is an alias) | *Views*: every result view (paper items and dashboards, ≈ 100; the `data` views are in the explorer's Atlas only), grouped by domain (Paper, Kinematics, EMG — time / frequency domain, IMU, Timing and intent cues, Uncertainty, Variability and sensitivity, Quality control, Intent decoding); the domain filter and the searchable view list select one. Controls appear only when the view understands them: perspective, statistic, participant, **phase**, alignment and the view's own **pickers** (e.g. SPM test, QC column, decoder configuration). The selection is kept in the URL (`/results?view=emg_psd&p=P3&phase=chewing`; the participant as its pseudonym, a legacy `pid=` link still resolves), written without a page reload (`dcc.Location(refresh="callback-nav")`); the page reads it when it is built, so a link reopens the same figure. Figures are cached per view, parameters and latest run (48 entries); the zoom survives theme and window changes (`uirevision` = the view and its parameters) and is reset when the parameters change. Build errors show as a card with details. Under each figure the structured caption and a **Sources** drawer: the stored tables it is drawn from, the store read (snapshot or live) and the paper items it appears as; views drawn by a paper item are marked "· paper" in the list. A view with depictions shows a *Depiction* picker (an08 §4). **Export → LaTeX** writes TikZ with the current zoom (`apply_zoom`) to `<out_dir>/viewer_exports` (a paper folder holds exactly its paper set); **PDF** / **PNG** write a matplotlib render to its `previews/`. *Tables*: any stored table (stage, table, participants) with sorting, filtering and CSV download, plus a quick strip plot of one numeric column by participant / task / muscle. |
| **Paper** (`/paper`; `pages/paper.py`) | The study's paper set (`paper_*.yaml`; several sets: a selector) for one target, in storyboard order under its section headers (`{section: …}` entries). Filters: target, section (one at a time by default, so only that section is built; or all), part (core / supporting / supplement). Each card: the view title with its parameters in reader words (P1, open-close, frames, depiction), a part badge and the LaTeX name (`\\figcaption{label}`), the figure or table with its caption and sources. **Gallery** mode: every depiction an item offers side by side; the one in the paper set is marked, **Use this** (after a confirmation) writes `form: …` into the paper set (`papers.set_item_param`: one line, comments kept, the label pinned, re-read and verified). **Export the paper set** is `biomech export --paper … --target …` (manifest, captions, numbers; optionally `preview.pdf` and the size check); **Export every view** is `biomech export --all` (an08 §4). Both run in the background, one at a time, with progress and cancel; the page polls only while one runs. |
| **Videos** (`/videos`; `/animation` is an alias) | Layout by reader name (multimodal, kinematics, tracking, EMG, cohort, the supplementary videos 1–3, the systems-paper video `validation`), participant, phases (played in order), optional t0/t1, speed (0.5) and fps (30), the scrolling window before/after the cursor (3 s / 1 s) and the landmark trail (1 s), size (720p default here), camera frames, and the video's own theme (dark by default, as `biomech animate`; the page theme does not change the video). **Preview** plays Plotly frames in the browser; **Still frame** renders one video frame at the given time exactly as in the file (same scene as the renderer); **Render video** writes the full-quality MP4 in the background (progress, **Cancel** leaves no partial file) to `<out_dir>/videos/`, named as by `biomech animate`, and links it (and its filmstrip) under `/video-files/`. The page polls only while a render runs. Camera frames show the participant and are off unless ticked (an11). |

**Implementation** (`biomech/viewer/`):
- `context.py`: shared state.
- `runs.py`: the run thread and events.
- `videos.py`: the video render thread (`VideoJob`: one render at a time, frame progress, cancel).
- `jobs.py`: background jobs for long exports of the Paper page (`Job`: one at a time,
  progress, cancel; a second start while one runs is refused).
- `validation_signals.py`: the *Both systems* tab of the Signals page (validation study).
- `app.py`: the page routes (`PAGES`) and the aliases `/explorer`, `/analysis`, `/animation`.
- `data.py`: explorer figures (one legend per row), frames, 3D — all through `DataContext`
  (cached EMG preparations closed on eviction; MVC phases read from the MVC take).
  A row with nothing to show says why instead of staying blank:
  - an MVC take (EMG only);
  - a stage the study does not run (the validation study has no `imu`, `kinematics`,
    `labels`; its signals are under Results → Validation);
  - missing results of one participant;
  - a flat EMG recording (the JTS-MM EMG of the comparison takes is constant zero: only the
    Delsys electrodes were applied);
  - EMG masked in every sample.
- `pages/*.py`: layout plus callbacks per page (`control` = Overview, `explorer` = Signals,
  `analysis` = Results, `paper` = Paper, `animation` = Videos).
- **Page environment** (`components.env_store`): the shell's theme and window width are
  copied into a store on each page by one pattern-matching callback, and page callbacks
  take only inputs of their own page. A page callback with a shell input fired while another
  page was shown, and Dash reported the page's components as nonexistent;
  `test_r7_viewer.py` checks that no callback mixes them.
- `assets/style.css`: light and dark tokens (the same as `viz/style.py` THEMES, checked by a
  test; figures sit on the card colour in both themes); the theme follows
  `prefers-color-scheme` unless overridden in the navigation bar, and figures are re-rendered
  with the matching template.
- `components.py`: `graph_box`, a `.graph-box` around a `dcc.Graph(responsive=True)` at 100 %
  height. The box keeps the figure's aspect ratio (CSS `aspect-ratio`) or a fixed height, and
  can be dragged taller or shorter (`resize: vertical`).
- `assets/resize.js`:
  - a ResizeObserver on every graph container, including graphs added later (found through a
    MutationObserver), calls `Plotly.Plots.resize` (debounced) when a box, the window or the
    column divider changes size;
  - it pushes the window width into the `viewport` store (debounced, ≥ 24 px change), so the
    figures are laid out again for the new width;
  - it drives the explorer's column divider.

**Sizing.** Results, paper board, atlas and animation preview lay the figure out at the card
width (`render_plotly.container_width_in`: the window width, capped by the 1500-px page). The
height comes from the paper's aspect ratio: `to_plotly(fit=True)` scales the spec's row height
so the layout has the height/width of its IEEE-width paper layout, clamped to 240–1100 px
(`fit_height`), and records width/height in `layout.meta["aspect"]` for the box. Equal-aspect
(mm-for-mm) and 3-D panels, such as `data_orthographic` and `data_jaw3d`, are therefore no
longer flattened to a strip on a wide screen. Multi-column figures stack into one column when
the card is narrower than 6.5 in (window below ~690 px). The explorer's signal figure fills a
980-px box (min 420 px), the side 3-D view a square box; both can be resized by dragging.

**Plotly rendering** (`viz/render_plotly.py`): hover templates carry the axis titles and units;
heatmaps show row · column · value with unit and a colourbar (sequential or diverging ramp,
zero-centred) with the unit; cell values switch to light ink on dark cells; reference lines and
spans are drawn in axis-domain units, so they stay full length after zooming; bands are drawn
per finite stretch, so they never bridge gaps; the dark theme uses the dark colourway; panels
whose legend policy is "none" never feed Plotly's default legend.

Figures render through `viz/render_plotly.py`; `apply_zoom` maps `relayoutData` back onto the
spec.
