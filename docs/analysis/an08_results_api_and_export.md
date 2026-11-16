<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an08 — Results API, run API and export

These contracts let the `biomech` command line (CLI), the paper build and the viewer share one
implementation: the run API executes the stages, the ResultStore is the only read and write
path of their outputs, and the export turns registered views into figures, tables and numbers.

## 1. Run API — `biomech.pipeline`

```python
from biomech.config import load_config
from biomech.pipeline import run, stage_status

cfg = load_config("experiments/icorr2027/study.yaml")
summary = run(cfg, stages=["qc_cohort"], pids=None, n_jobs=6,
              on_event=print, cancel=threading.Event(), scheduler="participant")
stage_status(cfg)          # DataFrame: stage, unit, status (missing|missing-input|failed|stale|up-to-date)
```

**Registering stages:**
- A stage is registered with `@stage(name, scope="participant"|"cohort", depends=..., version=...)`
  in `biomech/stages.py` (dataset stages), `imu/stage_camerafree.py` (camera-free),
  `validation/stages*.py` (`group="validation"`) or a plugin package (`biomech.stages` entry
  point, e.g. `software/ml`). A study runs the groups listed in its `pipeline.groups`;
  `biomech stages` lists every stage with its group and dependencies.
- It receives a context (`params`, `participant`/`participants`, `upstream()`) and returns a
  `StageResult(tables, arrays, extra)`.
- It never writes files itself. **Bump `version` whenever a code change alters results.**
- Sessions and the EMG preparation come from the context: `ctx.session("motion"|"mvc")`
  (open `SessionData`, closed by the run), `ctx.emg_prep()` (the participant's `EmgPrep` under
  the run's `signals.emg`) and `ctx.memo(key, make)` (any other pure per-participant value,
  e.g. the pose chain `jaw_motion:<KinParams>`). Within one participant's chain they are built
  once (`UnitCache`), so `qc`, `emg`, `timing` and `ml_windows` clean the EMG (mains fit) once
  instead of four times.
- Random draws use `ctx.seed` (participant stages: the study seed combined with the participant
  id, `pipeline.participant_seed`, so participants never share draws) or `ctx.config.seed`
  (cohort stages). The study seed is part of every cache key.
- Cohort stages see `ctx.participants` (the participants they may use, see *Failure handling*),
  `ctx.excluded` (pid → reason) and `ctx.cancelled()` (poll it between long steps; raise to
  stop).

**Scheduling** (`scheduler=`):
- `"participant"` (default): participant stages are grouped into *waves* (a wave holds the
  participant stages whose cohort-stage inputs lie in earlier waves; with the built-in stages
  there is one wave). Per wave, each participant's whole chain is one task (`-j N` spawned
  worker processes; joblib's threading backend gives threads, used by tests; BLAS pinned to
  one thread per worker), with one `UnitCache`. A cohort stage starts in a thread once the
  participant chains of its inputs have returned, so independent cohort stages (`stats_*`,
  `validation`, `variability`) overlap with each other and with `ml_eval`.
- `"stage"`: the earlier order, one stage at a time across participants.
- Both give the same store: `tests/test_scheduler_equality.py` runs the synthetic study both
  ways and compares every table and array (wall-clock columns excepted). Permutation tests
  that use numpy's global RNG (spm1d) run under a lock with a fixed seed
  (`stats.waveforms.seeded_global_rng`), so concurrent cohort stages stay deterministic.

**Cache key** of a unit is a hash of:
- the stage name and version,
- the **code hash**: every analysis source file of biomech (by exact relative path; `viz/`,
  `viewer/`, `results/`, `anim/`, `check/`, `cli.py`, `pipeline.py`, `catalog.py`,
  `config.py` and the read-side helpers `io/context.py`, `ensembles.py`, `report.py`,
  `sensitivity.py` are presentation/orchestration and excluded, so editing a view never
  invalidates analyses), its package YAML (e.g. `norms.yaml`), the jtsmm signal packages
  (`config`, `emg`, `imu`, `io`) plus the single jtsmm files the analysis imports from the
  viewer/tools packages (`pipeline.JTSMM_FILES`, e.g. `viewer/processing.py`), and plugin
  packages except their `cli`, `labeling` and `training` subpackages (`.py`, `.yaml`, `.yml`,
  `.json` files; `testing/synth.py` and `norms/` are included, so editing the synthetic
  generator also invalidates the caches; `etc/catalog.yaml` is not),
- the study **seed**,
- the stage's parameter block **and the shared `signals` block**,
- the participant record,
- the input files' size and mtime,
- the *expected* keys of every upstream unit.

Changing anything upstream therefore marks all dependants `stale` before anything re-runs.

**Failure handling:** a failure is stored with its traceback (an unwritable result counts as a
failure of that unit), other units continue, and participant dependants are skipped; the skip is
stored as a unit with status `skipped` and the reason. A dependant is also skipped when an
upstream unit exists but is stale (its key differs from the expected one).
**Cohort stages** run on the included participants whose participant-scope upstream is current
(`KeyResolver.coverage`). A participant whose upstream *failed or was skipped* under the current
key, or whose input session is missing, is left out with its reason; a participant whose upstream
is merely stale or not computed yet **blocks** the stage (a statistic is never computed silently
from a subset because some participants were not run). Every cohort unit stores a `coverage`
table (pid, used, reason; view `pipeline_coverage`), and its key includes the units used, so it
re-runs when a left-out participant recovers.
**Cancellation** is checked about every second: tasks that have not started are dropped,
running participant chains and cohort stages finish (no worker is killed in the middle of a
write; long cohort stages such as `ml_eval`/`ml_online` poll `ctx.cancelled()` between
configurations and stop), and every unit that did not run is reported as `cancelled`.

**Store integrity:** a unit is replaced by rename-aside → rename-in → delete, so readers never see
a gap; dot-prefixed leftovers of a killed process are invisible and removed by `biomech gc`, which
first restores a unit that was moved aside but not replaced (process killed between the two
renames). One run at a time may write a store (`.run.lock`; a dead owner's lock is taken over).

**Snapshots** (`results/snapshot.py`): `biomech snapshot NAME [--note …]` freezes the live store
into `<out>/snapshots/NAME/` — `cache/` copied (not hard-linked; a snapshot never changes), `snapshot.json` (git state, code hash, versions, resolved config,
run ids, unit status at freeze time), `schema.json` (data dictionary), and copies of the study
YAML, cohort manifest and catalog. `view`, `export`, `check-outputs`, `report`, `animate` and
`describe` take `--snapshot NAME` (the `DataContext` reads the snapshot; outputs still go under
`<out>`); a paper item list may pin `snapshot: NAME`. `biomech compare OLD [NEW]` lines up every
cohort table of two snapshots (`live` = the live store) value by value (as sensitivity runs) and
writes a CSV — the source of the old → new values of errata and deviations.

**Reporting population** (D35): a paper set may name `variant: reporting_population`; it then
reads that sensitivity variant's store, and its `snapshot:` is a snapshot *of that store*
(`biomech snapshot --variant reporting_population NAME` → `<out>/sensitivity/reporting_population/snapshots/NAME/`,
with the filtered cohort manifest, the variant's own `comparison.parquet` and the primary's
comparisons; `--out-dir` pointing at the variant folder would freeze the primary manifest).
`sensitivity.store_config(cfg, variant, snapshot)` is the one resolver: `export` (and
`--gallery`), `numbers` (`--dry-run`: print only), `verify-export` (variant and snapshot from the
manifest's provenance), `check-outputs` (every paper-set check reads the set's store,
`papers.paper_context`) and `snapshot` go through it; `--variant NAME|primary` and `--snapshot`
override a paper set's choice. Exports record the variant in the manifest's provenance, and
`check-paper` fails when a manuscript's exports come from different populations.

**Data dictionary** (`results/schema.py`): `biomech describe [STAGE [TABLE|ARRAY]]` lists every
stored table column (label, unit, description from the catalog; uncatalogued columns get label
and unit from their name) and every array family (unit, shape, description from
`schema.ARRAYS`; time stamps in µs, positions in the maxillary frame). A test fails on an array
family without an entry.

**Events:** each run writes `runs/<run_id>/events.jsonl`: `run_start`, `stage_start`,
`unit_done`, `unit_skipped`, `stage_end`, then `run_end` or `cancelled`. These drive the progress
display. `runs/<run_id>/manifest.json` records the git SHA (and dirty flag), config hash and
source, the cohort, the stages, package versions and the final statuses.

**Sensitivity runs** (`biomech/sensitivity.py`): a named set of dotted config overrides in the
`sensitivity` block of `biomech.yaml` (or the study file), e.g.

```yaml
sensitivity:
  lowfreq_mask:
    description: "D14 sensitivity: the low-frequency rule excludes samples"
    set:
      signals.emg.mask.analysis_rules: [rail, leadoff, envelope, lowfreq]
```

`biomech sensitivity lowfreq_mask -j 12` runs the whole pipeline with the overrides into
`<out_dir>/sensitivity/lowfreq_mask/` (own cache, seeded with copies of the primary's
units, `sensitivity.seed_cache`: keys are content addressed, so only the stages whose inputs
the overrides change re-run -- `c2_muscle_mean` takes seconds; the primary store is untouched; without
NAME the variants are listed, `--compare-only` only rewrites the comparison) and writes
`comparison.parquet`: every cohort table that finished in both stores, lined up value by value
(`stage, table, key, column, primary, variant, diff, rel_diff`, `rel_diff = diff / |primary|`).
The key (`a=1|b=2`) is built from the text columns whose value *sets* are identical in both
runs (task, muscle, test, rule, …; a text column that changed, e.g. a cluster list, is
dropped) plus the numeric parameter columns in `sensitivity.PARAM_COLUMNS` (`drop_db`,
`hold_s`, `primary`, `k`, `w_emg`, `cycle`, `window_s`, `detector_k`, `seed`, `fold`); a
repeated key gets its position appended (`|#n`). Every other shared numeric or boolean column
is a value. The Results page shows it (`sensitivity_changes`, `sensitivity_table`, an13 §4).
`load_config(..., overrides={"a.b": v})` applies the same overrides programmatically, with
the same unknown-key and type checks as a study file.

**Skipping stages (D22):** `skip_stages: [ml_windows]` in a variant leaves that stage and
every stage downstream of it (`sensitivity.skipped_stages`; here `ml_eval`, `ml_online`) out
of the variant's run and out of its comparison, even when the variant's store holds older
units of them. The comparison records the skipped stages (`sensitivity.not_run(df)`, kept in
the parquet metadata), so a view can show "not re-run" instead of a missing row. A study file
may add `skip_stages` to a package variant (`sensitivity.<name>.skip_stages`).

**Cancelling a run:** Ctrl-C in `biomech run` sets the pipeline's cancel event: running units
finish (cohort stages such as `ml_eval` check it between configurations and folds), no new
unit starts, an interrupted cohort unit is recorded as failed ("cancelled") and never stored
half-written; the exit code is 130. A second Ctrl-C exits at once (store writes are atomic;
`biomech gc` restores leftovers).

**Dataset release** (`results/release.py`): `biomech export-data --to DIR [--format csv|hdf5]
[--arrays] [--snapshot NAME]` writes every stored table of the included participants and the
cohort with pseudonyms (`P<n>`, the figure numbering) instead of study ids (participants
outside the cohort dropped, other ids in text cells become "excluded"), never per-person
demographics (cohort aggregates are kept), plus `data_dictionary.csv` (label, unit, decimals,
description per column; with `--arrays`, per array family) and `release.json` (provenance,
file list). `csv`: `<stage>/<table>.csv`; `hdf5`: `results.h5` with
`/<stage>/<table>/<column>` (attributes label, unit, description) and, with `--arrays`,
`/arrays/<stage>/<P<n>|cohort>/<key>`. The release is scanned for study ids and refused if
one is left.

**Timing** (`biomech bench`, `results/bench.py:run_timing`): per stage the units run
(`n_units`, `n_failed`), summed and maximal unit time (`sum_s`, `max_s`), the stage's wall
span (first `stage_start` to last `unit_done`, `span_s`) and its share of the summed unit time
(`share`), sorted by `sum_s`, plus the run's wall time; from the event log of a run (default:
the latest), also written to `<out_dir>/bench/<run_id>.csv`.

## 2. ResultStore — `biomech.results`

The only read and write path for outputs:

```
data/analysis/<study>/cache/<stage>/<unit>/{meta.json, <table>.parquet, arrays.npz}
data/analysis/<study>/runs/<run_id>/{manifest.json, events.jsonl}
```

- `read_table(stage, name, units=None)` concatenates over units and adds a `unit` column.
- `read_arrays(stage, unit)`.
- `status(stage, unit, expected_key)`, `meta(...)`, `units(stage)`, `stages()`, `runs()`,
  `run_manifest()`, `run_events()`.
- **Read tracking** (`results/access.py`): inside `with track() as log:` every table read
  lands in `log.tables` as `(stage, table)` and every array key a caller looks up in
  `log.arrays` (arrays come back as a `TrackingDict`, so only the keys actually used count);
  `family(key)` maps a key to its pattern (`open_close__norm_pct` → `{task}__norm_pct`). The
  view-coverage check (an12) compares these families with what the stages store.

Writes go to a temp directory and are renamed into place, so an interrupted run never leaves a
unit that looks complete.

**Table convention:** tidy, one row per participant × (task × cycle ×) metric entity, with snake_case
column names that match `biomech/etc/catalog.yaml`.

## 3. Catalog — `biomech.catalog`

`biomech/etc/catalog.yaml` maps each output column to a label, LaTeX symbol, siunitx unit, decimals and a
one-line definition. Tables, TikZ export and viewer format through `catalog.get(name)`. Add an
entry whenever a stage gains a column.

## 4. Figure/table export (`biomech/viz/`)

**Spec model** (`spec.py`): `FigureSpec` → `Panel` → `Series`, plus `TableSpec` / `Column`.
Series kinds: line, scatter (optionally coloured by `c`), band, bar/barh, errorbar, hline/vline
(labelled in place), hspan, heatmap, text, colorline, arrow, fill, image, sigbar (SPM
supra-threshold clusters as bars in reserved headroom), and in 3-D panels
line3d/scatter3d/arrow3d. Colour bars (`Colorbar`) use the sequential or the diverging ramp. Panels carry a legend policy (`auto`: none for one labelled series,
below the panel for two or more; `inside:<pos>` only on request), `equal` (mm for mm; figures
with `meta.equal_scale = "shared"` give all such panels one scale), `sharex` / `sharey` (panels
with the same key share the x / y range; `sharey` includes set limits and the SPM-bar head-room,
2026-11-05), `colorbar`, `note`,
and `tag` -- the panel letter, drawn bold at a fixed corner (top left of the panel cell,
outside the axis, on the first title line). Without a `tag`, `Panel.heading()` takes it from
an "(a) ..." title prefix, so views keep writing "(a) Title" and every renderer shows **a**
plus the short title; the export letters multi-panel figures that have no tags at all
(`export.auto_tags`). Specs hold data and meaning; appearance comes from roles and tokens
(`style.py`: fixed Okabe–Ito order, single-hue sequential ramp, light/dark themes, venue
typography).

**Venues** (`style.VENUES`). `FigureSpec.width` stays `column` / `text`; the active
venue maps it to inches and sets the absolute font sizes (pt) and the figure font family:

| Venue | Column / text | Label, title / tick, legend / note / tag | Figure font |
|---|---|---|---|
| `ieee` (default) | 3.5 / 7.16 in | 8 / 7 / 6 / 8 bold | the document's (IEEE: Times) |
| `springer` | 85 / 170 mm | 8 / 7 / 7 / 8 bold | sans (`\sffamily`) |
| `nature` | 89 / 183 mm | 7 / 6 / 5 / 8 bold | sans |

Select it with `export(..., venue=...)`, `biomech export --venue ieee|springer|nature`, or a
`venue:` key in the paper yaml (the flag wins; without either: `ieee`, so existing exports do
not change). Inside an export `style.use_venue` makes it the default for `layout.compute`,
the pgf style, the matplotlib previews, the size check and the preview page, so all
renderers lay out with the same sizes (the viewer keeps its own fonts and Inter). The pgf
style writes the sizes as `\fontsize{..}{..}\selectfont` macros (`\jtsmmfont<label|tick|
title|legend|note|tag>`), never `\footnotesize`, so an 11- or 12-pt document class does not
change a figure; `\jtsmmfigsetup` (first line of every picture) selects the figure family
and lets siunitx set units in the surrounding text font.

**Style rules** (`style.ROLES`): one meaning per colour -- `accent` (and its `band`)
is the cohort summary in the ink colour, never a categorical hue; muscles always wear their
entity colour and the opener / closer groups the hue of their lead muscle (digastric R
vermillion, masseter R blue); protocols (`protocol:loso` orange, `protocol:blocked` sky) and
feature sets (`set:*`: orange, sky, green, purple, purple dashed) avoid the opener / closer
hues; `reference` and `threshold` lines are grey dashed and labelled at the line; within a
role family no two roles share colour and marker (`v` is drawn as a turned-over `^` in pgf
too). Every renderer prints U+2212 for a sign (`style.unicode_minus`: tick labels, in-panel
text, legends, table cells; pgf `\jtsmmminus` = the text font's `\textminus`). Reader text:
- **No plan codes in reader-facing text.** Views keep codes out of captions and labels at
  the source. `etc/codes.yaml` lists the code families (C, S, A, V → pre-specified; D, VD →
  post hoc; X → exploratory; E, EV, G, W, R → dropped). `vocab.reader_text` applies them
  only as a safety net, in the export and in the viewer alike.
- **Value vocabularies.** `etc/vocab.yaml` holds the reader name of every stored
  categorical value: tasks, muscles, landmarks, bodies, transforms, systems, estimators,
  feature sets, onset rules, clock and tip variants, and so on.
  - Each entry has a label, short and title form: `vocab.term(domain, value, form)`.
  - A `columns:` map lets `vocab.cell` and `blocks.frame_table` translate table cells.
    Booleans read yes/no.
  - Module-level label dicts derive from it, so there is no second copy.
- **Both files are presentation data:** they never enter a cache key.
- **`check/text.py` flags:**
  - plan codes;
  - raw identifiers (snake_case words);
  - raw values (`nan`, `n/a`, a lone `True`);
  - labels guessed from column names (`catalog.misses`);
  - categorical values without a vocabulary entry (`vocab.unknown_terms`);
  - labels, tick labels or table cells emptied by dropping a code that has no reader word,
    and codes dropped mid-sentence (`dropped_code_issues`, comparing the built spec with its
    reader copy).

  `check-outputs` fails a paper-set item with any of them (`check_paper_text`) and reports
  dashboards as warnings.
- **Structured captions** (`viz/caption.py`): `CaptionSpec` holds what the item shows, the
  meaning of each mark, n, the statistic and test, the evidence level and notes (plan codes
  only as provenance).
  - `caption.attach(spec, cap)` renders `spec.caption` and keeps the parts in
    `spec.meta["caption"]` (tables too: `TableSpec.meta`).
  - Every caption gets its abbreviations spelled out at first use, from `etc/glossary.yaml`
    (in `reader_spec`, so the export and the viewer agree).
  - `check-outputs` warns for paper items without a structured caption until the migration
    is done.

**Layout engine** (`layout.py`) — one geometry for every renderer. It resolves limits, ticks and
tick labels (plain numbers on log axes), wraps titles to their cell, rotates crowded category
labels to vertical, sizes and wraps legends below each panel, and returns every box in inches.
Renderers place their axes exactly there, so nothing is left for a backend to auto-place on the
data, and the paper, the viewer and the videos share geometry.

**Registry** (`views/registry.py`): every view is registered with a kind (`paper`, `dashboard`,
`data`, `animation`), the parameters it understands (participant, phase, perspective,
alignment, statistic, window), a `domain` (Results-page section), the `phases` export and the
checks expand, and pickers (`choices`, §7). View modules are discovered automatically
(`view_modules()`: every module of the package except `registry.NON_VIEW_MODULES`:
`registry`, `blocks`, `vocab`, `forms`).
Builders take a `ViewRequest` with a `DataContext` (included participants only, one global
pseudonym map). Exported ids carry every non-default parameter and picker value
(`fig_kinematics--pid-P3--persp-participant`, `emg_envelope--units-norm`); the participant
appears as its pseudonym `pid-P<n>`, never as the study id.

**Perspectives** (`biomech/ensembles.py`): trials on a common axis (`cycle`, `angle`, `time`,
`event`, `path`) are aggregated at `trials`, `participant` or `cohort` level (`mean_sd`,
`median_iqr`, `mean_ci`); cohort statistics run over participant centres and need at least half
of the participants at each point.

**Renderers:**
- `render_pgf.py`: `jtsmm-pgfstyle.tex` (venue fonts, colours, the colormaps `jtseq` and
  `jtdiv`, roles and marker styles in both key trees), `figures/<id>/fig.tex` — absolutely
  positioned `axis` environments (`scale only axis`), explicit ticks, axis labels, titles,
  panel tags and legends as nodes in the layout's bands — with external `.dat` files (`nan` = break; LTTB thinning per finite run
  with a budget by run length, keeping each run's ends and the NaN between runs, so a gap
  stays a gap; 3-D RDP), heatmap colour ranges that are never degenerate (symmetric about 0
  for the diverging ramp), `tables/<id>.tex` (`tabular*` to the column or text width,
  booktabs, siunitx, a note line), `preview.tex` (the venue's page), and `standalone_<id>.tex`
  for the size check.
- `render_mpl.py`: PNG/PDF exports, the layout tests (`text_overlaps`) and video frames
  (unfilled markers `|`, `x` are drawn by their edge in the series colour).
- `render_plotly.py`: the viewer — themes; panels at the layout's domains; one legend per
  panel below it (a panel with legend policy `none` adds no entry); hovers name the axis
  titles and units (heatmaps: row · column and the value with the colour-bar unit);
  colour bars at the layout's colour-bar box; hline/vline span the axis in domain units, so
  zooming never clips them; a band is one polygon per finite stretch (never bridges a gap);
  heatmap cell values in a contrasting ink; `fit_height` (paper aspect at the card width,
  240–1100 px), `reflow` to one column on narrow screens, `apply_zoom` and `merge_relayout`.

```bash
make -C experiments/icorr2027 paper   # export + LuaLaTeX preview + per-figure size check
```

**Everything, not only the paper items:** `biomech export --all` (or `make export-all`, or
*Export every view* on the viewer's paper board) writes every registered view -- paper,
dashboards and the data atlas, each perspective, per-participant views for every included
participant -- as TikZ plus PNG/PDF previews to `<out>/export_all/`. `--kind`,
`--perspectives default` and `--pid` narrow it; `--to` sets the directory. Spec ids carry the
non-default parameters, so no two items share a file.

`biomech export --check-size` compiles every figure alone (venue typeface) and fails when any
of these holds:
- its natural width exceeds the venue's column or text width (IEEE: 3.5 / 7.16 in);
- its height exceeds the venue's `max_height_in` (8.5 in);
- any glyph is smaller than the venue's minimum font size (`min_pt`: IEEE 6 pt, Springer
  7 pt). A LuaTeX `pre_shipout_filter` hook logs the smallest glyph size
  (`JTSMM-MINFONT`);
- the log shows overfull boxes or pgfplots warnings.

Tables are checked by the preview compile (`--compile`): an overfull table is a problem. `--standalone` compiles every figure alone with LuaLaTeX
into `figures/<id>/<id>.pdf` for the journal upload: the page is exactly the layout box
(column or text width), fonts embedded (IEEE: TeX Gyre Termes, Springer / Nature: TeX Gyre
Heros).

```bash
biomech export --config experiments/icorr2027/study.yaml --paper experiments/icorr2027/paper_dataset.yaml \
    --target journal --compile --check-size --standalone        # Springer (the target's venue)
biomech export --config experiments/icorr2027/study.yaml --paper experiments/icorr2027/paper_dataset.yaml \
    --target conference --compile --check-size --standalone     # IEEE conference cut
biomech verify-export --config experiments/icorr2027/study.yaml data/analysis/icorr2027/paper_dataset/journal
```

**Paper sets** (`biomech/papers.py`):
- **Format:** a paper set (`paper_*.yaml` next to the study file) pins its snapshot and
  lists `targets` (venue and output folder each) and `items`.
- **Item keys:** `id` is the view and `label` the LaTeX name (letters only); `part` is
  core, supporting or supplement; `in` lists the targets (default all). Every other key is
  a view parameter.
- **Legacy sets:** a set with `status: legacy` is history and left out of the boards.
- **Older form:** the single-target form (`venue`, `out_dir` at the top) still loads, as
  the target `default`.

**Manifest and captions** (`viz/manifest.py`):
- **`manifest.json`:** every export writes one. Per item it holds the label, part, view,
  parameters, kind, id, caption (plain and LaTeX), sources and the sha256 of every file. It
  also holds the provenance and the paper set and target.
- **`captions.tex`:** labelled items get `\figcaption{label}` and `\figsources{label}`. It
  needs `jtsmm-pgfstyle.tex`.
- **Verification:** `check/integrity.py` checks an export against its manifest and the
  pinned snapshot (an12).
- **Pruning:** a paper folder holds exactly its set. An export removes the files of the
  items its previous manifest listed that are no longer in the set. Single-view exports from
  the viewer go to `<out>/viewer_exports/`.
- **One function for the paper set:** `export_paper_set` exports a paper-set target (items,
  manifest, numbers) for the CLI and the viewer's paper board alike.

**Depictions and the gallery** (`viz/views/forms.py`):
- **Mechanism:** a view offers several depictions of the same numbers through a picker
  named `form`, using the existing picker mechanism.
  - Its values, default, id suffix (`--form-box`), viewer dropdown ("Depiction") and paper
    set key (`{id: x, form: box}`) need no further code.
  - The coverage check builds every form.
  - `unc` (uncertainty shown) and `colour` (colour by) follow the same convention.
- **Drawers:** `forms.summary(form, labels, values, ...)` draws one row per label as strip,
  box, violin, raincloud, ECDF or bar. It returns the panel and the meaning of its marks for
  the structured caption.
  - Only existing series kinds are used, and the stats stages' own interval routine.
  - On a log axis, densities are estimated in log space; a bar needs a linear axis.
- **Reader names:** pickers and their values come from `vocab.picker_label` /
  `choice_label` (`etc/vocab.yaml` domains `picker`, `form`).
- **Gallery:** `biomech export --paper … --gallery` exports every item in every depiction
  it offers to `<target folder>_gallery` with a compiled preview, to choose from. The
  choice is then written into the paper set.
- **Use:** the first view with depictions was `imu_agreement`; most paper items now offer
  several.

**In-text numbers** (`viz/numbers.py`):
- **Declaration:** every number a manuscript states is declared once in the paper set's
  numbers file (`numbers_dataset.yaml`, `numbers_systems.yaml`). Each entry gives the key
  (letters only), the stage:table source, a filter that selects exactly one row (or `agg`
  over the selected rows), the column, the CI and n columns, the format, the siunitx unit
  and a text.
- **Other sources:** `config:<path>` (a method parameter as the store was computed),
  `sensitivity:<variant>` (a variant's comparison), `hw:<table>` / `bench:<table>` (hardware
  and bench tables), and `derived:<name>` (2026-11-05): a small table computed from the stored
  results with the figures' own helpers (`viz/derived.py`, per paper `viz/derived_dataset.py`
  / `viz/derived_systems.py`; presentation code, not hashed, no stage re-run), e.g. the
  translation at 10° (`coupling_curve`), the share outside the design box
  (`workspace_box`), the condyle-point path and axis direction (`condyle_path`), the IMU lags,
  least-products agreement, recorder metadata and mounting sessions of the systems paper.
  A `cite:` group reads another study's pinned snapshot, also a sensitivity variant's.
- **Format `fmt: req`:** a requirement macro takes Table I's rounding rule
  (`paper.req_decimals`), so text and table print the same value.
- **Output:** the export resolves them from the pinned snapshot into `numbers.json` (full
  values, source, provenance) and `numbers.tex`:
  - `\numX` (value with unit), `\numXVal`, `\numXCI` ("[lo, hi] unit") and `\numXN`;
  - siunitx v3 is needed.
- **Failure:** an unresolvable key or a filter that does not select one row fails the
  export.
- **Integrity:** the integrity check re-resolves every number and compares.
- **Additional files:** a paper set's `supplementary_interactive` views are written as
  self-contained interactive HTML (`additional/<label>.html`) and its rendered
  `supplementary_videos` are recorded in the export manifest; `biomech paper-sync --additional`
  (`make sync-paper-additional`) copies both to the manuscript's `results/additional/`.
- **Commands:**
  - `biomech numbers --paper …` prints and writes the numbers;
  - `biomech check-paper TEX_DIR --export DIR` lists every `\numX` and
    `\figcaption{label}` the manuscript uses but no export defines (exit 1), and the unused
    numbers.
- **Never type a result into a manuscript.**

**In the paper:**

```latex
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\usepackage{booktabs,siunitx,array,tabularx}\input{jtsmm-pgfstyle}
...
\begin{figure*}\centering\input{figures/fig_kinematics/fig.tex}\caption{...}\end{figure*}
```

**Spec and source data next to every export** (`viz/serialize.py`): each figure writes
`figures/<id>/spec.json` (the whole FigureSpec -- panels, axes and units, series kinds and
roles, caption, sources -- with arrays referenced into) `spec.npz` (exact dtypes and shapes)
and `source_data.csv` (the 1-D series in long form: panel, series, kind, label, x, y, y2, …;
the journals' "source data"); each table writes `tables/<id>.csv` (plain headers with units)
and `tables/<id>.json`. All carry a provenance record (snapshot or live store, newest run,
git state, export time), also as a `% Data:` line in the pgf files. `biomech render
figures/<id>/spec.json --to pgf|pdf|png|html [--out PATH]` re-renders a figure (or a table
json to booktabs) from these files alone; every view re-renders identically
(tests/test_r4_serialize.py).

Compile with LuaLaTeX (dynamic memory for 3-D and colour-line figures). Set the target's
`out_dir` in the paper set (`paper_dataset.yaml`, `paper_systems.yaml`) to the paper
directory, so that the data paths resolve from the directory LaTeX compiles in. Items in a paper yaml may fix a perspective, e.g.
`- {id: fig_kinematics, perspective: cohort, stat: median_iqr}`.

## 5. Synthetic ground truth and tests (`biomech/testing/synth.py`)

`biomech synth --out DIR` (or `write_study(root, specs)`) writes, per synthetic participant, a
motion take and an MVC take in the real `session.h5` format whose every signal follows from one
physical model, plus `cohort.yaml` and `study.yaml`; `Truth` holds what the analysis must
recover:

- **anatomy** in the maxillary frame (incisors, second molars, skin-palpated condyles);
- **motion:** open_close and chewing as a hinge about the condyle axis plus a *delayed*
  translation along u (rotation first, translation from 40 % of the opening;
  `translation_mm` = 0 is a pure hinge), protrusion and laterotrusion as translations,
  free_motion as a rich mixture (identifies the IMU calibration);
- **optical poses** with white noise (`pose_noise_mm`, `pose_noise_deg`), the CT tip resting on
  each landmark during its hold;
- **two IMUs** with known mounting and heading offset: game-rotation-vector quaternions and raw
  acceleration including gravity, consistent with the poses;
- **six EMG channels:** bursts whose onsets precede the mechanical onset by `lead_ms`, mains hum
  with harmonics at `mains_hz`, DC offsets, a tissue low-pass `emg_band_hz`, optional lead-off.

`DEFAULT_SPECS`: s01 pure hinge, lead 60 ms; s02 4 mm translation, 90 ms, lead-off on
digastric_r; s03 6 mm, 120 ms, 28° opening; s04 3 mm, 75 ms, 60 Hz grid, EMG band 20–250 Hz.
`tests/test_e2e_synthetic.py` runs the full stage graph on them and asserts the recovery of the
anatomy, cycles and clinical metrics, hinge residual vs the noise null, translation, IMU
mounting and agreement, the EMG lead under both onset rules and the ensemble rise (±40 ms),
references, mains frequency (60 Hz for s04) and lead-off, label classes, spectral shape and
bandwidth, and finite cohort statistics. Run the suite with `pytest software/biomech`
(`-m "not slow"` skips the full-graph tests).

## 6. Command reference (`biomech`)

Every command except `stages`, `synth`, `cohort draft`, `render` and `check-paper` takes
`--config STUDY.yaml` (required); most also take `--data-dir` (session root;
precedence `--data-dir` > `$JTSMM_DATA_DIR` > `paths.data_dir` in the study > the repo default
`data/pilot_study/jtsmm_data`) and `--out-dir` (results root, default `data/analysis/<study>`;
not `view` and `cohort check`).

| Command | Options | Does |
|---|---|---|
| `biomech stages` | — | list the registered stages in dependency order |
| `biomech cohort check` / `cohort draft` | `--config` / `--study NAME --data-dir DIR` | validate the manifest against the session folders / print a draft manifest |
| `biomech run` | `--stage S` (repeatable; dependencies first), `--subject sNN` (repeatable), `--force`, `-j N` (default: one per participant, at most the performance cores + 4; 1 = in-process) | bring stages up to date; cached units are skipped |
| `biomech status` | — | stage × participant status against the current configuration (up-to-date / stale / missing / missing-input / failed) |
| `biomech show STAGE TABLE` | `--csv FILE`, `--snapshot` | print or save one table across participants |
| `biomech gc` | — | remove leftovers of interrupted writes (restores a unit moved aside by a killed write) |
| `biomech snapshot [NAME]` | `--note`, `--overwrite`, `--list`, `--variant NAME` (a sensitivity variant's store) | freeze the current results (§1) |
| `biomech compare OLD [NEW]` | `--csv FILE` | compare the cohort tables of two snapshots (`live` = live store) |
| `biomech describe [STAGE [NAME]]` | `--json`, `--snapshot` | data dictionary of the stored results |
| `biomech sensitivity [NAME]` | `-j N` (default as `run`), `--compare-only` | run a sensitivity variant and compare it with the primary run (no NAME: list the variants) (§1) |
| `biomech bench` | `--run RUN_ID` (default: the latest run) | per-stage timing of a run (§1) |
| `biomech export-data` | `--to DIR` (required), `--format csv/hdf5`, `--arrays` (hdf5), `--snapshot` | pseudonymized dataset release with data dictionary (§1) |
| `biomech render SPEC` | `--to pgf/pdf/png/html`, `--out PATH` | re-render an exported figure/table from its spec files (§4) |
| `biomech export` | `--paper paper_dataset.yaml` (with `--target NAME`, `--gallery`) **or** `--all` (`--kind paper/dashboard/data`, repeatable; `--perspectives all/default`; `--pid sNN`, repeatable); `--to DIR`; `--venue ieee/springer/nature` (default: the target's venue, else ieee); `--compile` (LuaLaTeX `preview.pdf`); `--check-size`; `--standalone` (`figures/<id>/<id>.pdf`); `--snapshot`, `--variant` (override the paper set's store); `-j N` (0 = up to 8) | TikZ/pgfplots figures, booktabs tables, PNG/PDF previews, manifest, captions and numbers (§4) |
| `biomech numbers` | `--paper FILE` (required), `--target`, `--snapshot`, `--variant`, `--dry-run` (print only) | resolve the paper set's in-text numbers and write `numbers.tex` / `numbers.json` (§4) |
| `biomech verify-export FOLDER` | `--snapshot`, `--variant` (default: from the manifest's provenance) | rebuild every exported item from its store and compare (an12) |
| `biomech check-paper TEX_DIR` | `--export DIR` (required, repeatable); no `--config` | undefined `\num…` / `\figcaption` uses, mixed reporting populations, unused numbers (an12) |
| `biomech report` | `--pid sNN` (required), `--out DIR`, `--compile/--no-compile`, `--snapshot` | participant report of every data view (an10) |
| `biomech animate` | `--pid`, `--layout multimodal/kinematics/tracking/emg/cohort/v1_hinge/v2_intent/v3_release/validation` (V1–V3: supplementary videos; `validation`: the systems-paper video, validation study only; an11), `--from-clip FILE` (re-render from `<stem>.clip.npz` / `.clip.json` alone, no config needed), `--captions FILE`, `--phase P` (repeatable, played in order), `--t0/--t1` (s from the phase start), `--speed` (0.5), `--fps` (30), `--size 1080p/720p/4k/print/<W>/<W>x<H>` (1080p), `--theme dark/light` (dark), `--with-camera`, `--window BEFORE AFTER` (scrolling panels, s around the cursor; 3 1), `--trail S` (fading landmark path; 1), `--out FILE.mp4/.webm/.gif` (default `<out>/videos/<stem>.mp4`, an11 *File names*), `-j N` (0 = up to 8), `--filmstrip`, `--animate-tex`, `--snapshot` | synchronized video (an11) |
| `biomech check-outputs` | `--report-dir DIR`, `--no-figures`, `--no-pgf`, `--no-videos`, `--snapshot`, `-j N` (0 = up to 8) | check every output (an12); exit code 1 on a failure |
| `biomech synth` | `--out DIR` (required), `--run/--no-run`, `-j N` (default 4), `--emg-fs HZ` (default 2000) | write (and run) a synthetic study with a known truth (§5) |
| `biomech view` | `--port` (8060), `--host` (127.0.0.1), `--debug`, `--verbose` (keep the per-request access log), `--snapshot` | analysis viewer (an09) |

**Makefile** (`experiments/icorr2027/Makefile`): `check`, `qc`, `all`, `status`, `view`,
`paper` (= `paper-journal` + `paper-conference`: `export --paper paper_dataset.yaml --target
journal` / `--target conference`, each `--compile --check-size --standalone`), `export-all` (`export --all`), `check-all`
(`check-outputs`), `video` (`animate … --fps 30 --size 1080p --filmstrip`). Variables:
`BIOMECH=` (executable; default `biomech` on PATH, else the repo venv), `DATA=` (session root),
`JOBS=` (default 6), and for `video` `PID=` (s01), `LAYOUT=` (multimodal), `PHASE=` (chewing),
`SPEED=` (0.5).

## 7. Adding an analysis mode

A new analysis is a stage output plus a view; nothing else keeps a list.

1. **Stage** (`biomech/stages.py`, or a plugin via the `biomech.stages` entry point): compute
   in an analysis module, return `StageResult(tables, arrays)`. Use `ctx.emg_prep()` /
   `ctx.session()` / `ctx.memo()` for shared inputs (§1). Name per-task arrays
   `<task>__<quantity>` and per-participant cohort arrays `<quantity>__<pid>__...`, so the
   coverage guard groups them into families.
2. **Catalog** (`etc/catalog.yaml`): label, unit, decimals (and a plausible range) for every
   new column; views and tables read labels and units from here (`views.vocab.axis(col)`).
   Declare every new table in `etc/tables.yaml` (title and caption: its complete table view,
   an12 *Completeness*) and give every new categorical value a reader name in
   `etc/vocab.yaml`; `check-outputs` fails a paper item, and warns for other views, until
   both exist.
3. **View** (any module in `biomech/viz/views/`, discovered automatically):

   ```python
   from biomech.viz.spec import FigureSpec
   from biomech.viz.views import blocks as B, vocab as V
   from biomech.viz.views.registry import ViewRequest, register

   @register("my_metric", "dashboard", "My metric", domain="emg_freq",
             params=("phase",), phase="chewing", phases=("chewing", "open_close"),
             choices={"muscle": V.MUSCLES})
   def my_metric(req: ViewRequest) -> FigureSpec:
       t = req.ctx.table("my_stage", "my_table")
       t = t[(t.task == req.phase) & (t.muscle == req.options["muscle"])]
       p = B.strip_by(t, "my_value", "task", "My metric per participant")
       return FigureSpec("my_metric", "My metric", [p], caption="...",
                         sources=["my_stage:my_table"])
   ```

   `domain` puts it in a Results-page section; `phases` are the phases export, the checks and
   the coverage guard expand; `choices` become dropdowns in the viewer (a list, or
   `lambda ctx: [...]` from the data) and reach the builder as `req.options[name]`. The shared
   panels (`views/blocks.py`: strips, heatmaps, curves with bands, spectra, SPM curves, forest,
   dumbbell, event raster, histogram, table) keep the visual language consistent.
4. **Tests:** build the view on the synthetic study (`synth_study` fixture) and render it with
   the three renderers. `tests/test_view_coverage.py` fails until every stored table and array
   family of the new stage is read by some view (or listed in `check/coverage.py:INTERNAL`
   with the reason it is only an input of another stage).

The view then appears in the viewer (Results page, or the explorer Atlas for `data` views),
in `biomech export --all`, in `check-outputs` and in the coverage report without further
registration.
