<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an12 — Output check (`biomech check-outputs`)

One command checks everything the study produced before it goes into the paper. It reads the
store, renders every registered view, re-exports the paper items, probes the videos, and writes
`<out>/check/report.html` (+ `report.json`, contact sheets).

```bash
biomech check-outputs --config experiments/icorr2027/study.yaml       # make -C experiments/icorr2027 check-all
```

Options: `--report-dir DIR` (default `<out>/check`), `--no-figures` (skips the figure, the
view-coverage, the paper-text and the export-integrity areas), `--no-pgf` (skips the LuaLaTeX
export), `--no-videos`, `--snapshot NAME` (check a frozen store instead of the live one),
`-j/--jobs N` (check workers: 0 = up to 8, 1 = in-process; the same report). The checked
groups are the study's `pipeline.groups` (dataset or validation); the dataset-only number
checks (ensembles, reference table) run for dataset studies only. Every item is pass / warn /
fail / skip with a detail line (and a link to the PNG where useful).

**Which store each check reads.** The store-level areas (coverage, numbers, figures, views)
read the configured store (`--snapshot` or live). Every **paper-set** check — paper text,
caption n, in-text n, export integrity — reads the paper set's **own** store: its `variant`
(e.g. `reporting_population` for the dataset paper, D35) at its pinned `snapshot`
(`papers.paper_context`, resolved in one place by `PaperSet.store_config` /
`sensitivity.store_config`). A paper set whose store cannot be opened fails its check.

| Area | Checks |
|---|---|
| `coverage` | every stage × included participant up to date (`stage_status`); units of participants outside the cohort flagged as warn (stored, never read) |
| `numbers` | per task (open-close, chewing) the cycle counts agree between the kinematics cycles table and the ensembles (an10), and every EMG profile cycle exists in the cycles table; Table I (`table_reference`, mean and SD) equals `stats_kinematics.reference_values`; values of the participant-stage tables outside the catalog's plausible ranges (`biomech/etc/catalog.yaml`, `range`; any value outside = warn, > 5 % outside = fail) |
| `figures` | every view × perspective (default phase and pickers; per-participant views for the first included participant) builds; no overlapping or clipped text in the matplotlib render (titles, panel tags, labels, ticks, in-panel text, legends); the layout check; Plotly light and dark; in every `paper` figure no two legend entries of a panel look alike (lines: colour + dash, markers: colour + marker, areas: colour; `legend_conflicts`); the paper items (every `paper` view) export and compile with LuaLaTeX and every figure compiled alone fits 3.5 in / 7.16 in with no overfull box or pgfplots warning (skip without `lualatex`); the exported paper items carry no internal plan codes (D6, X2, C1, E17, …) in captions, notes, titles, legends or table cells (`exported_code_leaks`, lists the offending items); contact sheets (one montage per view kind, 12 per sheet) for the visual pass |
| `views` | view coverage (below), including column-level completeness and the complete table views |
| `text` | reader text (`check/text.py:item_issues`, an08 §4). The issues: plan codes, raw identifiers (snake_case), raw values (`nan`, a lone `True`), labels guessed from column names (no catalog entry), categorical values without a vocabulary entry, labels / tick labels / table cells emptied by dropping a code that has no reader word ("E3b") and codes dropped mid-sentence ("the slope of E2 against time" → "the slope of against time"; `dropped_code_issues` compares the built spec with its reader copy). One meaning per colour within a figure (`colour_meaning_issues`, 2026-11-05: a hue may carry one series role; dashed / dotted variants, explicit colours, colormaps and the labelled phase lanes exempt). Every item of the study's paper sets, in every depiction (`form`) the gallery offers, must have none of them (`check_paper_text`, **fail**); every other view is reported (warn). Every paper caption structured (marks, n, statistic; warn while not yet) |
| `n` | participant counts (below) |
| `integrity` | every exported target of the study's paper sets (`check_exports`; `check/integrity.py`), read from the paper set's store (variant + pinned snapshot): the pinned store matches the current code (stale stages named), the files match `manifest.json` (sha256; no stale or unlisted items), the export came from that store, and every item rebuilt now from it equals the exported one (series values, labels, captions, table cells). A target not exported yet is a warn. `biomech verify-export DIR` runs the same check on one folder (the store taken from the manifest's provenance, or `--variant` / `--snapshot`) |
| `videos` | for every `<out>/videos/*.mp4`: ffprobe (H.264, even size, duration > 0); the manifest (`.json`) next to the video; duration = the timeline's expected duration (±1.5 frames); the embedded timecode read back at up to 8 sampled frames equals the timeline (sync); skip without videos or ffprobe |
| `docs` | the repository docs test (`tests/test_docs_links.py` at the repository root: links, anchors, stale defaults) |
| `privacy` | no subject data tracked by git (`data/pilot_study`, `data/analysis`); no participant ids (`sNN`) in any view's text (titles, captions, labels, ticks, legends, table cells) or in the exported TeX; each video manifest declares `with_camera` |

**Participant counts (area `n`; `check_numbers_n`, `caption_n_issues`, `check_stale_n`).**
- **In-text numbers** (`numbers_*.yaml`, resolved from the paper set's store,
  `number_n_issues`): a participant n never exceeds that store's cohort — the dataset paper on
  `reporting_population`: **13**; a number cited from the other study: that study's cohort (systems: 2);
  the `primary` column of a variant comparison (`sensitivity:<variant>`, e.g. the "including
  the participant recorded outside the protocol" robustness values): the analysis of record, **14**; an reporting-population
  number (key with `Adults`, or the `reporting_population` comparison's variant column): the adults.
  The item names the paper set's population and the three bounds ("reporting_population: 13; adults
  13; analysis 14").
- **Captions:** the participant count a structured caption states ("k participants", or the
  distinct pseudonyms P1, P2, … it names) is at most the cohort size (reporting-population captions: the
  adults), equals the participants the item labels (P1 … Pn as ticks, legend entries,
  panel titles or cells), and is not above the participants of the item's source tables
  (the `stage:table` entries of `spec.sources`, read from the paper set's store;
  `source_participants`: distinct ids of the `pid` / `pid_*` columns of a per-participant
  table, else the largest participant count of a cohort table, `n_participants`,
  `n_subjects`, `n_pid` or `n`; a caption may state fewer, the usable subset). A
  single-participant item names its participant first ("P1, comparison take"). Only an item
  that states a count none of these can check is listed as "not checkable" (warn; since
  2026-11-01 none in either paper set).
- **Stale n:** no "n = 12" / "n ≈ 12" (the cohort size of earlier rounds) as a participant
  count in the paper-writing notes and the active paper sets and numbers files (warn; "n = 12 cycles" and
  other units are not counted).
- The reporting population is the included participants not flagged `outside_protocol`
  (`cohort_sizes`).

**Manuscript check (`biomech check-paper TEX_DIR --export DIR …`).** Every `\num…` macro and
`\figcaption{label}` the manuscript uses must be defined by the exports, and all exports must
come from **one reporting population** (the same sensitivity variant): a mix is
reported as "MIXED POPULATION" and exits 1; unused numbers are listed.

Exit code 1 when anything fails. `tests/test_check_outputs.py` corrupts one artefact of each kind
(a table cell, a clipped figure with a leaked id, an oversized pgf figure, an off-sync video, a
stray unit outside the cohort) and requires the check to catch it; a clean synthetic study must
pass.

**View coverage** (area `views`, with figures enabled; `check/coverage.py`): every registered
`paper`, `dashboard` and `data` view is built under read tracking (`results/access.py`, an08
§2) for every phase of `registry.expand_phases` × every perspective (participant views for
the first included participant), plus every picker value once on top of the first variant.
Failures: a variant that does not build, and a stored table or array family (ok units only;
array keys reduced to families with the task, muscle, pid and model as placeholders) that no
view reads, minus `coverage.INTERNAL` — the outputs that are only inputs of other stages, each
with its reason: the `ml_windows` feature matrices, labels, time stamps, phase index and
standardization (`X_*`, `y`, `t_end_us`, `phase`, `norm_*`), the per-window spectra
`emg:spectral__{task}__win`, the specification-curve inputs `kinematics:spec`, `emg:spec`,
the pose matrices `kinematics:{task}__T`, `T_max_hp`, `T_closed`, and the SPM test inputs
`stats_emg:spm1__left|right`, `spm2__openers|closers`. The same check runs in
`tests/test_view_coverage.py` on the synthetic study.

**Completeness:**
- **Every stored result column reaches an artefact.**
  - Under read tracking, tables come back as a `TrackingFrame` (`results/access.py`), which
    notes the columns a view uses: indexing, attributes, `.loc`/`.at`, `get`. Whole-frame
    reads (iteration, `to_dict`, `values`) count as reading every column.
  - Every column of every shown table must be used by some view.
  - The only exceptions are listed with a reason in `check/coverage_exempt.yaml`:
    bookkeeping (`unit`) and the private columns withheld from every artefact.
- **Every stored table has its complete table view.** `etc/tables.yaml` declares one entry
  per stored stage table, with a title and caption. `viz/views/tables.py` registers it
  (domain "Result tables"): every column, reader headers and cells, privacy redaction,
  study ids in cells pseudonymized.
- **A new stage table fails the check until it is declared.**
- Both stores must report 0 unused columns.
- `coverage_matrix.csv` (next to the report): per stored column the artefacts that use it.
- `text_gaps.json`: every guessed label, value without vocabulary and raw identifier (with the
  views and columns), the full lists behind the reader-text warnings.
- Exports: each paper set's pinned snapshot is checked against the current code (stale stages
  are named, a warning) and every exported item is rebuilt from it (`verify-export`).
- Videos: the clip file next to each rendered video is compared with the same clip loaded
  from the store (`integrity.verify_clip`).
- `tests/test_r7_integrity.py` seeds the failures and requires them to be caught: an edited
  `.dat`, changed array values with consistent hashes, an edited caption and table cell, an
  unlisted export, an unread column.
