# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``biomech check-outputs``: one consistent check of everything the study produced (W8).

Areas (each item is pass / warn / fail with a detail line and, where useful, a
link to the artefact):

* **coverage** -- every stage x included participant up to date; units of
  excluded participants are flagged (they exist but are never read);
* **numbers** -- trial counts agree across the cycles table, the ensembles and the
  EMG profiles; Table I (mean, SD) equals an independent recomputation from the
  participant values; values outside the catalog's plausible ranges (participant-stage
  tables);
* **figures** -- every registered view x perspective builds, renders without
  overlapping or clipped text, passes the layout check, renders in Plotly, and
  carries no participant id; in paper figures no two legend entries of a panel look alike
  (colour + marker / dash); paper figures compiled alone fit their IEEE width; the exported
  paper items carry no internal plan codes (D6, X2, C1, E17, ...) in reader-facing text;
  contact sheets (one montage per view kind) for the visual pass;
* **videos** -- ffprobe (codec, size, fps, duration), the embedded timecode read
  back at sampled frames matches the timeline, the manifest records the camera flag; every
  video a paper set lists (``supplementary_videos``) is rendered, from the store the set pins;
* **views** -- every view variant (phase, perspective, picker) builds and every stored
  table and array family is read by some view (``check/coverage.py``);
* **docs and privacy** -- the docs link test; no subject data tracked by git; no
  participant ids or demographics in exported text;
* **n** -- every in-text number counting participants is at most the
  cohort size of the store its paper set reads (the analysis of record: 14 dataset, 2
  systems; a paper set with ``variant: reporting_population``: 13, its numbers taken from the
  primary column of a variant comparison -- the all-participant robustness values --
  at most the analysis's 14; reporting-population numbers at most the adults); the participant count
  each structured caption states is at most that cohort size and equals the participants
  shown where the item labels them (P1, P2, ...); no stale "n = 12" in the paper docs
  (the paper notes) and paper sets. Every paper-set check reads the paper set's own store (its
  variant at its pinned snapshot, :func:`biomech.papers.paper_context`);
* **integrity** -- every paper-set export equals a rebuild from its pinned store; every pinned
  snapshot (``snapshot:``, ``record_snapshot:``) has a bundle that passes ``biomech bundle
  --verify --against`` (J9.4; not bundled yet: warning).

Writes ``<out>/report.json``, ``<out>/report.html`` and ``<out>/text_gaps.json`` (every
guessed label and value without vocabulary, with the views that need it).
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

PID_RE = re.compile(r"(?<![A-Za-z0-9_])s\d{2}(?![A-Za-z0-9_])")
# Ids and file names join words with "_", "-", "." and "/": a study id between those counts.
NAME_RE = re.compile(r"(?<![A-Za-z0-9])s\d{2}(?![A-Za-z0-9])")
STATUS_ORDER = {"fail": 0, "warn": 1, "pass": 2, "skip": 3}
# Folders that hold subject data (recordings, derived features, EMG windows, analysis outputs);
# git must track nothing in them (the .gitignore names the same folders).
SUBJECT_DATA_DIRS = ("data/pilot_study", "data/analysis", "data/jtsmm_qualisys_delsys_eval",
                     "data/raw", "data/processed", "data/recorder", "data/optical")


@dataclass
class Item:
    area: str
    name: str
    status: str
    detail: str = ""
    link: str | None = None


@dataclass
class Report:
    study: str
    started: float = field(default_factory=time.time)
    items: list[Item] = field(default_factory=list)

    def add(self, area: str, name: str, ok: bool | str, detail: str = "",
            link: str | None = None) -> None:
        status = ok if isinstance(ok, str) else ("pass" if ok else "fail")
        self.items.append(Item(area, name, status, detail, link))

    @property
    def n_fail(self) -> int:
        return sum(i.status == "fail" for i in self.items)

    @property
    def n_warn(self) -> int:
        return sum(i.status == "warn" for i in self.items)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({"study": self.study, "started": self.started,
                                    "n_fail": self.n_fail, "n_warn": self.n_warn,
                                    "items": [asdict(i) for i in self.items]}, indent=1),
                        encoding="utf-8")

    def to_html(self, path: Path, sheets: list[str]) -> None:
        rows = []
        for i in sorted(self.items, key=lambda i: (STATUS_ORDER.get(i.status, 9), i.area,
                                                    i.name)):
            link = f' <a href="{html.escape(i.link)}">open</a>' if i.link else ""
            rows.append(f"<tr class='{i.status}'><td><span class='chip'>{i.status}</span></td>"
                        f"<td>{html.escape(i.area)}</td><td>{html.escape(i.name)}</td>"
                        f"<td>{html.escape(i.detail)}{link}</td></tr>")
        imgs = "".join(f"<figure><img src='{html.escape(s)}' alt='{html.escape(s)}'>"
                       f"<figcaption>{html.escape(s)}</figcaption></figure>" for s in sheets)
        doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Output check</title><style>
:root {{ --bg:#fff; --fg:#1f2328; --fg2:#57606a; --line:#e6e8eb; --ok:#1a7f37; --warn:#9a6700;
        --bad:#cf222e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --fg2:#a8b3bd;
        --line:#262c34; --ok:#3fb950; --warn:#d29922; --bad:#f85149; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif;
        margin:0 auto; padding:16px; max-width:1300px; }}
table {{ border-collapse:collapse; width:100%; }} td {{ border-bottom:1px solid var(--line);
        padding:4px 8px; vertical-align:top; }}
.chip {{ font-weight:600; }} tr.pass .chip {{ color:var(--ok); }}
tr.warn .chip {{ color:var(--warn); }} tr.fail .chip {{ color:var(--bad); }}
tr.skip .chip {{ color:var(--fg2); }} a {{ color:var(--ok); }}
figure {{ margin:12px 0; }} img {{ max-width:100%; border:1px solid var(--line); }}
figcaption {{ color:var(--fg2); font-size:12px; }}
</style></head><body>
<h1>Output check: {html.escape(self.study)}</h1>
<p>{self.n_fail} failing, {self.n_warn} warnings, {len(self.items)} checks.</p>
<table>{''.join(rows)}</table><h2>Contact sheets</h2>{imgs}</body></html>"""
        path.write_text(doc, encoding="utf-8")


# -- coverage ---------------------------------------------------------------------------------
def check_coverage(rep: Report, ctx) -> None:
    from biomech.pipeline import registry

    if getattr(ctx.config, "snapshot", None):
        from biomech.results import snapshot as SN

        meta = SN.info(ctx.config.out_dir, ctx.config.snapshot)
        n = int(meta.get("stale_units", 0))
        st = meta.get("status")
        known = unavailable_units(ctx.config.cohort_path)
        doc = []
        if isinstance(st, list):         # documented-unavailable units do not count (live rule)
            doc = [f"{r['stage']}/{r['unit']}" for r in st if r.get("status") == "failed"
                   and (r.get("unit"), r.get("stage")) in known]
            n = sum(r.get("status") != "up-to-date" for r in st) - len(doc)
        rep.add("coverage", f"snapshot {meta['name']} complete when frozen", n == 0,
                f"frozen {meta['created_utc']}, git {meta.get('git', {}).get('sha', '?')[:9]}, "
                f"{n} unit(s) were not current"
                + (f"; documented unavailable: {', '.join(doc)}" if doc else ""))
    else:
        _check_live_coverage(rep, ctx)
    included = set(ctx.pids())
    for name, s in registry().items():
        if s.scope != "participant" or s.group not in ctx.config.groups:
            continue
        units = set(ctx.store.units(name))
        extra = sorted(units - included)
        if extra:
            rep.add("coverage", f"{name}: units outside the cohort", "warn",
                    f"{', '.join(extra)} stored but never read (excluded or removed)")


def _check_live_coverage(rep: Report, ctx) -> None:
    from biomech.pipeline import stage_status

    st = stage_status(ctx.config)
    bad = st[st.status != "up-to-date"]
    # A stage the cohort manifest documents as unavailable for a participant (missing data,
    # `unavailable: {stage: reason}`) is reported, not failed -- flagged, never hidden.
    known = unavailable_units(ctx.config.cohort_path)
    doc = bad[[(r.unit, r.stage) in known and r.status == "failed" for r in bad.itertuples()]]
    bad = bad.drop(doc.index)
    rep.add("coverage", "stages up to date", bad.empty,
            "all units current" if bad.empty else
            "; ".join(f"{r.stage}/{r.unit}: {r.status}" for r in bad.itertuples())[:600])
    if not doc.empty:
        rep.add("coverage", "documented unavailable units", "warn",
                "; ".join(f"{r.stage}/{r.unit}: {known[(r.unit, r.stage)]}"
                          for r in doc.itertuples())[:600])


def unavailable_units(cohort_path) -> dict[tuple[str, str], str]:
    """(pid, stage) -> reason, from the ``unavailable`` entries of the cohort manifest."""
    import yaml

    try:
        doc = yaml.safe_load(Path(cohort_path).read_text(encoding="utf-8")) or {}
    except (OSError, TypeError, yaml.YAMLError):
        return {}
    return {(str(r["pid"]).lower(), str(st)): str(why)
            for r in doc.get("participants") or [] if isinstance(r, dict)
            for st, why in (r.get("unavailable") or {}).items()}


# -- numbers ----------------------------------------------------------------------------------
def check_view_coverage(rep: Report, ctx, *, reads: dict[tuple, tuple] | None = None,
                        n_jobs: int = 1, groups: Any = None, out: Path | None = None) -> None:
    """Every stored output is read by some view; every view variant builds (an13 §4).
    ``reads``: store reads of variants already built by :func:`check_figures` (not built
    again); ``n_jobs``: the other variants in worker processes."""
    from biomech.check import coverage as C

    r = C.report(ctx, prebuilt=reads, n_jobs=n_jobs, groups=groups)
    for name, err in r["errors"][:40]:
        rep.add("views", f"{name} builds", False, err[:300])
    rep.add("views", "every view variant (phase, perspective, picker) builds",
            not r["errors"], f"{len(r['errors'])} failed" if r["errors"] else "")
    missing = [f"{st}:{n}" for st, n in r["tables"] + r["arrays"]]
    rep.add("views", "every stored table and array family is shown by a view", not missing,
            ", ".join(missing[:20]))
    und = r.get("undeclared", [])
    rep.add("views", "every stored table has a complete table view (etc/tables.yaml)", not und,
            ", ".join(und[:20]))
    cols = [f"{st}:{t}:{c}" for st, t, c in r.get("columns", [])]
    rep.add("views", "every stored result column is used by a view (completeness)", not cols,
            f"{len(cols)} unused: " + ", ".join(cols[:40]) if cols else "")
    if out is not None and r.get("matrix"):
        n = C.write_matrix(r["matrix"], Path(out) / "coverage_matrix.csv")
        rep.add("views", "coverage matrix (stored column -> artefacts using it)", True,
                f"{n} columns", link="coverage_matrix.csv")
    from biomech.viz.views import registry as R

    raw = sorted(v.id for v in R.views(groups=ctx if groups is None else groups)
                 if v.needs_session)
    rep.add("views", "views that need the recorded session (not reproducible from the store)",
            True, ", ".join(raw) or "none")


def check_counts(rep: Report, ctx) -> None:
    from biomech import ensembles as E

    cyc = ctx.table("kinematics", "cycles")
    for task in ("open_close", "chewing"):
        if cyc.empty:
            break
        ens = E.kinematic(ctx, task, "theta", "cycle")
        want = cyc[(cyc.task == task)].groupby("pid").size()
        got = ens.trials.groupby("pid").size() if ens.n_trials else want * 0
        diff = (want - got.reindex(want.index).fillna(0)).abs()
        rep.add("numbers", f"{task}: cycles in ensembles = cycles table", bool((diff == 0).all()),
                f"max difference {int(diff.max()) if len(diff) else 0} cycles")
        prof = E.emg_profiles(ctx, task)
        if prof.n_trials:
            extra = []
            for pid, g in prof.trials.groupby("pid"):
                ids = set(cyc[(cyc.pid == pid) & (cyc.task == task)].cycle)
                extra += [f"{pid}:{c}" for c in g.cycle if c not in ids]
            rep.add("numbers", f"{task}: EMG profile cycles exist in kinematics", not extra,
                    ", ".join(extra[:10]) or "all matched")


def check_tables(rep: Report, ctx) -> None:
    """Paper tables against an independent recomputation from the stats tables."""
    from biomech.viz.views import registry as R

    try:
        spec = R.build("table_reference", R.request(ctx, "table_reference"))
    except Exception as exc:
        rep.add("numbers", "table_reference builds", False, f"{type(exc).__name__}: {exc}")
        return
    rv = ctx.table("stats_kinematics", "reference_values")
    from biomech.viz.views.paper import REF_ROWS

    bad = []
    rows = {r["metric"].split(" (")[0]: r for r in spec.rows if not r.get("_rule")}
    for key, label, _unit in REF_ROWS:
        g = rv[(rv.metric == key) & (rv.group == "all")]
        if g.empty or label not in rows:
            continue
        for col in ("mean", "sd"):
            a, b = float(rows[label][col]), float(g[col].iloc[0])
            if not (np.isclose(a, b, rtol=1e-9, equal_nan=True)):
                bad.append(f"{label} {col}: table {a:.4g} vs stats {b:.4g}")
    rep.add("numbers", "table_reference = stats_kinematics.reference_values", not bad,
            "; ".join(bad) or f"{len(rows)} rows checked")


def check_ranges(rep: Report, ctx) -> None:
    from biomech import catalog
    from biomech.pipeline import registry

    cat = {k: e for k, e in catalog.load().items() if e.range}
    for name, s in registry().items():
        if s.scope != "participant" or s.group not in ctx.config.groups:
            continue
        meta = ctx.store.meta(name, ctx.pids()[0]) if ctx.pids() else None
        for table in (meta.tables if meta else ()):
            df = ctx.table(name, table)
            for col in set(df.columns) & set(cat):
                v = df[col].to_numpy(float) if df[col].dtype != object else np.empty(0)
                v = v[np.isfinite(v)]
                lo, hi = cat[col].range
                out = int(((v < lo) | (v > hi)).sum())
                if out:
                    frac = out / max(v.size, 1)
                    rep.add("numbers", f"{name}.{table}.{col} in plausible range",
                            "fail" if frac > 0.05 else "warn",
                            f"{out}/{v.size} outside [{lo:g}, {hi:g}]")


# -- figures ----------------------------------------------------------------------------------
def pid_leaks(texts, rx: re.Pattern = PID_RE) -> list[str]:
    """Study ids (``s01``) and session folder names (which encode id, sex, age and
    recording time) found in ``texts`` (sorted, unique)."""
    from biomech.results.release import SESSION_RE

    found = {m.group(0) for t in texts if t for m in rx.finditer(str(t))}
    found |= {m.group(0) for t in texts if t for m in SESSION_RE.finditer(str(t))}
    return sorted(found)


def _spec_text(spec) -> list[str]:
    from biomech.viz.spec import FigureSpec

    # The id names the exported files (figures/<id>/, previews/<id>.png): scanned too, with
    # "_"/"-" as word separators ("fig--pid-s03", "x_s03").
    sid = getattr(spec, "id", "")
    out = [" ".join(m.group(0) for m in NAME_RE.finditer(sid)) if sid else "",
           getattr(spec, "title", ""), getattr(spec, "caption", ""), getattr(spec, "notes", "")]
    if isinstance(spec, FigureSpec):
        for p in spec.panels:
            out += [p.title, p.x.label, p.y.label, p.note, *(p.x.ticklabels or []),
                    *(p.y.ticklabels or [])]
            for s in p.series:
                out += [s.label or "", *(s.text or [])]
    else:
        out += [str(v) for r in spec.rows for v in r.values()]
    return [t for t in out if t]


def legend_conflicts(spec) -> list[str]:
    """Legend entries of one panel that share their look (D2): lines by colour and dash,
    marker series by colour and marker, areas by colour. Unlabelled series (grey context)
    may repeat freely."""
    from biomech.viz import style
    from biomech.viz.spec import FigureSpec

    if not isinstance(spec, FigureSpec):
        return []
    out = []
    for i, p in enumerate(spec.panels):
        seen: dict[tuple, str] = {}
        for s in p.legend_entries():
            col, dash, mark = style.resolve(s)
            if s.kind in ("scatter", "errorbar", "scatter3d"):
                key = ("mark", col, s.style.get("ms", mark))
            elif s.kind in ("band", "fill", "bar", "barh"):
                key = ("area", col)
            else:
                key = ("line", col, dash)
            if key in seen and seen[key] != s.label:
                tag = p.heading()[0] or str(i)
                out.append(f"panel {tag}: {seen[key]!r} and {s.label!r} look alike {key[1:]}")
            seen.setdefault(key, s.label or "")
    return out


def code_leaks(texts) -> list[str]:
    """Internal plan codes (``D6``, ``X2``, ``C1``, ``E17``) in reader-facing ``texts``."""
    from biomech.viz.views import vocab as V

    return sorted({c for t in texts if t for c in V.internal_codes(str(t))})


def exported_code_leaks(root: Path) -> dict[str, list[str]]:
    """Exported items (``figures/<id>/spec.json``, ``tables/<id>.json``) whose reader-facing
    text still carries internal plan codes: item id -> codes."""
    from biomech.viz import serialize

    bad: dict[str, list[str]] = {}
    for f in sorted(Path(root).glob("figures/*/spec.json")) + sorted(
            Path(root).glob("tables/*.json")):
        try:
            spec = (serialize.load_figure(f) if f.name == "spec.json"
                    else serialize.load_table(f))
        except Exception:                      # not a spec file
            continue
        texts = _spec_text(spec)[1:]           # [0] = study ids in the item id
        codes = code_leaks(texts)
        if codes:
            bad[spec.id] = codes
    return bad


def _req_key(vid: str, req: Any) -> tuple:
    """Identity of a view request (two equal keys build the same spec)."""
    return (vid, req.pid, repr(req.phase), req.perspective, req.align, req.stat,
            repr(req.window_s), repr(sorted(req.options.items(), key=lambda kv: str(kv[0]))))


def _view_figure_checks(ctx, vid: str, pid: str | None, png_dir: Path
                        ) -> tuple[list[tuple], list[str], dict[tuple, tuple]]:
    """The figure checks of one view (every perspective): ``Report.add`` argument tuples in
    order, the PNGs written (in order), and the store reads of each build by request key
    (reused by the view-coverage check instead of building the same variant again)."""
    import matplotlib.pyplot as plt

    from biomech.results.access import track
    from biomech.viz import layout as L
    from biomech.viz import render_mpl
    from biomech.viz.render_plotly import to_plotly
    from biomech import catalog
    from biomech.check.text import raw_id_cells, text_issues
    from biomech.viz.spec import FigureSpec
    from biomech.viz.views import registry as R
    from biomech.viz.views import vocab as V

    v = R.get(vid)
    adds: list[tuple] = []
    pngs: list[str] = []
    reads: dict[tuple, tuple] = {}
    for persp in (v.perspectives or (None,)):
        label = f"{v.id}" + (f" [{persp}]" if persp else "")
        key = None
        catalog.misses(reset=True)
        V.unknown_terms(reset=True)
        with track() as log:
            try:
                req = R.request(ctx, v.id, perspective=persp,
                                pid=pid if (persp not in (None, "cohort") or "pid" in v.params)
                                else None)
                key = _req_key(v.id, req)
                spec = R.build(v.id, req)
                err = None
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        if key is not None:
            reads[key] = (set(log.tables), set(log.arrays), err, set(log.columns))
        if err is not None:
            adds.append(("figures", f"{label} builds", False, err))
            continue
        leaked = pid_leaks(_spec_text(spec))
        adds.append(("privacy", f"{label}: no participant ids", not leaked, ", ".join(leaked)))
        # Reader text: dashboards are reported, paper items fail in
        # check_paper_text.
        iss = text_issues(V.reader_spec(spec))
        miss = sorted(catalog.misses(reset=True))
        unk = sorted(f"{d}:{x}" for d, x in V.unknown_terms(reset=True))
        iss += ([f"labels guessed: {', '.join(miss[:8])}"] if miss else []) + \
               ([f"values without vocabulary: {', '.join(unk[:8])}"] if unk else [])
        adds.append(("text", f"{label}: reader text", "pass" if not iss else "warn",
                     "; ".join(iss)[:400]))
        ids = raw_id_cells(V.reader_spec(spec)) if any("raw identifiers" in x for x in iss) \
            else {}
        if miss or unk or ids:          # the full lists, collected into text_gaps.json
            adds.append((GAPS, label, "pass", json.dumps({"labels": miss, "values": unk,
                                                          "identifiers": ids})))
        if not isinstance(spec, FigureSpec):
            adds.append(("figures", f"{label} builds", bool(spec.rows), f"{len(spec.rows)} rows"))
            continue
        if v.kind == "paper":
            conf = legend_conflicts(spec)
            adds.append(("figures", f"{label}: legend entries distinguishable", not conf,
                         "; ".join(conf[:3])))
        try:
            b = render_mpl.build(spec)
            ov = render_mpl.text_overlaps(b)
            lay = L.check(b.layout)
            path = png_dir / f"{spec.id}.png"
            b.fig.savefig(path, dpi=110, facecolor=b.fig.get_facecolor())
            plt.close(b.fig)
            pngs.append(str(path))
            adds.append(("figures", f"{label}: no overlapping or clipped text", not ov,
                         "; ".join(ov[:3]), f"png/{path.name}"))
            adds.append(("figures", f"{label}: layout check", not lay, "; ".join(lay[:3])))
            for theme in ("light", "dark"):
                to_plotly(spec, theme=theme)
            adds.append(("figures", f"{label}: Plotly light/dark", True))
        except Exception as exc:
            adds.append(("figures", f"{label} renders", False, f"{type(exc).__name__}: {exc}"))
    return adds, pngs, reads


_WORKER_CTX: dict[Any, Any] = {}


def _worker_ctx(config, cohort):
    """One DataContext per worker process (and study), reused across its tasks."""
    from biomech.io.context import DataContext

    k = (str(config.source), str(config.read_dir))
    if k not in _WORKER_CTX:
        _WORKER_CTX[k] = DataContext(config, cohort)
    return _WORKER_CTX[k]


def _figure_task(config, cohort, vids: list[str], pid: str | None, png_dir: Path
                 ) -> list[tuple[str, tuple]]:
    import warnings

    ctx = _worker_ctx(config, cohort)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [(vid, _view_figure_checks(ctx, vid, pid, png_dir)) for vid in vids]


def _pool(n_jobs: int):
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    n = min(8, os.cpu_count() or 1) if n_jobs == 0 else max(1, int(n_jobs))
    return ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn")), n


def check_figures(rep: Report, ctx, out: Path, *, pgf: bool = True,
                  kinds: tuple[str, ...] = ("paper", "dashboard", "data"),
                  n_jobs: int = 1, reads: dict[tuple, tuple] | None = None,
                  groups: Any = None) -> list[str]:
    """Every view x perspective builds, renders (no overlaps, layout, Plotly) and leaks no id;
    the paper export compiles. ``n_jobs`` (1 = in-process, 0 = up to 8): views in worker
    processes, results added in view order (the same report). ``reads``, when given,
    collects the store reads of every build (``check_view_coverage`` reuses them)."""
    from biomech.viz.views import registry as R

    png_dir = out / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    pngs: dict[str, list[Path]] = {}
    pid = ctx.pids()[0] if ctx.pids() else None
    groups = R.study_groups(ctx if groups is None else groups)
    views = [v for v in R.views(groups=groups) if v.kind in kinds]
    results: dict[str, tuple] = {}
    # Worker processes see the registered views (not ones patched in at run time).
    if n_jobs != 1 and len(views) > 1:
        pool, n = _pool(n_jobs)
        with pool:
            chunks = [[v.id for v in views[k::n * 2]] for k in range(n * 2)]
            futs = [pool.submit(_figure_task, ctx.config, ctx.cohort, c, pid, png_dir)
                    for c in chunks if c]
            for f in futs:
                results.update(f.result())
        paths = [p for v in views for p in results[v.id][1]]
        if len(paths) != len(set(paths)):          # two views write one PNG: keep the order
            results = {}
    if not results:
        results = {v.id: _view_figure_checks(ctx, v.id, pid, png_dir) for v in views}
    for v in views:
        adds, files, rd = results[v.id]
        for a in adds:
            rep.add(*a)
        pngs.setdefault(v.kind, []).extend(Path(f) for f in files)
        if reads is not None:
            reads.update(rd)
    if pgf and shutil.which("lualatex"):
        from biomech.viz.export import export

        res = export(ctx, None, out / "paper", previews=False, compile_pdf=True,
                     check_size=True, n_jobs=n_jobs, groups=groups)
        rep.add("figures", "paper export compiles (LuaLaTeX)", not res["errors"],
                "; ".join(res["errors"])[:400], link="paper/preview.pdf")
        rep.add("figures", "paper figures fit their IEEE width, no overfull boxes",
                not res["problems"], "; ".join(res["problems"])[:400])
        texts = [p.read_text(errors="ignore") for p in (out / "paper").rglob("*.tex")]
        leaked = pid_leaks(texts)
        rep.add("privacy", "exported TeX has no participant ids", not leaked, ", ".join(leaked))
        codes = exported_code_leaks(out / "paper")
        rep.add("figures", "exported paper items: no internal plan codes in reader-facing text",
                not codes, "; ".join(f"{k}: {', '.join(v)}" for k, v in codes.items())[:400])
        check_file_names(rep, out / "paper", "exported files")
    elif pgf:
        rep.add("figures", "paper export compiles", "skip", "LuaLaTeX not installed")
    return contact_sheets(pngs, out, n_jobs=n_jobs)


def _sheet(chunk: list[Path], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    cols = 3
    rows = -(-len(chunk) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 3.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), chunk):
        ax.imshow(mpimg.imread(f))
        ax.set_title(f.stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def contact_sheets(pngs: dict[str, list[Path]], out: Path, per_sheet: int = 12,
                   n_jobs: int = 1) -> list[str]:
    """One montage per view kind (and page), for the visual pass (``n_jobs``: sheets in
    worker processes; the same files)."""
    jobs = []
    for kind, files in pngs.items():
        for page in range(0, len(files), per_sheet):
            jobs.append((files[page:page + per_sheet], f"sheet_{kind}_{page // per_sheet + 1}.png"))
    if n_jobs != 1 and len(jobs) > 1:
        pool, _ = _pool(n_jobs)
        with pool:
            for f in [pool.submit(_sheet, chunk, out / name) for chunk, name in jobs]:
                f.result()
    else:
        for chunk, name in jobs:
            _sheet(chunk, out / name)
    return [name for _, name in jobs]


# -- videos -------------------------------------------------------------------------------------
def _ffprobe(path: Path) -> dict[str, Any]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=codec_name,width,height,r_frame_rate,"
                        "nb_read_frames:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, timeout=300)
    return json.loads(r.stdout or "{}")


def check_file_names(rep: Report, root: Path, what: str) -> None:
    """Privacy: no study id in any file or directory name below ``root``."""
    if not root.is_dir():
        return
    names = [str(p.relative_to(root)) for p in root.rglob("*")]
    leaked = pid_leaks(names, NAME_RE)
    rep.add("privacy", f"{what}: no participant ids in file names", not leaked,
            ", ".join(leaked) or f"{len(names)} names checked")


def check_video_privacy(rep: Report, video_dir: Path) -> None:
    """Privacy: video file names and their manifests (JSON) carry no study id."""
    if not video_dir.is_dir():
        return
    check_file_names(rep, video_dir, "videos")
    for man in sorted(video_dir.glob("*.json")):
        leaked = pid_leaks([man.read_text(errors="ignore")])
        rep.add("privacy", f"{man.name}: no participant ids in the manifest", not leaked,
                ", ".join(leaked))


def check_videos(rep: Report, video_dir: Path,
                 read_timecode: Callable[[np.ndarray], float | None] | None = None,
                 n_jobs: int = 4) -> None:
    check_video_privacy(rep, video_dir)
    files = sorted(video_dir.glob("*.mp4")) if video_dir.is_dir() else []
    if not files:
        rep.add("videos", "videos present", "skip", f"no videos in {video_dir}")
        return
    if shutil.which("ffprobe") is None:
        rep.add("videos", "ffprobe available", "skip", "ffprobe not installed")
        return
    # ffprobe (-count_frames decodes the whole stream) for all files in threads; the items
    # are added in file order as before.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, min(int(n_jobs), len(files)))) as ex:
        probes = list(ex.map(_ffprobe, files))
    for f, info in zip(files, probes):
        st = (info.get("streams") or [{}])[0]
        dur = float((info.get("format") or {}).get("duration", "nan"))
        w, h = int(st.get("width", 0)), int(st.get("height", 0))
        ok = st.get("codec_name") == "h264" and w % 2 == 0 and h % 2 == 0 and dur > 0
        rep.add("videos", f"{f.name}: stream", ok,
                f"{st.get('codec_name')} {w}x{h} {st.get('r_frame_rate')} {dur:.2f} s, "
                f"{st.get('nb_read_frames')} frames")
        man = f.with_suffix(".json")
        meta = json.loads(man.read_text()) if man.is_file() else None
        rep.add("videos", f"{f.name}: manifest", meta is not None,
                "" if meta else "no manifest next to the video")
        if meta is not None and "expected_duration_s" in meta:
            rep.add("videos", f"{f.name}: duration = timeline",
                    abs(dur - float(meta["expected_duration_s"])) < 1.5 / float(
                        meta.get("fps", 30)), f"{dur:.3f} vs {meta['expected_duration_s']:.3f} s")
        if meta is not None and read_timecode is not None and "frames" in meta:
            errs = _timecode_errors(f, meta, read_timecode)
            rep.add("videos", f"{f.name}: embedded timecode matches the timeline", not errs,
                    "; ".join(errs[:3]) or f"{len(meta['frames'])} frames checked")
        if meta is not None:
            rep.add("privacy", f"{f.name}: camera frames declared",
                    "with_camera" in meta, f"with_camera = {meta.get('with_camera')}")


def _timecode_errors(path: Path, meta: dict[str, Any],
                     read_timecode: Callable[[np.ndarray], float | None]) -> list[str]:
    """``meta['frames']``: [[frame_index, master_time_us], ...] sampled by the renderer."""
    from biomech.anim.render_video import decode

    errs = []
    fps = float(meta.get("fps", 30))
    want = [(int(idx), float(t_us)) for idx, t_us in meta["frames"][:8]]
    # Sequential decode, not ffmpeg -ss: a seek can land on a neighbouring frame, which at
    # slow motion is a timecode error of a whole frame.
    try:
        imgs = decode(path, [i for i, _ in want])
    except (OSError, ValueError, KeyError) as exc:
        return [f"not decodable: {exc}"]
    if len(imgs) < len(want):
        return [f"only {len(imgs)} of {len(want)} sampled frames decodable"]
    tol = float(meta.get("timecode_tol_us", 1e6 / fps))
    for (idx, t_us), img in zip(sorted(want), imgs):
        got = read_timecode(img)
        if got is None or abs(got - t_us) > tol:
            errs.append(f"frame {idx}: timecode {got} vs {t_us:.0f}")
    return errs


# -- docs / privacy -------------------------------------------------------------------------------
def check_repo(rep: Report, repo: Path | None) -> None:
    if repo is None:
        rep.add("privacy", "repository checks", "skip", "not running from a checkout")
        return
    r = subprocess.run(["git", "ls-files", *SUBJECT_DATA_DIRS], cwd=repo,
                       capture_output=True, text=True, timeout=60)
    tracked = [f for f in r.stdout.splitlines() if f]
    rep.add("privacy", "no subject data tracked by git", not tracked,
            ", ".join(tracked[:5]))
    test = repo / "tests" / "test_docs_links.py"
    if test.is_file():
        import sys

        r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(test)], cwd=repo,
                           capture_output=True, text=True, timeout=600)
        rep.add("docs", "links, anchors and stale defaults", r.returncode == 0,
                r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])


def check_hardware(rep: Report, study: str) -> None:
    """hardware/mechanical/parts.yaml: files and usage valid (fail); the study's part versions
    stated (warn while not: the setup figures then show the original JTS renders)."""
    from biomech import hwparts

    try:
        man = hwparts.load()
    except Exception as e:                      # noqa: BLE001 (report, do not stop the check)
        rep.add("hardware", "parts manifest readable", False, str(e)[:300])
        return
    if not man.parts:
        rep.add("hardware", "parts manifest", "skip", "no hardware/mechanical/parts.yaml")
        return
    errors, _ = man.problems()
    rep.add("hardware", "parts manifest valid", not errors, "; ".join(errors[:5]))
    if study in man.usage:
        missing = [p for p, v in man.usage[study].items() if not v]
        rep.add("hardware", "part versions of the study specified",
                "warn" if missing else True,
                ("not yet specified: " + ", ".join(missing)) if missing else "")


# -- entry point ----------------------------------------------------------------------------------
def check_paper_text(rep: Report, ctx) -> None:
    """Every item of the study's active paper sets reads on its own: no plan
    codes, raw identifiers or raw values, no label guessed from a column name, every
    categorical value in the vocabulary. Every depiction (``form``) of an item counts, as the
    gallery offers each. One check per paper set; any issue fails it."""
    from biomech import papers

    for ps in papers.study_paper_sets(getattr(ctx.config, "source", None)):
        try:
            with papers.paper_context(ctx, ps) as pctx:
                _check_paper_set_text(rep, pctx, ps)
        except Exception as exc:                 # the paper set's store cannot be opened
            rep.add("text", f"{ps.name}: items build from the paper set's store", False,
                    f"{type(exc).__name__}: {exc}"[:600])


def _check_paper_set_text(rep: Report, ctx, ps) -> None:
    """:func:`check_paper_text` for one paper set, ``ctx`` reading its store."""
    from biomech.check.text import UNSTRUCTURED, caption_rubric_issues, item_issues
    from biomech.viz.views import registry as R

    n_max, n_adults = cohort_sizes(ctx)
    bad, plain, n = [], [], 0
    rubric: list[str] = []
    n_bad, n_open = [], []
    items = ps.variants()
    for it in items:
        forms = ([None] if "form" in it.params
                 else R.get(it.view_id).choice_values("form", ctx) or [None])
        default = R.get(it.view_id).defaults.get("form")
        for form in forms:
            n += 1
            label = it.label if form in (None, default) else f"{it.label} [{form}]"
            if it.params != next(i for i in ps.items if i.index == it.index).params:
                label += " (" + ", ".join(f"{k}: {v}" for k, v in it.params.items()) + ")"
            params = it.params if form is None else {**it.params, "form": form}
            built: list[Any] = []
            try:
                iss = item_issues(ctx, it.view_id, params, built)
            except Exception as exc:
                iss = [f"build error {type(exc).__name__}: {exc}"]
            if UNSTRUCTURED in iss:
                plain.append(label)
                iss.remove(UNSTRUCTURED)
            if iss:
                bad.append(f"{label}: {'; '.join(iss)}")
            if built:
                rub = caption_rubric_issues(built[0])
                if rub:
                    rubric.append(f"{label}: {'; '.join(rub)}")
                probs, known = caption_n_issues(built[0], n_max, n_adults,
                                                source_participants(ctx, built[0]))
                n_bad += [f"{label}: {p}" for p in probs]
                if not known:
                    n_open.append(label)
    rep.add("text", f"{ps.name}: every item reads on its own ({len(items)} items, "
            f"{n} depictions)", not bad, " | ".join(bad)[:600])
    rep.add("text", f"{ps.name}: every caption structured (marks, n, statistic)",
            "pass" if not plain else "warn",
            f"{len(plain)} of {n} not yet: {', '.join(plain[:20])}" if plain else "")
    rep.add("text", f"{ps.name}: every caption self-explanatory (what, encodings, n, "
            "statistic, table notes, length)", "pass" if not rubric else "warn",
            f"{len(rubric)} of {n}: " + " | ".join(rubric)[:1500] if rubric else "")
    rep.add("n", f"{ps.name}: caption n within the cohort ({n_max}; adults {n_adults}), "
            "equal to the participants shown and not above those of its source tables",
            not n_bad, " | ".join(n_bad)[:600])
    if n_open:
        rep.add("n", f"{ps.name}: caption n not checkable against the data", "warn",
                f"{len(n_open)} of {n} items state a participant count but neither label "
                "their participants (P1, ...) nor name a source table with participants: "
                + ", ".join(n_open[:20]))


# -- n -------------------------------------------------------------------------------
# "14 participants", "13 shape-usable participants", "84 channels of 14 participants"
CAPTION_PARTICIPANTS_RE = re.compile(r"(?<![\w.])(\d+)\s+(?:[A-Za-z][\w-]*\s+){0,2}"
                                     r"participants?\b")
PSEUDONYM_RE = re.compile(r"(?<![\w])P(\d+)(?![\w])")
STALE_N_RE = re.compile(r"\bn\s*(?:=|≈|~)\s*(\d+)\b")
STALE_N = (12,)                               # cohort sizes of earlier rounds (r7: n = 12)
OTHER_UNITS = ("cycle", "channel", "window", "trial", "frame", "electrode", "take", "fold",
               "session", "segment", "task", "muscle", "hold", "landmark", "sample", "cell",
               "step", "slot", "condition", "point", "variant", "feature", "class")


def cohort_sizes(ctx: Any) -> tuple[int, int]:
    """(participants, adults) of the context's cohort: the second count is the participants not
    flagged ``outside_protocol`` (the ``reporting_population`` variant)."""
    inc = ctx.cohort.included()
    adults = [p for p in inc if "outside_protocol" not in (p.flags or [])]
    return len(inc), len(adults)


def caption_participants(text: str, n_max: int) -> list[int]:
    """Participant counts a caption's n part states: "k participants" (up to two words in
    between), else the distinct pseudonyms it names ("P1, comparison take" -> 1)."""
    ks = [int(m.group(1)) for m in CAPTION_PARTICIPANTS_RE.finditer(text or "")]
    if ks:
        return ks
    ps = {int(m.group(1)) for m in PSEUDONYM_RE.finditer(text or "") if int(m.group(1)) <= n_max}
    return [len(ps)] if ps else []


def spec_pseudonyms(spec: Any, n_max: int) -> set[int]:
    """Participants an item labels (P1 ... P<n_max> as tick labels, legend entries, panel
    titles or table cells, also with a status in parentheses: "P3 (flagged)"); "P95" and other
    non-pseudonyms above the cohort size are not."""
    from biomech.viz.spec import FigureSpec

    texts: list[str] = []
    if isinstance(spec, FigureSpec):
        for p in spec.panels:
            texts.append(p.title or "")
            for a in (p.x, p.y):
                if a is not None:
                    texts += [str(t) for t in (a.ticklabels or [])]
            texts += [s.label or "" for s in p.series]
    else:
        texts += [v for r in spec.rows for v in r.values() if isinstance(v, str)]
    out = set()
    for t in texts:
        m = re.fullmatch(r"\s*P(\d+)(?:\s*\([^()]*\))?\s*", t)
        if m and 1 <= int(m.group(1)) <= n_max:
            out.add(int(m.group(1)))
    return out


# Cohort-table columns that count participants (largest value = the participants a row uses).
N_COLUMNS = ("n_participants", "n_subjects", "n_pid", "n")


def source_participants(ctx: Any, spec: Any) -> int | None:
    """Participants in the item's source tables (``spec.sources``, "stage:table"): distinct
    ids over the ``pid`` / ``pid_*`` columns of a per-participant table, else the largest
    participant count of a cohort table (:data:`N_COLUMNS`, the first present); the largest
    over the sources. A spec built from arrays of one named participant (``meta["pid"]``)
    counts as 1. None when no source carries participants (schematics, cohort arrays)."""
    import pandas as pd

    best = None
    for src in getattr(spec, "sources", None) or []:
        stage, _, table = str(src).partition(":")
        try:
            df = ctx.table(stage, table)
        except Exception:                       # noqa: BLE001 -- arrays, other study, absent
            continue
        if not len(getattr(df, "columns", ())) or df.empty:
            continue
        ids = [c for c in df.columns if c == "pid" or c.startswith("pid_")]
        if ids:
            v = pd.concat([df[c] for c in ids]).dropna().astype(str)
            n = int(v[v != ""].nunique())
        else:
            col = next((c for c in N_COLUMNS if c in df.columns), None)
            x = pd.to_numeric(df[col], errors="coerce") if col else pd.Series(dtype=float)
            if not x.notna().any():
                continue
            n = int(x.max())
        best = n if best is None else max(best, n)
    pid = (getattr(spec, "meta", None) or {}).get("pid")
    if best is None and isinstance(pid, str) and pid:
        return 1
    return best


def caption_n_issues(spec: Any, n_max: int, n_adults: int,
                     data_n: int | None = None) -> tuple[list[str], bool]:
    """Problems of the participant count a structured caption states: above the cohort size
    (reporting-population captions: above the adults), unequal to the participants the item labels
    (P1, ...), or above the participants in the item's source tables (``data_n``, see
    :func:`source_participants`; a caption may state fewer: the usable subset). Second
    value: False only when the caption states a count that none of these can check."""
    cap = (getattr(spec, "meta", None) or {}).get("caption") or {}
    text = str(cap.get("n") or "")
    ks = caption_participants(text, n_max)
    bound = n_adults if re.search(r"\badults?\b", text) else n_max
    probs = [f"caption says {k} participants, more than {bound}" for k in ks if k > bound]
    shown = spec_pseudonyms(spec, n_max)
    if ks and len(shown) >= 2 and len(shown) not in ks:
        probs.append(f"caption says n = {', '.join(map(str, ks))}, the item labels "
                     f"{len(shown)} participants")
    if ks and data_n is not None:
        probs += [f"caption says {k} participants, its source tables hold {data_n}"
                  for k in ks if k > data_n]
    # a single-participant item names its participant first ("P1, comparison take"; a
    # percentile such as "P95" inside the text is not one)
    named = [int(m.group(1)) for m in [re.match(r"\s*P(\d+)\b", text)] if m]
    probs += [f"caption names P{k}, the cohort has {n_max}" for k in named if k > n_max]
    known = not ks or len(shown) >= 2 or data_n is not None or bool(named)
    return probs, known


def number_n_issues(nums: list[Any], n_max: int, n_adults: int,
                    cited_max: dict[str, int] | None = None,
                    n_analysis: int | None = None) -> list[str]:
    """In-text numbers whose n (participants) exceeds the cohort of the store they come from:
    ``n_max`` (the paper set's store; 13 for an ``reporting_population`` paper), cited
    numbers their own study's cohort (``cited_max`` by study file), the ``primary`` column
    of a variant comparison (``sensitivity:<variant>``: the primary run's value, e.g. the
    all-participant robustness numbers) the analysis of record (``n_analysis``, default
    ``n_max``), and an reporting-population number (key with ``Adults``, or the ``variant`` column of
    the ``reporting_population`` comparison) the adults."""
    out = []
    for n in nums:
        if n.n is None or getattr(n, "n_of", "participants") != "participants":
            continue
        cite = getattr(n, "cite", None)
        src = str(n.source)
        bound = (cited_max or {}).get(cite["study"], n_max) if cite else n_max
        what = "the cited study's participants" if cite else "participants"
        if src.startswith("sensitivity:") and getattr(n, "column", "") == "primary":
            bound, what = (n_analysis if n_analysis is not None else n_max), \
                "participants of the analysis"
        elif "Adults" in n.key or src.startswith("sensitivity:reporting_population"):
            bound, what = min(bound, n_adults), "adults"
        if n.n > bound:
            out.append(f"{n.key}: n = {n.n} > {bound} {what}")
    return out


def check_numbers_n(rep: Report, ctx: Any) -> None:
    """Every in-text number of the study's paper sets, resolved from the paper set's store
    (its variant at its pinned snapshot): a participant n never exceeds that store's cohort
    (dataset paper on ``reporting_population``: 13), the primary values of a variant comparison the
    analysis of record (``ctx``: 14), reporting-population numbers the adults."""
    from biomech import papers
    from biomech.viz import numbers as N

    n_analysis = cohort_sizes(ctx)[0]
    for ps in papers.study_paper_sets(getattr(ctx.config, "source", None)):
        if ps.numbers is None:
            continue
        sizes: dict[str, int] = {}

        def opener(study: str, snap: str, variant: str | None = None) -> Any:
            c = N.open_cited(study, snap, variant)
            sizes[study] = len(c.pids())
            return c

        try:
            with papers.paper_context(ctx, ps) as pctx:
                nums = N.resolve(pctx, N.load(ps.numbers), opener)
                n_max, n_adults = cohort_sizes(pctx)
            probs = number_n_issues(nums, n_max, n_adults, sizes, n_analysis)
            k = sum(n.n is not None for n in nums)
            pop = f"{ps.variant}: " if ps.variant else ""
            rep.add("n", f"{ps.name}: in-text n within the cohort ({pop}{n_max}; adults "
                    f"{n_adults}; analysis {n_analysis})", not probs,
                    " | ".join(probs)[:600] if probs else f"{k} numbers with an n")
        except Exception as exc:
            rep.add("n", f"{ps.name}: in-text numbers resolve", False,
                    f"{type(exc).__name__}: {exc}"[:600])


def stale_n_lines(paths: list[Path], stale: tuple[int, ...] = STALE_N) -> list[str]:
    """``file:line`` of every "n = 12" / "n ≈ 12" (a cohort size of an earlier round) in
    ``paths``; a count of other units ("n = 12 cycles") is not one."""
    out = []
    for p in paths:
        try:
            lines = Path(p).read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for m in STALE_N_RE.finditer(line):
                if int(m.group(1)) not in stale:
                    continue
                nxt = line[m.end():].strip().lower()
                if nxt.startswith(OTHER_UNITS):
                    continue
                out.append(f"{Path(p).name}:{i}")
    return out


def check_stale_n(rep: Report, repo: Path | None, ctx: Any = None) -> None:
    """No stale cohort size in the paper notes and the active paper sets (warn)."""
    from biomech import papers

    if repo is None:
        return
    paths = sorted((Path(repo) / "docs" / "analysis").glob("an1[5-7]*.md"))
    study = getattr(getattr(ctx, "config", None), "source", None) if ctx is not None else None
    for ps in papers.study_paper_sets(study):
        paths.append(ps.path)
        if ps.numbers is not None:
            paths.append(ps.numbers)
    hits = stale_n_lines(paths)
    rep.add("n", "no stale n (n = 12) in the paper notes and the paper sets",
            "pass" if not hits else "warn", ", ".join(hits[:30]))


def check_exports(rep: Report, ctx) -> None:
    """Every exported target of the study's paper sets is consistent with the results (plan
    R7 integrity): files as in the manifest, and every item rebuilt from the pinned snapshot
    equals the exported one. A target not exported yet is reported, not failed."""
    from biomech import papers
    from biomech.check import integrity as I

    for ps in papers.study_paper_sets(getattr(ctx.config, "source", None)):
        store = (f"{ps.variant} " if ps.variant else "") + f"snapshot {ps.snapshot}"
        try:
            with papers.paper_context(ctx, ps) as pctx:
                if ps.snapshot:
                    stale = snapshot_stale_stages(pctx.config)
                    rep.add("integrity", f"{ps.name}: pinned {store} matches the current "
                            "analysis code", "warn" if stale else True,
                            ("stale stages (re-run and freeze before the final export): "
                             + ", ".join(stale)) if stale else "")
                for name, tgt in ps.targets.items():
                    label = f"{ps.name}/{name}: export consistent with {store}"
                    if not (tgt.out_dir / "manifest.json").is_file():
                        rep.add("integrity", label, "warn", f"not exported yet ({tgt.out_dir})")
                        continue
                    probs = I.verify(tgt.out_dir, pctx)
                    rep.add("integrity", label, not probs, " | ".join(probs)[:600])
        except Exception as exc:                 # the paper set's store cannot be opened
            rep.add("integrity", f"{ps.name}: the paper set's store ({store}) opens", False,
                    f"{type(exc).__name__}: {exc}"[:600])


def check_bundles(rep: Report, ctx) -> None:
    """Every pinned snapshot of the study's paper sets (``snapshot:`` in its variant's store,
    ``record_snapshot:`` in the primary store) has a bundle that passes ``biomech bundle
    --verify --against`` (J9.4). A snapshot not bundled yet is a warning."""
    import yaml

    from biomech import papers
    from biomech import sensitivity as S
    from biomech.results import bundle as BD

    pins: list[tuple[str | None, str]] = []
    for ps in papers.study_paper_sets(getattr(ctx.config, "source", None)):
        if ps.snapshot:
            pins.append((ps.variant, ps.snapshot))
        rec = (yaml.safe_load(ps.path.read_text(encoding="utf-8")) or {}).get("record_snapshot")
        if rec:
            pins.append((None, str(rec)))
    for variant, snap in dict.fromkeys(pins):
        label = f"bundle of {(variant + ' ') if variant else ''}snapshot {snap}"
        try:
            snap_dir = S.store_config(ctx.config, variant, snap).read_dir
        except Exception as exc:
            rep.add("integrity", f"{label}: snapshot opens", False, f"{exc}"[:400])
            continue
        p = BD.bundle_path(snap_dir)
        if not p.is_file():
            rep.add("integrity", f"{label} verified against the snapshot", "warn",
                    f"not bundled yet: biomech bundle {snap}"
                    + (f" --variant {variant}" if variant else ""))
            continue
        probs = BD.verify(p, against=snap_dir)
        rep.add("integrity", f"{label} verified against the snapshot", not probs,
                " | ".join(probs)[:600])


def check_clip_data(rep: Report, ctx, video_dir: Path) -> None:
    """Every rendered video's data equals the results: the clip file
    next to each video against the same clip loaded from the store being checked."""
    from biomech.check.integrity import verify_clip

    clips = sorted(Path(video_dir).glob("*.clip.json")) if Path(video_dir).is_dir() else []
    for p in clips:
        try:
            probs = verify_clip(p, ctx)
        except Exception as exc:
            probs = [f"{type(exc).__name__}: {exc}"]
        rep.add("videos", f"{p.name.removesuffix('.clip.json')}: video data = results",
                not probs, " | ".join(probs)[:400])


def check_paper_videos(rep: Report, ctx) -> None:
    """The supplementary videos each paper set lists (``supplementary_videos``, J2.7):
    rendered (``<stem>.mp4`` with its manifest in the videos folder of the paper set's store or
    of the study), and rendered from the store the set pins (manifest ``snapshot`` and
    ``variant``). Not rendered: warn; another store or an unknown layout: fail."""
    from biomech import papers
    from biomech.anim.layouts import COHORT_LAYOUTS, LAYOUTS

    for ps in papers.study_paper_sets(getattr(ctx.config, "source", None)):
        if not ps.videos:
            continue
        try:
            with papers.paper_context(ctx, ps) as pctx:
                dirs = papers.video_dirs(pctx.config, ctx.config)
                pids = set(pctx.pids())
                for v in ps.videos:
                    name = f"{ps.name}: supplementary video {v.describe()}"
                    if v.layout not in LAYOUTS:
                        rep.add("videos", name, False, f"unknown layout {v.layout!r}")
                        continue
                    if v.layout not in COHORT_LAYOUTS and v.pid not in pids:
                        rep.add("videos", name, False, "its participant is not in the paper "
                                "set's store" if v.pid else "the layout needs a pid")
                        continue
                    mp4, meta = papers.find_video(v, dirs, pctx.pseudonym)
                    if mp4 is None:
                        rep.add("videos", name, "warn", "listed but not rendered (looked in "
                                + ", ".join(str(d) for d in dirs) + "; expected "
                                + papers.video_stem_for(v, pctx.pseudonym) + ".mp4)")
                        continue
                    probs = papers.video_store_problems(meta, ps)
                    rep.add("videos", name, not probs, "; ".join(probs) or
                            f"{mp4.name}, snapshot {meta.get('snapshot')}", link=str(mp4))
        except Exception as exc:                 # the paper set's store cannot be opened
            rep.add("videos", f"{ps.name}: supplementary videos checked", False,
                    f"{type(exc).__name__}: {exc}"[:600])


def snapshot_stale_stages(config) -> list[str]:
    """Stages of a frozen snapshot whose stored units the current code and configuration
    would compute differently (their keys changed since the freeze)."""
    from biomech.pipeline import stage_status

    try:
        st = stage_status(config, read=True)
    except Exception:                       # an unreadable snapshot is reported elsewhere
        return []
    return sorted(st[st["status"] == "stale"]["stage"].unique()) if not st.empty else []


GAPS = "text-gaps"           # internal rows: the full lists behind the "reader text" warnings


def write_text_gaps(rep: Report, path: Path) -> dict[str, Any]:
    """Move the full lists of guessed labels and values without vocabulary out of the
    report (the warnings show only the first few) into ``text_gaps.json``: per name the
    views that need it, so the catalog and vocabulary can be completed in one pass."""
    labels: dict[str, list[str]] = {}
    values: dict[str, list[str]] = {}
    idents: dict[str, list[str]] = {}
    keep = []
    for i in rep.items:
        if i.area != GAPS:
            keep.append(i)
            continue
        d = json.loads(i.detail)
        for n in d.get("labels", []):
            labels.setdefault(n, []).append(i.name)
        for n in d.get("values", []):
            values.setdefault(n, []).append(i.name)
        for col, ids in (d.get("identifiers") or {}).items():
            idents[f"{i.name} / {col}"] = ids
    rep.items = keep
    doc = {"labels": dict(sorted(labels.items())), "values": dict(sorted(values.items())),
           "identifiers": dict(sorted(idents.items()))}
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return doc


def check_outputs(config, out: str | Path | None = None, *, figures: bool = True,
                  pgf: bool = True, videos: bool = True, repo: Path | None = None,
                  kinds: tuple[str, ...] = ("paper", "dashboard", "data"),
                  n_jobs: int = 1, groups: Any = None) -> Report:
    """All checks (module docstring). ``n_jobs`` (1 = in-process, 0 = up to 8): views,
    view-coverage variants, contact sheets, export items in worker processes, ffprobe in
    threads -- the same report. ``groups`` (default: ``config.groups``): the study groups
    whose views are built and exported and whose outputs must be covered; the dataset
    number checks (ensembles, reference table) run for dataset studies only."""
    from biomech.io.context import DataContext
    from biomech.paths import repo_root

    ctx = DataContext(config)
    out = Path(out) if out else Path(config.out_dir) / "check"
    out.mkdir(parents=True, exist_ok=True)
    rep = Report(config.study)
    groups = tuple(groups) if groups else tuple(config.groups)
    check_coverage(rep, ctx)
    if "dataset" in groups:
        check_counts(rep, ctx)
        check_tables(rep, ctx)
    check_ranges(rep, ctx)
    reads: dict[tuple, tuple] = {}
    sheets = check_figures(rep, ctx, out, pgf=pgf, kinds=kinds, n_jobs=n_jobs,
                           reads=reads, groups=groups) if figures else []
    if figures:
        check_view_coverage(rep, ctx, reads=reads, n_jobs=n_jobs, groups=groups, out=out)
        check_paper_text(rep, ctx)
        check_exports(rep, ctx)
    if videos:
        try:
            from biomech.anim import read_timecode
        except Exception:                       # animator not installed
            read_timecode = None
        check_videos(rep, Path(config.out_dir) / "videos", read_timecode,
                     n_jobs=8 if n_jobs == 0 else max(4, n_jobs))
        check_clip_data(rep, ctx, Path(config.out_dir) / "videos")
        check_paper_videos(rep, ctx)
    check_numbers_n(rep, ctx)
    check_bundles(rep, ctx)
    check_repo(rep, repo if repo is not None else repo_root())
    check_hardware(rep, config.study)
    check_stale_n(rep, repo if repo is not None else repo_root(), ctx)
    write_text_gaps(rep, out / "text_gaps.json")
    rep.to_json(out / "report.json")
    rep.to_html(out / "report.html", sheets)
    ctx.close()
    return rep
