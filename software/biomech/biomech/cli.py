# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``biomech`` command line: a thin wrapper over the run API and the ResultStore."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import click
import pandas as pd

from biomech.cohort import draft_manifest, dump_manifest, load_cohort
from biomech.config import ConfigError, load_config
from biomech.pipeline import registry, run, stage_status
from biomech.results import ResultStore

_config_opt = click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
                           required=True, help="Study YAML (e.g. experiments/icorr2027/study.yaml).")
_data_opt = click.option("--data-dir", type=click.Path(file_okay=False), default=None,
                         help="Session root (overrides $JTSMM_DATA_DIR and the config).")
_out_opt = click.option("--out-dir", type=click.Path(file_okay=False), default=None,
                        help="Results root (default: data/analysis/<study>).")


_snap_opt = click.option("--snapshot", default=None,
                         help="Read results from a frozen snapshot (biomech snapshot) "
                              "instead of the live store.")


_variant_opt = click.option("--variant", default=None,
                            help="Read the store of a sensitivity variant (e.g. reporting_population; "
                                 "'primary': the primary analysis); --snapshot then names a "
                                 "snapshot of that store. With --paper: overrides the paper "
                                 "set's variant.")


def _load(config_path: str, data_dir: str | None, out_dir: str | None,
          snapshot: str | None = None, variant: str | None = None, pset=None):
    """The configuration a command reads: the store of ``variant`` (None: primary) at
    ``snapshot`` (None: live) through :func:`biomech.sensitivity.store_config`, or with a paper
    set ``pset`` the set's store (its variant and pinned snapshot; ``variant`` / ``snapshot``
    override them, :meth:`biomech.papers.PaperSet.store_config`)."""
    from biomech import sensitivity as S

    try:
        cfg = load_config(config_path, data_dir=data_dir, out_dir=out_dir)
        if pset is not None:
            return pset.store_config(cfg, snapshot, variant)
        return S.store_config(cfg, variant, snapshot)
    except (ConfigError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc


def _store_label(cfg) -> str:
    """The store a command read, for its summary line ("reporting_population store, snapshot r11a")."""
    snap = getattr(cfg, "snapshot", None)
    return (f"{getattr(cfg, 'variant', None) or 'primary'} store, "
            + (f"snapshot {snap}" if snap else "live"))


@click.group()
def main() -> None:
    """Jaw biomechanics analysis over JTS-MM sessions (docs/analysis/)."""


@main.command("stages")
def stages_cmd() -> None:
    """List registered stages in dependency order."""
    for name, st in registry().items():
        deps = f" <- {', '.join(st.depends)}" if st.depends else ""
        click.echo(f"{name:<16} [{st.scope}, {st.group}] v{st.version}{deps}  "
                   f"{st.description}")


@main.group("cohort")
def cohort_grp() -> None:
    """Cohort manifest tools."""


@cohort_grp.command("draft")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--study", default="")
def cohort_draft(data_dir: str, study: str) -> None:
    """Print a manifest draft (first take per participant is primary)."""
    click.echo(dump_manifest(draft_manifest(data_dir, study)), nl=False)


@cohort_grp.command("check")
@_config_opt
@_data_opt
def cohort_check(config_path: str, data_dir: str | None) -> None:
    """Validate the manifest against the session folders; list the session folders the
    manifest does not name (informational: they are not part of the study)."""
    cfg = _load(config_path, data_dir, None)
    cohort = load_cohort(cfg.cohort_path, cfg.data_dir)
    problems = cohort.validate()
    for p in problems:
        click.echo(f"PROBLEM  {p}")
    for s in cohort.unlisted_sessions():
        click.echo(f"UNLISTED {s.pid}: {s.name} ({s.kind}"
                   + (f", {s.tag}" if s.tag else "") + ")")
    d = cohort.demographics()
    click.echo(f"included n={d['n']} (f={d['n_female']}, m={d['n_male']}), age "
               f"{d['age_mean']:.1f} ± {d['age_sd']:.1f} [{d['age_min']}–{d['age_max']}]")
    sys.exit(1 if problems else 0)


@contextlib.contextmanager
def _sigint_cancel():
    """Ctrl-C -> the pipeline's safe cancel (running units finish -- cohort stages such as
    ml_eval check it between configurations -- no new ones start, nothing half-written);
    a second Ctrl-C exits at once (store writes are atomic, gc restores leftovers)."""
    import signal
    import threading

    cancel = threading.Event()

    def on_sigint(signum, frame) -> None:
        if cancel.is_set():
            click.echo("second interrupt: exiting now", err=True)
            os._exit(130)
        cancel.set()
        click.echo("cancelling: running units finish, no new ones start "
                   "(Ctrl-C again to exit at once)", err=True)

    old = signal.signal(signal.SIGINT, on_sigint)
    try:
        yield cancel
    finally:
        signal.signal(signal.SIGINT, old)


@main.command("run")
@_config_opt
@_data_opt
@_out_opt
@click.option("--stage", "stages", multiple=True, help="Stage(s) to bring up to date (default: all).")
@click.option("--subject", "pids", multiple=True, help="Participant id(s), e.g. s01 (default: all).")
@click.option("--force", is_flag=True, help="Re-run the requested stages even if cached.")
@click.option("-j", "--jobs", "n_jobs", type=int, default=None,
              help="Parallel workers (default: one per participant, at most the "
                   "performance cores + 4; 1 = in-process).")
def run_cmd(config_path: str, data_dir: str | None, out_dir: str | None, stages: tuple[str, ...],
            pids: tuple[str, ...], force: bool, n_jobs: int) -> None:
    """Run stages (and their dependencies) for the cohort."""
    cfg = _load(config_path, data_dir, out_dir)

    def show(ev: dict) -> None:
        k = ev["kind"]
        if k == "stage_start":
            click.echo(f"[{ev['stage']}] {ev['n_todo']} to run, {ev['n_cached']} cached")
        elif k == "unit_done":
            msg = f"  {ev['unit']:<8} {ev['status']:<6} {ev['elapsed_s']:>7.2f}s"
            click.echo(msg + (f"  {ev['error']}" if ev.get("error") else ""))
        elif k == "unit_skipped":
            click.echo(f"  {ev['unit']:<8} skipped ({ev['reason']})")

    try:
        with _sigint_cancel() as cancel:
            summary = run(cfg, stages or None, pids or None, force=force, n_jobs=n_jobs,
                          on_event=show, cancel=cancel)
    except (RuntimeError, ValueError) as exc:          # concurrent run, invalid cohort
        raise click.ClickException(str(exc)) from exc
    click.echo(f"run {summary.run_id}: {summary.n_failed} failed"
               + (" (cancelled)" if cancel.is_set() else "") + f" -> {cfg.out_dir}")
    if cancel.is_set():
        sys.exit(130)
    sys.exit(1 if summary.n_failed else 0)


@main.command("sensitivity")
@_config_opt
@_data_opt
@_out_opt
@click.argument("name", required=False)
@click.option("--compare-only", is_flag=True, help="Only (re)write the comparison.")
@click.option("-j", "--jobs", "n_jobs", type=int, default=None,
              help="Parallel workers (default: one per participant, at most the "
                   "performance cores + 4; 1 = in-process).")
def sensitivity_cmd(config_path: str, data_dir: str | None, out_dir: str | None,
                    name: str | None, compare_only: bool, n_jobs: int) -> None:
    """Run a sensitivity variant into <out_dir>/sensitivity/NAME and compare it with the
    primary run, or with its base variant's run (``base:``, D36) (without NAME: list
    the variants)."""
    from biomech import sensitivity as S

    cfg = _load(config_path, data_dir, out_dir)
    if not name:
        for k, v in S.variants(cfg).items():
            skip = S.skipped_stages(cfg, k)
            click.echo(f"{k:<16} {v.get('description', '')}"
                       + (f" (not run: {', '.join(skip)})" if skip else ""))
        return
    try:
        vcfg = S.variant_config(config_path, name, data_dir=data_dir, out_dir=out_dir)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    if not compare_only:
        src = S.base_config(cfg, name)          # the base variant's store (D36) or the primary
        if not Path(src.out_dir, "cache").is_dir():
            raise click.ClickException(f"variant {name} builds on {S.base_name(cfg, name)}: "
                                       f"run `biomech sensitivity {S.base_name(cfg, name)}` first")
        n = S.seed_cache(src, vcfg)
        if n:
            click.echo(f"variant {name}: {n} units copied from "
                       f"{S.base_name(cfg, name) or 'the primary'} (re-used where unchanged)")
        skip = S.skipped_stages(cfg, name)
        if skip:
            click.echo(f"variant {name}: not run (skip_stages): {', '.join(skip)}")
        with _sigint_cancel() as cancel:
            summary = run(vcfg, S.variant_stages(cfg, name), None, n_jobs=n_jobs,
                          cancel=cancel)
        click.echo(f"variant {name}: run {summary.run_id}, {summary.n_failed} failed"
                   + (" (cancelled)" if cancel.is_set() else "") + f" -> {vcfg.out_dir}")
        if cancel.is_set():
            sys.exit(130)
        if summary.n_failed:
            # units the cohort manifest documents as unavailable (missing data) fail in the
            # primary run too: compare anyway, as check-outputs reports them (warn, not fail)
            from biomech.check.outputs import unavailable_units

            known = {**unavailable_units(cfg.cohort_path), **unavailable_units(vcfg.cohort_path)}
            bad = [f"{st}/{u}" for st, d in summary.statuses.items() for u, s in d.items()
                   if s == "failed" and (u, st) not in known]
            if bad:
                sys.exit(1)
            click.echo(f"variant {name}: only documented-unavailable units failed; comparing")
    df = S.write_comparison(cfg, name)
    changed = df[df["diff"].abs() > 1e-9] if not df.empty else df
    click.echo(f"{len(df)} values compared, {len(changed)} differ -> "
               f"{S.variant_dir(cfg, name) / S.COMPARISON_FILE}")


@main.command("bench")
@_config_opt
@_data_opt
@_out_opt
@click.option("--run", "run_id", default=None, help="Run id (default: the latest run).")
def bench_cmd(config_path: str, data_dir: str | None, out_dir: str | None,
              run_id: str | None) -> None:
    """Per-stage timing of a run (sum, max per unit, wall time) from its event log."""
    from biomech.results.bench import run_timing

    cfg = _load(config_path, data_dir, out_dir)
    store = ResultStore(cfg.out_dir)
    run_id = run_id or next(iter(store.runs()), None)
    if run_id is None:
        raise click.ClickException("no runs in this store")
    df, wall = run_timing(store.run_events(run_id))
    if df.empty:
        click.echo(f"run {run_id}: nothing ran")
        return
    path = cfg.out_dir / "bench" / f"{run_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    click.echo(df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    click.echo(f"run {run_id}: wall {wall:.1f} s -> {path}")


@main.command("snapshot")
@_config_opt
@_data_opt
@_out_opt
@click.argument("name", required=False)
@click.option("--note", default="", help="Free text stored with the snapshot.")
@click.option("--overwrite", is_flag=True, help="Replace an existing snapshot of that name.")
@click.option("--list", "list_", is_flag=True, help="List the snapshots.")
@click.option("--delete", "delete_", is_flag=True,
              help="Delete snapshot NAME (asks first; refused for a snapshot that a paper set "
                   "or numbers file pins).")
@click.option("--yes", is_flag=True, help="With --delete: do not ask.")
@click.option("--variant", default=None,
              help="Freeze (or list) the store of this sensitivity variant (e.g. reporting_population) "
                   "into <out>/sensitivity/VARIANT/snapshots/NAME, with its filtered cohort "
                   "manifest and comparison.")
def snapshot_cmd(config_path: str, data_dir: str | None, out_dir: str | None,
                 name: str | None, note: str, overwrite: bool, list_: bool,
                 delete_: bool, yes: bool, variant: str | None) -> None:
    """Freeze the current results as snapshot NAME (a copy; no re-run needed later).

    --out-dir is the primary results root also with --variant (pointing it at the variant
    directory would freeze the primary cohort manifest with the variant's results)."""
    from biomech.results import snapshot as SN

    cfg = _load(config_path, data_dir, out_dir, variant=variant)
    if list_ or not name:
        for n in SN.available(cfg.out_dir):
            m = SN.info(cfg.out_dir, n)
            click.echo(f"{n:<20} {m['created_utc']}  git {(m['git'].get('sha') or '?')[:9]}"
                       f"{'+dirty' if m['git'].get('dirty') else ''}  "
                       f"stale units {m.get('stale_units', 0)}  {m.get('note', '')}")
        return
    if delete_:
        if name not in SN.available(cfg.out_dir):
            raise click.ClickException(f"no snapshot {name} in {cfg.out_dir}")
        pins = SN.pinned(config_path).get(name)
        if pins:
            raise click.ClickException(f"snapshot {name} is pinned by {', '.join(pins)}; "
                                       "re-pin first")
        target = SN.path(cfg.out_dir, name)
        if not yes and not click.confirm(f"delete {target}?", default=False):
            raise click.Abort()
        SN.delete(cfg.out_dir, name)
        click.echo(f"deleted {target}")
        return
    try:
        p = SN.create(cfg, name, note=note, overwrite=overwrite)
    except (FileExistsError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    m = SN.info(cfg.out_dir, name)
    click.echo(f"snapshot {name}: {m['n_files']} files, {len(m['runs'])} run(s), "
               f"{m['stale_units']} stale unit(s) -> {p}")


@main.command("bundle")
@_config_opt
@_data_opt
@_out_opt
@click.argument("name", required=False)
@click.option("--variant", default=None,
              help="The snapshot belongs to this sensitivity variant's store (e.g. "
                   "reporting_population).")
@click.option("--verify", "verify_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Check a bundle (.h5) or archive (.tar.zst) instead of "
                                 "writing one.")
@click.option("--against", default=None,
              help="With --verify: compare with this snapshot (of --variant's store) value "
                   "for value.")
@click.option("--archive", is_flag=True,
              help="Also write <NAME>.tar.zst of the snapshot (SHA256SUMS inside, sidecar).")
@click.option("--overwrite", is_flag=True, help="Replace an existing bundle / archive.")
def bundle_cmd(config_path: str, data_dir: str | None, out_dir: str | None, name: str | None,
               variant: str | None, verify_path: str | None, against: str | None,
               archive: bool, overwrite: bool) -> None:
    """Bundle snapshot NAME into one lossless HDF5 file with checksums and provenance
    (bundle_NAME.h5 in the snapshot directory; J9.4), verified against the snapshot
    before it gets its final name. --verify FILE [--against NAME] checks one. Exit 1 on any
    problem."""
    from biomech.results import bundle as BD
    from biomech.results import snapshot as SN

    if verify_path:
        snap_dir = None
        if against:
            snap_dir = _load(config_path, data_dir, out_dir, against, variant).read_dir
        fn = BD.verify_archive if verify_path.endswith(BD.ARCHIVE_SUFFIX) else BD.verify
        probs = fn(verify_path, against=snap_dir)
        for p in probs:
            click.echo(f"PROBLEM {p}", err=True)
        click.echo(f"{Path(verify_path).name}: {len(probs)} problems"
                   + (f" (against {snap_dir})" if snap_dir else ""))
        sys.exit(1 if probs else 0)
    if not name:
        raise click.UsageError("name a snapshot (or --verify FILE)")
    cfg = _load(config_path, data_dir, out_dir, name, variant)
    pins = SN.pinned(config_path).get(name, [])
    if not pins:
        click.echo(f"note: snapshot {name} is not pinned by a paper set; readers refuse an "
                   "unpinned bundle by default", err=True)
    try:
        p = BD.write(cfg.read_dir, overwrite=overwrite, pinned_by=pins)
        click.echo(f"{p}: {p.stat().st_size / 1e6:.1f} MB, verified against snapshot {name}; "
                   f"sha256 {BD.sidecar(p).read_text().split()[0][:16]}")
        if archive:
            a = BD.write_archive(cfg.read_dir, overwrite=overwrite)
            click.echo(f"{a}: {a.stat().st_size / 1e6:.1f} MB, verified")
    except (FileExistsError, BD.BundleError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("describe")
@_config_opt
@_data_opt
@_out_opt
@_snap_opt
@click.argument("stage", required=False)
@click.argument("name", required=False)
@click.option("--json", "as_json", is_flag=True, help="Print the dictionary as JSON.")
def describe_cmd(config_path: str, data_dir: str | None, out_dir: str | None,
                 snapshot: str | None, stage: str | None, name: str | None,
                 as_json: bool) -> None:
    """Data dictionary of the stored results: tables (columns, units) and array families."""
    import json

    from biomech.results.schema import describe
    from biomech.viz.style import plain_unit

    cfg = _load(config_path, data_dir, out_dir, snapshot)
    d = describe(ResultStore(cfg.read_dir), stage)
    if as_json:
        click.echo(json.dumps(d, indent=1, default=str))
        return
    for st, v in d.items():
        for t, cols in v["tables"].items():
            if name and name != t:
                continue
            click.echo(f"{st}.{t}")
            for c, e in cols.items():
                u = f" [{plain_unit(e['unit'])}]" if e["unit"] else ""
                click.echo(f"    {c:<32} {e['label']}{u}  {e['description']}")
        for f, e in v["arrays"].items():
            if name and name != f:
                continue
            click.echo(f"{st} array {f}  {e.get('shape', '')} [{e.get('unit', '')}] "
                       f"{e.get('description', '')}")


@main.command("compare")
@_config_opt
@_data_opt
@_out_opt
@click.argument("old")
@click.argument("new", default="live")
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False), default=None)
def compare_cmd(config_path: str, data_dir: str | None, out_dir: str | None, old: str,
                new: str, csv_path: str | None) -> None:
    """Compare every cohort table of snapshot OLD with NEW (default: the live store)."""
    from biomech import sensitivity as S
    from biomech.results import snapshot as SN

    cfg = _load(config_path, data_dir, out_dir)
    try:
        a, b = ResultStore(SN.path(cfg.out_dir, old)), ResultStore(SN.path(cfg.out_dir, new))
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    df = S.compare(a, b)
    target = Path(csv_path) if csv_path else SN.root(cfg.out_dir) / f"compare_{old}_vs_{new}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    changed = df[df["diff"].abs() > 1e-12] if not df.empty else df
    click.echo(f"{len(df)} values compared, {len(changed)} differ -> {target}")


@main.command("gc")
@_config_opt
@_data_opt
@_out_opt
def gc_cmd(config_path: str, data_dir: str | None, out_dir: str | None) -> None:
    """Remove leftovers of interrupted writes from the result store."""
    cfg = _load(config_path, data_dir, out_dir)
    try:
        removed = ResultStore(cfg.out_dir).gc()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"removed {len(removed)} leftover(s)")


@main.command("status")
@_config_opt
@_data_opt
@_out_opt
def status_cmd(config_path: str, data_dir: str | None, out_dir: str | None) -> None:
    """Stage x participant status matrix against the current config."""
    cfg = _load(config_path, data_dir, out_dir)
    df = stage_status(cfg)
    click.echo(df.pivot(index="unit", columns="stage", values="status").fillna("").to_string())


@main.command("show")
@_config_opt
@_data_opt
@_out_opt
@click.argument("stage")
@click.argument("table")
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False), default=None)
@click.option("--snapshot", default=None,
              help="Read a frozen snapshot (e.g. r7, v5) instead of the live store.")
def show_cmd(config_path: str, data_dir: str | None, out_dir: str | None, stage: str, table: str,
             csv_path: str | None, snapshot: str | None) -> None:
    """Print (or save) one result table across participants (live store or --snapshot)."""
    cfg = _load(config_path, data_dir, out_dir, snapshot)
    df = ResultStore(cfg.read_dir).read_table(stage, table)
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    with pd.option_context("display.max_rows", 500, "display.width", 200,
                           "display.max_columns", 40, "display.float_format", "{:.3g}".format):
        click.echo(df.to_string(index=False))


@main.command("export")
@_config_opt
@_data_opt
@_out_opt
@click.option("--paper", "paper_yaml", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Paper set (paper_dataset.yaml, paper_systems.yaml) listing the items.")
@click.option("--target", default=None,
              help="Output target of the paper set (e.g. journal, conference; default: the "
                   "set's default_target).")
@click.option("--gallery", is_flag=True,
              help="With --paper: every item in every depiction it offers (form picker), to "
                   "<target folder>_gallery with a compiled preview and gallery.html (all "
                   "depictions, tables, the study's videos and hardware images side by side; "
                   "with --standalone the pictures are the compiled figures), to choose the "
                   "forms from.")
@click.option("--all", "export_all", is_flag=True,
              help="Every registered view instead of the paper items (see --kind, "
                   "--perspectives, --pid).")
@click.option("--kind", "kinds", multiple=True,
              type=click.Choice(["paper", "dashboard", "data"]),
              help="With --all: view kinds (repeatable; default: all three).")
@click.option("--perspectives", type=click.Choice(["all", "default"]), default="all",
              show_default=True, help="With --all: every perspective or only the default one.")
@click.option("--pid", "pids", multiple=True,
              help="With --all: participants for per-participant views (default: all).")
@click.option("--to", "target_dir", type=click.Path(file_okay=False), default=None,
              help="Target directory (default: the paper yaml out_dir, or <out>/export_all).")
@click.option("--compile", "compile_pdf", is_flag=True, help="Compile preview.pdf (LuaLaTeX).")
@click.option("--check-size", is_flag=True,
              help="Compile each figure alone and check it fits its column/text width.")
@click.option("--standalone", is_flag=True,
              help="Also compile each figure alone into figures/<id>/<id>.pdf (journal upload).")
@click.option("--venue", type=click.Choice(["ieee", "springer", "nature"]), default=None,
              help="Figure profile: widths, absolute font sizes, font family (default: the "
                   "paper yaml's 'venue:', else ieee).")
@_snap_opt
@_variant_opt
@click.option("-j", "--jobs", "n_jobs", default=0, show_default=True,
              help="Export workers (0 = up to 8, 1 = in-process); the same files.")
def export_cmd(config_path: str, data_dir: str | None, out_dir: str | None,
               paper_yaml: str | None, target: str | None, gallery: bool, export_all: bool,
               kinds: tuple[str, ...],
               perspectives: str, pids: tuple[str, ...], target_dir: str | None,
               compile_pdf: bool, check_size: bool, standalone: bool, venue: str | None,
               snapshot: str | None, variant: str | None, n_jobs: int) -> None:
    """Export figures (TikZ/pgfplots + PNG/PDF previews) and tables (booktabs).

    Either the items of a paper set for one target (--paper paper_dataset.yaml [--target
    journal]) or every registered view (--all).
    """
    from biomech import papers
    from biomech.io.context import DataContext
    from biomech.viz.export import all_items, export

    if bool(paper_yaml) == export_all:
        raise click.UsageError("give exactly one of --paper or --all")
    pset = papers.load(paper_yaml) if paper_yaml else None
    # A paper set names its store (variant, pinned snapshot); --variant / --snapshot override.
    cfg = _load(config_path, data_dir, out_dir, snapshot, variant, pset)
    ctx = DataContext(cfg)
    if export_all:
        items = all_items(ctx, tuple(kinds) or ("paper", "dashboard", "data"), perspectives,
                          list(pids) or None,
                          groups=getattr(getattr(ctx, "config", None), "groups", None))
        dest = Path(target_dir) if target_dir else cfg.out_dir / "export_all"
        click.echo(f"{len(items)} items -> {dest}")
        venue = venue or "ieee"
        from biomech.viz.style import VENUES

        if venue not in VENUES:
            raise click.UsageError(f"unknown venue {venue!r}; known: {', '.join(VENUES)}")
        res = export(ctx, items, dest, compile_pdf=compile_pdf, check_size=check_size,
                     standalone=standalone, venue=venue, n_jobs=n_jobs,
                     groups=getattr(getattr(ctx, "config", None), "groups", None),
                     meta={"all": True})
    else:
        from biomech.viz.export import export_paper_set

        try:
            tgt = pset.target(target)
        except ValueError as e:
            raise click.UsageError(str(e))
        venue = venue or tgt.venue
        click.echo(f"{pset.name} / {tgt.name}: {len(pset.items_for(tgt.name))} items "
                   f"({_store_label(cfg)})")
        if gallery:
            from biomech.viz.export import export_gallery

            res = export_gallery(ctx, pset, tgt.name, to=target_dir, venue=venue,
                                 check_size=check_size, standalone=standalone, n_jobs=n_jobs,
                                 groups=getattr(getattr(ctx, "config", None), "groups", None))
            res.setdefault("numbers", [])
            for g in res.get("gallery_html", []):
                click.echo(f"gallery page: {g}")
        else:
            res = export_paper_set(ctx, pset, tgt.name, to=target_dir, venue=venue,
                                   compile_pdf=compile_pdf, check_size=check_size,
                                   standalone=standalone, n_jobs=n_jobs,
                                   groups=getattr(getattr(ctx, "config", None), "groups", None))
        dest = Path(res["target_dir"][0])
        if res["numbers"]:
            click.echo(f"numbers: {len(res['numbers'])} -> numbers.tex")
    target_path = dest
    click.echo(f"venue: {venue}")
    click.echo(f"figures: {len(res['figures'])} ({', '.join(res['figures'][:12])}"
               f"{', ...' if len(res['figures']) > 12 else ''})\n"
               f"tables: {', '.join(res['tables'])}")
    if standalone:
        click.echo(f"standalone PDFs: {len(res['standalone'])}")
    for e in res["errors"]:
        click.echo(f"ERROR {e}", err=True)
    for e in res["problems"]:
        click.echo(f"PROBLEM {e}", err=True)
    click.echo(f"-> {target_path}")
    sys.exit(1 if res["errors"] or res["problems"] else 0)


@main.command("verify-export")
@_config_opt
@_data_opt
@_out_opt
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@_snap_opt
@_variant_opt
def verify_export_cmd(config_path: str, data_dir: str | None, out_dir: str | None, folder: str,
                      snapshot: str | None, variant: str | None) -> None:
    """Check an export FOLDER against the results: the files against its
    manifest.json, and every item rebuilt from the store it was exported from (its variant and
    snapshot, or --variant / --snapshot) against the exported one. Exit 1 on any difference."""
    from biomech.check import integrity as I
    from biomech.io.context import DataContext
    from biomech.viz import manifest as M

    man = M.read(Path(folder))
    prov = man.get("provenance") or {}
    cfg = _load(config_path, data_dir, out_dir, snapshot or prov.get("snapshot"),
                variant or prov.get("variant"))
    ctx = DataContext(cfg)
    try:
        probs = I.verify(Path(folder), ctx)
    finally:
        ctx.close()
    for p in probs:
        click.echo(f"PROBLEM {p}", err=True)
    click.echo(f"{len(man.get('items', []))} items, {len(probs)} problems "
               f"({_store_label(cfg)})")
    sys.exit(1 if probs else 0)


@main.command("numbers")
@_config_opt
@_data_opt
@_out_opt
@click.option("--paper", "paper_yaml", type=click.Path(exists=True, dir_okay=False),
              required=True, help="Paper set whose numbers file to resolve.")
@click.option("--target", default=None, help="Target folder to write numbers.tex/json to.")
@_snap_opt
@_variant_opt
@click.option("--dry-run", is_flag=True, help="Resolve and print only; write nothing.")
def numbers_cmd(config_path: str, data_dir: str | None, out_dir: str | None, paper_yaml: str,
                target: str | None, snapshot: str | None, variant: str | None,
                dry_run: bool) -> None:
    """Resolve the paper set's in-text numbers from its store (its variant at its pinned
    snapshot), print them and write numbers.tex / numbers.json to the target
    folder."""
    from biomech import papers
    from biomech.io.context import DataContext
    from biomech.viz import numbers as N
    from biomech.viz import serialize

    pset = papers.load(paper_yaml)
    if pset.numbers is None:
        raise click.UsageError(f"{paper_yaml} names no numbers file (numbers: ...)")
    ctx = DataContext(_load(config_path, data_dir, out_dir, snapshot, variant, pset))
    try:
        nums = N.resolve(ctx, N.load(pset.numbers))
        tgt = pset.target(target)
        if not dry_run:
            N.write(tgt.out_dir, nums, serialize.provenance(ctx), pset.numbers)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise click.ClickException(f"{pset.numbers.name} ({_store_label(ctx.config)}): "
                                   f"{exc}") from exc
    finally:
        ctx.close()
    for n in nums:
        ci = f" [{n.ci_text[0]}, {n.ci_text[1]}]" if n.ci_text else ""
        click.echo(f"\\{n.macro():28s} {n.text}{ci}{' n=' + str(n.n) if n.n is not None else ''}")
    click.echo(f"{len(nums)} numbers ({_store_label(ctx.config)})"
               + ("" if dry_run else f" -> {tgt.out_dir / 'numbers.tex'}"))


@main.command("check-docs")
@click.argument("docs", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--verbose", is_flag=True, help="List every hit, not only the counts.")
@click.option("--fill", is_flag=True,
              help="Rewrite the values quoted after macros with the exports' current texts "
                   "(one number per macro), then check.")
def check_docs_cmd(docs: tuple[str, ...], verbose: bool, fill: bool) -> None:
    """Docs against the results and the code (J8): undefined or mis-quoted
    \\num macros, broken file references and missing index entries fail; hand-typed result
    numbers, stale snapshot names and reporting-population mentions in the paper-facing docs are
    listed. Default: every doc of docs/analysis and both analysis plans."""
    from biomech.check.docs import DOCS, REPO, check_docs, fill_doc, macro_texts

    if fill:
        macros = macro_texts()
        targets = [Path(d) for d in docs] or sorted(DOCS.glob("an1[4-9]*.md")) + [
            REPO / "experiments/icorr2027/analysis_plan.md",
            REPO / "experiments/jtsmm_validation/analysis_plan.md"]
        for d in targets:
            n, left = fill_doc(d, macros)
            if n or left:
                click.echo(f"{d.name}: {n} values updated, {len(left)} groups to check by hand")
                for x in left[:20] if verbose else []:
                    click.echo(f"  {x}")
    reps, missing = check_docs([Path(d) for d in docs] or None)
    fail = len(missing)
    for r in reps:
        kinds = (("undefined", r.undefined), ("mismatched", r.mismatched),
                 ("bad refs", r.bad_refs), ("hand-typed", r.hand_typed),
                 ("stale snapshots", r.stale_snapshots), ("population", r.population))
        if any(v for _, v in kinds):
            click.echo(f"{r.doc}: " + ", ".join(f"{k} {len(v)}" for k, v in kinds if v))
            if verbose:
                for k, v in kinds:
                    for x in v:
                        click.echo(f"  {k}: {x}")
        fail += r.failures()
    for m in missing:
        click.echo(f"not in docs/analysis/README.md: {m}")
    from biomech.check.docs import claim_coverage

    cov_fail, cov_warn = claim_coverage()
    for x in cov_fail:
        click.echo(f"claim coverage: {x}")
    for x in cov_warn if verbose else []:
        click.echo(f"claim coverage (warn): {x}")
    fail += len(cov_fail)
    click.echo(f"{len(reps)} docs, {fail} failures"
               + (f"; {len(cov_warn)} paper items without a claim row (-v)" if cov_warn else ""))
    sys.exit(1 if fail else 0)


@main.command("check-paper")
@click.argument("tex_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--export", "exports", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Export folder(s) whose numbers.tex / captions.tex the manuscript uses.")
def check_paper_cmd(tex_dir: str, exports: tuple[str, ...]) -> None:
    """Check a LaTeX manuscript (TEX_DIR) against the exports: every \\numX macro and
    \\figcaption{label} it uses must be defined, and all exports must come from one
    reporting population (the same sensitivity variant) -- exit 1 otherwise;
    unused numbers are listed."""
    from biomech.viz import numbers as N

    r = N.check_manuscript(tex_dir, list(exports))
    for u in r["undefined"]:
        click.echo(f"UNDEFINED {u}", err=True)
    pops = sorted(set(r["populations"].values()))
    if len(pops) > 1:
        for d, v in r["populations"].items():
            click.echo(f"MIXED POPULATION {d}: {v}", err=True)
    for m in r["other_population"]:
        click.echo(f"OTHER POPULATION \\{m}: the analysis of record (all participants), not "
                   "the paper's population (D36)", err=True)
    if r["unused"]:
        click.echo(f"unused numbers ({len(r['unused'])}): {', '.join(r['unused'])}")
    click.echo(f"{len(r['undefined'])} undefined; reporting population: "
               f"{', '.join(pops) or 'unknown'}")
    sys.exit(1 if r["undefined"] or len(pops) > 1 or r["other_population"] else 0)


@main.command("paper-sync")
@click.argument("export_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--to", "dest", required=True, type=click.Path(file_okay=False),
              help="The manuscript's results folder (e.g. <manuscript>/results); updated in "
                   "place, only files this command wrote are replaced or removed.")
@click.option("--manuscript", type=click.Path(exists=True, file_okay=False), default=None,
              help="Manuscript folder: run check-paper against the synced folder afterwards.")
@click.option("--source-data", is_flag=True,
              help="Also copy each item's source data (CSV) to source_data/<label>.csv.")
@click.option("--additional", is_flag=True,
              help="Also copy the additional files (interactive HTML, supplementary videos; "
                   "tens of MB) to additional/<label>.html|.mp4.")
@click.option("--dry-run", is_flag=True, help="List what would change; write nothing.")
def paper_sync_cmd(export_dir: str, dest: str, manuscript: str | None, source_data: bool,
                   additional: bool, dry_run: bool) -> None:
    """Copy a paper export (EXPORT_DIR, from `biomech export --paper ...`) into a manuscript
    repository: numbers, captions, caption-free figure PDFs and table bodies named by their
    paper labels, with a manifest (sha256, snapshot, code) and a README. Every source file is
    checked against the export manifest first."""
    from biomech.viz import numbers as N
    from biomech.viz import paper_sync as P

    cmd = " ".join(["biomech paper-sync", str(Path(export_dir).resolve()), "--to",
                    str(Path(dest).resolve())]
                   + (["--manuscript", str(Path(manuscript).resolve())] if manuscript else [])
                   + (["--source-data"] if source_data else [])
                   + (["--additional"] if additional else []))
    try:
        r = P.sync(export_dir, dest, source_data=source_data, command=cmd, dry_run=dry_run,
                   additional=additional)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        click.echo(f"paper-sync: {e}", err=True)
        sys.exit(1)
    verb = "would write" if dry_run else "written"
    for rel in r["written"]:
        click.echo(f"{verb}: {rel}")
    for rel in r["removed"]:
        click.echo(f"{'would remove' if dry_run else 'removed'}: {rel}")
    click.echo(f"{len(r['written'])} {verb}, {len(r['unchanged'])} unchanged, "
               f"{len(r['removed'])} removed -> {dest}")
    if manuscript and not dry_run:
        c = N.check_manuscript(Path(manuscript).resolve(), [Path(dest).resolve()])
        for u in c["undefined"]:
            click.echo(f"UNDEFINED {u}", err=True)
        for m in c["other_population"]:
            click.echo(f"OTHER POPULATION \\{m} (D36)", err=True)
        click.echo(f"check-paper: {len(c['undefined'])} undefined, "
                   f"{len(c['unused'])} numbers not used yet")
        sys.exit(1 if c["undefined"] or c["other_population"] else 0)


@main.command("export-data")
@_config_opt
@_data_opt
@_out_opt
@click.option("--to", "target", type=click.Path(file_okay=False), required=True)
@click.option("--format", "fmt", type=click.Choice(["csv", "hdf5"]), default="csv",
              show_default=True)
@click.option("--arrays", is_flag=True, help="Also the stored arrays (hdf5 only).")
@_snap_opt
def export_data_cmd(config_path: str, data_dir: str | None, out_dir: str | None, target: str,
                    fmt: str, arrays: bool, snapshot: str | None) -> None:
    """Pseudonymized release of every result table (+ data dictionary): P<n> instead of
    study ids, participants outside the cohort dropped, no per-person demographics."""
    from biomech.io.context import DataContext
    from biomech.results.release import export_data

    ctx = DataContext(_load(config_path, data_dir, out_dir, snapshot))
    try:
        meta = export_data(ctx, target, fmt, arrays)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{len(meta['files'])} files, {meta['participants']} participants "
               f"({meta['snapshot']}) -> {target}")
    for k, v in meta["dropped_private_columns"].items():
        click.echo(f"  not released (per-person demographics): {k}: {', '.join(v)}")


@main.command("render")
@click.argument("spec_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--to", "fmt", type=click.Choice(["pgf", "pdf", "png", "html"]), default="pgf",
              show_default=True)
@click.option("--out", "out_path", type=click.Path(), default=None,
              help="Target directory (pgf) or file (default: next to the spec).")
def render_cmd(spec_path: str, fmt: str, out_path: str | None) -> None:
    """Re-render an exported figure (figures/<id>/spec.json) or table (tables/<id>.json)
    from its spec files alone -- no result store, no recomputation."""
    from biomech.viz import render_mpl, render_pgf, serialize

    p = Path(spec_path)
    kind = json.loads(p.read_text(encoding="utf-8")).get("kind")
    if kind == "table":
        if fmt != "pgf":
            raise click.UsageError("tables render to pgf (booktabs) only")
        target = Path(out_path) if out_path else p.parent.parent
        click.echo(f"-> {render_pgf.render_table(serialize.load_table(p), target)}")
        return
    spec = serialize.load_figure(p)
    if fmt == "pgf":
        target = Path(out_path) if out_path else p.parent / "rerender"
        render_pgf.write_style(target)
        click.echo(f"-> {render_pgf.render_figure(spec, target)}")
    elif fmt in ("pdf", "png"):
        target = Path(out_path) if out_path else p.parent / f"{spec.id}.{fmt}"
        render_mpl.render_figure(spec, target)
        click.echo(f"-> {target}")
    else:
        from biomech.viz.render_plotly import to_plotly

        target = Path(out_path) if out_path else p.parent / f"{spec.id}.html"
        to_plotly(spec).write_html(target, include_plotlyjs=True)   # self-contained
        click.echo(f"-> {target}")


@main.command("check-outputs")
@_config_opt
@_data_opt
@_out_opt
@click.option("--report-dir", type=click.Path(file_okay=False), default=None,
              help="Where report.html/json and contact sheets go (default: <out>/check).")
@click.option("--no-figures", is_flag=True, help="Skip rendering every view.")
@click.option("--no-pgf", is_flag=True, help="Skip the LuaLaTeX export and size checks.")
@click.option("--no-videos", is_flag=True, help="Skip the video checks.")
@_snap_opt
@click.option("-j", "--jobs", "n_jobs", default=0, show_default=True,
              help="Check workers (0 = up to 8, 1 = in-process); the same report.")
def check_outputs_cmd(config_path: str, data_dir: str | None, out_dir: str | None,
                      report_dir: str | None, no_figures: bool, no_pgf: bool,
                      no_videos: bool, snapshot: str | None, n_jobs: int) -> None:
    """Check every generated analysis, table, figure and video (pass/warn/fail report)."""
    from biomech.check.outputs import check_outputs

    cfg = _load(config_path, data_dir, out_dir, snapshot)
    rep = check_outputs(cfg, report_dir, figures=not no_figures, pgf=not no_pgf,
                        videos=not no_videos, n_jobs=n_jobs, groups=cfg.groups)
    for i in rep.items:
        if i.status in ("fail", "warn"):
            click.echo(f"{i.status.upper():5} {i.area:9} {i.name}: {i.detail}")
    target = Path(report_dir) if report_dir else cfg.out_dir / "check"
    click.echo(f"{rep.n_fail} failing, {rep.n_warn} warnings, {len(rep.items)} checks -> "
               f"{target / 'report.html'}")
    sys.exit(1 if rep.n_fail else 0)


@main.command("animate")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Study YAML (e.g. experiments/icorr2027/study.yaml); not "
                                 "needed with --from-clip.")
@_data_opt
@_out_opt
@click.option("--pid", default=None, help="Participant id, e.g. s01 (not for the cohort "
                                          "layouts cohort and workspace3d).")
@click.option("--layout", default="multimodal", show_default=True,
              type=click.Choice(["multimodal", "kinematics", "tracking", "emg", "cohort",
                                 "v1_hinge", "v2_intent", "v3_release",
                                 "validation", "workspace3d", "emg_spectrogram",
                                 "onset_rules"]))  # = anim.LAYOUTS
@click.option("--phase", "phases", multiple=True,
              help="Phase(s) to play, in order (repeat for cuts between phases). Default: "
                   "chewing (v1_hinge, v2_intent, validation, onset_rules: open_close; "
                   "v3_release: open_close and chewing); for cohort the task (default: "
                   "open_close; workspace3d: every task). validation: the systems-paper "
                   "video, validation study only.")
@click.option("--t0", type=float, default=None, help="Window start, s from the phase start.")
@click.option("--t1", type=float, default=None, help="Window end, s from the phase start.")
@click.option("--speed", type=float, default=None,
              help="Playback speed (0.25 = slow motion, 4 = time-lapse). Default 0.5 "
                   "(v2_intent: 1, simulated real time; --from-clip: the clip's).")
@click.option("--fps", type=float, default=None, help="Frame rate (default 30; --from-clip: "
                                                      "the clip's).")
@click.option("--size", default=None,
              help="1080p (default), 720p, 4k, print (7.16 in at 300 dpi), a width or WxH "
                   "(--from-clip: the size of the clip's render).")
@click.option("--theme", type=click.Choice(["dark", "light"]), default=None,
              help="dark (default) or light (--from-clip: the clip's).")
@click.option("--with-camera", is_flag=True,
              help="Include the camera previews (they show the participant; off by default).")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None,
              help="Output .mp4/.webm/.gif (default: <out-dir>/videos/<layout>_P<n>_<phases>[setting suffixes].mp4, anim.video_stem).")
@click.option("-j", "--jobs", "n_jobs", default=0, help="Render workers (0 = up to 8).")
@click.option("--window", "window", type=(float, float), default=None,
              help="Scrolling panels: seconds before and after the cursor (default 3 1; "
                   "v2_intent shows only the past: AFTER is 0).")
@click.option("--trail", type=float, default=None,
              help="Length (s) of the fading path behind moving landmarks (default 1).")
@click.option("--captions", "captions_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="JSON list of [start_s, end_s, text] cues in video seconds "
                                 "(replaces the layout's default burned-in captions).")
@click.option("--from-clip", "from_clip", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Re-render from <stem>.clip.npz/.clip.json alone (no result "
                                 "store, no session); --speed/--fps/--theme/--size/--window/"
                                 "--trail/--captions override the clip's settings.")
@click.option("--filmstrip", is_flag=True, help="Also write a pgfplots filmstrip of key frames.")
@click.option("--animate-tex", is_flag=True,
              help="Also write PNG frames + an \\animategraphics snippet (LaTeX animate).")
@_snap_opt
@_variant_opt
def animate_cmd(config_path: str | None, data_dir: str | None, out_dir: str | None,
                pid: str | None, layout: str, phases: tuple[str, ...], t0: float | None,
                t1: float | None, speed: float | None, fps: float | None, size: str | None,
                theme: str | None, with_camera: bool, out_path: str | None, n_jobs: int,
                window: tuple[float, float] | None, trail: float | None,
                captions_path: str | None, from_clip: str | None, filmstrip: bool,
                animate_tex: bool, snapshot: str | None, variant: str | None) -> None:
    """Render a synchronized animation of the data to video (and LaTeX).

    Every render also writes <stem>.clip.npz/.clip.json (re-render with --from-clip) and,
    for the supplementary layouts (v1_hinge, v2_intent, v3_release, validation, workspace3d,
    emg_spectrogram, onset_rules), a PNG filmstrip. --variant/--snapshot pick the store (a
    paper set's: e.g. --variant reporting_population --snapshot r13a); the manifest records both.
    """
    import json

    from biomech.anim import default_phases, default_speed, make, video_stem
    from biomech.anim.layouts import COHORT_LAYOUTS
    from biomech.anim.render_video import Cancelled, render_video

    win = (-abs(window[0]), abs(window[1])) if window is not None else None
    caps = None
    if captions_path:
        try:
            caps = tuple((float(c[0]), float(c[1]), str(c[2]))
                         for c in json.loads(Path(captions_path).read_text(encoding="utf-8")))
        except (ValueError, TypeError, IndexError) as exc:
            raise click.BadParameter(f"{captions_path}: {exc}", param_hint="--captions")
    ctx = None
    if from_clip:
        from biomech.anim.clipfile import clip_stem, load_clip_file, read_clip_doc

        try:
            anim = load_clip_file(from_clip)
            doc = read_clip_doc(from_clip)
        except (OSError, ValueError, KeyError) as exc:
            raise click.ClickException(str(exc)) from exc
        over = {k: v for k, v in (("speed", speed), ("fps", fps), ("theme", theme),
                                  ("window_s", win), ("trail_s", trail), ("captions", caps))
                if v is not None}
        anim = anim.with_(**over)
        size = size or doc.get("render", {}).get("size") or "1080p"
        size = tuple(size) if isinstance(size, list) else size
        stem = clip_stem(anim, doc, size=size)
        out = Path(out_path) if out_path else Path(from_clip).parent / f"{stem}.mp4"
        layout = anim.layout
    else:
        if not config_path:
            raise click.UsageError("--config is required (unless --from-clip)")
        from biomech.io.context import DataContext

        cfg = _load(config_path, data_dir, out_dir, snapshot, variant)
        if layout not in COHORT_LAYOUTS and not pid:
            raise click.UsageError("--pid is required for this layout")
        speed = default_speed(layout) if speed is None else speed
        fps = 30.0 if fps is None else fps
        theme = theme or "dark"
        size = size or "1080p"
        ctx = DataContext(cfg)
        try:
            if layout in COHORT_LAYOUTS:
                task = phases[0] if phases else "open_close"
                anim = make(ctx, None, layout, speed=speed, fps=fps, theme=theme, task=task)
                stem = video_stem(layout, None, [task], speed=speed, theme=theme, size=size,
                                  fps=fps)
            else:
                ph = list(phases) or default_phases(layout)
                win = win or (-3.0, 1.0)
                trail = 1.0 if trail is None else trail
                anim = make(ctx, pid, layout, ph, t0, t1, speed=speed, fps=fps, theme=theme,
                            with_camera=with_camera, window_s=win, trail_s=trail,
                            captions=caps)
                stem = video_stem(layout, pid, ph, with_camera, pseudonym=ctx.pseudonym(pid),
                                  t0_s=t0, t1_s=t1, speed=speed, theme=theme, size=size,
                                  window_s=win, trail_s=trail, fps=fps)
            anim.compose()                  # a layout without its data fails here, not later
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        out = Path(out_path) if out_path else cfg.out_dir / "videos" / f"{stem}.mp4"
    tl = anim.timeline
    click.echo(f"{layout}: {tl.n_frames} frames, {tl.duration_s:.1f} s at {anim.fps:g} fps "
               f"({anim.speed:g}x) -> {out}")
    last = [-1]

    def progress(done: int, total: int) -> None:
        pct = int(100 * done / max(total, 1))
        if pct // 10 != last[0]:
            last[0] = pct // 10
            click.echo(f"  {pct:3d} %")

    try:
        res = render_video(anim, out, size=size, n_jobs=n_jobs or None, on_progress=progress)
    except (Cancelled, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote {res.path} ({res.size[0]}x{res.size[1]}, {res.n_frames} frames, "
               f"{res.elapsed_s:.1f} s, {res.n_chunks} chunks)")
    if res.clip_files:
        click.echo(f"clip -> {res.clip_files[0].name}, {res.clip_files[1].name}")
    if res.filmstrip:
        click.echo(f"filmstrip PNG -> {res.filmstrip}")
    if filmstrip:
        from biomech.anim.render_latex import filmstrip as film

        p = film(anim, out.parent / f"{out.stem}_tex", fig_id=f"film_{stem}".replace("+", "_"))
        click.echo(f"filmstrip -> {p}")
    if animate_tex:
        from biomech.anim.render_latex import animate_frames

        p = animate_frames(anim, out.parent / f"{out.stem}_animate")
        click.echo(f"animate frames -> {p}")
    if ctx is not None:
        ctx.close()


@main.command("report")
@_config_opt
@_data_opt
@_out_opt
@click.option("--pid", required=True, help="Participant id, e.g. s01.")
@click.option("--out", "report_dir", type=click.Path(file_okay=False), default=None,
              help="Report folder (default: <out_dir>/reports/<pid>, git-ignored).")
@click.option("--compile/--no-compile", "compile_pdf", default=False,
              help="Compile report_<pid>.tex with LuaLaTeX.")
@_snap_opt
def report_cmd(config_path: str, data_dir: str | None, out_dir: str | None, pid: str,
               report_dir: str | None, compile_pdf: bool, snapshot: str | None) -> None:
    """Participant report: every data view (TikZ) and the key tables in report_<pid>.tex."""
    from biomech.report import build_report

    cfg = _load(config_path, data_dir, out_dir, snapshot)
    try:
        res = build_report(cfg, pid, report_dir, compile_pdf=compile_pdf, log=click.echo)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{len(res['figures'])} figures, {len(res['tables'])} tables -> {res['tex']}")
    if res["pdf"]:
        click.echo(f"pdf -> {res['pdf']}")
    for e in res["errors"]:
        click.echo(f"ERROR {e}", err=True)
    sys.exit(1 if res["errors"] else 0)


@main.command("synth")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), required=True,
              help="Folder for the synthetic sessions, cohort, study.yaml and results.")
@click.option("--run/--no-run", default=True, show_default=True,
              help="Run every stage after writing the data.")
@click.option("-j", "--jobs", "n_jobs", default=4, show_default=True)
@click.option("--emg-fs", default=2000, show_default=True, help="EMG rate of the takes.")
def synth_cmd(out_dir: str, run: bool, n_jobs: int, emg_fs: int) -> None:
    """Write a synthetic study with a known truth (no subject data), e.g. to try the viewer."""
    from dataclasses import replace

    from biomech.pipeline import run as run_pipeline
    from biomech.testing.synth import DEFAULT_SPECS, write_study

    path, truths = write_study(out_dir, [replace(s, emg_fs=emg_fs) for s in DEFAULT_SPECS])
    click.echo(f"wrote {len(truths)} participants -> {path}")
    if run:
        summary = run_pipeline(_load(str(path), None, None), None, n_jobs=n_jobs,
                               on_event=lambda ev: ev["kind"] == "stage_start" and click.echo(
                                   f"[{ev['stage']}] {ev['n_todo']} to run"))
        click.echo(f"run {summary.run_id}: {summary.n_failed} failed")
    click.echo(f"view it:  biomech view --config {path}")


@main.command("view")
@_config_opt
@_data_opt
@click.option("--port", default=8060, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--debug", is_flag=True)
@click.option("--verbose", is_flag=True, help="Keep the per-request access log.")
@_snap_opt
def view_cmd(config_path: str, data_dir: str | None, port: int, host: str, debug: bool,
             verbose: bool, snapshot: str | None) -> None:
    """Analysis viewer: control center, data explorer, results, paper board, animation."""
    from biomech.viewer.app import main as run_viewer

    try:
        run_viewer(config_path, port=port, host=host, debug=debug, data_dir=data_dir,
                   verbose=verbose, snapshot=snapshot)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
