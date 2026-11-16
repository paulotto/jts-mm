# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Export registered views to the paper: TikZ/pgfplots + booktabs, PNG/PDF previews, checks.

``items`` are view ids or dicts ``{id, perspective, align, stat, pid, phase}``
(the paper sets: paper_dataset.yaml, paper_systems.yaml). ``venue`` (``ieee`` default, ``springer``, ``nature``; :data:`style.VENUES`)
sets the column / text widths, the absolute font sizes and the figure font family for the
pgf export, the previews and the checks alike. Every figure is also compiled alone
(``standalone``) when ``check_size`` is set, and its natural page size must fit the venue's
column or text width -- the size test of plan W2; ``standalone`` additionally writes the
journal-upload PDF ``figures/<id>/<id>.pdf`` (exactly the layout box, venue typeface).
Reader-facing text passes through :func:`vocab.reader_text` (no internal plan codes).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from biomech.viz import render_mpl, render_pgf, style
from biomech.viz.spec import FigureSpec, TableSpec
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V

LATEX = "lualatex"            # dynamic memory: 3-D and colour-line figures need it


def all_items(ctx: Any, kinds: tuple[str, ...] = ("paper", "dashboard", "data"),
              perspectives: str = "all", pids: list[str] | None = None,
              groups: Any = None) -> list[dict[str, Any]]:
    """Every registered view of the study's groups (``groups``; default: the groups of
    ``ctx``'s study) as export items: each perspective (``"all"``) or only the default one,
    and once per participant for views that take one (default: every included participant;
    cohort-level views once)."""
    pids = list(pids) if pids else list(ctx.pids())
    items: list[dict[str, Any]] = []
    for v in R.views(groups=ctx if groups is None else groups):
        if v.kind not in kinds:
            continue
        persps = list(v.perspectives) if (perspectives == "all" and v.perspectives) else [None]
        for persp in persps:
            needs_pid = "pid" in v.params and persp != "cohort"
            for pid in (pids if needs_pid else [None]):
                d = {"id": v.id}
                if persp is not None:
                    d["perspective"] = persp
                if pid is not None:
                    d["pid"] = pid
                items.append(d)
    return items


def _item(entry: Any) -> tuple[str, dict[str, Any]]:
    """(view id, view parameters) of an export entry; keys starting with ``_`` (``_label``,
    ``_part`` from a paper set) describe the item and are not view parameters."""
    if isinstance(entry, str):
        return entry, {}
    d = {k: v for k, v in dict(entry).items() if not str(k).startswith("_")}
    return d.pop("id"), d


def export_paper_set(ctx: Any, pset: Any, target: str | None = None, *,
                     to: str | Path | None = None, venue: str | None = None,
                     **kw: Any) -> dict[str, Any]:
    """Export one target of a paper set (:mod:`biomech.papers`): its items with their labels
    and parts, the manifest with the paper set and target, and the in-text numbers of the
    set (``numbers.tex``) resolved from the same store. ``ctx`` reads the paper set's store
    (:meth:`biomech.papers.PaperSet.store_config`: its variant at its pinned snapshot); the
    manifest's provenance records which. ``to`` overrides the target folder,
    ``venue`` its venue; other keywords go to :func:`export`. The result has
    ``target_dir`` and ``numbers`` (count) besides the export lists."""
    from biomech.viz import numbers as N
    from biomech.viz import serialize

    tgt = pset.target(target)
    dest = Path(to) if to else tgt.out_dir
    meta = {"paper": pset.name, "paper_file": str(pset.path), "target": tgt.name,
            "snapshot_pinned": pset.snapshot, "variant": pset.variant}
    over = pset.caption_overrides(tgt.name)
    entries = []
    for i in pset.items_for(tgt.name):
        e = i.entry()
        if i.label in over:
            e["_caption_tex"] = over[i.label]["text"]
            e["_caption_panels"] = over[i.label]["panels"]
        entries.append(e)
    res = export(ctx, entries, dest, venue=venue or tgt.venue, meta=meta, **kw)
    res["target_dir"] = [str(dest)]
    res["numbers"] = []
    if pset.numbers is not None and not res.get("cancelled"):
        try:
            nums = N.resolve(ctx, N.load(pset.numbers))
            N.write(dest, nums, serialize.provenance(ctx), pset.numbers)
            res["numbers"] = [n.key for n in nums]
        except Exception as exc:
            res["errors"].append(f"numbers: {exc}")
    if over and not res.get("cancelled"):
        res["errors"] += caption_override_problems(res.get("items", []), over,
                                                   set(res["numbers"]))
    if not res.get("cancelled") and (dest / "manifest.json").is_file():
        try:
            res["additional"] = [e["label"] for e in _additional(ctx, pset, dest)]
        except Exception as exc:
            res["errors"].append(f"additional files: {exc}")
    return res


def caption_words(tex: str) -> int:
    """Words of a LaTeX caption as a reader counts them (commands dropped, their braced
    arguments and number macros counted as words)."""
    s = re.sub(r"\([a-z](?:(?:, | to | and )[a-z])*\)", " ", tex)     # panel letters
    s = re.sub(r"\\num[A-Z]\w*", " N ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)
    s = re.sub(r"[{}$~]", " ", s)
    return len([w for w in s.split() if re.search(r"\w", w)])


def caption_override_problems(items: list[dict[str, Any]], over: dict[str, Any],
                              numbers: set[str]) -> list[str]:
    """Checks of the written captions (:meth:`biomech.papers.PaperSet.caption_overrides`):
    the word limit of the item's part and kind, every number macro defined by the set, and
    no hand-typed result (a number with a decimal point outside a macro)."""
    limits = over.get("_limits") or {}
    out = []
    for r in items:
        tex = r.get("caption_tex")
        if not tex:
            continue
        lab, kind = r.get("label"), "figure" if r.get("figure") else "table"
        lim = (limits.get(r.get("part")) or {}).get(kind)
        n = caption_words(tex)
        if lim and n > int(lim):
            out.append(f"caption of {lab}: {n} words, the limit for a {r.get('part')} {kind} "
                       f"is {lim}")
        for m in re.findall(r"\\num([A-Z]\w*)", tex):
            key = m[:1].lower() + m[1:]
            if not any(k in numbers for k in (key, re.sub(r"(Val|CI|N)$", "", key))):
                out.append(f"caption of {lab}: \\num{m} is not a number of the set")
        # method constants are typed on purpose in math or \SI / \qty (e.g. $2.5$ s of
        # calibration); a bare decimal number in the prose is a result typed by hand
        prose = re.sub(r"\$[^$]*\$|\\(?:SI|qty|num)\{[^}]*\}(\{[^}]*\})?|\\num[A-Z]\w*", "",
                       tex)
        if re.search(r"(?<![\w.])\d+\.\d+(?![\w])", prose):
            out.append(f"caption of {lab}: a typed decimal number in the prose (use a macro, "
                       "or $...$ / \\SI for a method constant)")
    return out


def _additional(ctx: Any, pset: Any, dest: Path) -> list[dict[str, Any]]:
    """The paper set's additional files (2026-11-05), recorded in the export manifest under
    ``additional``: each ``supplementary_interactive`` view as a self-contained interactive
    HTML file in ``additional/`` (plotly inlined, light theme), and each rendered
    ``supplementary_videos`` entry that fits the set's store by its path (the MP4 stays where
    ``biomech animate`` wrote it). ``biomech paper-sync --additional`` copies them."""
    import hashlib

    from biomech import papers
    from biomech.viz import gallery, render_plotly

    def sha(f: Path) -> str:
        return hashlib.sha256(f.read_bytes()).hexdigest()

    out: list[dict[str, Any]] = []
    add = dest / "additional"
    for it in getattr(pset, "interactive", None) or []:
        spec = R.build(it.view_id, R.request(ctx, it.view_id, **it.params))
        fig = render_plotly.to_plotly(spec, theme="light")
        add.mkdir(parents=True, exist_ok=True)
        f = add / f"{it.label}.html"
        fig.write_html(str(f), include_plotlyjs=True, full_html=True,
                       config={"displaylogo": False})
        out.append({"label": it.label, "kind": "interactive", "id": it.view_id,
                    "file": f"additional/{f.name}", "sha256": sha(f),
                    "caption": getattr(spec, "caption", "")})
    for it in getattr(pset, "data", None) or []:
        spec = R.build(it.view_id, R.request(ctx, it.view_id, **it.params))
        if not isinstance(spec, TableSpec):
            raise ValueError(f"supplementary_data {it.label}: {it.view_id} is not a table")
        add.mkdir(parents=True, exist_ok=True)
        f = add / f"{it.label}.csv"
        keys = [c.key for c in spec.columns]
        rows = [r for r in spec.rows if not r.get("_rule") and not r.get("_head")]
        import csv

        with f.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([c.header + (f" ({style.plain_unit(c.unit)})" if c.unit else "")
                        for c in spec.columns])
            for r in rows:
                w.writerow([r.get(k, "") for k in keys])
        out.append({"label": it.label, "kind": "data", "id": it.view_id,
                    "file": f"additional/{f.name}", "sha256": sha(f),
                    "caption": spec.caption})
    out_root = Path(getattr(ctx.config, "out_dir", dest))
    vdirs = list(dict.fromkeys(papers.video_dirs(ctx.config) + [out_root / "videos"]))
    for e in gallery.listed_videos(pset, vdirs, getattr(ctx, "pseudonym", None)):
        if e["status"] == "rendered" and e["path"] is not None:
            out.append({"label": e["label"], "kind": "video", "source": str(e["path"]),
                        "sha256": sha(Path(e["path"])), "describe": e["describe"]})
    mf = dest / "manifest.json"
    man = json.loads(mf.read_text())
    man["additional"] = out
    mf.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    return out


# Pickers the gallery enumerates besides ``form`` (J2.6: every perspective of an
# item, e.g. own vs aligned frames); per-muscle / per-task pickers stay at their default (the
# dashboards show those).
GALLERY_PICKERS = ("form", "frames", "mode", "rate", "metric", "margin", "rule", "body",
                   "scope")
GALLERY_MAX = 24                     # depictions per item at most (the first ones in order)


def gallery_items(ctx: Any, entries: list[Any]) -> list[dict[str, Any]]:
    """Every entry once per combination of its view's gallery pickers (:data:`GALLERY_PICKERS`:
    the form and the perspective pickers.6: the authors pick a depiction from
    the gallery). A picker the entry already sets stays fixed; entries without any stay as
    they are; at most :data:`GALLERY_MAX` combinations per entry."""
    import itertools

    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for e in entries:
        d = {"id": e} if isinstance(e, str) else dict(e)
        v = R.get(d["id"])
        axes = [(k, v.choice_values(k, ctx)) for k in GALLERY_PICKERS
                if k in v.choices and k not in d]
        axes = [(k, vals) for k, vals in axes if vals]
        if not axes:
            sig = tuple(sorted((k, str(x)) for k, x in d.items() if not k.startswith("_")))
            if sig not in seen:
                seen.add(sig)
                out.append(d)
            continue
        combos = itertools.islice(itertools.product(*(vals for _, vals in axes)), GALLERY_MAX)
        for combo in combos:
            g = {**d, **{k: val for (k, _), val in zip(axes, combo)}}
            # one depiction once: an expanded picker can reproduce another entry of the set
            # (two parallel compiles of one id would race on the same files)
            sig = tuple(sorted((k, str(v)) for k, v in g.items() if not k.startswith("_")))
            if sig in seen:
                continue
            seen.add(sig)
            if d.get("_label"):
                g["_label"] = d["_label"] + "".join(
                    w.capitalize() for val in combo
                    for w in re.split(r"[^A-Za-z0-9]+", str(val)) if w)
            out.append(g)
    return out


def export_gallery(ctx: Any, pset: Any, target: str | None = None, *,
                   to: str | Path | None = None, **kw: Any) -> dict[str, Any]:
    """Every item of a paper-set target in every depiction it offers, into
    ``<target folder>_gallery`` (or ``to``) with a compiled preview: the page to choose the
    forms from (the choice is then written into the paper set: ``{id: ..., form: ...}``)."""
    tgt = pset.target(target)
    dest = Path(to) if to else tgt.out_dir.with_name(tgt.out_dir.name + "_gallery")
    items = gallery_items(ctx, [i.entry() for i in pset.items_for(tgt.name)])
    kw.setdefault("compile_pdf", True)
    res = export(ctx, items, dest, venue=kw.pop("venue", None) or tgt.venue,
                 meta={"paper": pset.name, "target": tgt.name, "gallery": True}, **kw)
    res["target_dir"] = [str(dest)]
    if not res.get("cancelled") and (dest / "manifest.json").is_file():
        from biomech import hwparts
        from biomech.viz import gallery

        out_root = Path(getattr(ctx.config, "out_dir", dest))
        try:            # every part render and design-analysis figure of hardware/mechanical
            man = hwparts.load()
            imgs = [(f"Original JTS render: {a.title.lower()} (no participant)",
                     a.legacy_render) for a in man.assemblies.values()
                    if a.legacy_render and a.legacy_render.is_file()] + man.images()
        except Exception:                       # no manifest (installed package)
            imgs = []
        from biomech import papers

        vdirs = papers.video_dirs(ctx.config) + [out_root / "videos"]
        vdirs = list(dict.fromkeys(vdirs))
        pseud = getattr(ctx, "pseudonym", None)
        res["gallery_html"] = [str(gallery.write_html(
            dest, title=f"Gallery: {pset.name}, {tgt.name}", videos=vdirs[:1],
            images=imgs, pseudonym=pseud,
            listed=gallery.listed_videos(pset, vdirs, pseud)))]
        if out_root.parent.is_dir():          # data/analysis: one page linking all galleries
            res["gallery_html"].append(str(gallery.write_index(out_root.parent)))
    return res


def _item_meta(entry: Any) -> dict[str, Any]:
    return {k[1:]: v for k, v in dict(entry).items() if str(k).startswith("_")} \
        if isinstance(entry, dict) else {}


def _compile(tex: Path, timeout: int = 600) -> tuple[bool, str]:
    exe = shutil.which(LATEX) or shutil.which("pdflatex")
    if exe is None:
        return False, "no LaTeX engine found"
    r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                       cwd=tex.parent, capture_output=True, text=True, timeout=timeout)
    log = tex.with_suffix(".log")
    text = log.read_text(errors="replace") if log.exists() else r.stdout
    return r.returncode == 0, text


def tex_problems(log: str) -> list[str]:
    """Overfull boxes and pgfplots warnings in a LaTeX log."""
    out = [m.group(0) for m in re.finditer(r"Overfull \\[hv]box[^\n]*", log)]
    out += [m.group(0) for m in re.finditer(r"Package pgfplots Warning:[^\n]*", log)]
    return out


def pdf_size_in(pdf: Path) -> tuple[float, float] | None:
    exe = shutil.which("pdfinfo")
    if exe is None or not pdf.exists():
        return None
    r = subprocess.run([exe, str(pdf)], capture_output=True, text=True, timeout=30)
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", r.stdout)
    return (float(m.group(1)) / 72.0, float(m.group(2)) / 72.0) if m else None


def check_figure_size(out_dir: Path, spec: FigureSpec,
                      venue: str | style.Venue | None = None) -> list[str]:
    """Compile the figure alone and compare its natural width with the venue's target width
    (default venue: the active one)."""
    Vn = style.venue(venue)
    tex = render_pgf.standalone_figure(out_dir, spec.id, Vn)
    ok, log = _compile(tex)
    if not ok:
        return [f"{spec.id}: standalone compile failed: {_tex_error(log)}"]
    probs = [f"{spec.id}: {p}" for p in tex_problems(log)]
    size = pdf_size_in(tex.with_suffix(".pdf"))
    limit = Vn.width_in(spec.width)
    if size and size[0] > limit + 1.5 / 72:
        probs.append(f"{spec.id}: {size[0]:.2f} in wide > {limit:.2f} in ({Vn.name} {spec.width})")
    if size and size[1] > Vn.max_height_in:
        probs.append(f"{spec.id}: {size[1]:.2f} in high > {Vn.max_height_in:.2f} in ({Vn.name})")
    m = re.findall(r"JTSMM-MINFONT ([0-9.]+)", log)
    if m and float(m[-1]) < Vn.min_pt - 0.05:
        probs.append(f"{spec.id}: text down to {float(m[-1]):.1f} pt < {Vn.min_pt:g} pt "
                     f"({Vn.name} minimum)")
    for ext in (".aux", ".log", ".tex", ".pdf"):     # only the measurement is kept (J1.3)
        tex.with_suffix(ext).unlink(missing_ok=True)
    return probs


def _tex_error(log: str) -> str:
    """The first TeX error of a failed compile ("! ..." with its next lines), else the log
    tail: the problem line names the cause, not the node-memory dump after it."""
    m = re.search(r"^! .*(?:\n.*){0,3}", log, re.M)
    return re.sub(r"\s+", " ", m.group(0))[:400] if m else log[-400:]


def standalone_pdf(out_dir: Path, spec: FigureSpec,
                   venue: str | style.Venue | None = None) -> tuple[Path | None, list[str]]:
    """The journal-upload PDF of one figure: ``figures/<id>/<id>.pdf``, compiled alone with
    LuaLaTeX in the venue's typeface, page = the layout box (column or text width).

    Returns the PDF (``None`` on failure) and the problems (compile errors, overfull boxes,
    pgfplots warnings, a page wider than the venue allows)."""
    Vn = style.venue(venue)
    out_dir = Path(out_dir)
    name = f"upload_{spec.id}"
    tex = render_pgf.standalone_figure(out_dir, spec.id, Vn, natural=False, name=name)
    ok, log = _compile(tex)
    probs = [] if ok else [f"{spec.id}: upload PDF compile failed: {_tex_error(log)}"]
    probs += [f"{spec.id} (upload): {p}" for p in tex_problems(log)] if ok else []
    target = None
    if ok and tex.with_suffix(".pdf").is_file():
        target = out_dir / "figures" / spec.id / f"{spec.id}.pdf"
        shutil.move(str(tex.with_suffix(".pdf")), target)
        size = pdf_size_in(target)
        limit = Vn.width_in(spec.width)
        if size and size[0] > limit + 1.5 / 72:
            probs.append(f"{spec.id}: upload PDF {size[0]:.2f} in wide > {limit:.2f} in")
    for ext in (".aux", ".log", ".tex"):
        tex.with_suffix(ext).unlink(missing_ok=True)
    return target, probs


def captioned_pdf(out_dir: Path, kind: str, item_id: str, caption: str, sources: list[str],
                  prov: dict[str, Any] | None, width: str = "column",
                  venue: str | style.Venue | None = None,
                  label: str | None = None) -> tuple[Path | None, list[str]]:
    """``captioned/<id>.pdf``: the item with its caption, sources and provenance (J1.1),
    compiled alone at the venue's column or text width. Returns the PDF (``None`` on
    failure) and the problems."""
    from biomech.viz import manifest as M

    out_dir = Path(out_dir)
    tex = render_pgf.captioned_item(out_dir, kind, item_id, M.caption_latex(caption),
                                    M.caption_latex(", ".join(sources)), prov, venue,
                                    width=width, label=label)
    ok, log = _compile(tex)
    probs = [] if ok else [f"{item_id}: captioned PDF compile failed: {_tex_error(log)}"]
    target = None
    if ok and tex.with_suffix(".pdf").is_file():
        target = out_dir / "captioned" / f"{item_id}.pdf"
        target.parent.mkdir(exist_ok=True)
        shutil.move(str(tex.with_suffix(".pdf")), target)
    for ext in (".aux", ".log", ".tex"):
        tex.with_suffix(ext).unlink(missing_ok=True)
    return target, probs


def auto_tags(spec: FigureSpec) -> FigureSpec:
    """Panel letters a, b, c, ... for a multi-panel figure that has none (neither ``tag`` nor
    an "(a) ..." title prefix): every exported multi-panel figure is lettered. Also one
    legend for identical legends (:func:`shared_legend`). In place."""
    if len(spec.panels) > 1 and not any(p.heading()[0] for p in spec.panels):
        for i, p in enumerate(spec.panels[:26]):
            p.tag = "abcdefghijklmnopqrstuvwxyz"[i]
    return shared_legend(spec)


def shared_legend(spec: FigureSpec) -> FigureSpec:
    """When every panel that shows a legend shows the same one (same labels, roles and
    marks), only the last of them keeps it (the same legend 3-6 times per figure).
    In place."""
    shown = [p for p in spec.panels if p.legend_mode() != "none"]
    if len(shown) < 2:
        return spec

    def sig(p: Any) -> tuple:
        return tuple((s.label, s.role, s.kind, str(s.style.get("ms")), str(s.style.get("dash")))
                     for s in p.legend_entries())

    if len({sig(p) for p in shown}) == 1:
        for p in shown[:-1]:
            p.legend = "none"
    return spec


def export(ctx: Any, items: list[Any] | None, out_dir: str | Path, *, previews: bool = True,
           compile_pdf: bool = False, check_size: bool = False, standalone: bool = False,
           venue: str | None = None,
           on_progress: Callable[[int, int, str], None] | None = None,
           cancel: Callable[[], bool] | None = None, n_jobs: int = 1,
           groups: Any = None, meta: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Export ``items`` (default: every paper view) to ``out_dir`` for ``venue`` (default:
    the active venue, IEEE unless set).

    ``groups``: the study's view groups (default: those of ``ctx``'s study,
    ``StudyConfig.groups``). Only views of these groups are exported: the default items are
    the paper views of the groups, and an item of another group is an error (a dataset
    study never exports a validation view and vice versa).

    ``standalone``: also the journal-upload PDF per figure (``figures/<id>/<id>.pdf``,
    ``done["standalone"]``). ``on_progress(done, total, label)`` is called before each
    item; ``cancel()`` is polled between items -- a cancelled export keeps what it wrote
    and reports ``done["cancelled"]`` (the viewer runs this as a background job).

    ``n_jobs`` (1 = in-process; 0 = up to 8): items are built in worker processes, grouped
    by participant (one ``DataContext`` and EMG preparation per participant and worker);
    every file and the returned lists are those of the in-process export (items write
    only their own files; lists are merged in item order). The LuaLaTeX compiles of
    ``check_size`` / ``standalone`` run in a thread pool either way.

    Every export writes ``manifest.json`` (per item the view, parameters, label,
    part, caption, sources and the sha256 of every file; the provenance; ``meta``, e.g. the
    paper set and target) and, for items with a ``_label``, ``captions.tex``
    (``\\figcaption{label}``). ``check/integrity.py`` verifies an export against both.
    """
    from biomech.viz import manifest as M

    with style.use_venue(venue) as Vn:
        done = _export(ctx, items, Path(out_dir), Vn, previews=previews,
                       compile_pdf=compile_pdf, check_size=check_size, standalone=standalone,
                       on_progress=on_progress, cancel=cancel, n_jobs=n_jobs,
                       groups=R.study_groups(ctx if groups is None else groups))
        if not done["cancelled"]:
            M.write(Path(out_dir), done, ctx, venue=Vn.name,
                    meta={**(meta or {}), "standalone": bool(standalone)})
        return done


def _export_one(ctx: Any, entry: Any, out: Path, Vn: style.Venue, prov: dict[str, Any],
                previews: bool) -> dict[str, Any]:
    """Build and write one item: {"errors": [...], "figure" | "table": id, "width"}."""
    from biomech.viz import serialize

    vid, params = _item(entry)
    res: dict[str, Any] = {"errors": [], "view": vid, "params": params, **_item_meta(entry)}
    try:
        spec = R.build(vid, R.request(ctx, vid, **params))
    except Exception as exc:              # one broken item must not stop the export
        res["errors"].append(f"{vid}: {type(exc).__name__}: {exc}")
        return res
    want = res.get("caption_panels")
    if want and isinstance(spec, FigureSpec) and len(spec.panels) != int(want):
        res["errors"].append(f"{res.get('label') or vid}: the written caption is for {want} "
                             f"panels, the figure has {len(spec.panels)}: rewrite it")
    if res.get("width") in ("column", "text"):   # the paper set's width for this item
        spec.width = res["width"]
    spec = V.reader_spec(spec)              # no internal plan codes in reader-facing text
    if isinstance(spec, FigureSpec):
        auto_tags(spec)
    if isinstance(spec, FigureSpec):
        # Files of an earlier export of this figure (data files of series that no longer
        # exist, an upload PDF of older results) never survive a re-export (J1.3).
        fdir = out / "figures" / spec.id
        if fdir.is_dir():
            for f in fdir.iterdir():
                if f.is_file():
                    f.unlink()
        (out / "captioned" / f"{spec.id}.pdf").unlink(missing_ok=True)
        try:
            render_pgf.render_figure(spec, out, prov=prov, venue=Vn)
            # The spec and its data next to the figure (G3): re-render without the store.
            serialize.write_figure(spec, out / "figures" / spec.id, prov)
        except Exception as exc:
            res["errors"].append(f"{spec.id} (pgf): {type(exc).__name__}: {exc}")
            return res
        if previews:
            try:
                render_mpl.render_figure(spec, out / "previews" / f"{spec.id}.png", venue=Vn)
                render_mpl.render_figure(spec, out / "previews" / f"{spec.id}.pdf", venue=Vn)
            except Exception as exc:
                res["errors"].append(f"{spec.id} (preview): {type(exc).__name__}: {exc}")
        res.update(figure=spec.id, width=spec.width, caption=spec.caption,
                   sources=list(spec.sources))
    elif isinstance(spec, TableSpec):
        (out / "captioned" / f"{spec.id}.pdf").unlink(missing_ok=True)
        render_pgf.render_table(spec, out, prov=prov)
        serialize.write_table(spec, out, prov)
        res.update(table=spec.id, width=spec.width, caption=spec.caption,
                   sources=list(spec.sources))
    return res


class _Fig:
    """What the size check and the upload PDF need of a written figure."""

    def __init__(self, fig_id: str, width: str) -> None:
        self.id, self.width = fig_id, width


def _compile_item(out: Path, r: dict[str, Any], Vn: style.Venue, check_size: bool,
                  standalone: bool, prov: dict[str, Any] | None
                  ) -> tuple[list[str], str | None, str | None]:
    """(problems, upload PDF, captioned PDF; paths relative to ``out``) of one written item.
    ``standalone`` writes the caption-free upload PDF of a figure and the captioned PDF of
    every figure and table."""
    probs: list[str] = []
    rel = cap = None
    kind = "figure" if r.get("figure") else "table"
    item_id = r.get("figure") or r.get("table")
    if kind == "figure":
        fig = _Fig(item_id, r.get("width", "column"))
        if check_size:
            probs += check_figure_size(out, fig, Vn)
        if standalone:
            pdf, p2 = standalone_pdf(out, fig, Vn)
            probs += p2
            rel = str(pdf.relative_to(out)) if pdf is not None else None
    if standalone:
        pdf, p3 = captioned_pdf(out, kind, item_id, r.get("caption", ""),
                                r.get("sources", []), prov, r.get("width", "column"), Vn,
                                label=r.get("label"))
        probs += p3
        cap = str(pdf.relative_to(out)) if pdf is not None else None
    return probs, rel, cap


def _export_group(config: Any, cohort: Any, entries: list[tuple[int, Any]], out: Path,
                  venue: str, prov: dict[str, Any], previews: bool
                  ) -> list[tuple[int, dict[str, Any]]]:
    """Worker: export ``entries`` [(index, item)] with its own DataContext."""
    from biomech.io.context import DataContext

    ctx = DataContext(config, cohort)
    try:
        with style.use_venue(venue) as Vn:
            return [(i, _export_one(ctx, e, out, Vn, prov, previews)) for i, e in entries]
    finally:
        ctx.close()


def _groups(todo: list[Any], n_workers: int) -> list[list[tuple[int, Any]]]:
    """Items by participant (one group each), the cohort-level items in ``n_workers``
    interleaved groups; largest first."""
    by_pid: dict[Any, list[tuple[int, Any]]] = {}
    rest: list[tuple[int, Any]] = []
    for i, e in enumerate(todo):
        pid = e.get("pid") if isinstance(e, dict) else None
        (by_pid.setdefault(pid, []) if pid is not None else rest).append((i, e))
    out = list(by_pid.values()) + [g for g in (rest[k::n_workers] for k in range(n_workers)) if g]
    return sorted(out, key=len, reverse=True)


def _export(ctx: Any, items: list[Any] | None, out: Path, Vn: style.Venue, *, previews: bool,
            compile_pdf: bool, check_size: bool, standalone: bool,
            on_progress: Callable[[int, int, str], None] | None,
            cancel: Callable[[], bool] | None, n_jobs: int = 1,
            groups: tuple[str, ...] | None = None) -> dict[str, list[str]]:
    import os
    from concurrent.futures import ThreadPoolExecutor

    out.mkdir(parents=True, exist_ok=True)
    render_pgf.write_style(out, Vn)
    from biomech.viz import serialize

    prov = serialize.provenance(ctx)
    done: dict[str, list[Any]] = {"figures": [], "tables": [], "errors": [], "problems": [],
                                  "cancelled": [], "standalone": [], "captioned": [],
                                  "venue": [Vn.name],
                                  "items": []}
    todo = list(items or [v.id for v in R.views("paper", groups)])
    if groups is not None and items:
        keep = []
        for entry in todo:
            vid = _item(entry)[0]
            g = R.REGISTRY[vid].group if vid in {v.id for v in R.views()} else None
            if g is not None and g not in groups:
                done["errors"].append(f"{vid}: a {g} view; this study's groups are "
                                      f"{', '.join(groups)}")
                continue
            keep.append(entry)
        todo = keep
    results: dict[int, dict[str, Any]] = {}
    compiles: dict[int, Any] = {}
    n_cpu = os.cpu_count() or 1
    latex = ThreadPoolExecutor(max_workers=max(1, min(8, n_cpu))) \
        if (check_size or standalone) else None

    def got(i: int, r: dict[str, Any]) -> None:
        results[i] = r
        if latex is not None and (r.get("figure") or r.get("table")):
            compiles[i] = latex.submit(_compile_item, out, r, Vn, check_size, standalone, prov)

    try:
        n_workers = min(8, n_cpu) if n_jobs == 0 else max(1, int(n_jobs))
        if n_workers > 1 and len(todo) > 1:
            _export_parallel(ctx, todo, out, Vn, prov, previews, n_workers, got, done,
                             on_progress, cancel)
        else:
            for n, entry in enumerate(todo):
                if cancel is not None and cancel():
                    done["cancelled"].append(f"cancelled after {n} of {len(todo)} items")
                    break
                if on_progress is not None:
                    on_progress(n, len(todo), _item(entry)[0])
                got(n, _export_one(ctx, entry, out, Vn, prov, previews))
        ids = [r.get("figure") or r.get("table") for r in results.values()]
        ids = [x for x in ids if x]
        if len(ids) != len(set(ids)) and n_workers > 1:
            # Two items wrote the same files: only the in-process order defines the result.
            for f in compiles.values():
                f.result()
            if latex is not None:
                latex.shutdown(wait=True)
            return _export(ctx, items, out, Vn, previews=previews, compile_pdf=compile_pdf,
                           check_size=check_size, standalone=standalone,
                           on_progress=on_progress, cancel=cancel, n_jobs=1, groups=groups)
        built = []
        for i in sorted(results):
            r = results[i]
            done["errors"] += r["errors"]
            if i in compiles:
                probs, rel, cap = compiles[i].result()
                done["problems"] += probs
                if rel is not None:
                    done["standalone"].append(rel)
                    r["standalone"] = rel
                if cap is not None:
                    done["captioned"].append(cap)
                    r["captioned"] = cap
            if r.get("figure"):
                done["figures"].append(r["figure"])
                built.append(("figure", r["figure"]))
            elif r.get("table"):
                done["tables"].append(r["table"])
                built.append(("table", r["table"]))
            if r.get("figure") or r.get("table"):
                done["items"].append({k: v for k, v in r.items() if k != "errors"})
    finally:
        if latex is not None:
            latex.shutdown(wait=True)
        (out / "jtsmm-minfont.lua").unlink(missing_ok=True)   # the size test's probe
    if on_progress is not None and not done["cancelled"]:
        on_progress(len(todo), len(todo), "preview" if compile_pdf else "")
    from biomech.viz import manifest as M

    caps = {(r.get("figure") or r.get("table")): (M.caption_latex(r.get("caption", "")),
                                                 M.caption_latex(", ".join(r.get("sources", []))))
            for r in results.values() if r.get("figure") or r.get("table")}
    tex = render_pgf.standalone_preview(out, built, venue=Vn, captions=caps, prov=prov)
    if compile_pdf and not done["cancelled"]:
        ok, log = _compile(tex)
        if not ok:
            done["errors"].append(f"LaTeX failed:\n{log[-3000:]}")
        else:
            done["problems"] += tex_problems(log)
        for ext in (".aux", ".log"):          # build leftovers (J1.3)
            tex.with_suffix(ext).unlink(missing_ok=True)
    return done


def _export_parallel(ctx: Any, todo: list[Any], out: Path, Vn: style.Venue,
                     prov: dict[str, Any], previews: bool, n_workers: int,
                     got: Callable[[int, dict[str, Any]], None], done: dict[str, list[str]],
                     on_progress: Callable[[int, int, str], None] | None,
                     cancel: Callable[[], bool] | None) -> None:
    """Item groups in worker processes (spawned); results handed to ``got`` as they come."""
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    groups = _groups(todo, n_workers)
    n_done = 0
    with ProcessPoolExecutor(max_workers=min(n_workers, len(groups)),
                             mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(_export_group, ctx.config, ctx.cohort, g, out, Vn.name, prov,
                               previews) for g in groups}
        stopped = False
        while pending:
            fin, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in fin:
                if fut.cancelled():
                    continue
                for i, r in fut.result():
                    got(i, r)
                    n_done += 1
                if on_progress is not None:
                    on_progress(n_done, len(todo), "")
            if not stopped and cancel is not None and cancel():
                stopped = True
                for fut in pending:
                    fut.cancel()                      # groups not started: dropped
                pending = {f for f in pending if not f.cancelled()}
                done["cancelled"].append(f"cancelled after {n_done} of {len(todo)} items")
