# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The gallery page: every depiction of every paper item side by side, to
choose from, in one local HTML file next to the gallery export (``gallery.html``).

:func:`write_html` reads what ``biomech export --gallery`` wrote (``manifest.json``, the
figures, previews and tables) and adds what a paper may use besides figures and tables:

* **figures**: one card per depiction, grouped by item (same view and parameters) under the
  storyboard sections of the paper set; the image is the compiled standalone figure
  (``figures/<id>/<id>.pdf`` from ``--standalone``, rasterized with ``pdftoppm``) or, without
  it, the matplotlib preview; under it the caption and the line that selects it in the paper
  set (``{id: ..., form: ...}``);
* **tables**: rendered from ``tables/<id>.json`` (the same cells as the LaTeX table);
* **videos**: first the supplementary videos the paper set lists (``supplementary_videos``,
  J2.7: label, the matched render, or "not rendered" / "other store" when it is
  missing or rendered from another snapshot than the set pins; :func:`listed_videos`), then
  every rendered video of the study (``<study>/videos/*.mp4``, by ``biomech animate``) with
  its layout, participant and duration, and its filmstrip PNG when one was written;
* **images**: hardware renders (no participant) that a setup figure can use.

Nothing is recomputed: the page only shows exported files, so it shows exactly what the
paper would get. Participant pseudonyms only (the export and the video manifests carry no
study ids).
"""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

PART_ORDER = {"core": 0, "supporting": 1, "supplement": 2}

CSS = """
:root { --bg:#fbfbfa; --card:#fff; --ink:#1d1f23; --ink2:#5b616b; --line:#e3e4e6;
        --accent:#0b62a4; --chip:#eef2f6; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16181b; --card:#1e2125; --ink:#e8e9eb; --ink2:#a3a8b0; --line:#2f3338;
          --accent:#6db3f2; --chip:#262a30; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding:20px 24px 8px; border-bottom:1px solid var(--line); }
h1 { font-size:20px; margin:0 0 4px; }
h2 { font-size:17px; margin:28px 24px 8px; padding-top:8px; border-top:1px solid var(--line); }
h3 { font-size:14px; margin:18px 24px 6px; color:var(--ink2); font-weight:600; }
.meta { color:var(--ink2); font-size:12.5px; }
nav { padding:8px 24px; display:flex; flex-wrap:wrap; gap:6px; position:sticky; top:0;
      background:var(--bg); border-bottom:1px solid var(--line); z-index:2; }
nav a { font-size:12px; padding:3px 8px; border-radius:10px; background:var(--chip);
        color:var(--ink); text-decoration:none; }
.row { display:flex; gap:14px; overflow-x:auto; padding:4px 24px 12px; align-items:flex-start; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px;
        flex:0 0 auto; width:min(560px, 92vw); }
.card.wide { width:min(900px, 94vw); }
.card img { width:100%; height:auto; background:#fff; border-radius:4px; }
.card video { width:100%; border-radius:4px; background:#000; }
.tag { display:inline-block; font-size:11.5px; padding:1px 7px; border-radius:9px;
       background:var(--chip); margin-right:4px; color:var(--ink2); }
.tag.default { background:var(--accent); color:#fff; }
.cap { font-size:12.5px; color:var(--ink2); margin-top:6px; max-height:9.5em; overflow:auto; }
code { font-size:11.5px; background:var(--chip); padding:1px 4px; border-radius:3px;
       user-select:all; }
table.t { border-collapse:collapse; font-size:12px; width:100%; }
table.t th, table.t td { border-bottom:1px solid var(--line); padding:3px 6px; text-align:left;
                          vertical-align:top; }
table.t th { color:var(--ink2); font-weight:600; }
.scroll { max-height:520px; overflow:auto; }
"""


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


def _rasterize(pdf: Path, png: Path, dpi: int = 160) -> bool:
    """The first page of ``pdf`` as ``png`` (pdftoppm); False when not available."""
    exe = shutil.which("pdftoppm")
    if exe is None or not pdf.is_file():
        return False
    if png.is_file() and png.stat().st_mtime >= pdf.stat().st_mtime:
        return True
    png.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([exe, "-png", "-r", str(dpi), "-singlefile", str(pdf),
                        str(png.with_suffix(""))], capture_output=True)
    return r.returncode == 0 and png.is_file()


def _image(out: Path, fid: str) -> str | None:
    """Relative path of the best picture of figure ``fid``: the compiled standalone figure
    (rasterized) or the preview PNG."""
    pdf = out / "figures" / fid / f"{fid}.pdf"
    png = out / "gallery_png" / f"{fid}.png"
    if _rasterize(pdf, png):
        return os.path.relpath(png, out)
    prev = out / "previews" / f"{fid}.png"
    return os.path.relpath(prev, out) if prev.is_file() else None


def _cell(col: dict[str, Any], v: Any) -> str:
    if isinstance(v, float):
        return "–" if v != v else f"{v:.{int(col.get('decimals', 2))}f}"
    if isinstance(v, bool):
        return "✓" if v else "–"
    return _esc(v).replace("$", "")


def _table_html(path: Path) -> str:
    try:
        t = json.loads(path.read_text(encoding="utf-8"))["table"]
    except (OSError, ValueError, KeyError):
        return "<p class='meta'>table file missing</p>"
    cols = t.get("columns", [])
    head = "".join(
        f"<th>{_esc(c.get('header', c['key'])).replace('$', '')}"
        + (f" ({_esc(c['unit'])})" if c.get("unit") else "") + "</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_cell(c, r.get(c['key']))}</td>" for c in cols)
                   + "</tr>" for r in t.get("rows", []))
    notes = f"<p class='cap'>{_esc(t.get('notes', ''))}</p>" if t.get("notes") else ""
    return (f"<div class='scroll'><table class='t'><tr>{head}</tr>{body}</table></div>"
            + notes)


def _select_line(item: dict[str, Any]) -> str:
    """What to add to the item's line in the paper set to choose this depiction (the other
    parameters stay as they are; participants are never named here, see ``_shown``)."""
    form = item.get("params", {}).get("form")
    return f"form: {form}" if form and "--form-" in item["id"] else "(current: no form key)"


def _shown(params: dict[str, Any], pseudonym: Any) -> dict[str, Any]:
    """Parameters as shown on the page: participant ids as pseudonyms (P3)."""
    out = {}
    for k, v in params.items():
        if k == "pid" and pseudonym is not None:
            try:
                v = f"P{pseudonym(v)}"
            except Exception:
                v = "P?"
        out[k] = v
    return out


def _group_key(item: dict[str, Any]) -> tuple:
    p = {k: v for k, v in item.get("params", {}).items() if k != "form"}
    return item["view"], tuple(sorted((k, str(v)) for k, v in p.items()))


def _videos(dirs: list[Path], out: Path) -> list[dict[str, Any]]:
    vids = []
    for d in dirs:
        for mp4 in sorted(Path(d).glob("*.mp4")):
            meta: dict[str, Any] = {}
            js = mp4.with_suffix(".json")
            if js.is_file():
                try:
                    meta = json.loads(js.read_text(encoding="utf-8"))
                except ValueError:
                    meta = {}
            strip = [q for q in (mp4.parent / f"{mp4.stem}_filmstrip.png",) if q.is_file()]
            vids.append({"path": os.path.relpath(mp4, out), "name": mp4.stem, "meta": meta,
                         "filmstrip": [os.path.relpath(p, out) for p in strip]})
    return vids


def listed_videos(pset: Any, dirs: list[Path], pseudonym: Any = None) -> list[dict[str, Any]]:
    """The paper set's ``supplementary_videos`` with their renders: per entry the label, a
    description (no study id), the matched video and its filmstrip (or None), the manifest
    and why it does not fit the set's store (:func:`biomech.papers.find_video`,
    :func:`biomech.papers.video_store_problems`)."""
    from biomech import papers

    out = []
    for v in getattr(pset, "videos", None) or []:
        try:
            mp4, meta = papers.find_video(v, dirs, pseudonym)
        except Exception as exc:                # unknown layout, no pseudonym, ...
            mp4, meta, err = None, None, str(exc)
        else:
            err = ""
        probs = papers.video_store_problems(meta, pset) if mp4 is not None else []
        strip = mp4.with_name(f"{mp4.stem}_filmstrip.png") if mp4 is not None else None
        out.append({"label": v.label, "describe": v.describe(), "path": mp4,
                    "filmstrip": strip if strip is not None and strip.is_file() else None,
                    "meta": meta or {}, "problems": probs,
                    "status": "not rendered" + (f" ({err})" if err else "") if mp4 is None
                    else ("other store" if probs else "rendered")})
    return out


def _listed_html(listed: list[dict[str, Any]], out: Path) -> list[str]:
    parts = ["<h3>Listed in the paper set (supplementary_videos)</h3><div class='row'>"]
    for e in listed:
        ok = e["status"] == "rendered"
        tag = (f"<span class='tag{' default' if ok else ''}'>{_esc(e['status'])}</span>")
        body = ""
        if e["path"] is not None:
            rel = os.path.relpath(e["path"], out)
            body = f"<video controls preload='metadata' src='{_esc(rel)}'></video>"
            if e["filmstrip"] is not None:
                fr = os.path.relpath(e["filmstrip"], out)
                body += f"<a href='{_esc(fr)}'><img loading='lazy' src='{_esc(fr)}' alt='filmstrip'></a>"
        snap = e["meta"].get("snapshot")
        cap = "; ".join(e["problems"]) or (f"snapshot {snap}" if snap else "")
        parts.append(f"<div class='card'>{tag}<span class='tag'>{_esc(e['label'])}</span>"
                     f"<code>{_esc(e['describe'])}</code>{body}<div class='cap'>{_esc(cap)}"
                     "</div></div>")
    parts.append("</div>")
    return parts


def write_html(out: str | Path, *, title: str, videos: list[Path] | None = None,
               images: list[tuple[str, Path]] | None = None, pseudonym: Any = None,
               listed: list[dict[str, Any]] | None = None) -> Path:
    """``out/gallery.html`` from the gallery export in ``out`` (see the module doc)."""
    from biomech.viz.views import vocab as V

    out = Path(out)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    items = man.get("items", [])
    prov = man.get("provenance", {})
    sections: dict[str, dict[tuple, list[dict[str, Any]]]] = {}
    for it in items:
        sections.setdefault(it.get("section") or "Items", {}).setdefault(
            _group_key(it), []).append(it)
    n_fig = sum(1 for i in items if i.get("kind") == "figure")
    n_tab = sum(1 for i in items if i.get("kind") == "table")
    vids = _videos(videos or [], out)
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>",
             f"<header><h1>{_esc(title)}</h1><div class='meta'>"
             f"snapshot {_esc(prov.get('snapshot', '?'))} · venue {_esc(man.get('venue'))} · "
             f"{n_fig} figure depictions, {n_tab} tables, {len(vids)} videos · exported "
             f"{_esc(prov.get('exported_utc', ''))}. Every depiction shows the same stored "
             "results; choose one per item and put its line into the paper set (or use "
             "the viewer's Paper → Gallery, which writes it).</div></header><nav>"]
    anchors = [(f"s{i}", s) for i, s in enumerate(sections)]
    if vids or listed:
        anchors.append(("videos", "Videos"))
    if images:
        anchors.append(("images", "Images"))
    parts += [f"<a href='#{a}'>{_esc(s)}</a>" for a, s in anchors]
    parts.append("</nav>")
    for (anchor, sec), groups in zip(anchors, sections.values()):
        parts.append(f"<h2 id='{anchor}'>{_esc(sec)}</h2>")
        for (view, _p), group in groups.items():
            first = group[0]
            params = _shown({k: v for k, v in first.get("params", {}).items() if k != "form"},
                            pseudonym)
            if "pid" in params and pseudonym is None:
                params.pop("pid")                     # never a study id on the page
            ptxt = ", ".join(f"{k} {v}" for k, v in params.items())
            parts.append(f"<h3>{_esc(first.get('label') or view)} · <code>{_esc(view)}</code>"
                         + (f" · {_esc(ptxt)}" if ptxt else "")
                         + f" · <span class='tag'>{_esc(first.get('part', ''))}</span>"
                         + (f"<span class='tag'>{len(group)} depictions</span>"
                            if len(group) > 1 else "") + "</h3><div class='row'>")
            for it in group:
                form = it.get("params", {}).get("form")
                is_default = "--form-" not in it["id"]
                name = V.choice_label("form", form) if form else "single depiction"
                tags = (f"<span class='tag{' default' if is_default else ''}'>"
                        f"{_esc(name)}{' · current' if is_default and form else ''}</span>")
                if it.get("kind") == "table":
                    body = _table_html(out / "tables" / f"{it['id']}.json")
                    wide = " wide"
                else:
                    img = _image(out, it["id"])
                    body = (f"<a href='{_esc(img)}'><img loading='lazy' src='{_esc(img)}' "
                            f"alt='{_esc(it.get('label', ''))} {_esc(name)}'></a>"
                            if img else "<p class='meta'>no picture</p>")
                    wide = ""
                parts.append(f"<div class='card{wide}'>{tags}<code>{_esc(_select_line(it))}"
                             f"</code>{body}<div class='cap'>{_esc(it.get('caption', ''))}"
                             "</div></div>")
            parts.append("</div>")
    if listed:
        parts.append("<h2 id='videos'>Videos</h2>")
        parts += _listed_html(listed, out)
    if vids:
        parts.append("<h2 id='videos'>Videos</h2><div class='row'>" if not listed else
                     "<h3>Every rendered video</h3><div class='row'>")
        for v in vids:
            m = v["meta"]
            lay = V.choice_label("video_layout", m.get("layout")) if m.get("layout") else ""
            dur = f"{float(m['duration_s']):.0f} s" if m.get("duration_s") else ""
            who = m.get("pseudonym", "")
            strips = "".join(f"<a href='{_esc(s)}'><img loading='lazy' src='{_esc(s)}' "
                             "alt='filmstrip'></a>" for s in v["filmstrip"])
            parts.append(f"<div class='card'><span class='tag default'>{_esc(lay or v['name'])}"
                         f"</span><span class='tag'>{_esc(who)}</span><span class='tag'>"
                         f"{_esc(', '.join(m.get('labels', [])))}</span><span class='tag'>"
                         f"{_esc(dur)}</span><video controls preload='metadata' "
                         f"src='{_esc(v['path'])}'></video>{strips}<div class='cap'>"
                         f"<code>{_esc(v['name'])}.mp4</code></div></div>")
        parts.append("</div>")
    if images:
        parts.append("<h2 id='images'>Images</h2><div class='row'>")
        for cap, p in images:
            rel = os.path.relpath(p, out)
            parts.append(f"<div class='card'><a href='{_esc(rel)}'><img loading='lazy' "
                         f"src='{_esc(rel)}' alt='{_esc(cap)}'></a><div class='cap'>"
                         f"{_esc(cap)}</div></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    dest = out / "gallery.html"
    dest.write_text("\n".join(parts), encoding="utf-8")
    return dest


def write_index(root: str | Path) -> Path:
    """``root/gallery.html``: a link to every gallery page under ``root`` (the analysis
    folder of all studies), so both papers' galleries open from one place."""
    root = Path(root)
    pages = sorted(p for p in root.glob("**/gallery.html")
                   if p.parent != root and "_archive" not in p.parts)
    rows = []
    for p in pages:
        try:
            man = json.loads((p.parent / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            man = {}
        meta, prov = man.get("meta", {}), man.get("provenance", {})
        n = len(man.get("items", []))
        rows.append(f"<div class='card'><h3 style='margin:0 0 6px'><a href='"
                    f"{_esc(os.path.relpath(p, root))}'>{_esc(meta.get('paper', p.parent.name))}"
                    f" · {_esc(meta.get('target', ''))}</a></h3><div class='meta'>{n} items · "
                    f"snapshot {_esc(prov.get('snapshot', '?'))} · venue "
                    f"{_esc(man.get('venue', ''))} · {_esc(prov.get('exported_utc', ''))}</div>"
                    "</div>")
    html_ = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width, initial-scale=1'><title>Paper galleries</title>"
             f"<style>{CSS}</style></head><body><header><h1>Paper galleries</h1><div "
             "class='meta'>Every depiction of every paper item, the tables, videos and images, "
             "per paper. Choose per item; the choice goes into the paper set (form).</div>"
             "</header><div class='row' style='flex-wrap:wrap'>" + "".join(rows)
             + "</div></body></html>")
    dest = root / "gallery.html"
    dest.write_text(html_, encoding="utf-8")
    return dest
