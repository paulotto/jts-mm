# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Integrity of exported artefacts against the analysis results (principle 5).

An export folder (paper set target, ``export --all``) is verified in three steps:

1. **Files** (:func:`verify_files`): ``manifest.json`` exists, every file it lists exists with
   the recorded sha256, and no exported figure or table is missing from it (a hand-edited
   or stale file is caught).
2. **Snapshot** (:func:`verify_rebuild`): the export came from the snapshot the paper set
   pins, and every item rebuilt now from that snapshot equals the exported item: the same
   panels, series, values (lossless arrays of ``spec.npz``), labels and caption; for tables
   every cell. A difference means the export is stale (results or code changed since) or a
   view is not deterministic -- both must be fixed before the numbers go into a paper.
3. In-text numbers and caption numbers are checked against the store by the numbers
   resolver (:mod:`biomech.viz.numbers`): every number re-resolved, cited
   numbers from the cited study's snapshot, means from their operands, with their n; a
   cited number must come from the snapshot the cited study's paper set pins
   (:func:`verify_cited`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

RTOL, ATOL = 1e-9, 1e-12
ARRAY_FIELDS = ("x", "y", "y2", "ylo", "c", "z")


def verify_files(out: Path) -> list[str]:
    """Problems of the files of an export folder against its manifest."""
    from biomech.viz import manifest as M

    out = Path(out)
    if not (out / "manifest.json").is_file():
        return [f"{out}: no manifest.json (export with this version of biomech)"]
    man = M.read(out)
    probs: list[str] = []
    listed: set[str] = set()
    for it in man.get("items", []):
        for rel, h in it.get("files", {}).items():
            listed.add(rel)
            p = out / rel
            if not p.is_file():
                probs.append(f"{it['id']}: missing file {rel}")
            elif M.sha256(p) != h:
                probs.append(f"{it['id']}: {rel} changed after the export (sha256 differs)")
    if man.get("meta", {}).get("standalone"):
        # an export with --standalone has the upload PDF of every figure and the captioned
        # PDF of every item, written by this export (J1.3)
        for it in man.get("items", []):
            want = [f"captioned/{it['id']}.pdf"]
            if it.get("kind") == "figure":
                want.append(f"figures/{it['id']}/{it['id']}.pdf")
            for rel in want:
                if rel not in it.get("files", {}):
                    probs.append(f"{it['id']}: no {rel} from this export (compile failed?)")
    on_disk = {str(p.relative_to(out)) for p in out.glob("figures/*/spec.json")}
    on_disk |= {str(p.relative_to(out)) for p in out.glob("tables/*.json")}
    extra = sorted(on_disk - listed)
    if extra:
        probs.append(f"exported items not in the manifest (left over from an earlier export?): "
                     f"{', '.join(extra[:10])}")
    return probs


def _num_equal(a: Any, b: Any) -> bool:
    fa = isinstance(a, (int, float, np.integer, np.floating)) and not isinstance(a, bool)
    fb = isinstance(b, (int, float, np.integer, np.floating)) and not isinstance(b, bool)
    if a is None and fb and math.isnan(float(b)):
        return True
    if b is None and fa and math.isnan(float(a)):
        return True
    if fa and fb:
        x, y = float(a), float(b)
        return (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def _plain(v: Any) -> Any:
    """A cell as the export serializes it (lists and arrays as plain lists)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, tuple):
        return list(v)
    return v


def _arr_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.allclose(a.astype(float), b.astype(float), rtol=RTOL, atol=ATOL,
                                equal_nan=True))
    return bool(np.array_equal(a, b))


def spec_diff(exported: Any, rebuilt: Any) -> list[str]:
    """Where two figure or table specs differ (values, labels, caption); [] if equal."""
    from biomech.viz.spec import FigureSpec

    d: list[str] = []
    if type(exported) is not type(rebuilt):
        return [f"kind {type(exported).__name__} vs {type(rebuilt).__name__}"]
    if exported.caption != rebuilt.caption:
        d.append("caption differs")
    if isinstance(exported, FigureSpec):
        if len(exported.panels) != len(rebuilt.panels):
            return d + [f"{len(exported.panels)} vs {len(rebuilt.panels)} panels"]
        for i, (p, q) in enumerate(zip(exported.panels, rebuilt.panels)):
            tag = f"panel {i + 1}"
            for attr in ("title", "note"):
                if getattr(p, attr) != getattr(q, attr):
                    d.append(f"{tag}: {attr} differs")
            for ax in ("x", "y"):
                if getattr(p, ax).label != getattr(q, ax).label:
                    d.append(f"{tag}: {ax} label differs")
            if len(p.series) != len(q.series):
                d.append(f"{tag}: {len(p.series)} vs {len(q.series)} series")
                continue
            for j, (s, t) in enumerate(zip(p.series, q.series)):
                if (s.kind, s.label, s.role) != (t.kind, t.label, t.role):
                    d.append(f"{tag} series {j + 1}: kind/label/role differ")
                for f in ARRAY_FIELDS:
                    if not _arr_equal(getattr(s, f, None), getattr(t, f, None)):
                        d.append(f"{tag} series {j + 1} ({s.label or s.kind}): {f} values differ")
                if list(s.text or []) != list(t.text or []):
                    d.append(f"{tag} series {j + 1}: text differs")
    else:
        if [c.key for c in exported.columns] != [c.key for c in rebuilt.columns]:
            return d + ["columns differ"]
        if [c.header for c in exported.columns] != [c.header for c in rebuilt.columns]:
            d.append("headers differ")
        keys = [c.key for c in exported.columns]
        # Rule rows (``_rule``: a midrule) are layout, not data; the export drops them.
        ra = [r for r in exported.rows if not r.get("_rule")]
        rb = [r for r in rebuilt.rows if not r.get("_rule")]
        if len(ra) != len(rb):
            return d + [f"{len(ra)} vs {len(rb)} rows"]
        bad = [(i, k) for i, (r, s) in enumerate(zip(ra, rb))
               for k in keys if not _num_equal(_plain(r.get(k)), _plain(s.get(k)))]
        if bad:
            i, k = bad[0]
            d.append(f"{len(bad)} cells differ (first: row {i + 1}, {k}: "
                     f"{ra[i].get(k)!r} vs {rb[i].get(k)!r})")
    return d


def rebuild(ctx: Any, item: dict[str, Any], venue: str) -> Any:
    """An exported item rebuilt as the export builds it (reader wording, panel letters)."""
    from biomech.viz import style
    from biomech.viz.export import auto_tags
    from biomech.viz.spec import FigureSpec
    from biomech.viz.views import registry as R
    from biomech.viz.views import vocab as V

    with style.use_venue(venue):
        spec = V.reader_spec(R.build(item["view"], R.request(ctx, item["view"],
                                                             **item.get("params", {}))))
        if isinstance(spec, FigureSpec):
            auto_tags(spec)
    return spec


def verify_rebuild(out: Path, ctx: Any) -> list[str]:
    """Problems of an export against the store ``ctx`` reads: reporting-population (variant,
    ) or snapshot mismatch and every item whose rebuilt spec differs from the
    exported one."""
    from biomech.viz import manifest as M
    from biomech.viz import serialize

    out = Path(out)
    man = M.read(out)
    probs: list[str] = []
    snap_exp = (man.get("provenance") or {}).get("snapshot")
    snap_now = getattr(ctx.config, "snapshot", None) or "live"
    pinned = (man.get("meta") or {}).get("snapshot_pinned")
    if pinned and snap_exp != pinned:
        probs.append(f"exported from {snap_exp}, but the paper set pins {pinned}")
    if snap_exp != snap_now:
        probs.append(f"exported from {snap_exp}; verified against {snap_now}")
    var_exp = (man.get("provenance") or {}).get("variant") or "primary"
    var_now = getattr(ctx.config, "variant", None) or "primary"
    var_pin = (man.get("meta") or {}).get("variant") or "primary"
    if "snapshot_pinned" in (man.get("meta") or {}) and var_exp != var_pin:   # a paper set
        probs.append(f"exported from the {var_exp} store, but the paper set reports {var_pin}")
    if var_exp != var_now:
        probs.append(f"exported from the {var_exp} store; verified against {var_now}")
    for it in man.get("items", []):
        try:
            new = rebuild(ctx, it, man.get("venue", "ieee"))
        except Exception as exc:
            probs.append(f"{it['id']}: rebuild failed: {type(exc).__name__}: {exc}")
            continue
        try:
            old = (serialize.load_figure(out / "figures" / it["id"] / "spec.json")
                   if it["kind"] == "figure" else serialize.load_table(
                       out / "tables" / f"{it['id']}.json"))
        except Exception as exc:
            probs.append(f"{it['id']}: exported spec unreadable: {type(exc).__name__}: {exc}")
            continue
        diff = spec_diff(old, new)
        if diff:
            probs.append(f"{it['id']}: differs from the results now: {'; '.join(diff[:4])}")
    return probs


def verify_cited(out: Path) -> list[str]:
    """Cited numbers (``numbers.json`` ``cited``) against the cited study: each must come
    from a snapshot that an active paper set of that study pins (a cited value from another
    snapshot than the one the other paper reports would differ from it)."""
    import json

    from biomech import papers

    jp = Path(out) / "numbers.json"
    if not jp.is_file():
        return []
    cited = (json.loads(jp.read_text(encoding="utf-8")).get("cited") or {})
    probs: list[str] = []
    for (study, snap), keys in _by_source(cited).items():
        try:
            pins = {ps.snapshot for ps in papers.study_paper_sets(study) if ps.snapshot}
        except Exception as exc:
            probs.append(f"cited from {study}: paper sets unreadable ({exc})")
            continue
        if pins and snap not in pins:
            probs.append(f"cited numbers {', '.join(keys[:5])}: from snapshot {snap}, but the "
                         f"cited study's paper set pins {', '.join(sorted(pins))}")
    return probs


def _by_source(cited: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for k, v in cited.items():
        out.setdefault((str(v.get("study")), str(v.get("snapshot"))), []).append(k)
    return out


def verify(out: Path, ctx: Any) -> list[str]:
    """Files, the rebuild against ``ctx``, the in-text numbers (own, cited and means,
    re-resolved) and the cited snapshots (empty list: the export is consistent with the
    results)."""
    from biomech.viz import numbers as N

    probs = verify_files(out)
    if any("no manifest.json" in p for p in probs):
        return probs
    return (probs + verify_rebuild(out, ctx) + N.verify(out, ctx) + verify_cited(out)
            + verify_bundle_tie(out, ctx))


def verify_bundle_tie(out: Path, ctx: Any) -> list[str]:
    """An export made from a bundled snapshot names that bundle (J9.4): the bundle
    recorded in its manifest must still be the snapshot's bundle (same file hash). Exports
    without a recorded bundle pass (bundle written after the export: re-export to tie them)."""
    from biomech.results.bundle import bundle_path, info
    from biomech.viz import manifest as M

    rec = (M.read(Path(out)).get("provenance") or {}).get("bundle")
    if not rec:
        return []
    root = Path(getattr(ctx.config, "read_dir", "") or "")
    now = info(bundle_path(root)) if getattr(ctx.config, "snapshot", None) else None
    if now is None:
        return [f"export names bundle {rec.get('file')}, which no longer exists in {root}"]
    if (now.get("sha256"), now.get("bundle_sha256")) != (rec.get("sha256"),
                                                         rec.get("bundle_sha256")):
        return [f"export names bundle {rec.get('file')} sha256 {str(rec.get('sha256'))[:12]}, "
                f"the snapshot's bundle is now {str(now.get('sha256'))[:12]}"]
    return []


# -- videos ----------------------------------------------------------------------------------
def _flat(d: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}/"))
    elif isinstance(d, np.ndarray) and d.dtype.kind in "fiub":
        out[prefix.rstrip("/")] = d
    return out


def verify_clip(path: Path, ctx: Any) -> list[str]:
    """A rendered video's data against the results: the clip file stored next to it
    (``<stem>.clip.npz/.json``, what the frames were drawn from) and the same clip loaded
    again from ``ctx`` (participant by pseudonym, the same phases and window) must hold the
    same arrays. Empty list = consistent."""
    from biomech.anim import make
    from biomech.anim.clipfile import load_clip_file, read_clip_doc

    doc = read_clip_doc(path)
    c, a = doc["clip"], doc["animation"]
    if c.get("kind") != "clip":
        return []                               # cohort clips: paths of all participants
    inv = {f"P{n}": pid for pid, n in ctx.pseudonyms().items()}
    pid = inv.get(str(c.get("pseudonym")))
    if pid is None:
        return [f"{Path(path).name}: participant {c.get('pseudonym')} not in this study"]
    saved = load_clip_file(path).clip
    t0_s, t1_s = (c.get("meta", {}).get("window_s") or [None, None])[:2]
    fresh = make(ctx, pid, a["layout"], [s[0] for s in c["segments"]], t0_s, t1_s,
                 speed=float(a["speed"]), fps=float(a["fps"]), theme=a["theme"]).clip
    probs = []
    for field in ("kin", "emg", "imu", "labels", "posterior", *sorted(saved.extra)):
        A = _flat(saved.extra.get(field) if field in saved.extra else getattr(saved, field, None))
        Bf = _flat(fresh.extra.get(field) if field in saved.extra else getattr(fresh, field, None))
        for k in sorted(set(A) & set(Bf)):
            x, y = A[k], Bf[k]
            if x.shape != y.shape or not np.allclose(x, y, rtol=1e-6, atol=1e-6, equal_nan=True):
                probs.append(f"{Path(path).name}: {field}/{k} differs from the results")
        if set(A) != set(Bf):
            probs.append(f"{Path(path).name}: {field} keys differ "
                         f"({sorted(set(A) ^ set(Bf))[:4]})")
    return probs
