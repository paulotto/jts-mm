# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Figures and tables of the parts (make analyses; needs make renders first).

    ../.venv/bin/python tools/analyses.py

figures/  lineage_teeth_attachment   teeth attachment from the JTS original to version 5, one scale
          marker_versions            head marker, mouth marker, calibration tool versions
          v5_variants                version-5 variants coloured by their distance to the saved CAD
          crosscheck                 top views: tag seats (CAD) vs. optical.yaml, IMU seats,
                                     calibration-tool rod vs. the pivot-calibrated tip
tables/   parts.{csv,tex}            every variant: status, CAD, size, volume, mass (solid, printed;
                                     from renders/index.json)
          v5_variants.{csv,tex}      max / p95 distance and moved area of the version-5 variants
          crosscheck.{csv,tex,json}  CAD vs. configuration numbers of the crosscheck figure (the
                                     json also holds the measured geometry, for the viewer)

Figures embed their renders at 300 dpi (common.save_fig). The CAD-vs-configuration numbers are a consistency check of the printed geometry against
software/optical/etc/optical.yaml and the pivot calibration; they change no pipeline input.
"""

from __future__ import annotations

import json
import sys
import textwrap

import numpy as np
import yaml
from common import (INDEX, MECH, PRINTED, REPO, TABLES, distances, load_mesh, manifest, mpl,
                    read_json, save_fig, write_json, write_text)
from PIL import Image

OPTICAL = REPO / "software" / "optical" / "etc"


# -- shared ------------------------------------------------------------------------------------
def cropped(png):
    """RGBA render cropped to its content; returns (array, crop box in px)."""
    im = Image.open(png).convert("RGBA")
    box = im.getbbox() or (0, 0, *im.size)
    return np.asarray(im.crop(box)), box


def show_render(ax, part, vid, index, man, title):
    """A render at its true scale (mm) in ``ax``; returns its width and height in mm."""
    rel = f"{part}/{vid}"
    img, _ = cropped(man.render(part, vid))
    s = index[rel]["mm_per_px"]
    h, w = img.shape[:2]
    ax.imshow(img, extent=(0, w * s, 0, h * s), interpolation="lanczos")
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    return w * s, h * s


def scale_bar(ax, mm, label=None):
    x0, x1 = ax.get_xlim()
    y0, _ = ax.get_ylim()
    ax.plot([x0, x0 + mm], [y0, y0], color="k", lw=2, solid_capstyle="butt", clip_on=False)
    ax.text(x0 + mm / 2, y0, (label or f"{mm:g} mm") + "\n", ha="center", va="bottom", fontsize=7)


def grid(man, index, rows, fname, bar=10):
    """Rows of (part, [variant, ...]) titled by the manifest; one scale per row."""
    plt = mpl()
    ncol = max(len(r[1]) for r in rows)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(1.55 * ncol, 1.85 * len(rows)),
                            squeeze=False)
    for i, (part, cells) in enumerate(rows):
        sizes = [show_render(axs[i][j], part, vid, index, man,
                             textwrap.fill(man.parts[part].variants[vid].title, 20))
                 for j, vid in enumerate(cells)]
        wmax, hmax = max(s[0] for s in sizes), max(s[1] for s in sizes)
        for j in range(ncol):
            ax = axs[i][j]
            if j >= len(cells):
                ax.set_axis_off()
                continue
            ax.set_xlim(-0.04 * wmax, wmax * 1.04)
            ax.set_ylim(-0.12 * hmax, hmax * 1.04)
        axs[i][0].text(-0.08, 0.5, man.parts[part].title, transform=axs[i][0].transAxes,
                       rotation=90, ha="right", va="center", fontsize=9)
        scale_bar(axs[i][0], bar if wmax < 120 else 50)
    fig.tight_layout(w_pad=0.4, h_pad=0.6)
    save_fig(fig, fname)
    plt.close(fig)


# -- lineage -------------------------------------------------------------------------------------
def lineage(man, index):
    grid(man, index, [("teeth_attachment",
                       ["jts", "v1", "v2", "v4", "v5_draft", "v5_5_5mm"])],
         "lineage_teeth_attachment")
    grid(man, index, [("head_marker", ["jts", "aruco_2026-08-18", "qualisys_aruco"]),
                      ("mouth_marker", ["jts", "v1", "aruco", "qualisys_aruco"]),
                      ("calibration_tool", ["jts", "aruco", "qualisys_aruco"])],
         "marker_versions")


# -- version-5 variants-----------------------------------------------------------------------------
def v5_variants(man):
    """Each version-5 variant coloured by its distance to the saved CAD state (5.5 mm)."""
    import pyvista as pv

    part = man.parts["teeth_attachment"]
    ref = load_mesh(part.variants["v5_5_5mm"].stl)
    order = ["v5_4mm", "v5_4_5mm", "v5_5mm", "v5_6_5mm", "v5_print_2026-10-03"]
    vmax = 4.0
    imgs, rows = [], []
    for vid in order:
        v = part.variants[vid]
        m = load_mesh(v.stl)
        d = distances(m.vertices, ref)
        rows.append({"variant": vid, "title": v.title, "max_mm": round(float(d.max()), 2),
                     "p95_mm": round(float(np.percentile(d, 95)), 2),
                     "moved_area_pct": round(100 * float(
                         m.area_faces[(d[m.faces] > 0.05).any(1)].sum() / m.area), 1)})
        mesh = pv.wrap(m)
        mesh["d"] = d
        pl = pv.Plotter(off_screen=True, window_size=(1400, 1400), lighting="light_kit")
        pl.add_mesh(pv.wrap(ref), color="#bdbdbd", opacity=0.2)
        pl.add_mesh(mesh, scalars="d", cmap="YlOrRd", clim=(0, vmax), show_scalar_bar=False,
                    smooth_shading=True, split_sharp_edges=True)
        pl.enable_parallel_projection()
        c = ref.bounds.mean(0)
        pl.camera_position = [tuple(c + np.array([-260.0, -420.0, 380.0])), tuple(c), (0, 0, 1)]
        pl.camera.parallel_scale = float(np.linalg.norm(np.ptp(ref.bounds, 0))) / 2 * 0.9
        imgs.append(pl.screenshot(None, transparent_background=True, return_img=True))
        pl.close()
    plt = mpl()
    fig, axs = plt.subplots(1, len(order), figsize=(1.75 * len(order), 2.1))
    for ax, img, r in zip(axs, imgs, rows):
        im = Image.fromarray(img)
        im = im.crop(im.getbbox())
        ax.imshow(np.asarray(im))
        ax.set_title(textwrap.fill(r["title"], 18) + f"\nmax {r['max_mm']:.1f} mm", fontsize=7)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=list(axs), fraction=0.025, pad=0.02)
    cb.set_label("distance to version 5, 5.5 mm (saved CAD) / mm", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    save_fig(fig, "v5_variants")
    plt.close(fig)
    return rows


# -- parts table ---------------------------------------------------------------------------------
def parts_rows(man, index):
    """One row per variant from renders/index.json (masses: common.masses, via render.py)."""
    out = []
    for part in man.parts.values():
        for v in part.variants.values():
            g = index.get(f"{part.id}/{v.id}")
            if g is None:
                continue
            out.append({"part": part.id, "part_title": part.title, "variant": v.id,
                        "title": v.title, "status": v.status,
                        "cad": "pending" if v.cad_pending else ("yes" if v.fcstd_body else "no"),
                        "size_mm": " x ".join(f"{x:.1f}" for x in g["size_mm"]),
                        "volume_cm3": round(g["volume_mm3"] / 1000, 2),
                        **{k: g.get(k) for k in ("material", "mass_solid_g", "mass_printed_g",
                                                 "triangles")}})
    return out


# -- CAD vs configuration ------------------------------------------------------------------------
def planar_seats(m, side_lo, side_hi, normal=(0, 0, 1)):
    """Upward planar facets whose in-plane bounding box is about square (side_lo..side_hi):
    the tag seats. Returns [(centre, side, area, angle_deg)], largest and highest first."""
    n0 = np.asarray(normal, float)
    out = []
    for f, n, a in zip(m.facets, m.facets_normal, m.facets_area):
        if n @ n0 < 0.999:
            continue
        pts = m.vertices[np.unique(m.faces[f])][:, :2]
        best = None
        for ang in np.arange(0.0, 90.0, 0.1):          # the box angle that fits tightest
            r = np.radians(ang)
            rot = pts @ np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
            ext = np.ptp(rot, 0)
            if best is None or ext.prod() < best[0]:
                lo = rot.min(0)
                c2 = lo + ext / 2
                ctr = c2 @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
                best = (ext.prod(), ext, ctr, ang)
        _, ext, ctr, ang = best
        if side_lo <= ext.min() and ext.max() <= side_hi and ext.max() / ext.min() < 1.06:
            z = float(m.vertices[np.unique(m.faces[f])][:, 2].mean())
            out.append((np.array([ctr[0], ctr[1], z]), float(ext.mean()), float(a), ang))
    out.sort(key=lambda s: (-round(s[0][2], 1), -s[2]))
    return out


def rect_seat(m, w, h, tol=0.6):
    """Upward planar facet with an axis-aligned w x h bounding box (the IMU seat)."""
    for f, n, a in zip(m.facets, m.facets_normal, m.facets_area):
        if n[2] < 0.999:
            continue
        pts = m.vertices[np.unique(m.faces[f])]
        ext = np.ptp(pts[:, :2], 0)
        if sorted(np.abs(np.sort(ext) - np.sort([w, h]))) [-1] < tol:
            lo = pts.min(0)
            return np.array([lo[0] + ext[0] / 2, lo[1] + ext[1] / 2, pts[:, 2].mean()]), ext
    return None, None


def body_frame(seats, axis_hint=None):
    """Origin at the midpoint of the two tag seats, x from one seat to the other, z up."""
    a, b = seats[0][0], seats[1][0]
    if a[0] > b[0]:
        a, b = b, a
    x = b - a
    x[2] = 0
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0, 1])
    y = np.cross(z, x)
    if axis_hint is not None and y @ axis_hint > 0:     # body -y must point along the hint
        x, y = -x, -y
    return (a + b) / 2, np.vstack([x, y, z])            # rows: body axes in CAD coordinates


def crosscheck(man):
    opt = yaml.safe_load((OPTICAL / "optical.yaml").read_text())["markers"]
    mounts = opt["mounts_T_body_tag"]
    tip = json.loads((OPTICAL / "ct_tip_calibration.json").read_text())

    def cfg(body):
        ids = [str(i) for i in opt["rigid_bodies"][body]["tag_ids"]]
        T = [np.array(mounts[i], float) for i in ids]
        return ids, T

    targets = [("HP", "head_marker", "qualisys_aruco", (36, 44)),
               ("MP", "mouth_marker", "qualisys_aruco", (28, 32)),
               ("CT", "calibration_tool", "qualisys_aruco", (36, 44)),
               ("eval_array", "trueness_array", "v1", (38, 44))]
    rows, geo = [], {}
    for body, part, vid, (lo, hi) in targets:
        m = load_mesh(man.parts[part].variants[vid].stl)
        seats = planar_seats(m, lo, hi)[:2]
        ids, T = cfg(body)
        d_cfg = float(np.linalg.norm(T[0][:3, 3] - T[1][:3, 3]))
        if len(seats) < 2:
            rows.append({"body": body, "quantity": "tag spacing", "config": round(d_cfg, 2),
                         "cad": None, "note": "tag seats not found"})
            continue
        d_cad = float(np.linalg.norm(seats[0][0][:2] - seats[1][0][:2]))
        rows.append({"body": body, "part": part, "variant": vid, "quantity": "tag spacing / mm",
                     "config": round(d_cfg, 2), "cad": round(d_cad, 2),
                     "diff": round(d_cad - d_cfg, 2) + 0.0})          # + 0.0: no "-0.0"
        geo[body] = {"part": part, "variant": vid, "seats": [s[0].tolist() for s in seats],
                     "seat_side": [round(s[1], 1) for s in seats], "angles": [s[3] for s in seats]}
        if body == "CT":
            ang_cfg = float(np.degrees(np.arctan2(T[0][1, 0], T[0][0, 0])))
            a_cad = seats[0][3] % 90
            rows.append({"body": body, "part": part, "variant": vid,
                         "quantity": "tag rotation in plane / deg", "config": round(abs(ang_cfg), 1),
                         "cad": round(min(a_cad, 90 - a_cad) if abs(ang_cfg) < 45 else
                                      max(a_cad, 90 - a_cad), 1)})
            # rod: the long thin end of the tool; its axis from the last 30 mm of vertices
            v = m.vertices
            mid = np.mean([s[0] for s in seats], 0)
            far = v[np.argmax(np.linalg.norm(v[:, :2] - mid[:2], axis=1))]
            u = (far - mid)
            u[2] = 0
            u /= np.linalg.norm(u)
            along = (v - mid) @ u
            end = along.max()
            sel = v[along > end - 30]
            axis_pt = sel.mean(0)
            origin, R = body_frame(seats, axis_hint=u)
            # body origin sits off the seat midpoint by the configured tag offset (y = -8.38)
            off_cfg = np.mean([t[:3, 3] for t in T], 0)
            origin = origin - R.T @ off_cfg
            tip_cad = origin + R.T @ np.array(tip["tip_body_mm"])
            rod_end = mid + u * end
            rod_end[2] = axis_pt[2]
            rel = tip_cad - axis_pt
            lateral = rel - u * (rel @ u)
            rows += [
                {"body": body, "part": part, "variant": vid,
                 "quantity": "tip beyond printed rod end / mm", "config": None,
                 "cad": round(float((tip_cad - rod_end) @ u), 1),
                 "note": "pivot-calibrated tip vs. CAD rod end (metal dart point)"},
                {"body": body, "part": part, "variant": vid,
                 "quantity": "tip off the rod axis, in plane / mm", "config": None,
                 "cad": round(float(lateral[:2] @ np.cross([0, 0, 1], u)[:2]), 2)},
                {"body": body, "part": part, "variant": vid,
                 "quantity": "rod axis below tag plane / mm",
                 "config": round(-float(tip["tip_body_mm"][2]), 2),
                 "cad": round(float(origin[2] - axis_pt[2]), 2),
                 "note": "config: pivot tip z in the body frame"}]
            geo[body].update({"rod_axis_pt": axis_pt.tolist(), "rod_dir": u.tolist(),
                              "rod_end": rod_end.tolist(), "tip_cad": tip_cad.tolist(),
                              "origin": origin.tolist(), "R": R.tolist()})
    for body, part, vid, wh in (("HP", "head_marker", "qualisys_aruco", (28.0, 25.4)),
                                ("MP", "mouth_marker", "qualisys_aruco", (23.9, 25.4))):
        m = load_mesh(man.parts[part].variants[vid].stl)
        c, ext = rect_seat(m, *wh)
        if c is None or body not in geo:
            continue
        mid = np.mean(np.array(geo[body]["seats"]), 0)
        rows.append({"body": body, "part": part, "variant": vid,
                     "quantity": "IMU seat centre from tag midpoint / mm", "config": None,
                     "cad": round(float(np.linalg.norm(c[:2] - mid[:2])), 1),
                     "note": f"in plane; seat floor {round(mid[2] - c[2], 1) + 0.0:.1f} mm "
                             "below the tags"})
        geo[body]["imu_seat"] = c.tolist()
        geo[body]["imu_seat_size"] = ext.tolist()
    return rows, geo


def crosscheck_figure(man, geo):
    plt = mpl()
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon, Rectangle

    order = [b for b in ("HP", "MP", "CT", "eval_array") if b in geo]
    fig, axs = plt.subplots(1, len(order), figsize=(2.3 * len(order), 2.6))
    names = {"HP": "Head marker", "MP": "Mouth marker", "CT": "Calibration tool",
             "eval_array": "Trueness array"}
    for ax, body in zip(np.atleast_1d(axs), order):
        g = geo[body]
        m = load_mesh(man.parts[g["part"]].variants[g["variant"]].stl)
        up = m.face_normals[:, 2] > 0.3
        ax.add_collection(PolyCollection(m.vertices[m.faces[up]][:, :, :2], facecolor=PRINTED,
                                         edgecolor="none", alpha=0.35,
                                         rasterized=True))      # thousands of triangles
        for c, side, ang in zip(g["seats"], g["seat_side"], g["angles"]):
            r = np.radians(ang)
            R = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2 @ R + c[:2]
            ax.add_patch(Polygon(sq, fill=False, ec="k", lw=0.9))
            sq30 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 15.0 @ R + c[:2]
            ax.add_patch(Polygon(sq30, fill=False, ec="#c0392b", lw=0.8, ls="--"))
        if "imu_seat" in g:
            c, (w, h) = g["imu_seat"], g["imu_seat_size"]
            ax.add_patch(Rectangle((c[0] - w / 2, c[1] - h / 2), w, h, fill=False,
                                   ec="#e8742a", lw=1.0))
        if "tip_cad" in g:
            p, e = np.array(g["tip_cad"]), np.array(g["rod_end"])
            ax.plot(*e[:2], "o", ms=3, color="k")
            ax.plot(*p[:2], "x", ms=6, mew=1.5, color="#c0392b")
            ax.plot([e[0], p[0]], [e[1], p[1]], color="#c0392b", lw=0.8)
        lo, hi = m.bounds[0][:2], m.bounds[1][:2]
        if "tip_cad" in g:
            lo = np.minimum(lo, np.array(g["tip_cad"])[:2] - 5)
            hi = np.maximum(hi, np.array(g["tip_cad"])[:2] + 5)
        pad = 4
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(names[body], fontsize=8)
        ax.set_ylim(lo[1] - 3 * pad, hi[1] + pad)
        ax.plot([lo[0], lo[0] + 20], [lo[1] - 1.5 * pad] * 2, color="k", lw=2)
        ax.text(lo[0] + 22, lo[1] - 1.5 * pad, "20 mm", ha="left", va="center", fontsize=6)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color="k", lw=0.9, label="tag seat (CAD)"),
               Line2D([], [], color="#c0392b", lw=0.8, ls="--", label="30 mm tag"),
               Line2D([], [], color="#e8742a", lw=1.0, label="IMU seat (CAD)"),
               Line2D([], [], color="#c0392b", marker="x", ls="none",
                      label="pivot-calibrated tip")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(fig, "crosscheck")
    plt.close(fig)


# -- tables --------------------------------------------------------------------------------------
def tex_escape(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(name, rows, cols, caption):
    import csv
    import io

    TABLES.mkdir(exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=[c for c, _ in cols], extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    write_text(TABLES / f"{name}.csv", buf.getvalue())
    lines = [f"% {caption} (generated by hardware/mechanical/tools/analyses.py)",
             r"\begin{tabular}{" + "".join(
                 "r" if all(isinstance(r.get(c), (int, float)) or r.get(c) is None for r in rows)
                 else "l" for c, _ in cols) + "}", r"\toprule",
             " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(tex_escape("" if r.get(c) is None else r.get(c))
                                for c, _ in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write_text(TABLES / f"{name}.tex", "\n".join(lines))


def main() -> int:
    man = manifest()
    index = read_json(INDEX, {})
    if not index:
        print("no renders/index.json: run make renders first")
        return 1
    lineage(man, index)
    v5 = v5_variants(man)
    mass = parts_rows(man, index)
    write_table("parts", mass, [("part_title", "part"), ("title", "variant"),
                                ("status", "status"), ("cad", "CAD"),
                                ("size_mm", "size / mm"), ("volume_cm3", "volume / cm³"),
                                ("material", "material"), ("mass_solid_g", "solid / g"),
                                ("mass_printed_g", "printed / g")],
                "Part variants: size, volume, mass (solid and printed estimate)")
    write_table("v5_variants", v5, [("title", "variant"), ("max_mm", "max / mm"),
                                    ("p95_mm", "p95 / mm"), ("moved_area_pct", "moved area / %")],
                "V5 variants vs. the saved CAD state (V5 5.5 mm)")
    xrows, geo = crosscheck(man)
    write_table("crosscheck", xrows, [("body", "body"), ("quantity", "quantity"),
                                      ("config", "config"), ("cad", "CAD"), ("diff", "diff")],
                "CAD geometry vs. optical configuration and pivot calibration")
    write_json(TABLES / "crosscheck.json", {"rows": xrows, "geometry": geo})
    crosscheck_figure(man, geo)
    for r in xrows:
        print(f"  {r['body']:10s} {r['quantity']:42s} config {r.get('config')!s:>8} "
              f"CAD {r.get('cad')!s:>8}")
    print(f"figures -> {MECH.name}/figures, tables -> {MECH.name}/tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
