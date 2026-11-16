# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Small 2D plotting helpers shared by the figure scripts."""
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def fill(ax, geom, **kw):
    """Fill a shapely (Multi)Polygon with holes as one matplotlib patch per polygon."""
    if geom is None or geom.is_empty: return
    for g in ([geom] if geom.geom_type == 'Polygon' else list(geom.geoms)):
        if g.geom_type != 'Polygon': continue
        v, c = [], []
        for ring in [g.exterior] + list(g.interiors):
            r = np.asarray(ring.coords); v += list(r); c += [Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(v, c), **kw))

def section_axes(ax, xlabel="lingual  ←  mm  →  buccal", xlim=(-10.5, 8.5), ylim=(13.5, 0.3)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.grid(alpha=.2)
    ax.set_xlabel(xlabel); ax.set_ylabel("mm below the cusp tips")

# ---------------------------------------------------------------- dimensioning
DIM_COL = '#1a1a1a'
def dim(ax, p1, p2, off, text, fs=7.5, col=DIM_COL, ext=0.6, textoff=0.35, lw=0.6):
    """Linear dimension between p1 and p2, dimension line shifted by `off` along the
    left-hand normal of p1->p2 (negative = other side)."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float); d = p2 - p1; L = np.linalg.norm(d)
    if L < 1e-9: return
    u = d / L; nrm = np.array([-u[1], u[0]]); s = np.sign(off) if off else 1.0
    q1, q2 = p1 + nrm * off, p2 + nrm * off
    for p, q in ((p1, q1), (p2, q2)):
        ax.plot([p[0], q[0] + nrm[0] * ext * s], [p[1], q[1] + nrm[1] * ext * s], color=col, lw=lw * 0.8, zorder=20)
    ax.annotate('', xy=q2, xytext=q1, arrowprops=dict(arrowstyle='<|-|>', color=col, lw=lw, mutation_scale=7,
                                                        shrinkA=0, shrinkB=0), zorder=20)
    ang = np.degrees(np.arctan2(u[1], u[0]))
    if ang > 90: ang -= 180
    if ang < -90: ang += 180
    m = (q1 + q2) / 2 + nrm * textoff * s
    ax.text(m[0], m[1], text, fontsize=fs, color=col, rotation=ang, ha='center', va='center', zorder=21,
            rotation_mode='anchor', bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.85))

def hdim(ax, x1, x2, y, text, **kw):   # horizontal dimension at height y
    dim(ax, (x1, y), (x2, y), 0.0, text, **kw)
def vdim(ax, y1, y2, x, text, **kw):   # vertical dimension at x
    dim(ax, (x, y1), (x, y2), 0.0, text, **kw)

def radius(ax, c, R, ang_deg, text, fs=7.5, col=DIM_COL, inside=True):
    a = np.radians(ang_deg); p = np.array(c) + R * np.array([np.cos(a), np.sin(a)])
    ax.annotate('', xy=p, xytext=c, arrowprops=dict(arrowstyle='-|>', color=col, lw=0.6, mutation_scale=7), zorder=20)
    m = (np.array(c) + p) / 2
    ax.text(m[0], m[1], text, fontsize=fs, color=col, ha='center', va='bottom', zorder=21,
            bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.85))
    ax.plot(*c, marker='+', color=col, ms=6, mew=0.8, zorder=20)

def leader(ax, p, q, text, fs=7.5, col=DIM_COL, ha='left'):
    ax.annotate(text, xy=p, xytext=q, fontsize=fs, color=col, ha=ha, va='center', zorder=21,
                arrowprops=dict(arrowstyle='-|>', color=col, lw=0.6, mutation_scale=7),
                bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.85))

def datum(ax, y, x0, x1, text, col='0.45'):
    ax.plot([x0, x1], [y, y], ls='-.', color=col, lw=0.7, zorder=5)
    ax.text(x1, y, ' ' + text, fontsize=7, color=col, va='center', ha='left')

def drawing_frame(ax, title=None):
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=9.5, loc='left')
