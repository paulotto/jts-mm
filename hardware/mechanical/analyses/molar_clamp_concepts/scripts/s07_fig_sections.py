# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""2D section through the right first molar for every concept (editable SVG).
->  latex/figures/{svg,pdf,png}/section_c*"""
import warnings; warnings.filterwarnings('ignore')
import pickle, numpy as np
from matplotlib.patches import Circle
from geom import *
from plot2d import fill, section_axes
plt = mpl_setup()
SV_ = TOOTH_MID['M1']
out = pickle.load(open(os.path.join(BUILD, 'concepts.pkl'), 'rb'))
LABEL = {'printed': 'printed, conforming', 'rigid': 'rigid frame (reference)', 'spring': 'printed spring',
         'tpu': 'TPU liner', 'wire': 'stainless wire'}

def crossing_points(pts, s):
    p, t, nrm = frame(s); ss = ((pts[:, :2] - p) * t).sum(1); res = []
    for i in np.where(np.diff(np.sign(ss)) != 0)[0]:
        f = ss[i] / (ss[i] - ss[i + 1]); q = pts[i] + (pts[i + 1] - pts[i]) * f
        res.append(np.array([((q[:2] - p) * nrm).sum(), q[2] - Z0]))
    return res

for nm, r in out.items():
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    fill(ax, section_geom(teeth_mesh(), SV_), fc=C['tooth'], ec=C['tooth_edge'], lw=1.2, zorder=1)
    handles = []
    for k in ('printed', 'rigid', 'spring', 'tpu'):
        if k in r and r[k] is not None:
            fill(ax, section_geom(r[k], SV_), fc=C[k], ec='k', lw=0.5, zorder=3)
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=C[k], ec='k', lw=.5, label=LABEL[k]))
    if 'wires' in r:
        from scipy.interpolate import splprep, splev
        tck, _ = splprep(r['wires'][0].T, s=0); pts = np.array(splev(np.linspace(0, 1, 4000), tck)).T
        for q in crossing_points(pts, SV_): ax.add_patch(Circle(q, 0.35, fc=C['wire'], ec='k', lw=.6, zorder=4))
        ax.text(8.3, 8.4, "arrowheads engage the\nembrasures in front of\nand behind this section", fontsize=8, ha='right', color='0.3')
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=C['wire'], ec='k', lw=.5, label=LABEL['wire']))
    if 'arch' in r:
        for key, a, lab in (('arch', 1.0, 'arch wire, mounted'), ('arch_free', 0.4, 'arch wire, free shape')):
            for q in crossing_points(r[key], SV_):
                if q[0] < 0: ax.add_patch(Circle(q, 0.45, fc=C['wire'] if a == 1 else C['spring'], ec='k', lw=.6, alpha=a, zorder=4))
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=C['wire'] if a == 1 else C['spring'], alpha=a, ec='k', lw=.5, label=lab))
        ax.annotate('', xy=(-5.3, 8.2), xytext=(-8.8, 8.2), arrowprops=dict(arrowstyle='-|>', color=C['red'], lw=1.8))
        ax.text(-9.0, 7.6, "preload", fontsize=8.5, color=C['red'])
    section_axes(ax, xlim=(-13.5, 10.0), ylim=(14.5, 0.8))
    ax.legend(handles=handles, loc='lower right', fontsize=7.5, frameon=True, framealpha=.9)
    save_fig(fig, 'section_' + nm); plt.close(fig); print(nm)
