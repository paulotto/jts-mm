# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mechanics figures: jaw above vs below the lingual height of contour, the
incisor section, and the undercut survey bar chart.  Each panel is its own
editable SVG/PDF (text stays text).   ->  latex/figures/{svg,pdf,png}/mech_*"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Polygon as MP
from geom import *
plt = mpl_setup()
SV = jload('survey.json')

from plot2d import fill, section_axes
def axes_style(ax, xlabel="lingual  ←  mm  →  buccal"):
    section_axes(ax, xlabel)

def jaw_panel(s, jaw_h, case, name, tooth_key):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    fill(ax, section_geom(teeth_mesh(), s), fc=C['tooth'], ec=C['tooth_edge'], lw=1.3, zorder=2)
    hs, bu, li = profile(s, hmax=13, dh=0.05)
    L = -li; v = ~np.isnan(L); hh = hs[v]; Ls = gaussian_filter1d(L[v], 3)
    gum = SV[tooth_key]['lingual']['gum_h']
    k = np.argmax(np.where((hh > 3.5) & (hh < gum), Ls, -1e9)); hc = hh[k]; nc = -Ls[k]
    ax.plot([nc, nc], [1.5, gum + 1.2], '--', color='0.35', lw=1.0, zorder=5)
    ax.text(nc - 0.15, 1.55, "path of insertion\n(survey line)", ha='right', va='top', fontsize=8.5, color='0.3')
    ax.plot([nc - 0.3, nc + 0.6], [hc, hc], '-', color='0.3', lw=1.1, zorder=6)
    ax.text(nc + 0.75, hc, "height of contour", ha='left', va='center', fontsize=8.5, color='0.25',
            bbox=dict(fc='white', ec='none', alpha=.7, pad=0.5), zorder=6)
    zz = (hh >= hc) & (hh <= gum)
    ax.fill_betweenx(hh[zz], -Ls[zz], np.full(zz.sum(), nc), color='#9ecae1', alpha=.9, zorder=3, lw=0)
    ud = Ls[k] - Ls[np.argmin(np.abs(hh - (gum - 0.2)))]
    ax.text(-10.2, 13.0, "undercut on this tooth: %.2f mm (blue)" % ud, color=C['blue'], fontsize=9, weight='bold')
    zj = (hh >= max(jaw_h - 1.3, 3.75)) & (hh <= jaw_h + 1.3)
    inner = np.column_stack([-Ls[zj], hh[zj]]); outer = inner.copy(); outer[:, 0] -= 1.5
    ax.add_patch(MP(np.vstack([inner, outer[::-1]]), fc=C['rigid'], ec='k', lw=.7, zorder=4))
    kj = np.argmin(np.abs(hh - jaw_h)); nj = -Ls[kj]
    dn = np.gradient(-Ls, hh)[kj]; nv = np.array([-1.0, dn]); nv /= np.linalg.norm(nv)
    col = C['red'] if case == 'eject' else C['green']
    ax.annotate('', xy=(nj - 1.6, jaw_h), xytext=(nj - 4.4, jaw_h),
                arrowprops=dict(arrowstyle='-|>', color=C['rigid'], lw=2, mutation_scale=16), zorder=7)
    ax.text(nj - 4.5, jaw_h - 0.3, "spring\nforce", ha='right', va='bottom', fontsize=8.5, color=C['rigid'])
    Pt = np.array([nj - 0.75, jaw_h]); R = nv * 3.4
    ax.annotate('', xy=Pt + R, xytext=Pt, arrowprops=dict(arrowstyle='-|>', color=col, lw=2.3, mutation_scale=16), zorder=8)
    ax.annotate('', xy=Pt + np.array([0, R[1]]) * 2.2, xytext=Pt,
                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6, ls='--', mutation_scale=13), zorder=8)
    lab = ("tooth pushes the jaw OCCLUSALLY\n→ the part is ejected" if R[1] < 0 else
           "tooth pushes the jaw GINGIVALLY\n→ the part is pulled onto the teeth")
    ax.text(-10.2, 10.6, lab, fontsize=9.5, color=col, weight='bold', va='top')
    axes_style(ax); save_fig(fig, name); plt.close(fig)

jaw_panel(-TOOTH_MID['M1'], 4.5, 'eject', 'mech_eject', 'LM1')
jaw_panel(-TOOTH_MID['M1'], 7.8, 'seat', 'mech_seat', 'LM1')

fig, ax = plt.subplots(figsize=(5.2, 4.4))
fill(ax, section_geom(teeth_mesh(), TOOTH_MID['I1']), fc=C['tooth'], ec=C['tooth_edge'], lw=1.3)
ax.text(-10.2, 12.9, "No lingual undercut: a lingual jaw always sits on a\nface that converges toward the incisal edge, so a\nspring there pushes the part off.",
        fontsize=8.5, color=C['red'], va='bottom')
axes_style(ax, "lingual  ←  mm  →  labial"); save_fig(fig, 'mech_incisor'); plt.close(fig)

fig, ax = plt.subplots(figsize=(10.5, 3.6))
order = ['M2', 'M1', 'PM2', 'PM1', 'C', 'I2', 'I1']; labels, bu, li = [], [], []
for t in order: labels.append('L-' + t); bu.append(SV['L' + t]['buccal']['undercut']); li.append(SV['L' + t]['lingual']['undercut'])
for t in order[::-1]: labels.append('R-' + t); bu.append(SV['R' + t]['buccal']['undercut']); li.append(SV['R' + t]['lingual']['undercut'])
x = np.arange(len(labels))
ax.bar(x - 0.2, li, 0.4, color=C['green'], label='lingual undercut')
ax.bar(x + 0.2, bu, 0.4, color='#e08214', label='buccal / labial undercut')
for y, t in ((0.25, '0.25 mm  typical for a stiff cast Co-Cr clasp'), (0.5, '0.5 mm  typical for wrought wire / flexible polymer')):
    ax.axhline(y, color='0.4', ls=':', lw=1); ax.text(4.5, y + 0.015, t, ha='center', va='bottom', fontsize=8.5, color='0.3')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('undercut depth [mm]')
ax.legend(loc='upper center', ncol=2, frameon=False); ax.grid(axis='y', alpha=.25); ax.set_ylim(0, 1.35)
ax.axvline(6.5, color='0.6', lw=0.8)
save_fig(fig, 'mech_undercut_bars'); plt.close(fig)
print('mechanics figures written')
