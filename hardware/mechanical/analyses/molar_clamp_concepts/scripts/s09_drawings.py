# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Dimensioned drawings for translating the concepts into FreeCAD.

Plan views: occlusal view in the Teeth_02 frame shifted to the midline
(X = x - MIDLINE_X, Y = y), anterior at the top.  Sections: perpendicular to the
arch at the first molar, n to the right = buccal, h downward from the lowest cusp
tip.  All part dimensions come from params.py; reference dimensions of the teeth
are measured on the model.   ->  latex/figures/*/drw_*, latex/generated/ref_dims.tex"""
import warnings; warnings.filterwarnings('ignore')
import pickle, numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Polygon as MP, Circle, Rectangle, Arc
from trimesh.path.polygons import projected
from geom import *
from plot2d import fill, dim, hdim, vdim, radius, leader, datum, drawing_frame
from params import COMMON, C1, C2, C3, C4, C5
plt = mpl_setup()
out = pickle.load(open(os.path.join(BUILD, 'concepts.pkl'), 'rb'))
SV = jload('survey.json'); SPR = jload('springs.json')
S_M1 = TOOTH_MID['M1']; EM, ED = EMBRASURES_R['PM2/M1'], EMBRASURES_R['M1/M2']
GAP = COMMON['gap']

# ---------------------------------------------------------------- tooth profile helpers
_prof = {}
def prof(s):
    if s not in _prof:
        hs, bu, li = profile(s, hmax=14, dh=0.05)
        ok_b, ok_l = ~np.isnan(bu), ~np.isnan(li)
        _prof[s] = (hs, lambda h, hs=hs[ok_b], b=gaussian_filter1d(bu[ok_b], 2): np.interp(h, hs, b),
                        lambda h, hs=hs[ok_l], l=gaussian_filter1d(li[ok_l], 2): np.interp(h, hs, l))
    return _prof[s]
def band(s, h0, h1, t, side, gap=GAP, n=60):
    """Polygon of a layer of thickness t on the lingual (side=-1) or buccal (+1) face between h0..h1."""
    _, nb, nl = prof(s); f = nl if side < 0 else nb; h = np.linspace(h0, h1, n)
    inner = f(h) + side * gap; outer = inner + side * t
    return np.vstack([np.column_stack([inner, h]), np.column_stack([outer, h])[::-1]])
def tooth(ax, s, **kw):
    fill(ax, section_geom(teeth_mesh(), s), fc=C['tooth'], ec=C['tooth_edge'], lw=0.9, zorder=1, **kw)
def poly(ax, P, key, **kw):
    ax.add_patch(MP(P, closed=True, fc=C[key], ec='k', lw=0.6, zorder=4, **kw))
def section_frame(ax, s, xlim=(-14, 10.5), ylim=(15.5, -0.5)):
    tooth(ax, s); ax.set_xlim(*xlim); ax.set_ylim(*ylim); drawing_frame(ax)
    datum(ax, 0.0, xlim[0], xlim[1] - 3.2, 'h = 0 (lowest cusp tip)')

# ---------------------------------------------------------------- reference dimensions
def pt(s, h, side, off=0.0):
    F_ = None
    _, nb, nl = prof(s); n = (nb(h) if side > 0 else nl(h)) + side * off
    return arch2xyz([s], [n], [h])[0]
REF = {}
REF['M1_lingual_width'] = np.linalg.norm(pt(S_M1, C2['contact_h'], -1)[:2] - pt(-S_M1, C2['contact_h'], -1)[:2])
REF['M1_buccal_width'] = np.linalg.norm(pt(S_M1, 6.0, 1)[:2] - pt(-S_M1, 6.0, 1)[:2])
REF['M2_lingual_width'] = np.linalg.norm(pt(48.0, 7.5, -1)[:2] - pt(-48.0, 7.5, -1)[:2])
REF['C_labial_width'] = np.linalg.norm(pt(TOOTH_MID['C'], 6.0, 1)[:2] - pt(-TOOTH_MID['C'], 6.0, 1)[:2])
pm, pl = arch2xyz([EM, EMBRASURES_L['PM2/M1']], [0, 0], [6, 6])[:, :2]; pi = pt(0.0, 7.0, 1)[:2]
a, b = pl - pm, pi - pm                     # 2-D cross product (np.cross needs 3-D in NumPy 2)
REF['arch_depth'] = abs(a[0] * b[1] - a[1] * b[0]) / np.linalg.norm(pl - pm)
REF['M1_length_R'] = ED - EM; REF['M1_length_L'] = EMBRASURES_L['PM2/M1'] - EMBRASURES_L['M1/M2']
REF['M1_crown_width'] = prof(S_M1)[1](6.5) - prof(S_M1)[2](6.5)
jdump({k: float(v) for k, v in REF.items()}, 'ref_dims.json')
with open(os.path.join(GEN, 'ref_dims.tex'), 'w') as f:
    rows = [('M1 lingual width at $h=%.1f$ (lingual jaw contact)' % C2['contact_h'], REF['M1_lingual_width']),
            ('M1 buccal width at $h=6$', REF['M1_buccal_width']), ('M2 lingual width at $h=7.5$', REF['M2_lingual_width']),
            ('canine labial width at $h=6$', REF['C_labial_width']), ('arch depth (incisors to M1 mesial line)', REF['arch_depth']),
            ('M1 mesio-distal length, right / left', None), ('M1 bucco-lingual crown width at $h=6.5$', REF['M1_crown_width'])]
    for lab, v in rows:
        if v is None: f.write('%s & \\num{%.1f} / \\num{%.1f} \\\\\n' % (lab, REF['M1_length_R'], REF['M1_length_L']))
        else: f.write('%s & \\num{%.1f} \\\\\n' % (lab, v))

# ---------------------------------------------------------------- plan views
def XY(p): p = np.atleast_2d(p); return np.column_stack([p[:, 0] - MIDLINE_X, p[:, 1]])
TEETH_PLAN = []
for hh in (3.0, 6.5):
    sec = teeth_mesh().section(plane_origin=[0, 0, Z0 + hh], plane_normal=[0, 0, 1])
    TEETH_PLAN.append([XY(sec.vertices[e.points][:, :2]) for e in sec.entities])
def plan_frame(ax):
    for P in TEETH_PLAN[1]: ax.plot(P[:, 0], P[:, 1], color=C['tooth_edge'], lw=0.6, zorder=1)
    ax.set_xlim(-40, 40); ax.set_ylim(182, 118); drawing_frame(ax)
def footprint(mesh, res=0.15):
    """Top-view outline of a mesh: rasterise its (dense) vertices, close gaps, contour."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    from skimage.measure import find_contours
    v = np.asarray(mesh.vertices)[:, :2]; lo = v.min(0) - 1.0; hi = v.max(0) + 1.0
    nx_, ny_ = (np.ceil((hi - lo) / res).astype(int))
    img = np.zeros((nx_, ny_), bool); ij = ((v - lo) / res).astype(int); img[ij[:, 0], ij[:, 1]] = True
    img = binary_closing(img, iterations=3)
    from shapely.geometry import Polygon
    g = None
    for c in find_contours(np.pad(img, 1).astype(float), 0.5):
        if len(c) < 12: continue
        pg = Polygon((c - 1) * res + lo).buffer(0)
        if pg.area < 0.3: continue
        g = pg if g is None else g.symmetric_difference(pg)      # even-odd: keeps the holes open
    return g
def plan_parts(ax, r):
    for k in ('printed', 'rigid', 'spring', 'tpu'):
        if k in r and r[k] is not None:
            g = footprint(r[k])
            if g is None: continue
            from shapely import affinity
            fill(ax, affinity.translate(g, -MIDLINE_X, 0), fc=C[k], ec='k', lw=0.4, zorder=3 if k != 'tpu' else 3.5, alpha=0.95)
    for w in r.get('wires', []): P = XY(w); ax.plot(P[:, 0], P[:, 1], color=C['wire'], lw=1.6, zorder=4)
    if 'arch' in r:
        P = XY(r['arch']); ax.plot(P[:, 0], P[:, 1], color=C['wire'], lw=1.6, zorder=4)
        P = XY(r['arch_free']); ax.plot(P[:, 0], P[:, 1], color=C['spring'], lw=1.0, ls='--', zorder=4)
def station_ticks(ax, stations, side=1, h=6.5):
    for lab, s in stations.items():
        a = XY(arch2xyz([s], [8.5], [h])[0]); b = XY(arch2xyz([s], [10.5], [h])[0])
        ax.plot([a[0, 0], b[0, 0]], [a[0, 1], b[0, 1]], color='0.4', lw=0.6)
        ax.text(b[0, 0], b[0, 1], ' %s\n s=%.1f' % (lab, s), fontsize=6, color='0.35', ha='left' if s > 0 else 'right', va='center')

# D0: reference plan
fig, ax = plt.subplots(figsize=(6.4, 5.6)); plan_frame(ax)
cl = XY(arch2xyz(SF[::20], np.zeros(len(SF[::20])), np.full(len(SF[::20]), 6.0)))
ax.plot(cl[:, 0], cl[:, 1], ls='-.', color='0.5', lw=0.7); ax.text(cl[len(cl) // 2, 0], cl[len(cl) // 2, 1] + 5, 'arch centreline', fontsize=6.5, color='0.4', ha='center')
station_ticks(ax, EMBRASURES_R); station_ticks(ax, EMBRASURES_L)
a, b = XY(pt(S_M1, C2['contact_h'], -1))[0], XY(pt(-S_M1, C2['contact_h'], -1))[0]
dim(ax, a, b, 0.0, 'M1 lingual %.1f' % REF['M1_lingual_width'])
a, b = XY(pt(S_M1, 6.0, 1))[0], XY(pt(-S_M1, 6.0, 1))[0]; dim(ax, a, b, 6.5, 'M1 buccal %.1f' % REF['M1_buccal_width'])
a, b = XY(pt(TOOTH_MID['C'], 6.0, 1))[0], XY(pt(-TOOTH_MID['C'], 6.0, 1))[0]; dim(ax, a, b, 7.5, 'canines %.1f' % REF['C_labial_width'])
mid = (XY(pm)[0] + XY(pl)[0]) / 2; dim(ax, XY(pi)[0], np.array([XY(pi)[0][0], mid[1]]), -26, 'arch depth %.1f' % REF['arch_depth'])
ax.plot([XY(pm)[0][0], XY(pl)[0][0]], [XY(pm)[0][1], XY(pl)[0][1]], ls=':', color='0.5', lw=0.6)
save_fig(fig, 'drw_ref_plan'); plt.close(fig)

# D1..D5: plan views of the concepts
for nm, r in out.items():
    fig, ax = plt.subplots(figsize=(6.4, 5.6)); plan_frame(ax); plan_parts(ax, r)
    station_ticks(ax, {'PM2/M1': EM, 'M1/M2': ED}); station_ticks(ax, {'PM2/M1': EMBRASURES_L['PM2/M1'], 'M1/M2': EMBRASURES_L['M1/M2']})
    if nm == 'c4':
        a, b = XY(pt(S_M1, C2['contact_h'], -1))[0], XY(pt(-S_M1, C2['contact_h'], -1))[0]
        dim(ax, a, b, 0.0, 'pads mounted %.1f' % REF['M1_lingual_width'])
        dim(ax, a, b, -4.0, 'pads free %.1f' % (REF['M1_lingual_width'] + 2 * C4['pre_spread']))
    elif nm in ('c2', 'c5'):
        a, b = XY(pt(S_M1, C2['contact_h'], -1))[0], XY(pt(-S_M1, C2['contact_h'], -1))[0]
        dim(ax, a, b, 0.0, 'jaw contact faces %.1f' % REF['M1_lingual_width'])
    a, b = XY(arch2xyz([COMMON['labial_bar']['s_half']], [prof(14.7)[1](7.2) + 1.0], [7.2])[0])[0], XY(arch2xyz([-COMMON['labial_bar']['s_half']], [prof(-14.7)[1](7.2) + 1.0], [7.2])[0])[0]
    dim(ax, a, b, 5.5, 'labial bar s = ±%.1f' % COMMON['labial_bar']['s_half'])
    save_fig(fig, 'drw_%s_plan' % nm); plt.close(fig)

# ---------------------------------------------------------------- sections with dimensions
S = S_M1; hs_, nb, nl = prof(S)
def n_l(h, off=0.0): return float(nl(h)) - GAP - off
def n_b(h, off=0.0): return float(nb(h)) + GAP + off

# C1
fig, ax = plt.subplots(figsize=(5.6, 4.4)); section_frame(ax, S)
top_s = float(top(S)); h0 = top_s + C1['flange_below_top']
poly(ax, band(S, h0, C1['flange_h_max'], C1['flange_t'], -1), 'printed')
poly(ax, band(S, *C1['buccal_arm_h'], C1['buccal_arm_t'], +1), 'printed')
hdim(ax, n_l(8.5, C1['flange_t']), n_l(8.5), 8.5, 't %.1f' % C1['flange_t'])
vdim(ax, 0, h0, -12.5, '%.1f' % h0); vdim(ax, 0, C1['flange_h_max'], -13.6, '%.1f' % C1['flange_h_max'])
hdim(ax, n_b(6.0), n_b(6.0, C1['buccal_arm_t']), 6.0, 't %.1f' % C1['buccal_arm_t'])
vdim(ax, 0, C1['buccal_arm_h'][0], 9.3, '%.1f' % C1['buccal_arm_h'][0]); vdim(ax, 0, C1['buccal_arm_h'][1], 10.3, '%.1f' % C1['buccal_arm_h'][1])
leader(ax, (n_l(9.0) - 0.4, 9.0), (-12.8, 12.8), 'engages the lingual undercut;\nblock out to 0.3–0.5 mm')
save_fig(fig, 'drw_c1_section'); plt.close(fig)

# C2
g = dict(nc=None)
lj, bj, lp = C2['lingual_jaw'], C2['buccal_jaw'], C2['loop']
n_surf = float(nl(C2['contact_h'])); n_back = n_surf - GAP - lj['t']; nc = n_back - lp['R'] + lp['t'] / 2; hc = lp['h_centre']
fig, ax = plt.subplots(figsize=(6.4, 4.6)); section_frame(ax, S, xlim=(-18, 10.5))
poly(ax, band(S, *lj['h'], lj['t'], -1), 'spring')
th = np.radians(np.linspace(lp['arc'][0], lp['arc'][1], 120))
ring = np.vstack([np.column_stack([nc + (lp['R'] + lp['t'] / 2) * np.cos(th), hc + (lp['R'] + lp['t'] / 2) * np.sin(th)]),
                  np.column_stack([nc + (lp['R'] - lp['t'] / 2) * np.cos(th), hc + (lp['R'] - lp['t'] / 2) * np.sin(th)])[::-1]])
poly(ax, ring, 'spring')
foot_top = hc + lp['R'] - lp['t'] / 2
poly(ax, np.array([[nc - lp['t'] / 2 - 0.4, foot_top], [n_l(11.5) - 0.1, foot_top], [n_l(11.5) - 0.1, C2['foot']['h'][1]], [nc - lp['t'] / 2 - 0.4, C2['foot']['h'][1]]]), 'rigid')
lb = COMMON['lingual_bar']; poly(ax, band(S, *lb['h'], lb['t'], -1), 'rigid')
poly(ax, band(S, *bj['h'], bj['t'], +1), 'rigid')
radius(ax, (nc, hc), lp['R'], 200, 'R %.1f' % lp['R'])
hdim(ax, nc - lp['R'] - lp['t'] / 2, nc - lp['R'] + lp['t'] / 2, hc + 1.6, 't %.1f' % lp['t'], textoff=0.5)
hdim(ax, nc, n_surf, hc - lp['R'] - 1.2, 'centre to tooth %.2f' % (n_surf - nc))
vdim(ax, 0, hc, -15.2, 'centre %.1f' % hc)
vdim(ax, 0, lj['h'][0], -16.3, '%.1f' % lj['h'][0]); vdim(ax, 0, lj['h'][1], -17.4, 'jaw %.1f' % lj['h'][1])
vdim(ax, 0, bj['h'][1], 9.6, '%.1f' % bj['h'][1])
hdim(ax, n_back, n_surf - GAP, lj['h'][1] + 0.8, 't %.1f' % lj['t'])
leader(ax, (nc, hc + lp['R']), (-17.5, 14.4), 'root: fillet r ≥ 0.5, rigid foot to the lingual bar')
leader(ax, (n_surf - GAP, C2['contact_h']), (-3.0, 12.6), 'contact line P at h = %.1f\n(loop centre at the same height, e = 0)' % C2['contact_h'])
hdim(ax, n_b(6.0), n_b(6.0, bj['t']), 6.0, 't %.1f' % bj['t'])
save_fig(fig, 'drw_c2_section'); plt.close(fig)

# C2 detail: loop module
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax = axs[0]; drawing_frame(ax, 'loop module, side view (section plane)')
R, t = lp['R'], lp['t']; c = np.array([0.0, 0.0])
for rr, lw in ((R + t / 2, 1.0), (R - t / 2, 1.0)):
    ax.add_patch(Arc(c, 2 * rr, 2 * rr, theta1=lp['arc'][0], theta2=lp['arc'][1], color='k', lw=lw))
th = np.radians(np.linspace(lp['arc'][0], lp['arc'][1], 150))
ax.fill(np.r_[(R + t / 2) * np.cos(th), (R - t / 2) * np.cos(th[::-1])], np.r_[(R + t / 2) * np.sin(th), (R - t / 2) * np.sin(th[::-1])], color=C['spring'], alpha=.9)
ax.add_patch(Rectangle((R - t / 2, -lj['h'][1] + hc), lj['t'], lj['h'][1] - lj['h'][0], fc=C['spring'], ec='k', lw=.6))
ax.add_patch(Rectangle((-R - t / 2 - 0.4, R - t / 2), 2 * R + 0.8, 1.8, fc=C['rigid'], ec='k', lw=.6))
radius(ax, c, R, 215, 'R %.1f' % R)
hdim(ax, R - t / 2, R + t / 2, -R - 0.9, 't %.1f' % t)
ax.annotate('', xy=(R + t / 2 + lj['t'] + 1.3, 0), xytext=(R + t / 2 + lj['t'] + 0.1, 0), arrowprops=dict(arrowstyle='<|-', color=C['red'], lw=1.4))
ax.text(R + lj['t'] + 1.6, 0.45, 'P (tooth)', color=C['red'], fontsize=7.5)
ax.plot([-R - 1.5, R + 3.5], [0, 0], ls='-.', color='0.5', lw=0.6); ax.text(-R - 1.5, -0.25, 'contact height = loop centre', fontsize=6.5, color='0.4')
ax.text(-R - 0.4, R + 1.2, 'rigid foot', fontsize=7, color='white', va='center')
ax.text(0, -R - 1.9, 'arc from −270° (root) to 0° (jaw), %d° swept' % (lp['arc'][1] - lp['arc'][0]), fontsize=7, ha='center')
ax.set_xlim(-R - 2.5, R + 5); ax.set_ylim(R + 2.6, -R - 2.4)
ax = axs[1]; drawing_frame(ax, 'loop module, view from lingual')
w = lp['w']; jl = lj['s'][1] - lj['s'][0]
ax.add_patch(Rectangle((-jl / 2, lj['h'][0]), jl, lj['h'][1] - lj['h'][0], fc=C['spring'], ec='k', lw=.6))
ax.add_patch(Rectangle((-w / 2, hc - R - t / 2), w, 2 * R + t, fc=C['spring'], ec='k', lw=.6, alpha=.6))
ax.add_patch(Rectangle((-w / 2 - 0.4, hc + R - t / 2), w + 0.8, 1.8, fc=C['rigid'], ec='k', lw=.6))
hdim(ax, -w / 2, w / 2, hc - R - t / 2 - 0.8, 'w %.1f' % w); hdim(ax, -jl / 2, jl / 2, lj['h'][1] + 0.9 + 1.8, 'jaw %.1f' % jl)
ax.set_xlim(-jl / 2 - 1.5, jl / 2 + 1.5); ax.set_ylim(hc + R + 4.5, hc - R - 2.5)
ax.text(0, hc + R + 3.8, 'print over-closed by %.1f mm; t = 2 × 0.4 mm lines' % lp['over_closure'], fontsize=7, ha='center')
fig.tight_layout(); save_fig(fig, 'drw_c2_loop'); plt.close(fig)

# C3 section + Adams elevation
fig, ax = plt.subplots(figsize=(5.6, 4.4)); section_frame(ax, S)
poly(ax, band(S, top_s + C1["flange_below_top"], C3['plate_h_max'], C3['plate_t'], -1), 'rigid')
nbr = n_b(C3['bridge_h'], C3['bridge_offset'] + C3['wire_d'] / 2)
ax.add_patch(Circle((nbr, C3['bridge_h']), C3['wire_d'] / 2, fc=C['wire'], ec='k', lw=.6, zorder=5))
hdim(ax, n_b(C3['bridge_h']), nbr - C3['wire_d'] / 2, C3['bridge_h'] - 1.0, '%.1f' % C3['bridge_offset'])
vdim(ax, 0, C3['bridge_h'], 9.5, '%.1f' % C3['bridge_h']); vdim(ax, 0, C3['plate_h_max'], -13.6, '%.1f' % C3['plate_h_max'])
hdim(ax, n_l(8.5, C3['plate_t']), n_l(8.5), 8.5, 't %.1f' % C3['plate_t'])
leader(ax, (nbr, C3['bridge_h']), (3.0, 13.5), 'Ø%.1f bridge, parallel to the buccal face' % C3['wire_d'])
save_fig(fig, 'drw_c3_section'); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 3.8)); drawing_frame(ax, 'Adams clasp, schematic buccal elevation (arc length s along the arch, height h)')
ta, tdd = float(top(EM)) - 0.45, float(top(ED)) - 0.45; a_, b_ = EM + 1.0, ED - 1.0; ah = C3['arrowhead_h']; bh = C3['bridge_h']
ax.axhspan(ah + 0.8, 11.8, color=C['tooth'], alpha=.35, lw=0); ax.text(EM - 2.3, 11.3, 'gingiva', fontsize=6.5, color='0.4')
def arrowhead(sc):   # loop pointing gingivally, tip at h = ah
    return [(sc - 0.6, bh + 1.4), (sc - 0.5, ah - 0.3), (sc, ah), (sc + 0.5, ah - 0.3), (sc + 0.6, bh + 1.4)]
P = [(EM, ta), (EM + 0.2, bh + 0.4)] + arrowhead(a_) + [(a_ + 1.4, bh), (b_ - 1.4, bh)] + arrowhead(b_) + [(ED - 0.2, bh + 0.4), (ED, tdd)]
P = np.array(P); ax.plot(P[:, 0], P[:, 1], color=C['wire'], lw=2.6, solid_joinstyle='round', solid_capstyle='round')
for se, lab, hh in ((EM, 'PM2/M1', ta), (ED, 'M1/M2', tdd)):
    ax.axvline(se, color='0.6', lw=0.6, ls=':'); ax.text(se, 0.6, lab, fontsize=7, ha='center', color='0.4')
    ax.plot(se, hh, 'o', ms=4, color=C['wire'])
hdim(ax, a_, b_, 10.9, 'arrowhead spacing %.1f (along the arch)' % (b_ - a_))
hdim(ax, EM, a_, 10.9, '1.0'); hdim(ax, b_, ED, 10.9, '1.0')
vdim(ax, 0, bh, EM - 1.3, 'bridge %.1f' % bh); vdim(ax, 0, ah, ED + 1.3, 'tip %.1f' % ah)
vdim(ax, 0, ta, EM - 2.3, 'crossing %.1f' % ta)
leader(ax, (b_ + 0.1, ah), ((EM + ED) / 2, 7.6), 'arrowheads in the embrasure undercut,\n0.5–1 mm below the contact point', ha='center')
leader(ax, (EM, ta), (EM + 1.6, 1.6), 'crossing on the marginal ridge → lingual tag in the plate')
ax.set_xlim(EM - 3.0, ED + 3.0); ax.set_ylim(12.0, 0.0); datum(ax, 0.0, EM - 3.0, ED - 1.0, 'h = 0')
save_fig(fig, 'drw_c3_adams'); plt.close(fig)

# C4 section
fig, ax = plt.subplots(figsize=(5.6, 4.4)); section_frame(ax, S)
poly(ax, band(S, *C4['pad_h'], C4['pad_t'], -1), 'printed')
nw = n_l(C4['wire_h'], C4['wire_offset'] + C4['wire_d'] / 2)
ax.add_patch(Circle((nw, C4['wire_h']), C4['wire_d'] / 2, fc=C['wire'], ec='k', lw=.6, zorder=5))
ax.add_patch(Circle((nw + C4['pre_spread'], C4['wire_h']), C4['wire_d'] / 2, fc=C['spring'], ec='k', lw=.6, alpha=.4, zorder=5))
hdim(ax, n_l(8.0, C4['pad_t']), n_l(8.0), 8.0, 't %.1f' % C4['pad_t'])
vdim(ax, 0, C4['pad_h'][0], -12.5, '%.1f' % C4['pad_h'][0]); vdim(ax, 0, C4['pad_h'][1], -13.6, '%.1f' % C4['pad_h'][1])
vdim(ax, 0, C4['wire_h'], 1.0, '%.1f' % C4['wire_h'])
leader(ax, (nw, C4['wire_h']), (-12.5, 13.8), 'Ø%.1f wire in a channel of the pad;\nfree position %.1f mm buccal (ghost)' % (C4['wire_d'], C4['pre_spread']))
save_fig(fig, 'drw_c4_section'); plt.close(fig)

# C5 section + insert
fig, ax = plt.subplots(figsize=(5.6, 4.4)); section_frame(ax, S)
poly(ax, band(S, *C5['liner_h'], C5['liner_t'], -1), 'tpu')
poly(ax, band(S, *C5['jaw_h'], C5['jaw_t'], -1, gap=GAP + C5['liner_t']), 'rigid')
poly(ax, band(S, *C5['buccal_jaw_h'], C5['buccal_jaw_t'], +1), 'rigid')
hdim(ax, n_l(8.0, C5['liner_t']), n_l(8.0), 8.0, '%.1f' % C5['liner_t'], textoff=-0.5)
hdim(ax, n_l(6.0, C5['liner_t'] + C5['jaw_t']), n_l(6.0, C5['liner_t']), 6.0, '%.1f' % C5['jaw_t'])
vdim(ax, 0, C5['jaw_h'][0], -13.0, '%.1f' % C5['jaw_h'][0]); vdim(ax, 0, C5['jaw_h'][1], -14.0, '%.1f' % C5['jaw_h'][1])
hdim(ax, n_b(6.0), n_b(6.0, C5['buccal_jaw_t']), 6.0, 't %.1f' % C5['buccal_jaw_t'])
vdim(ax, 0, C5['buccal_jaw_h'][1], 10.0, '%.1f' % C5['buccal_jaw_h'][1])
save_fig(fig, 'drw_c5_section'); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4))
ax = axs[0]; drawing_frame(ax, 'TPU insert, view from the tooth side')
L5 = C5['band_s'][1] - C5['band_s'][0]; H5 = C5['liner_h'][1] - C5['liner_h'][0]; nr = C5['n_ribs']
ax.add_patch(Rectangle((0, 0), L5, H5, fc=C['tpu'], ec='k', lw=.6, alpha=.35))
rw = C5['rib_fraction'] * L5 / nr
for i in range(nr):
    x0 = (i + 0.5) * L5 / nr - rw / 2; ax.add_patch(Rectangle((x0, 0), rw, H5, fc=C['tpu'], ec='k', lw=.6))
hdim(ax, 0, L5, H5 + 0.8, '%.1f' % L5); vdim(ax, 0, H5, -0.9, '%.1f' % H5)
hdim(ax, 0.5 * L5 / nr - rw / 2, 0.5 * L5 / nr + rw / 2, -0.8, 'rib %.1f' % rw)
ax.set_xlim(-2.5, L5 + 1.5); ax.set_ylim(H5 + 2.2, -2.0)
ax = axs[1]; drawing_frame(ax, 'section through insert and jaw (dovetail)')
tb, tr, dd = 1.0, C5['liner_t'] - 1.0, 1.0
ax.add_patch(MP([[0, 0], [tb, 0], [tb, H5], [0, H5]], fc=C['tpu'], ec='k', lw=.6))          # base
ax.add_patch(MP([[tb, 0.8], [tb + tr, 0.8], [tb + tr, H5 - 0.8], [tb, H5 - 0.8]], fc=C['tpu'], ec='k', lw=.6))  # rib
ax.add_patch(MP([[-C5['jaw_t'], -0.4], [0, -0.4], [0, 1.2], [-0.35, 1.5], [-0.35, H5 - 1.5], [0, H5 - 1.2], [0, H5 + 0.4], [-C5['jaw_t'], H5 + 0.4]],
                fc=C['rigid'], ec='k', lw=.6))
hdim(ax, tb, tb + tr, H5 + 1.1, 'rib %.1f' % tr); hdim(ax, 0, tb, H5 + 1.1 + 1.0, 'base %.1f' % tb)
hdim(ax, -C5['jaw_t'], 0, -1.2, 'jaw %.1f' % C5['jaw_t'])
leader(ax, (-0.2, H5 / 2), (-4.2, H5 / 2), 'dovetail 0.35 deep,\n60°, full length')
leader(ax, (tb + tr, H5 / 2), (tb + tr + 1.0, H5 / 2 - 1.2), 'interference\n%.2f mm' % C5['interference'])
ax.set_xlim(-6, 5.5); ax.set_ylim(H5 + 3.0, -2.2)
fig.tight_layout(); save_fig(fig, 'drw_c5_insert'); plt.close(fig)

# common: labial bar and marker stem
mk = COMMON['marker']; lb2 = COMMON['labial_bar']
fig, ax = plt.subplots(figsize=(6.4, 3.4)); drawing_frame(ax, 'labial bar and marker stem, section at the midline (s = 0)')
S0 = 0.0; tooth(ax, S0)
poly(ax, band(S0, *lb2['h'], lb2['t'], +1), 'rigid')
nb0 = float(prof(S0)[1](mk['h'])) + GAP + lb2['t']
ax.add_patch(Rectangle((nb0, mk['h'] - mk['stem_d'] / 2), mk['stem_len'], mk['stem_d'], fc='#666666', ec='k', lw=.6))
ax.add_patch(Rectangle((nb0 + mk['stem_len'], mk['h'] - mk['plate'][1] / 2), mk['plate'][2], mk['plate'][1], fc='#222222', ec='k', lw=.6))
hdim(ax, nb0, nb0 + mk['stem_len'], mk['h'] + mk['stem_d'] / 2 + 1.0, 'stem %.0f' % mk['stem_len'])
vdim(ax, mk['h'] - mk['stem_d'] / 2, mk['h'] + mk['stem_d'] / 2, nb0 + 3, 'Ø%.1f' % mk['stem_d'])
vdim(ax, mk['h'] - mk['plate'][1] / 2, mk['h'] + mk['plate'][1] / 2, nb0 + mk['stem_len'] + mk['plate'][2] + 1.0, 'marker %.0f' % mk['plate'][1])
vdim(ax, lb2['h'][0], lb2['h'][1], -6.0, 'bar %.1f–%.1f' % lb2['h']); vdim(ax, 0, mk['h'], -8.0, 'h %.1f' % mk['h'])
hdim(ax, float(prof(S0)[1](9.0)) + GAP, float(prof(S0)[1](9.0)) + GAP + lb2['t'], 10.2, 't %.1f' % lb2['t'])
ax.set_xlim(-10, 28); ax.set_ylim(16.5, -2.5); datum(ax, 0, -10, 18, 'h = 0')
save_fig(fig, 'drw_common_marker'); plt.close(fig)
print('drawings written; reference dims:', {k: round(float(v), 2) for k, v in REF.items()})

# ---------------------------------------------------------------- parameter tables from params.py
DESC = {'t': 'thickness', 'h': 'height range $h$ (from / to)', 's': 'station range $s$ (from / to)', 'w': 'width along the arch',
        'R': 'loop radius (to wall centre)', 'arc': 'arc angles, root / jaw [deg]', 'n_series': 'loops in series',
        'over_closure': 'printed over-closure', 'h_centre': 'loop centre height', 'contact_h': 'contact height of the tooth',
        's_half': 'half span $\\pm s$', 'flange_t': 'flange thickness', 'flange_below_top': 'flange starts below the cusp line',
        'flange_h_max': 'flange ends at $h$', 'buccal_arm_t': 'buccal arm thickness', 'buccal_arm_h': 'buccal arm $h$ range',
        'buccal_arm_s': 'buccal arm $s$ ranges (M1, M2)', 'plate_t': 'plate thickness', 'plate_s_half': 'plate half span',
        'plate_h_max': 'plate ends at $h$', 'wire_d': 'wire diameter', 'bridge_offset': 'bridge distance from the tooth',
        'bridge_h': 'bridge height', 'arrowhead_h': 'arrowhead tip height', 'tag_h': 'tag end height', 'pre_spread': 'pre-spread per side',
        'wire_h': 'wire height', 'wire_s_half': 'wire half span', 'wire_offset': 'wire distance from the tooth', 'pad_t': 'pad thickness',
        'pad_M1_s': 'M1 pad $s$ range', 'pad_M2_s': 'M2 pad $s$ range', 'pad_h': 'pad $h$ range', 'yoke_t': 'yoke thickness',
        'yoke_s_half': 'yoke half span', 'yoke_h': 'yoke $h$ range', 'liner_t': 'TPU liner thickness', 'liner_h': 'liner $h$ range',
        'jaw_t': 'rigid jaw thickness', 'jaw_h': 'jaw $h$ range', 'band_s': 'clamp $s$ range', 'buccal_jaw_t': 'buccal jaw thickness',
        'buccal_jaw_h': 'buccal jaw $h$ range', 'rib_fraction': 'rib area fraction', 'n_ribs': 'number of ribs',
        'interference': 'interference', 'gap': 'model offset from the teeth', 'relief': 'print relief',
        'length': 'length', 'n_half': 'half width across the tooth', 'stem_d': 'stem diameter', 'stem_len': 'stem length',
        'plate': 'marker plate (w, h, t)'}
def fmt(v):
    if isinstance(v, (tuple, list)):
        if v and isinstance(v[0], (tuple, list)): return '; '.join(fmt(x) for x in v)
        return ' / '.join('%g' % x for x in v)
    return '%g' % v
def rows_of(d, prefix=''):
    out = []
    for k, v in d.items():
        if isinstance(v, dict): out += rows_of(v, prefix + k.replace('_', ' ') + ': ')
        else: out.append((prefix + DESC.get(k, k.replace('_', ' ')), '\\texttt{%s}' % (k.replace('_', '\\_')), fmt(v)))
    return out
for name, d in (('common', COMMON), ('c1', C1), ('c2', C2), ('c3', C3), ('c4', C4), ('c5', C5)):
    with open(os.path.join(GEN, 'params_%s.tex' % name), 'w') as f:
        for a, b, c in rows_of(d): f.write('%s & %s & %s \\\\\n' % (a, b, c))
print('parameter tables written')
