# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Build the five concept geometries as conforming voxel solids (+ wire paths).

Every printed part is defined as a *mask* on the voxel grid: a shell at a given
distance from the teeth, restricted by arch station s, offset n, height h and by
which way the tooth surface faces (FN).  The masks are the editable design
parameters of each concept - change a number here and re-run s04..s07.
->  stl/c*_*.stl, build/concepts.pkl"""
import warnings; warnings.filterwarnings('ignore')
import pickle, numpy as np, pyvista as pv
from geom import *
from params import COMMON, C1, C2, C3, C4, C5

F = Field(); S, NN, H, FN, TOP = F.S, F.NN, F.H, F.FN, F.TOP
shell = F.shell
M1M2 = {1: EMBRASURES_R['M1/M2'], -1: EMBRASURES_L['M1/M2']}
PM2M1 = {1: EMBRASURES_R['PM2/M1'], -1: EMBRASURES_L['PM2/M1']}
CPM1 = {1: EMBRASURES_R['C/PM1'], -1: EMBRASURES_L['C/PM1']}

def LING(): return NN < np.where(np.abs(S) < 18, -1.5, -3.2)     # lingual of the crown
def BUCC(): return NN > np.where(np.abs(S) < 18, 0.8, 3.6)       # buccal / labial of the crown

# ---- building blocks ---------------------------------------------------------
def lingual_flange(t=C1['flange_t'], smax=C1['s_half'], hmax=C1['flange_h_max']):
    return shell(t) & (FN < -0.3) & LING() & (np.abs(S) <= smax) & (H >= TOP + C1['flange_below_top']) & (H <= hmax)
def labial_bar(p=COMMON['labial_bar']):
    return shell(p['t']) & (FN > 0.3) & BUCC() & (np.abs(S) <= p['s_half']) & (H >= p['h'][0]) & (H <= p['h'][1])
def crossover(se, hmax, t=COMMON['crossover']['t'], w=COMMON['crossover']['w'] / 2, buccal_hmax=None):
    m = shell(t) & (np.abs(S - se) < w) & (H <= hmax)
    if buccal_hmax is not None: m &= ~((NN > 0) & (H > buccal_hmax))
    return m
def pad(sg, s_lo, s_hi, h_lo, h_hi, face, t=1.2):
    lo, hi = sorted((sg * s_lo, sg * s_hi))
    f = ((FN < -0.3) & LING()) if face == 'L' else ((FN > 0.3) & BUCC())
    return shell(t) & f & (S >= lo) & (S <= hi) & (H >= h_lo) & (H <= h_hi)
def lingual_bar(t=COMMON['lingual_bar']['t'], smax=COMMON['lingual_bar']['s_half'],
                hlo=COMMON['lingual_bar']['h'][0], hhi=COMMON['lingual_bar']['h'][1]):
    return shell(t) & (FN < -0.3) & (NN < -1.0) & (np.abs(S) <= smax) & (H >= hlo) & (H <= hhi)
def rest(se, sg, p=COMMON['rest']):   # occlusal rest on the marginal ridge distal of embrasure se
    return shell(p['t']) & (np.abs(S - (se + sg * p['length'] / 2 - sg * 0.1)) < p['length'] / 2) & (np.abs(NN) < p['n_half']) & (H <= TOP + 1.6)
# ---- concepts ----------------------------------------------------------------
def c1():   # snap shell
    b = lingual_flange()
    for sg in (1, -1):
        for lo_, hi_ in C1['buccal_arm_s']: b |= pad(sg, lo_, hi_, *C1['buccal_arm_h'], 'B', C1['buccal_arm_t'])
        b |= crossover(PM2M1[sg], 7.4) | crossover(M1M2[sg], 7.4) | crossover(CPM1[sg], 9.4)
    return dict(printed=b | labial_bar())

def c_loop(sg, s_lo, s_hi, nc, hc, R, th, a1, a2):
    """C-loop in the (n,h) plane, swept along s. Angle a = atan2(h-hc, n-nc):
    0 deg points at the tooth (+n), -90 deg occlusally, +/-180 deg lingually."""
    lo, hi = sorted((sg * s_lo, sg * s_hi)); r = np.hypot(NN - nc, H - hc)
    ang = np.degrees(np.arctan2(H - hc, NN - nc)); ang = np.where(ang > a2, ang - 360, ang)
    return (S >= lo) & (S <= hi) & (np.abs(r - R) <= th / 2) & (ang <= a2) & (ang >= a1) & (F.dist > 0.05)

def c2_geometry(sg):
    """Derived loop placement for side sg: centre at the contact height, loop tangent to the jaw back."""
    p = C2; lj, lp = p['lingual_jaw'], p['loop']
    s_mid = sg * (lj['s'][0] + lj['s'][1]) / 2
    n_surf = F.surf_n(s_mid, p['contact_h'], -1, 0.0)            # lingual tooth surface at contact height
    n_jaw_back = n_surf - COMMON['gap'] - lj['t']
    nc = n_jaw_back - lp['R'] + lp['t'] / 2                       # loop centre line meets the jaw back
    return dict(s_mid=s_mid, n_surf=n_surf, n_jaw_back=n_jaw_back, nc=nc, hc=lp['h_centre'])

def c2():   # clothespin jaw
    rigid = lingual_bar() | labial_bar(); spring = np.zeros_like(rigid)
    bj, lj, lp = C2['buccal_jaw'], C2['lingual_jaw'], C2['loop']
    for sg in (1, -1):
        g = c2_geometry(sg)
        rigid |= pad(sg, *bj['s'], *bj['h'], 'B', bj['t'])                                  # rigid buccal jaw
        rigid |= crossover(PM2M1[sg], 13.0, 1.2, buccal_hmax=7.4)                            # bridge into the bar
        rigid |= crossover(CPM1[sg], 13.0, 1.2, buccal_hmax=9.4)
        spring |= pad(sg, *lj['s'], *lj['h'], 'L', lj['t'])                                 # lingual jaw
        sm = abs(g['s_mid']); w2 = lp['w'] / 2
        spring |= c_loop(sg, sm - w2, sm + w2, g['nc'], g['hc'], lp['R'], lp['t'], *lp['arc'])
        lo, hi = sorted((sg * (sm - w2), sg * (sm + w2)))
        rigid |= ((S >= lo) & (S <= hi) & (NN >= g['nc'] - lp['t'] / 2 - 0.4) & (NN <= g['nc'] + 3.0)
                  & (H >= g['hc'] + lp['R'] - lp['t'] / 2) & (H <= C2['foot']['h'][1]) & (F.dist > 0.05))  # foot
    return dict(rigid=rigid, spring=spring)

def adams_path(sg):
    """Adams clasp (0.7 mm) around M1, as a list of (s,n,h) way-points."""
    sm, sd = PM2M1[sg], M1M2[sg]; tm, td = float(top(sm)) - 0.45, float(top(sd)) - 0.45
    nl = lambda s, h: F.surf_n(s, h, -1, 0.45); nb = lambda s, h: F.surf_n(s, h, 1, 0.45)
    P = [(sm, nl(sm, 9.5) - 1.2, 9.8), (sm, nl(sm, 6.0), 6.0), (sm, -2.0, tm), (sm, 2.0, tm), (sm, nb(sm, 5.5), 5.5)]
    a = sm + sg * 1.0; P += [(a - sg * .5, nb(a, 8.3) + .1, 8.6), (a, nb(a, 8.9), 8.9), (a + sg * .5, nb(a, 8.3) + .1, 8.6)]
    P += [(s_, nb(s_, 5.6) + 0.7, 5.6) for s_ in np.linspace(sm + sg * 2.4, sd - sg * 2.4, 5)]
    b = sd - sg * 1.0; P += [(b - sg * .5, nb(b, 8.3) + .1, 8.6), (b, nb(b, 8.9), 8.9), (b + sg * .5, nb(b, 8.3) + .1, 8.6)]
    P += [(sd, nb(sd, 5.5), 5.5), (sd, 2.0, td), (sd, -2.0, td), (sd, nl(sd, 6.0), 6.0), (sd, nl(sd, 9.5) - 1.2, 9.8)]
    s, n, h = np.array(P).T; return arch2xyz(s, n, h)

def c3():   # wire clasps
    rigid = lingual_flange(t=C3['plate_t'], smax=C3['plate_s_half'], hmax=C3['plate_h_max']) | labial_bar()
    for sg in (1, -1): rigid |= crossover(CPM1[sg], 9.4) | rest(PM2M1[sg], sg)
    return dict(rigid=rigid, wires=[adams_path(1), adams_path(-1)])

def arch_path(pre_spread=0.0, h=C4['wire_h']):
    s = np.linspace(-C4['wire_s_half'], C4['wire_s_half'], 70)
    n = np.array([F.surf_n(si, h, -1, C4['wire_offset'] if abs(si) > 18 else 1.3) for si in s])
    n = n + pre_spread * (np.abs(s) / C4['wire_s_half']) ** 2          # free shape: legs spread outward
    return arch2xyz(s, n, np.full_like(s, h))

PRE_SPREAD = C4['pre_spread']   # mm per side, see the wire-sizing section
def c4():   # spring arch: 0.9 mm stainless arch + printed, resin-relined pads
    pads = np.zeros(F.dist.shape, bool)
    for sg in (1, -1):
        pads |= pad(sg, *C4['pad_M1_s'], *C4['pad_h'], 'L', C4['pad_t']) | pad(sg, *C4['pad_M2_s'], C4['pad_h'][0] - 0.4, C4['pad_h'][1], 'L', C4['pad_t'])
        pads |= shell(1.4) & (FN < -0.3) & (NN < -3.2) & (S * sg >= 41.0) & (S * sg <= 43.6) & (H >= 9.0) & (H <= 10.9)
    rigid = labial_bar() | (shell(C4['yoke_t']) & (FN < -0.3) & (np.abs(S) <= C4['yoke_s_half']) & (H >= C4['yoke_h'][0]) & (H <= C4['yoke_h'][1]))
    for sg in (1, -1): rigid |= crossover(CPM1[sg], 11.6, 1.2, buccal_hmax=9.4)
    return dict(rigid=rigid, printed=pads, arch=arch_path(0.0), arch_free=arch_path(PRE_SPREAD))

def c5():   # rigid clamp + TPU liner
    rigid = lingual_bar() | labial_bar(); tpu = np.zeros_like(rigid)
    for sg in (1, -1):
        lo, hi = sorted((sg * C5['band_s'][0], sg * C5['band_s'][1])); band = (S >= lo) & (S <= hi); ling = (FN < -0.3) & (NN < -3.2)
        d_l = COMMON['gap'] + C5['liner_t']
        tpu |= band & ling & (F.dist > COMMON['gap']) & (F.dist <= d_l) & (H >= C5['liner_h'][0]) & (H <= C5['liner_h'][1])
        rigid |= band & ling & (F.dist > d_l) & (F.dist <= d_l + C5['jaw_t']) & (H >= C5['jaw_h'][0]) & (H <= C5['jaw_h'][1])
        rigid |= shell(C5['buccal_jaw_t']) & band & (FN > 0.3) & (NN > 3.6) & (H >= C5['buccal_jaw_h'][0]) & (H <= C5['buccal_jaw_h'][1])
        rigid |= crossover(PM2M1[sg], 11.5, 1.2, buccal_hmax=7.4) | rest(PM2M1[sg], sg)
        rigid |= (np.abs(S - M1M2[sg]) < 0.9) & (F.dist > d_l) & (F.dist <= d_l + C5['jaw_t']) & (NN < -3.0) & (H >= 5.0) & (H <= 12.5)
        rigid |= crossover(CPM1[sg], 13.0, 1.2, buccal_hmax=9.4)
    return dict(rigid=rigid, tpu=tpu)

def tube(pts, r, n=300):
    return pv.Spline(np.asarray(pts), n).tube(radius=r, n_sides=18).triangulate()

if __name__ == '__main__':
    out = {}
    for nm, fn in (('c1', c1), ('c2', c2), ('c3', c3), ('c4', c4), ('c5', c5)):
        d = fn(); r = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.dtype == bool:
                mesh = F.to_mesh(v); r[k] = mesh; mesh.export(os.path.join(STLDIR, '%s_%s.stl' % (nm, k)))
            else:
                r[k] = v
        if 'wires' in r:
            w = tube(r['wires'][0], 0.35, 400).merge(tube(r['wires'][1], 0.35, 400))
            w.save(os.path.join(STLDIR, '%s_wire_0p7mm.stl' % nm))
        if 'arch' in r: tube(r['arch'], 0.45).save(os.path.join(STLDIR, '%s_arch_wire_0p9mm.stl' % nm))
        out[nm] = r; print(nm, 'built')
    pickle.dump(out, open(os.path.join(BUILD, 'concepts.pkl'), 'wb'))
