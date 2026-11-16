# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Geometry helpers on top of build/arch.npz and build/field.npz."""
import numpy as np, trimesh
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from skimage.measure import marching_cubes
from common import *

ARCH = np.load(os.path.join(BUILD, 'arch.npz'))
SF, CP, CT, CN = ARCH['sf'], ARCH['P'], ARCH['T'], ARCH['N']
Z0 = float(ARCH['z0'])
_TEETH = None
def teeth_mesh():
    global _TEETH
    if _TEETH is None: _TEETH = trimesh.load(TEETH_STL)
    return _TEETH

def frame(s):
    i = np.argmin(np.abs(SF - s)); return CP[i], CT[i], CN[i]

def arch2xyz(s, n, h):
    s, n, h = (np.atleast_1d(np.asarray(v, float)) for v in (s, n, h))
    p = np.column_stack([np.interp(s, SF, CP[:, 0]), np.interp(s, SF, CP[:, 1])])
    nn = np.column_stack([np.interp(s, SF, CN[:, 0]), np.interp(s, SF, CN[:, 1])])
    nn /= np.linalg.norm(nn, axis=1)[:, None]
    return np.column_stack([p + n[:, None] * nn, Z0 + h])

_os, _oh = ARCH['occ_s'], ARCH['occ_h']
_oh = gaussian_filter1d(np.where(np.isnan(_oh), np.nanmax(_oh), _oh), 2)
def top(s):
    """smoothed occlusal (cusp / marginal ridge) height at station s"""
    return np.interp(s, _os, _oh)

# ---- sections -------------------------------------------------------------
def section_polylines(mesh, s):
    """Cut `mesh` with the plane normal to the arch at station s -> list of (n,h) arrays."""
    p, t, nrm = frame(s)
    sec = mesh.section(plane_origin=[p[0], p[1], 0], plane_normal=[t[0], t[1], 0])
    out = []
    if sec is None: return out
    for e in sec.entities:
        q = sec.vertices[e.points]
        out.append(np.column_stack([((q[:, :2] - p) * nrm).sum(1), q[:, 2] - Z0]))
    return out

def section_geom(mesh, s):
    """Shapely geometry of the section (even-odd filled)."""
    from shapely.geometry import Polygon
    g = None
    for xy in section_polylines(mesh, s):
        if len(xy) < 4: continue
        pg = Polygon(xy).buffer(0)
        if pg.area < 1e-3: continue
        g = pg if g is None else g.symmetric_difference(pg)
    return g

def profile(s, hmax=13.0, dh=0.1):
    """Outermost buccal (max n) and lingual (min n) surface of the teeth section vs height."""
    polys = section_polylines(teeth_mesh(), s); hs = np.arange(0, hmax + 1e-9, dh); bu, li = [], []
    for hh in hs:
        xs = []
        for P in polys:
            a, b = P[:-1], P[1:]; c = ((a[:, 1] - hh) * (b[:, 1] - hh)) <= 0
            for (x1, y1), (x2, y2) in zip(a[c], b[c]):
                xs.append(x1 if y2 == y1 else x1 + (hh - y1) * (x2 - x1) / (y2 - y1))
        xs = np.array(xs); xs = xs[np.abs(xs) < 9.0] if len(xs) else xs
        bu.append(xs.max() if len(xs) else np.nan); li.append(xs.min() if len(xs) else np.nan)
    return hs, np.array(bu), np.array(li)

# ---- voxel field (lazy, large) ---------------------------------------------
class Field:
    def __init__(self):
        F = np.load(os.path.join(BUILD, 'field.npz'))
        self.dist = F['dist']; g = F['grid']
        self.x0, self.y0, self.h0, self.vs = g[:4]; self.nx, self.ny, self.nh = (int(v) for v in g[4:7])
        hs = self.h0 + (np.arange(self.nh) + 0.5) * self.vs
        sh = (self.nx, self.ny, self.nh)
        self.S = np.broadcast_to(F['S2'][:, :, None], sh)
        self.NN = np.broadcast_to(F['N2'][:, :, None], sh)
        self.H = np.broadcast_to(hs[None, None, :], sh)
        gx, gy, gh = np.gradient(self.dist, self.vs); gn = np.sqrt(gx**2 + gy**2 + gh**2) + 1e-6
        self.FN = (gx * F['NX2'][:, :, None] + gy * F['NY2'][:, :, None]) / gn   # + = faces buccally
        del gx, gy, gh
        self.TOP = top(self.S)
    def shell(self, t, gap=0.05):
        return (self.dist > gap) & (self.dist <= gap + t)
    def to_mesh(self, mask, smooth=0.7, level=0.5):
        f = gaussian_filter(mask.astype(np.float32), smooth)
        if f.max() < level: return None
        v, fc, _, _ = marching_cubes(f, level)
        v = v * self.vs; v[:, 0] += self.x0; v[:, 1] += self.y0; v[:, 2] += self.h0 + Z0
        return trimesh.Trimesh(v, fc, process=False)
    def dist_at(self, xyz):
        i = (xyz[:, 0] - self.x0) / self.vs - 0.5; j = (xyz[:, 1] - self.y0) / self.vs - 0.5
        k = (xyz[:, 2] - Z0 - self.h0) / self.vs - 0.5
        return map_coordinates(self.dist, [i, j, k], order=1, mode='nearest')
    def surf_n(self, s, h, side, offset):
        """n at which the distance to the teeth equals `offset` (side +1 buccal / -1 lingual)."""
        ns = np.arange(0, 12, 0.05) * side
        d = self.dist_at(arch2xyz(np.full_like(ns, s), ns, np.full_like(ns, h)))
        inside = np.where(d < 0.11)[0]
        if len(inside) == 0: return ns[0]
        for kk in range(inside[-1], len(ns)):
            if d[kk] >= offset: return ns[kk]
        return ns[-1]
