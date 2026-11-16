# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Build the arch centreline, a dense local frame (tangent / buccal normal) and
the occlusal-height profile along the arch.   ->  build/arch.npz"""
import numpy as np, trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from common import *

m = trimesh.load(TEETH_STL); V = m.vertices
z0 = V[:, 2].min(); H = V[:, 2] - z0
Cc = np.array(POLAR_CENTRE)

# 1. centreline: mid-point between outer and inner extent in 1-deg polar bins (crown band h<5)
P = V[H < 5.0]; d = P[:, :2] - Cc
ang = np.degrees(np.arctan2(d[:, 0], -d[:, 1])); r = np.hypot(*d.T)
A = np.arange(-92, 92.1, 1.0); cl = []
for a in A:
    b = np.abs(ang - a) < 0.8
    cl.append((r[b].max() + r[b].min()) / 2 if b.sum() > 20 else np.nan)
cl = np.array(cl); ok = ~np.isnan(cl); A = A[ok]; cl = gaussian_filter1d(cl[ok], 2)
t = np.radians(A); X = Cc[0] + cl * np.sin(t); Y = Cc[1] - cl * np.cos(t)
L = np.r_[0, np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]; L -= np.interp(0, A, L)

# 2. dense frame
sf = np.linspace(L[0], L[-1], 3000); Xf = np.interp(sf, L, X); Yf = np.interp(sf, L, Y)
T = np.gradient(np.column_stack([Xf, Yf]), axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
N = np.column_stack([T[:, 1], -T[:, 0]])
N *= np.sign(((np.column_stack([Xf, Yf]) - Cc) * N).sum(1))[:, None]   # point buccally

# 3. occlusal profile (lowest h per 0.5 mm station) and bucco-lingual width
Q = V[H < 9.0]; tree = cKDTree(np.column_stack([Xf, Yf])); _, ii = tree.query(Q[:, :2])
s = sf[ii]; n = ((Q[:, :2] - np.column_stack([Xf, Yf])[ii]) * N[ii]).sum(1); h = Q[:, 2] - z0
ss = np.arange(L[0], L[-1], 0.5); occ = []
for si in ss:
    b = np.abs(s - si) < 0.4
    occ.append(h[b].min() if b.sum() >= 20 else np.nan)
np.savez(os.path.join(BUILD, 'arch.npz'), z0=z0, sf=sf, P=np.column_stack([Xf, Yf]), T=T, N=N,
         occ_s=ss, occ_h=np.array(occ))
print('arch length %.1f mm per side; saved build/arch.npz' % (L[-1]))
