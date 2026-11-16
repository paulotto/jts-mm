# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Filled voxel model of the dentition, outside distance field and arch
coordinates per voxel column.   ->  build/field.npz  (~150 MB, regenerable)"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, trimesh, time
from skimage.draw import polygon as rpoly
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from common import *

m = trimesh.load(TEETH_STL); z0 = m.vertices[:, 2].min(); vs = VOXEL
x0, x1, y0, y1, h0, h1 = -149.0, -69.0, 120.0, 179.0, -2.4, 14.4
nx, ny, nh = (int(round((x1 - x0) / vs)), int(round((y1 - y0) / vs)), int(round((h1 - h0) / vs)))
occ = np.zeros((nx, ny, nh), bool); t = time.time()
for k in range(nh):
    hk = h0 + (k + 0.5) * vs
    if hk <= 0: continue
    sec = m.section(plane_origin=[0, 0, z0 + hk], plane_normal=[0, 0, 1])
    if sec is None: continue
    path, _ = sec.to_2D(to_2D=np.eye(4))
    for poly in path.polygons_full:              # exterior rings only -> hollow model gets filled
        ex = np.array(poly.exterior.coords)
        rr, cc = rpoly((ex[:, 0] - x0) / vs - 0.5, (ex[:, 1] - y0) / vs - 0.5, shape=(nx, ny))
        occ[rr, cc, k] = True
dist = (distance_transform_edt(~occ) * vs).astype(np.float32)

A = np.load(os.path.join(BUILD, 'arch.npz'))
xs = x0 + (np.arange(nx) + 0.5) * vs; ys = y0 + (np.arange(ny) + 0.5) * vs
Xg, Yg = np.meshgrid(xs, ys, indexing='ij'); XY = np.column_stack([Xg.ravel(), Yg.ravel()])
_, ii = cKDTree(A['P']).query(XY)
S2 = A['sf'][ii].reshape(nx, ny).astype(np.float32)
N2 = ((XY - A['P'][ii]) * A['N'][ii]).sum(1).reshape(nx, ny).astype(np.float32)
np.savez(os.path.join(BUILD, 'field.npz'), occ=occ, dist=dist, S2=S2, N2=N2,
         NX2=A['N'][ii, 0].reshape(nx, ny).astype(np.float32),
         NY2=A['N'][ii, 1].reshape(nx, ny).astype(np.float32),
         grid=np.array([x0, y0, h0, vs, nx, ny, nh, z0]))
print('voxelised %dx%dx%d in %.0fs; saved build/field.npz' % (nx, ny, nh, time.time() - t))
