# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""3D renders (PNG) of every concept on the teeth model, three views each.
3D renders stay raster: a vector export of ~10^5 triangles is neither small nor
editable.  Change the camera dictionary below to get other views.
->  latex/figures/png/render_<concept>_<view>.png"""
import warnings; warnings.filterwarnings('ignore')
import sys, pickle, numpy as np, pyvista as pv
from PIL import Image, ImageOps
from common import *
pv.OFF_SCREEN = True
TEETH = pv.read(TEETH_STL)
CAMS = {   # position, focal point, view-up (occlusal direction is -z)
    'obl':  [(-58, 52, -188), (-110, 150, -128), (0, 0, -1)],
    'molB': [(-38, 142, -152), (-86, 158, -127), (0, 0, -1)],
    'molL': [(-128, 168, -168), (-87, 158, -126), (0, 0, -1)],
}
DET_BOX = (-101.0, -70.0, 138.0, 180.0, -150.0, -100.0)   # x, y, z bounds of the detail views
ZOOM = {'obl': 1.0, 'molB': 1.35, 'molL': 1.35}
SOLO = {   # parts alone, no teeth
    'solo_obl': [(-30, 60, -175), (-110, 150, -122), (0, 0, -1)],
    'solo_top': [(-110, 150, -300), (-110, 150, -122), (0, -1, 0)],
    'solo_detL': [(-118, 196, -148), (-88, 156, -124), (0, 0, -1)],
    'solo_detB': [(-48, 150, -150), (-88, 156, -124), (0, 0, -1)],
}
def tm2pv(m):
    return pv.PolyData(np.asarray(m.vertices), np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel())
def tube(pts, r, n=300): return pv.Spline(np.asarray(pts), n).tube(radius=r, n_sides=18)
def marker(Fsurf):
    from geom import arch2xyz
    a = arch2xyz([0.0], [Fsurf], [7.2])[0]; b = arch2xyz([0.0], [Fsurf + 1], [7.2])[0]; nr = (b - a) / np.linalg.norm(b - a)
    st = pv.Cylinder(center=a + nr * 6.0, direction=nr, radius=1.5, height=12)
    pl = pv.Plane(center=a + nr * 12.6, direction=nr, i_size=16, j_size=16).extrude(nr * 1.6, capping=True)
    return [(st, '#555555', 1.0), (pl, '#222222', 1.0)]
def parts(r):
    P = []
    for k in ('printed', 'rigid', 'spring', 'tpu'):
        if k in r and r[k] is not None: P.append((tm2pv(r[k]), C[k], 1.0))
    for w in r.get('wires', []): P.append((tube(w, 0.35, 400), C['wire'], 1.0))
    if 'arch' in r:
        P.append((tube(r['arch'], 0.45), C['wire'], 1.0)); P.append((tube(r['arch_free'], 0.45), C['spring'], 0.45))
    return P + marker(3.6)
def render(P, view, fn, teeth=True):
    p = pv.Plotter(off_screen=True, window_size=(1600, 1230)); p.set_background('white')
    if teeth: p.add_mesh(TEETH, color=C['tooth'], smooth_shading=True, specular=0.25)
    for mesh, col, op in P:
        if view.startswith('solo_det'):            # right-molar region only
            mesh = mesh.clip_box(DET_BOX, invert=False)
            if mesh.n_points == 0: continue
        p.add_mesh(mesh, color=col, smooth_shading=True, opacity=op,
                                        specular=0.6 if col == C['wire'] else 0.2)
    p.camera_position = CAMS[view] if view in CAMS else SOLO[view]
    p.camera.zoom(ZOOM.get(view, 1.0) if teeth else 1.0)
    p.enable_anti_aliasing('ssaa')
    p.screenshot(fn); p.close()
    im = Image.open(fn).convert('RGB'); bb = ImageOps.invert(im).getbbox(); m = 20
    im.crop((max(bb[0] - m, 0), max(bb[1] - m, 0), min(bb[2] + m, im.width), min(bb[3] + m, im.height))).save(fn)
if __name__ == '__main__':
    out = pickle.load(open(os.path.join(BUILD, 'concepts.pkl'), 'rb'))
    names = sys.argv[1:] or sorted(out)
    for nm in names:
        for v in CAMS: render(parts(out[nm]), v, os.path.join(FIG, 'png', 'render_%s_%s.png' % (nm, v)))
        solo = [q for q in parts(out[nm]) if not (q[1] == C['spring'] and q[2] < 1.0)]   # drop the ghost
        for v in SOLO: render(solo, v, os.path.join(FIG, 'png', 'render_%s_%s.png' % (nm, v)), teeth=False)
        print(nm, 'rendered')
