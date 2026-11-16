# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Reference geometry for FreeCAD, in the Teeth_02 frame (import next to the STL).

Every part of the concepts is a profile (see the section drawings) swept along a
curve that follows the teeth.  This script writes those curves, plus section-plane
markers at the key stations, as DXF R12 (File > Import in FreeCAD, or Draft >
Import) and as CSV (x, y, z, s).   ->  cad/*.dxf, cad/*.csv, cad/README.txt"""
import warnings; warnings.filterwarnings('ignore')
import pickle, numpy as np
from scipy.ndimage import gaussian_filter1d
from geom import *
from params import COMMON, C2, C4
F = Field()
CAD = os.path.join(ROOT, 'cad'); os.makedirs(CAD, exist_ok=True)

class DXF:
    """Minimal DXF R12 writer: 3D polylines, lines, points, text."""
    def __init__(self): self.e = []
    def poly3d(self, P, layer='0', closed=False):
        self.e += ['0', 'POLYLINE', '8', layer, '66', '1', '10', '0', '20', '0', '30', '0', '70', str(8 | (1 if closed else 0))]
        for x, y, z in P: self.e += ['0', 'VERTEX', '8', layer, '10', '%.4f' % x, '20', '%.4f' % y, '30', '%.4f' % z, '70', '32']
        self.e += ['0', 'SEQEND', '8', layer]
    def line(self, a, b, layer='0'):
        self.e += ['0', 'LINE', '8', layer, '10', '%.4f' % a[0], '20', '%.4f' % a[1], '30', '%.4f' % a[2],
                   '11', '%.4f' % b[0], '21', '%.4f' % b[1], '31', '%.4f' % b[2]]
    def point(self, p, layer='0'):
        self.e += ['0', 'POINT', '8', layer, '10', '%.4f' % p[0], '20', '%.4f' % p[1], '30', '%.4f' % p[2]]
    def text(self, p, s, h=1.0, layer='0'):
        self.e += ['0', 'TEXT', '8', layer, '10', '%.4f' % p[0], '20', '%.4f' % p[1], '30', '%.4f' % p[2], '40', str(h), '1', s]
    def save(self, fn):
        layers = sorted({self.e[i + 1] for i, v in enumerate(self.e[:-1]) if v == '8'} | {'0'})
        hdr = ['0', 'SECTION', '2', 'HEADER', '9', '$ACADVER', '1', 'AC1009', '9', '$INSUNITS', '70', '4', '0', 'ENDSEC']
        tab = ['0', 'SECTION', '2', 'TABLES', '0', 'TABLE', '2', 'LAYER', '70', str(len(layers))]
        for i, l in enumerate(layers):
            tab += ['0', 'LAYER', '2', l, '70', '0', '62', str(1 + i % 7), '6', 'CONTINUOUS']
        tab += ['0', 'ENDTAB', '0', 'ENDSEC']
        body = ['0', 'SECTION', '2', 'ENTITIES'] + self.e + ['0', 'ENDSEC', '0', 'EOF']
        with open(fn, 'w') as f: f.write('\n'.join(hdr + tab + body) + '\n')

def surface_curve(s0, s1, h, side, offset, ds=0.25, sigma_mm=1.5):
    """Curve on the tooth surface at height h, offset by `offset`, lingual (side=-1) or
    buccal (+1), between stations s0..s1.  Built as a true 2D offset of the section
    polygon (shapely buffer) so it has no cusps where the arch is tightly curved, then
    smoothed and pushed back out wherever smoothing cut into the teeth."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union, nearest_points
    from scipy.spatial import cKDTree
    sec = teeth_mesh().section(plane_origin=[0, 0, Z0 + h], plane_normal=[0, 0, 1])
    path, _ = sec.to_2D(to_2D=np.eye(4))
    body = unary_union([Polygon(p.exterior) for p in path.polygons_full]).buffer(0)
    body = body.buffer(2.5, join_style=1).buffer(-2.5, join_style=1)   # close interdental notches (rolling ball 2.5 mm)
    off = body.buffer(offset + 0.02, join_style=1)
    rings = [off.exterior] if off.geom_type == 'Polygon' else [g.exterior for g in off.geoms]
    B = np.vstack([np.asarray(r.coords) for r in rings])
    _, ii = cKDTree(CP).query(B); sB = SF[ii]; nB = ((B - CP[ii]) * CN[ii]).sum(1)
    sel = (np.sign(nB) == side) & (sB >= s0 - 1) & (sB <= s1 + 1)
    Q = B[sel]; q = sB[sel]; o = np.argsort(q); Q, q = Q[o], q[o]
    # resample by arc length and smooth
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    keep = np.r_[True, np.diff(L) > 1e-6]; Q, q, L = Q[keep], q[keep], L[keep]
    Lu = np.arange(0, L[-1], ds); X = np.interp(Lu, L, Q[:, 0]); Y = np.interp(Lu, L, Q[:, 1]); su = np.interp(Lu, L, q)
    sg = sigma_mm / ds; X, Y = gaussian_filter1d(X, sg, mode='nearest'), gaussian_filter1d(Y, sg, mode='nearest')
    inner = body.buffer(offset); edge = inner.boundary
    for it in range(6):                       # push out along the curve normal, re-smooth, repeat
        T = np.gradient(np.column_stack([X, Y]), axis=0); T /= np.linalg.norm(T, axis=1)[:, None] + 1e-12
        Nc = np.column_stack([T[:, 1], -T[:, 0]])
        _, jj = cKDTree(CP).query(np.column_stack([X, Y]))
        Nc *= np.sign((Nc * CN[jj]).sum(1) * side)[:, None]          # point away from the teeth
        pen = np.array([edge.distance(Point(x_, y_)) if inner.contains(Point(x_, y_)) else 0.0 for x_, y_ in zip(X, Y)])
        if pen.max() < 0.02: break
        X = X + Nc[:, 0] * (pen + 0.01) * 1.2; Y = Y + Nc[:, 1] * (pen + 0.01) * 1.2
        X, Y = gaussian_filter1d(X, 0.6 / ds, mode='nearest'), gaussian_filter1d(Y, 0.6 / ds, mode='nearest')
    m = (su >= s0) & (su <= s1)
    return np.column_stack([X[m], Y[m], np.full(m.sum(), Z0 + h)]), su[m]

def write(name, P, s, layer, dxf):
    np.savetxt(os.path.join(CAD, name + '.csv'), np.column_stack([P, s]), delimiter=',', fmt='%.4f',
               header='x,y,z,s  (mm, Teeth_02 frame)', comments='')
    dxf.poly3d(P, layer); one = DXF(); one.poly3d(P, layer); one.save(os.path.join(CAD, name + '.dxf'))

g = COMMON['gap']; allx = DXF(); curves = []; starts = []
c = [('path_lingual_contact_h%.1f' % C2['contact_h'], -53, 53, C2['contact_h'], -1, g, 'lingual face of jaws, pads, flange'),
     ('path_lingual_bar_h11.8', -40, 40, 11.8, -1, g, 'inner face of the lingual bar'),
     ('path_buccal_R_h6.0', 31.0, 53.0, 6.0, 1, g, 'buccal jaws / arms, right'),
     ('path_buccal_L_h6.0', -53.0, -31.0, 6.0, 1, g, 'buccal jaws / arms, left'),
     ('path_labial_h7.2', -COMMON['labial_bar']['s_half'], COMMON['labial_bar']['s_half'], 7.2, 1, g, 'labial bar')]
for name, s0, s1, h, side, off, what in c:
    P, s = surface_curve(s0, s1, h, side, off); write(name, P, s, name, allx); curves.append((name, what))
    _, _, nrm0 = frame(s0); starts.append((name, P[0][0], P[0][1], P[0][2], nrm0[0], nrm0[1], h))
s = SF[::10]; P = arch2xyz(s, np.zeros_like(s), np.full_like(s, 6.0)); write('ref_centreline_h6.0', P, s, 'centreline', allx)
curves.append(('ref_centreline_h6.0', 'arch centreline (defines s and n)'))

out = pickle.load(open(os.path.join(BUILD, 'concepts.pkl'), 'rb'))
for key, lab in (('arch', 'c4_wire_mounted'), ('arch_free', 'c4_wire_free')):
    P = out['c4'][key]; write(lab, P, np.linspace(-C4['wire_s_half'], C4['wire_s_half'], len(P)), lab, allx)
    _, _, nrm0 = frame(-C4['wire_s_half']); starts.append((lab, P[0][0], P[0][1], P[0][2], nrm0[0], nrm0[1], C4['wire_h']))
    curves.append((lab, 'concept 4 wire centreline (%s)' % key))
for i, side in enumerate(('R', 'L')):
    P = out['c3']['wires'][i]; write('c3_adams_%s' % side, P, np.arange(len(P)), 'c3_adams', allx)
    curves.append(('c3_adams_%s' % side, 'concept 3 Adams clasp way-points (wire centreline)'))

# stations: point on the centreline, and a 24 mm line along the local normal = trace of the section plane
st = DXF(); rows = []
for side in (EMBRASURES_R, EMBRASURES_L):
    for lab0, sv in side.items():
        lab = '%s %s' % (lab0, 'R' if sv > 0 else 'L')
        for d in (st, allx):
            p0 = arch2xyz([sv], [0.0], [6.0])[0]; a = arch2xyz([sv], [-12.0], [6.0])[0]; b = arch2xyz([sv], [12.0], [6.0])[0]
            d.point(p0, 'stations'); d.line(a, b, 'section_planes'); d.text(b, '%s s=%.1f' % (lab, sv), 0.8, 'stations')
        p, t, nrm = frame(sv); rows.append((lab, sv, p[0], p[1], t[0], t[1], nrm[0], nrm[1]))
for sv, lab in ((TOOTH_MID['M1'], 'M1 mid R'), (-TOOTH_MID['M1'], 'M1 mid L'), (0.0, 'midline')):
    p, t, nrm = frame(sv); rows.append((lab, sv, p[0], p[1], t[0], t[1], nrm[0], nrm[1]))
    a = arch2xyz([sv], [-12.0], [6.0])[0]; b = arch2xyz([sv], [12.0], [6.0])[0]
    for d in (st, allx): d.line(a, b, 'section_planes'); d.text(b, lab, 0.8, 'stations')
# tooth outline in every section plane, in sketch coordinates (n, h)
from scipy.interpolate import interp1d
for lab, sv, *_ in rows:
    polys = section_polylines(teeth_mesh(), sv)
    with open(os.path.join(CAD, 'section_%s.csv' % lab.replace('/', '-').replace(' ', '_')), 'w') as f:
        f.write('loop,n,h\n')
        for k, P in enumerate(polys):
            P = P[(P[:, 1] < 16.0)]
            if len(P) < 3: continue
            step = max(1, len(P) // 400)
            for n_, h_ in P[::step]: f.write('%d,%.4f,%.4f\n' % (k, n_, h_))
with open(os.path.join(CAD, 'path_starts.csv'), 'w') as f:
    f.write('curve,x,y,z,normal_x,normal_y,h\n')
    for r in starts: f.write('%s,%.4f,%.4f,%.4f,%.5f,%.5f,%.2f\n' % r)
st.save(os.path.join(CAD, 'stations.dxf')); allx.save(os.path.join(CAD, 'all_reference_geometry.dxf'))
with open(os.path.join(CAD, 'stations.csv'), 'w') as f:
    f.write('label,s,x,y,tangent_x,tangent_y,normal_x,normal_y  (z of the h=0 plane: %.4f)\n' % Z0)
    for r in rows: f.write('%s,%.2f,%.4f,%.4f,%.5f,%.5f,%.5f,%.5f\n' % r)
with open(os.path.join(CAD, 'README.txt'), 'w') as f:
    f.write('Reference geometry in the Teeth_02 frame (mm). z of the h = 0 plane (lowest cusp tip) = %.4f.\n'
            'h in the drawings = z - (%.4f).  n > 0 is buccal.\n\n' % (Z0, Z0))
    for name, what in curves: f.write('%-32s %s\n' % (name + '.dxf/.csv', what))
    f.write('%-32s %s\n' % ('stations.dxf/.csv', 'embrasure stations, section-plane traces (24 mm lines along the normal)'))
    f.write('%-32s %s\n' % ('all_reference_geometry.dxf', 'everything above in one file, one layer per curve'))
import shutil
shutil.copy(os.path.join(ROOT, 'scripts', 'cad_macro_template.FCMacro'), os.path.join(CAD, 'import_reference_geometry.FCMacro'))
print('wrote', len(curves) + 2, 'reference files + import macro to cad/')
