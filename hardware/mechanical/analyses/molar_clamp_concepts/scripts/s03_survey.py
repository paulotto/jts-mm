# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Survey every tooth: height of contour, gingival margin and undercut depth on
the buccal and lingual face, for a vertical path of insertion (model z axis).
->  build/survey.json, latex/generated/survey_table.tex"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.ndimage import gaussian_filter1d
from geom import *

def survey_face(hs, prof):
    v = ~np.isnan(prof); H, o = hs[v], prof[v]; top_ = H.min(); os_ = gaussian_filter1d(o, 1.5)
    gum = None
    for k in range(len(H) - 3):                       # gingival margin: first outward step >0.25 mm
        if H[k] >= top_ + 3.0 and o[k + 3] - o[k] > 0.25: gum = H[k]; break
    if gum is None: gum = min(top_ + 7.5, H.max())
    z = (H >= top_ + 0.5) & (H <= gum)
    kc = np.argmax(np.where(z, os_, -1e9)); kg = np.argmin(np.abs(H - (gum - 0.2)))
    return dict(top=float(top_), contour_h=float(H[kc]), gum_h=float(gum),
                undercut=float(max(os_[kc] - os_[kg], 0.0)))

res = {}
for side, sg in (('R', 1), ('L', -1)):
    for name, a in TOOTH_MID.items():
        hs, bu, li = profile(sg * a, hmax=13, dh=0.1)
        rec = dict(buccal=survey_face(hs, bu), lingual=survey_face(hs, -li))
        if name in ('I1', 'I2', 'C'): rec['lingual']['undercut'] = 0.0   # cingulum widens monotonically
        res[side + name] = rec
jdump(res, 'survey.json')

rows = []
for name in ('M2', 'M1', 'PM2', 'PM1', 'C', 'I2', 'I1'):
    r = [res['L' + name]['lingual']['undercut'], res['L' + name]['buccal']['undercut'],
         res['R' + name]['lingual']['undercut'], res['R' + name]['buccal']['undercut']]
    rows.append('%s & %s \\\\' % (name, ' & '.join('\\num{%.2f}' % x for x in r)))
open(os.path.join(GEN, 'survey_table.tex'), 'w').write('\n'.join(rows) + '\n')
for k, v in res.items():
    print('%-5s lingual %.2f  buccal %.2f' % (k, v['lingual']['undercut'], v['buccal']['undercut']))
