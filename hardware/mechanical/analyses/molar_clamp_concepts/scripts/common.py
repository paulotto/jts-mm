# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared paths, constants and figure helpers for the molar-clamp concept study.

Coordinate conventions (identical to Teeth_02 in the FreeCAD files of the parts,
exported with its placement applied):
    x  mesio-distal (dental midline at x = MIDLINE_X)
    y  arch depth, anterior teeth at small y
    z  tooth axis; the cusp tips are at the SMALLEST z (occlusal = -z)
Derived quantities:
    h  = z - z_min            height below the lowest cusp tip  [mm]
    s  = arc length along the arch centreline, 0 at the midline, + toward +x
    n  = offset from the centreline, + = buccal/labial, - = lingual
"""
import os, json
import numpy as np

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(ROOT, 'data')
BUILD  = os.path.join(ROOT, 'build')
STLDIR = os.path.join(ROOT, 'stl')
FIG    = os.path.join(ROOT, 'latex', 'figures')
GEN    = os.path.join(ROOT, 'latex', 'generated')
for d in (BUILD, STLDIR, GEN, *(os.path.join(FIG, k) for k in ('svg', 'pdf', 'png'))):
    os.makedirs(d, exist_ok=True)

# the scan is shared with the rest of hardware/mechanical (one copy, ../../scan)
TEETH_STL   = os.path.join(ROOT, '..', '..', 'scan', 'teeth_02.stl')
MIDLINE_X   = -109.75          # from the central-incisor embrasure (symmetry of I1/I2)
POLAR_CENTRE = (MIDLINE_X, 172.0)   # helper point inside the U, only used to build the centreline
VOXEL       = 0.2              # mm

# arc-length stations used throughout (right side +, left side -)
EMBRASURES_R = {'I2/C': 11.4, 'C/PM1': 17.9, 'PM1/PM2': 24.4, 'PM2/M1': 31.4, 'M1/M2': 42.4}
EMBRASURES_L = {'I2/C': -10.6, 'C/PM1': -17.6, 'PM1/PM2': -24.6, 'PM2/M1': -31.1, 'M1/M2': -42.5}
TOOTH_MID    = {'I1': 2.8, 'I2': 8.4, 'C': 14.7, 'PM1': 21.2, 'PM2': 27.9, 'M1': 37.0, 'M2': 48.0}

# colours (also used in the LaTeX legend)
C = dict(tooth='#f3e6d2', tooth_edge='#a8824e', printed='#3f7cc9', rigid='#35507a',
         spring='#e8742a', tpu='#4caf50', wire='#8a96a3', red='#c0392b', green='#17724a',
         blue='#2166ac', ink='#222222')

def mpl_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'svg.fonttype': 'none',        # keep text as <text>: editable in Inkscape/Illustrator
        'pdf.fonttype': 42,
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    return plt

def save_fig(fig, name, dpi=200):
    """Write svg (editable), pdf (for LaTeX) and png (preview)."""
    fig.savefig(os.path.join(FIG, 'svg', name + '.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'pdf', name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'png', name + '.png'), bbox_inches='tight', dpi=dpi)

def jdump(obj, name):
    with open(os.path.join(BUILD, name), 'w') as f: json.dump(obj, f, indent=1)

def jload(name):
    with open(os.path.join(BUILD, name)) as f: return json.load(f)
