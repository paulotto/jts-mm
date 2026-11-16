# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""OPTIONAL. Export Teeth_02 from the FreeCAD document with its placement applied.

Run with FreeCAD's command-line interpreter, e.g.
    freecadcmd s00_export_teeth_freecad.py -- ../../parts/teeth_attachment/teeth_attachment.FCStd
Without a path it looks for the combined FreeCAD file in hardware/mechanical/source/, which
is not part of the repository.
The repository already contains hardware/mechanical/scan/teeth_02.stl, so this is only needed
for a different subject or a changed model.
"""
import sys, os
import FreeCAD as App, Mesh
args = [a for a in sys.argv if a.endswith('.FCStd')]
src = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'source',
                                         'JTS_Models_simple_aruco.FCStd')
doc = App.openDocument(src)
obj = doc.getObject('Teeth_02')
m = obj.Mesh.copy()                 # Mesh.Mesh in local coordinates
m.transform(obj.Placement.toMatrix())
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', '..', 'scan',
                   'teeth_02.stl')
m.write(out)
print('wrote', out, m.CountFacets, 'facets')
