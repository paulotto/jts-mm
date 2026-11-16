# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Single source of truth for every design dimension (mm, deg).

Heights h are measured from the lowest cusp tip toward the gum; stations s are
arc lengths along the arch (see common.py); n is the offset from the arch
centreline (+ buccal).  "t" is always a wall / material thickness normal to the
tooth surface.  s04_concepts.py builds the geometry from these numbers,
s09_drawings.py dimensions the drawings with them, s08_springs.py sizes the
springs with them - change a value here and re-run the scripts.
"""
COMMON = dict(
    gap=0.05,                 # voxel offset of every shell from the teeth (print relief is added in CAD)
    relief=0.1,               # recommended relief on non-contacting surfaces when printing
    labial_bar=dict(t=1.6, s_half=16.5, h=(5.0, 9.4)),
    crossover=dict(t=1.1, w=1.8),                 # bridge over an embrasure, width along the arch
    rest=dict(t=1.0, length=2.6, n_half=3.2),     # occlusal rest on the marginal ridge
    lingual_bar=dict(t=2.0, s_half=40.0, h=(10.4, 13.2)),
    marker=dict(stem_d=3.0, stem_len=12.0, plate=(16.0, 16.0, 1.6), h=7.2),
)
C1 = dict(flange_t=1.5, flange_below_top=1.4, flange_h_max=10.6, s_half=53.0,
          buccal_arm_t=1.3, buccal_arm_h=(4.6, 7.2), buccal_arm_s=((32.2, 41.6), (43.2, 52.0)))
C2 = dict(buccal_jaw=dict(t=1.4, s=(32.4, 41.4), h=(4.8, 7.2)),
          lingual_jaw=dict(t=1.2, s=(33.4, 40.6), h=(6.3, 9.3)),
          loop=dict(R=3.0, t=0.8, w=5.5, h_centre=7.8, arc=(-270.0, 0.0), n_series=1,
                    over_closure=0.2),
          foot=dict(h=(10.6, 12.4)),                  # rigid block from the loop root to the lingual bar
          contact_h=7.8)                              # where the tooth pushes on the jaw
C3 = dict(plate_t=2.0, plate_s_half=45.5, plate_h_max=11.0,
          wire_d=0.7, bridge_offset=0.7, bridge_h=5.6, arrowhead_h=8.9, tag_h=9.8)
C4 = dict(wire_d=0.9, pre_spread=0.7, wire_h=10.3, wire_s_half=38.5, wire_offset=1.0,
          pad_t=1.4, pad_M1_s=(33.0, 41.2), pad_M2_s=(43.4, 51.0), pad_h=(6.4, 10.9),
          yoke_t=1.6, yoke_s_half=19.0, yoke_h=(9.6, 12.0))
C5 = dict(liner_t=1.6, liner_h=(5.4, 10.0), jaw_t=1.4, jaw_h=(5.0, 10.8), band_s=(32.6, 41.8),
          buccal_jaw_t=1.1, buccal_jaw_h=(4.8, 7.2), rib_fraction=0.35, n_ribs=4, interference=0.15)
