<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Incisor attachment analysis (2026-09-22)

Why the teeth attachment needs an incisal stop and an angled pad, and why straight wings do not
follow the lower arch. Measured on the lower-arch scan
[`../../scan/teeth_02.stl`](../../scan/teeth_02.stl); the findings shaped V5 of the teeth
attachment (see [`../../parts.yaml`](../../parts.yaml), `inspired_by`) and motivated the molar-clamp
concept study ([`../molar_clamp_concepts/`](../molar_clamp_concepts/README.md)).

| File | What it shows |
|---|---|
| [`stop_and_pad_explained.png`](stop_and_pad_explained.png) | (1) A front-to-back section of a lower central incisor: the crown is a wedge, 2.1 mm thick at the edge and 6.6 mm at the base, with a 3° labial and a 33° lingual face. (2) Flat jaws squeezing that wedge push the part up and off the tooth unless the friction coefficient reaches tan 33° = 0.65. (3) The fix: an incisal stop that takes the push-off force and a pad lying on the 33° lingual slope. |
| [`arch_vs_wings.png`](arch_vs_wings.png) | The lower arch from above: a straight wing fits only near the midline (stand-off from the teeth 1.3, 2.9 and 9.4 mm at ±8, ±13 and ±19 mm), so a wing past the canine has to follow the arch. It also marks the screw positions (±3.2 mm, central incisors; ±18.9 mm, first premolars) and shows the premolar cusps 1.5–2 mm above the incisal plane. |
| [`design_review_2026-09-22.svg`](design_review_2026-09-22.svg) | Review markup of the incisor attachment (Inkscape; screenshots with notes): thicker where the screws sit so the part does not bend, a slightly angled and ~2 mm thicker section, a 7.5 mm dimension. |

Provenance: the two PNGs were made with an ad-hoc script that was not kept (same palette as the
concept study, one day earlier); they are kept as figures. If they need to change, re-create them
from the scan with the concept-study helpers (`../molar_clamp_concepts/scripts/common.py`).
