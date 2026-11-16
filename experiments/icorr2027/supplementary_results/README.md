<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Supplementary results of the JNER data-set paper

Figures and tables that the paper states in one sentence each and cites from here
("supplementary_results" of the public release). Generated from the paper export by
`scripts/paper/build_supplementary_results.py`; do not edit by hand.

- `index.csv`: one row per item: `item` (the paper's label), `figure` (PDF), `data` (CSV),
  `claim` (the paper's claim label; "methods" = a quality check of the methods, with the
  analysis-plan entry), `sources` (store tables of the analysis pipeline), `shows`.
- `<item>.pdf`: the figure or table with its full caption; the line under the caption names
  the store tables, the snapshot and the code version it was exported from.
- `<item>.csv`: the data behind it. The first line is a provenance comment (`#`); figures
  are in long format (`panel`, `series`, `kind`, `x`, `y`, ...), tables as printed.
- Data files without a figure (`figure` empty in the index), e.g. `workspaceEnvelope.csv`:
  the coordinates as the paper cites them.

Population: the 13 adult participants (pseudonymised P1 to P13, store `reporting_population`, snapshot
`r17a`). Claim labels and the analysis plan: `experiments/icorr2027/analysis_plan.md`.

## Data snapshot

Every item was computed from one frozen snapshot of the processed data:

- snapshot: `r17a` (store `reporting_population`)
- bundle file: `bundle_r17a.h5`
- checksum of the content (sha256 over the bundle's manifest, which lists the sha256 of every
  table, and its provenance): `47f8694465fbac002dd97f625e47b49eb4410d98c054b1bd86c82c299615dcee`
- checksum of the bundle file (sha256): `39b1e32992791802f03336b51d45bad94aba0fd9dc0c5d798d8b129d3023bb73`

The data set is not part of this release. A copy of the bundle can be checked against these
values with `biomech bundle --verify <file>`.
