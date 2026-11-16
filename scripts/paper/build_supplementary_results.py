#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Build the release folder ``experiments/icorr2027/supplementary_results/`` from the JNER export.

The JNER shortening round (2026-11-11, the authors) and the split into two papers
(2026-11-12) moved the items below out of Additional file 1 Part B into the public release: the paper states each outcome in one
sentence and cites this folder. Per item the folder holds

* ``<label>.pdf``: the figure or table with its full caption, sources, snapshot and code
  version (the export's ``captioned/`` PDF, the view's generated caption);
* ``<label>.csv``: the data behind it (the figure's ``source_data.csv`` or the table's CSV;
  first line: a provenance comment);

plus ``index.csv`` (item, files, claim label of the paper, store tables, what it shows) and a
``README.md``. Items, claims and the one-line descriptions are fixed in ``RELEASE`` below;
every file comes from the export (``biomech export --paper paper_dataset.yaml``), never from
hand edits.

A file is rewritten only when its content changed (CSV: the rows after the provenance line;
PDF: the text before the provenance line), so a re-export with unchanged results leaves the
repository clean.

    software/jtsmm/.venv/bin/python scripts/paper/build_supplementary_results.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "data/analysis/icorr2027/paper_dataset/journal"
OUT = ROOT / "experiments/icorr2027/supplementary_results"

# (export label, claim label of the paper, one line on what it shows); paper order A, B, C
RELEASE = [
    ("suppTabAcquisition", "methods (X11)",
     "Acquisition quality per take: camera frame drops, camera time-base residual, lag of the "
     "jaw IMU behind the optics, epipolar error of the stereo calibration in use."),
    ("suppLowpassBound", "S2",
     "Peak opening angular velocity under the primary pose low-pass and the low-pass "
     "sensitivity runs: the velocity requirement is a lower bound."),
    ("suppBandwidth", "X1",
     "Jaw angular-velocity bandwidth per task and participant: raw IMU power quantiles and the "
     "part confirmed by the IMU-optical coherence."),
    ("suppNullVariants", "C1",
     "Hinge residual against every noise-only null (bench, in vivo, sensitivity nulls, hinge at "
     "the joint) and the condyle-based outcomes under the plausibility variants."),
    ("suppImpedance", "methods (D21)",
     "Electrode-skin impedance per channel from the check before each take, and against the "
     "rest signal level of the motion take."),
    ("tabImpedance", "methods (D21)",
     "Correlations between the pre-take electrode impedance and the rest signal of the motion "
     "take (pooled, within participants, within muscles)."),
    ("suppActivationMap", "S1",
     "Median normalized EMG envelope per task and muscle over the participants."),
    ("suppVariability", "S2",
     "Opening-angle waveform similarity (CMC) within and between participants, and the cycles "
     "needed for a stable participant mean."),
    ("suppUsabilityFlow", "C2",
     "Channel-cycles through the gates of the onset analysis (first-onset rule): the "
     "denominator of the C2 test."),
    ("suppDetectors", "C2",
     "Opening lead under every combination of onset detector, mechanical onset and onset rule."),
    ("suppReleaseSweep", "X3",
     "Causal closer-release detector over its drop threshold and hold time: detection above "
     "chance and decision time."),
    ("suppHTwoPower", "X3",
     "Simulated power of a future confirmatory closer-release test against the number of "
     "participants."),
    ("suppCompute", "X4",
     "Simulated real time: compute per 20-ms chunk, its ratio to signal time, the one-off "
     "calibration and the decision delay."),
    ("tabRequirementsIntent", "C2, X3",
     "Requirements from muscle timing and intent: leads, closer release and decoding relative "
     "to the jaw motion, with the shift null."),
    ("timingCues", "C2",
     "Every timing cue relative to the motion onset (opener and closer cues, causal IMU "
     "detector) and the lag from EMG to the acceleration it drives."),
    ("tabSequence", "X6",
     "Cohort tests of the order of muscle and motion events with supporting and structure rows "
     "and the decision of the hypothesis family."),
    ("release", "X3",
     "Causal closer-release detector: detection against chance, anticipation and false "
     "releases offline and in simulated real time, every setting."),
    ("online", "X4",
     "Simulated real-time decoding: balanced accuracy, latency, false activations at rest and "
     "compute per chunk."),
    ("tabDecodingRealtime", "X4",
     "Seven-class decoding in simulated real time: balanced accuracy, latency, false "
     "activations, comparison with IMU alone."),
    ("suppAblation", "X10",
     "Decoding with the EMG features of one muscle group left out, per participant."),
    ("tabCameraFree", "X5",
     "Camera-free estimates per step: opening and incisor RMSE against the optics with the "
     "verdict against the bounds and the paired contrasts."),
    ("robustnessRuns", "S2",
     "Opening lead and closer release in the primary analysis and in each sensitivity run of "
     "the robustness table."),
]

# Data files of the export's `additional` list released as they are (label, claim, shows)
RELEASE_DATA = [
    ("workspaceEnvelope", "requirements",
     "Coordinates of the cohort workspace envelope of the incisal point in the sagittal and the "
     "frontal plane (the blue area of the requirements figure (a, b)), mm from the rest position."),
]

README = """# Supplementary results of the JNER data-set paper

Figures and tables that the paper states in one sentence each and cites from here
(\"supplementary_results\" of the public release). Generated from the paper export by
`scripts/paper/build_supplementary_results.py`; do not edit by hand.

- `index.csv`: one row per item: `item` (the paper's label), `figure` (PDF), `data` (CSV),
  `claim` (the paper's claim label; \"methods\" = a quality check of the methods, with the
  analysis-plan entry), `sources` (store tables of the analysis pipeline), `shows`.
- `<item>.pdf`: the figure or table with its full caption; the line under the caption names
  the store tables, the snapshot and the code version it was exported from.
- `<item>.csv`: the data behind it. The first line is a provenance comment (`#`); figures
  are in long format (`panel`, `series`, `kind`, `x`, `y`, ...), tables as printed.
- Data files without a figure (`figure` empty in the index), e.g. `workspaceEnvelope.csv`:
  the coordinates as the paper cites them.

Population: the 13 adult participants (pseudonymised P1 to P13, store `reporting_population`, snapshot
`{snapshot}`). Claim labels and the analysis plan: `experiments/icorr2027/analysis_plan.md`.

## Data snapshot

Every item was computed from one frozen snapshot of the processed data:

- snapshot: `{snapshot}` (store `{variant}`)
- bundle file: `{bundle_file}`
- checksum of the content (sha256 over the bundle's manifest, which lists the sha256 of every
  table, and its provenance): `{bundle_sha256}`
- checksum of the bundle file (sha256): `{file_sha256}`

The data set is not part of this release. A copy of the bundle can be checked against these
values with `biomech bundle --verify <file>`.
"""


def _csv_body(f: Path) -> str:
    """The CSV without its provenance comment (first line, when it starts with #)."""
    if not f.exists():
        return ""
    t = f.read_text(encoding="utf-8")
    return t.split("\n", 1)[-1] if t.startswith("#") else t


def _pdf_text(f: Path) -> str:
    """The PDF's text without the provenance line (its code version changes every export)."""
    if not f.exists():
        return ""
    t = subprocess.run(["pdftotext", str(f), "-"], capture_output=True, text=True).stdout
    # whitespace normalized: the layout of the text shifts with the length of that line
    return " ".join(re.split(r"·\s*Sources:", t, maxsplit=1)[0].split()).rsplit(" ", 1)[0]


FORCE_PDF = False       # --force-figures: a design change leaves the PDF text unchanged


def _copy_if_changed(src: Path, dst: Path, same) -> bool:
    if dst.exists() and same(src) == same(dst) and not (FORCE_PDF and dst.suffix == ".pdf"):
        return False
    shutil.copyfile(src, dst)
    return True


def _write_if_changed(dst: Path, text: str) -> bool:
    if dst.exists() and dst.read_text(encoding="utf-8") == text:
        return False
    dst.write_text(text, encoding="utf-8")
    return True


def build(export: Path, out: Path) -> list[str]:
    manifest = json.loads((export / "manifest.json").read_text())
    by_label = {it["label"]: it for it in manifest["items"]}
    missing = [lab for lab, _, _ in RELEASE if lab not in by_label]
    if missing:
        raise SystemExit(f"not in the export: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    changed, rows = [], []
    for lab, claim, shows in RELEASE:
        it = by_label[lab]
        files = list(it["files"])
        pdf = next(f for f in files if f.startswith("captioned/"))
        data = next(f for f in files if f.endswith(".csv"))
        if _copy_if_changed(export / pdf, out / f"{lab}.pdf", _pdf_text):
            changed.append(f"{lab}.pdf")
        if _copy_if_changed(export / data, out / f"{lab}.csv", _csv_body):
            changed.append(f"{lab}.csv")
        rows.append({"item": lab, "figure": f"{lab}.pdf", "data": f"{lab}.csv",
                     "kind": it["kind"], "claim": claim,
                     "sources": "; ".join(it.get("sources") or []), "shows": shows})
    extra = {a["label"]: a for a in manifest.get("additional", [])}
    for lab, claim, shows in RELEASE_DATA:
        if lab not in extra or "file" not in extra[lab]:
            raise SystemExit(f"not in the export's additional files: {lab}")
        a = extra[lab]
        if _copy_if_changed(export / a["file"], out / f"{lab}.csv", _csv_body):
            changed.append(f"{lab}.csv")
        rows.append({"item": lab, "figure": "", "data": f"{lab}.csv", "kind": "data",
                     "claim": claim, "sources": "; ".join(a.get("sources") or []), "shows": shows})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    if _write_if_changed(out / "index.csv", buf.getvalue()):
        changed.append("index.csv")
    prov = manifest.get("provenance") or {}
    snap = prov.get("snapshot") or "r16a"
    bundle = prov.get("bundle") or {}
    readme = README.format(snapshot=snap, variant=prov.get("variant") or "primary",
                           bundle_file=bundle.get("file") or "not recorded",
                           bundle_sha256=bundle.get("bundle_sha256") or "not recorded",
                           file_sha256=bundle.get("sha256") or "not recorded")
    if _write_if_changed(out / "README.md", readme):
        changed.append("README.md")
    keep = {lab for lab, _, _ in RELEASE} | {lab for lab, _, _ in RELEASE_DATA}
    stale = sorted(p.name for p in out.iterdir() if p.suffix in (".pdf", ".csv")
                   and p.stem not in keep and p.name != "index.csv")
    if stale:
        raise SystemExit(f"files of items no longer released (remove them by hand): {stale}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", type=Path, default=EXPORT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force-figures", action="store_true",
                    help="copy every PDF again (after a change of the figure design, which "
                         "the text comparison cannot see)")
    a = ap.parse_args()
    global FORCE_PDF
    FORCE_PDF = a.force_figures
    changed = build(a.export, a.out)
    print(f"{a.out}: {len(RELEASE) + len(RELEASE_DATA)} items; changed: {', '.join(changed) or 'nothing'}")


if __name__ == "__main__":
    main()
