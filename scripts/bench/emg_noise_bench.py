#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shorted-input noise bench check B2 (dataset analysis plan, "Shorted-input noise bench check
(B2)", written before the measurement on 2026-11-04).

Input: one folder with the tables of ``jtsmm emg check --no-impedance`` saved with ``tee``::

    <folder>/check_<n>.txt        one capture each (inputs shorted, joined to the DRL input)

Per capture and channel the check's 20-450 Hz noise RMS with the mains removed (``noise µV``)
and its line peak (``50Hz µV``) are read from the table. The analysis is the one fixed in the
plan: per channel the median and range over the captures; with ``--store`` (a paper-store
snapshot directory) the same quantities of the pre-take checks of the participants
(``qc:emg_check``, motion take) per channel, the ratio bench / in vivo (median over
participants) and the in-vivo noise not explained by the board,
sqrt(max(n_vivo^2 - n_bench^2, 0)) (noise adds in quadrature). The reading: a channel whose
bench noise exceeds its check limit cannot meet the limit on this board; below half the limit,
the in-vivo failures come from electrodes, cables and participant. Descriptive; no test.

Output: ``hardware/bench/tables/emg_noise_captures.csv`` and ``emg_noise_channels.csv`` (one
row per channel plus ``channel = all``) for the paper numbers (``source: bench:<table>``).

    software/jtsmm/.venv/bin/python scripts/bench/emg_noise_bench.py \\
        data/bench/emg_noise_2026-11-04 \\
        --store data/analysis/icorr2027/sensitivity/reporting_population/snapshots/r15a
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "hardware" / "bench" / "tables"
FILE = re.compile(r"^check_(?P<rep>[A-Za-z0-9]+)\.txt$")
# One channel row of jtsmm.emg.check.format_check_table:
#   CH  name  leadoff  |Z| kOhm  noise uV  50Hz uV  status
ROW = re.compile(r"^\s*(?P<ch>[1-6])\s+(?P<name>\S+)\s+(?P<leadoff>PASS|FAIL)\s+(?P<z>\S+)\s+"
                 r"(?P<noise>\S+)\s+(?P<line>\S+)\s+(?P<status>.*)$")
CHANNELS = (1, 2, 3, 4, 5, 6)


def _num(x: str) -> float:
    try:
        return float(x)
    except ValueError:                      # "n/a"
        return float("nan")


def read_captures(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.txt")):
        m = FILE.match(p.name)
        if m is None:
            print(f"skipped (name): {p.name}", file=sys.stderr)
            continue
        found = 0
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            r = ROW.match(line)
            if r is None:
                continue
            found += 1
            rows.append({"rep": m["rep"], "channel": int(r["ch"]), "name": r["name"],
                         "noise_uv": _num(r["noise"]), "line_peak_uv": _num(r["line"]),
                         "status": r["status"].strip(), "file": p.name})
        if found != len(CHANNELS):
            raise ValueError(f"{p.name}: {found} channel rows, expected {len(CHANNELS)} "
                             "(saved the whole table of jtsmm emg check?)")
    cols = ["rep", "channel", "name", "noise_uv", "line_peak_uv", "status", "file"]
    return pd.DataFrame(rows, columns=cols)


def in_vivo(store: Path) -> pd.DataFrame:
    """The pre-take checks of the participants (motion take): pid, channel, noise, line peak."""
    from biomech.io.session import MUSCLES
    from biomech.results import ResultStore

    chk = ResultStore(store).read_table("qc", "emg_check")
    chk = chk[chk["take"] == "motion"]
    return pd.DataFrame({"pid": chk["pid"].to_numpy(),
                         "channel": chk["muscle"].map(lambda m: MUSCLES.index(m) + 1).to_numpy(),
                         "noise_uv": chk["noise_rms_uv"].to_numpy(float),
                         "line_peak_uv": chk["line_peak_uv"].to_numpy(float)})


def channel_table(cap: pd.DataFrame, vivo: pd.DataFrame | None) -> pd.DataFrame:
    from jtsmm.emg.check import NOISE_LIMIT_UV

    rows = []
    for ch, g in cap.groupby("channel"):
        n = g["noise_uv"].to_numpy(float)
        lim = float(NOISE_LIMIT_UV.get(int(ch), 10.0))
        row = {"channel": str(ch), "n_captures": int(np.isfinite(n).sum()),
               "noise_median_uv": float(np.nanmedian(n)), "noise_min_uv": float(np.nanmin(n)),
               "noise_max_uv": float(np.nanmax(n)),
               "line_peak_median_uv": float(np.nanmedian(g["line_peak_uv"])),
               "limit_uv": lim}
        row["above_limit"] = bool(row["noise_median_uv"] > lim)
        row["below_half_limit"] = bool(row["noise_median_uv"] < lim / 2)
        if vivo is not None:
            v = vivo[vivo["channel"] == ch]["noise_uv"].to_numpy(float)
            v = v[np.isfinite(v)]
            b = row["noise_median_uv"]
            row.update({
                "n_vivo": int(v.size),
                "vivo_noise_median_uv": float(np.median(v)) if v.size else np.nan,
                "vivo_above_limit_n": int((v > lim).sum()),
                "ratio_bench_vivo": float(np.median(b / v)) if v.size else np.nan,
                "vivo_excess_median_uv": float(np.median(np.sqrt(np.maximum(v ** 2 - b ** 2, 0))))
                if v.size else np.nan})
        rows.append(row)
    per = pd.DataFrame(rows)
    allr = {"channel": "all", "n_captures": int(per["n_captures"].sum()),
            "noise_median_uv": float(per["noise_median_uv"].median()),
            "noise_min_uv": float(per["noise_median_uv"].min()),
            "noise_max_uv": float(per["noise_median_uv"].max()),
            "line_peak_median_uv": float(per["line_peak_median_uv"].median()),
            "above_limit_n": int(per["above_limit"].sum()),
            "below_half_limit_n": int(per["below_half_limit"].sum()), "n_channels": len(per)}
    if vivo is not None:
        allr.update({"n_vivo": int(per["n_vivo"].sum()),
                     "vivo_noise_median_uv": float(np.nanmedian(vivo["noise_uv"])),
                     "vivo_above_limit_n": int(per["vivo_above_limit_n"].sum()),
                     "ratio_bench_vivo": float(per["ratio_bench_vivo"].median()),
                     "vivo_excess_median_uv": float(per["vivo_excess_median_uv"].median())})
    return pd.concat([per, pd.DataFrame([allr])], ignore_index=True)


def report(cap: pd.DataFrame, tab: pd.DataFrame) -> str:
    out = [f"{cap['rep'].nunique()} captures, {cap['channel'].nunique()} channels"]
    vivo = "vivo_noise_median_uv" in tab
    out.append("\nCH     bench noise µV (range)   line µV   limit   " +
               ("in vivo µV   bench/vivo   excess µV" if vivo else ""))
    for _, r in tab.iterrows():
        ch = "all" if r["channel"] == "all" else f"CH{r['channel']}"
        lim = f"{r['limit_uv']:5.0f}" if pd.notna(r.get("limit_uv")) else "    -"
        s = (f"{ch:<6} {r['noise_median_uv']:6.2f} ({r['noise_min_uv']:.2f}-"
             f"{r['noise_max_uv']:.2f})   {r['line_peak_median_uv']:7.2f}   {lim}")
        if vivo:
            s += (f"   {r['vivo_noise_median_uv']:9.2f}   {r['ratio_bench_vivo']:10.2f}"
                  f"   {r['vivo_excess_median_uv']:9.2f}")
        out.append(s)
    a = tab[tab["channel"] == "all"].iloc[0]
    out.append(f"\nchannels with bench noise above their check limit: {int(a['above_limit_n'])}"
               f" of {int(a['n_channels'])}; below half the limit: "
               f"{int(a['below_half_limit_n'])}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EMG shorted-input noise bench check (B2)")
    ap.add_argument("folder", type=Path, help="folder with the saved check_<n>.txt tables")
    ap.add_argument("--store", type=Path, default=None,
                    help="paper-store snapshot directory: compare with the participants' "
                         "pre-take checks (qc:emg_check, motion take)")
    ap.add_argument("--tables", type=Path, default=TABLES,
                    help="where the macro tables go (default hardware/bench/tables)")
    args = ap.parse_args(argv)

    cap = read_captures(args.folder)
    if cap.empty:
        print(f"no captures in {args.folder}", file=sys.stderr)
        return 1
    vivo = in_vivo(args.store) if args.store is not None else None
    tab = channel_table(cap, vivo)
    args.tables.mkdir(parents=True, exist_ok=True)
    cap.to_csv(args.tables / "emg_noise_captures.csv", index=False)
    tab.to_csv(args.tables / "emg_noise_channels.csv", index=False)
    print(report(cap, tab))
    print(f"\nwritten: {args.tables / 'emg_noise_captures.csv'}, "
          f"{args.tables / 'emg_noise_channels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
