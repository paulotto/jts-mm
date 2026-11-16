#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Electrode-impedance scale bench check B1 (dataset analysis plan, "Electrode-impedance scale
bench check (B1)", written before the measurement on 2026-11-03).

Input: one folder with the JSON exports of ``jtsmm emg impedance --export`` and the resistor
values measured with a multimeter::

    <folder>/R<kΩ>_CH<n>_<I>nA_<a|b>.json     e.g. R22_CH3_6nA_a.json (resistor across CH3)
    <folder>/resistors.csv                    nominal_kohm, channel, multimeter_kohm
                                              (channel empty or "all": the same resistor
                                              was moved over every channel)

Per capture only the channel named in the file name is read (the others are open or powered
down). Only the repetitions ``a`` / ``b`` enter the analysis; other suffixes (e.g. ``pre1``)
are listed against them. The
analysis is the one fixed in the plan:

* per capture the relative error e = (|Z| - R) / R, R the multimeter value;
* per channel and lead-off current the scale k = sum(|Z| R) / sum(R^2) (least squares through
  the origin), the largest |e| and the repeatability (largest difference of the a / b
  captures of one resistor);
* across channels the median and range of k, and the decision rule: every channel
  |k - 1| <= 0.10 and every |e| <= 0.10 at the dataset's current (6 nA) -> the stored |Z|
  stand; otherwise a per-channel correction (|Z| / k) becomes an erratum. The rank analyses
  stay unchanged unless the channels' k differ by more than 0.10 from each other.

Output: ``<folder>/captures.csv`` (every capture) and two small tables for the paper numbers
(``source: bench:<table>`` in the numbers files), committed with the repository:
``hardware/bench/tables/emg_impedance_captures.csv`` and ``emg_impedance_channels.csv``
(one row per channel and current, plus a row ``channel = all`` with the cohort of channels
and the decision). Descriptive; no test.

    software/jtsmm/.venv/bin/python scripts/bench/emg_impedance_bench.py \\
        data/bench/emg_impedance_2026-11-03
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "hardware" / "bench" / "tables"
NAME = re.compile(r"^R(?P<r>\d+(?:\.\d+)?)k?_CH(?P<ch>[1-6])_(?P<i>\d+)nA_(?P<rep>[A-Za-z0-9]+)"
                  r"\.json$")
CHANNELS = (1, 2, 3, 4, 5, 6)
NOMINAL = (10.0, 22.0, 47.0, 100.0)       # kΩ, the plan's values
REPS = ("a", "b")
PRIMARY_NA = 6                            # the dataset's lead-off current
K_TOL = 0.10                              # decision rule: |k - 1| and |e| at most 10 %
SPREAD_TOL = 0.10                         # channels' k differ by more: rank analyses affected


def read_resistors(path: Path) -> pd.DataFrame:
    """``resistors.csv`` -> nominal_kohm, channel (int, or 0 for every channel),
    multimeter_kohm."""
    r = pd.read_csv(path, dtype=str).fillna("")
    missing = {"nominal_kohm", "channel", "multimeter_kohm"} - set(r.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    ch = r["channel"].str.strip().str.lower().str.removeprefix("ch")
    return pd.DataFrame({
        "nominal_kohm": r["nominal_kohm"].astype(float),
        "channel": [0 if c in ("", "all", "*") else int(c) for c in ch],
        # an empty cell (template not filled in yet) is a missing value, not an error here
        "multimeter_kohm": pd.to_numeric(r["multimeter_kohm"].str.strip().replace("", np.nan)),
    })


def true_value(res: pd.DataFrame, nominal: float, channel: int) -> float:
    """Multimeter value of the resistor used for ``nominal`` on ``channel`` (a row for that
    channel wins over a row for every channel); NaN when none is recorded."""
    m = res[np.isclose(res["nominal_kohm"], nominal) & res["multimeter_kohm"].notna()]
    for sel in (m[m["channel"] == channel], m[m["channel"] == 0]):
        if len(sel) == 1:
            return float(sel["multimeter_kohm"].iloc[0])
        if len(sel) > 1:
            raise ValueError(f"resistors.csv: {len(sel)} rows for {nominal:g} kΩ, CH{channel}")
    return float("nan")


def read_captures(folder: Path, res: pd.DataFrame, allow_nominal: bool) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.json")):
        m = NAME.match(p.name)
        if m is None:
            print(f"skipped (name): {p.name}", file=sys.stderr)
            continue
        nominal, ch = float(m["r"]), int(m["ch"])
        d = json.loads(p.read_text(encoding="utf-8"))
        meta = d.get("meta", {})
        i_na = int(m["i"])
        if meta.get("current_na") is not None and int(meta["current_na"]) != i_na:
            raise ValueError(f"{p.name}: file name says {i_na} nA, the export {meta['current_na']}")
        c = next((c for c in d.get("channels", []) if int(c.get("channel", -1)) == ch), None)
        if c is None:
            raise ValueError(f"{p.name}: no CH{ch} in the export")
        r_true = true_value(res, nominal, ch)
        r_source = "multimeter"
        if not np.isfinite(r_true):
            if not allow_nominal:
                raise ValueError(f"{p.name}: no multimeter value for {nominal:g} kΩ on CH{ch} "
                                 "in resistors.csv (--allow-nominal uses the nominal value)")
            r_true, r_source = nominal, "nominal"
        z = c.get("z_kohm")
        z = float(z) if z is not None else np.nan
        rows.append({
            "nominal_kohm": nominal, "channel": ch, "current_na": i_na, "rep": m["rep"],
            "r_true_kohm": r_true, "r_source": r_source, "z_kohm": z,
            "v_tone_uv": float(c.get("v_tone_uv", np.nan)), "quality": c.get("quality", ""),
            "note": c.get("note", ""), "rel_err": (z - r_true) / r_true, "file": p.name,
        })
    cols = ["nominal_kohm", "channel", "current_na", "rep", "r_true_kohm", "r_source", "z_kohm",
            "v_tone_uv", "quality", "note", "rel_err", "file"]
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["current_na", "channel", "nominal_kohm", "rep"], ignore_index=True)


def scale(z: np.ndarray, r: np.ndarray) -> float:
    """k = sum(|Z| R) / sum(R^2), least squares through the origin (finite pairs)."""
    ok = np.isfinite(z) & np.isfinite(r)
    return float(np.sum(z[ok] * r[ok]) / np.sum(r[ok] ** 2)) if ok.any() else float("nan")


def repeatability(g: pd.DataFrame) -> tuple[float, float]:
    """Largest |Z| difference between the captures of one resistor (kΩ, and % of R)."""
    d_k, d_p = [], []
    for _, s in g.groupby("nominal_kohm"):
        z = s["z_kohm"].to_numpy(float)
        z = z[np.isfinite(z)]
        if len(z) >= 2:
            d_k.append(float(z.max() - z.min()))
            d_p.append(100.0 * d_k[-1] / float(s["r_true_kohm"].iloc[0]))
    return (max(d_k), max(d_p)) if d_k else (np.nan, np.nan)


def channel_table(cap: pd.DataFrame) -> pd.DataFrame:
    """One row per lead-off current and channel, plus ``channel = all`` per current."""
    rows = []
    for i_na, gi in cap.groupby("current_na"):
        per = []
        for ch, g in gi.groupby("channel"):
            k = scale(g["z_kohm"].to_numpy(float), g["r_true_kohm"].to_numpy(float))
            e = g["rel_err"].to_numpy(float)
            rk, rp = repeatability(g)
            row = {
                "current_na": int(i_na), "channel": str(ch),
                "n_captures": int(len(g)), "n_missing_z": int(g["z_kohm"].isna().sum()),
                "n_values": int(g["nominal_kohm"].nunique()),
                "k": k, "k_dev_pct": 100.0 * (k - 1.0),
                "max_abs_err_pct": 100.0 * float(np.nanmax(np.abs(e))) if np.isfinite(e).any()
                else np.nan,
                "mean_err_pct": 100.0 * float(np.nanmean(e)) if np.isfinite(e).any() else np.nan,
                "repeat_max_kohm": rk, "repeat_max_pct": rp,
            }
            row["within_tol"] = bool(abs(row["k"] - 1.0) <= K_TOL
                                     and row["max_abs_err_pct"] <= 100.0 * K_TOL
                                     and row["n_missing_z"] == 0)
            per.append(row)
        rows += per
        ks = np.array([r["k"] for r in per], float)
        ok = all(r["within_tol"] for r in per)
        spread = float(np.nanmax(ks) - np.nanmin(ks)) if np.isfinite(ks).any() else np.nan
        rows.append({
            "current_na": int(i_na), "channel": "all",
            "n_captures": int(sum(r["n_captures"] for r in per)),
            "n_missing_z": int(sum(r["n_missing_z"] for r in per)),
            "n_values": int(gi["nominal_kohm"].nunique()), "n_channels": len(per),
            "k": float(np.nanmedian(ks)), "k_min": float(np.nanmin(ks)),
            "k_max": float(np.nanmax(ks)), "k_spread": spread,
            "k_dev_pct": 100.0 * (float(np.nanmedian(ks)) - 1.0),
            "k_dev_max_pct": 100.0 * float(np.nanmax(np.abs(ks - 1.0))),
            "max_abs_err_pct": float(np.nanmax([r["max_abs_err_pct"] for r in per])),
            "mean_err_pct": 100.0 * float(np.nanmean(gi["rel_err"])),
            "repeat_max_kohm": float(np.nanmax([r["repeat_max_kohm"] for r in per])),
            "repeat_max_pct": float(np.nanmax([r["repeat_max_pct"] for r in per])),
            "within_tol": ok,
            # the rule is for the dataset's current; other currents are reported only
            "decision": ("n/a (not the dataset's current)" if int(i_na) != PRIMARY_NA
                         else "stands" if ok else "erratum: divide |Z| by the channel's k"),
            "rank_affected": bool(spread > SPREAD_TOL),
        })
    return pd.DataFrame(rows)


def completeness(cap: pd.DataFrame) -> list[str]:
    """Captures the plan asks for at the dataset's current that are not in the folder."""
    prim = cap[cap["current_na"] == PRIMARY_NA]
    have = set(zip(prim["nominal_kohm"], prim["channel"], prim["rep"]))
    return [f"R{r:g}_CH{c}_{PRIMARY_NA}nA_{p}" for r in NOMINAL for c in CHANNELS for p in REPS
            if (r, c, p) not in have]


def report(cap: pd.DataFrame, tab: pd.DataFrame, missing: list[str]) -> str:
    out = [f"{len(cap)} captures, {cap['channel'].nunique()} channels, "
           f"currents {sorted(cap['current_na'].unique().tolist())} nA"]
    if (cap["r_source"] == "nominal").any():
        out.append(f"NOTE: {int((cap['r_source'] == 'nominal').sum())} captures use the "
                   "nominal resistor value (no multimeter value)")
    if missing:
        out.append(f"missing at {PRIMARY_NA} nA ({len(missing)}): {', '.join(missing)}")
    for i_na, g in tab.groupby("current_na"):
        out.append(f"\n{i_na} nA     k     k-1 %   max|e| %   mean e %   repeat kΩ (%)   ok")
        for _, r in g.iterrows():
            ch = "all" if r["channel"] == "all" else f"CH{r['channel']}"
            out.append(f"  {ch:<6} {r['k']:6.3f} {r['k_dev_pct']:+7.1f} "
                       f"{r['max_abs_err_pct']:9.1f} {r['mean_err_pct']:+9.1f}   "
                       f"{r['repeat_max_kohm']:6.2f} ({r['repeat_max_pct']:4.1f})   "
                       f"{'yes' if r['within_tol'] else 'NO'}")
        a = g[g["channel"] == "all"].iloc[0]
        out.append(f"  k median {a['k']:.3f}, range {a['k_min']:.3f}-{a['k_max']:.3f} "
                   f"(spread {a['k_spread']:.3f}); decision: {a['decision']}"
                   + ("; channels' k differ by > 0.10: rank analyses affected"
                      if a["rank_affected"] else ""))
    return "\n".join(out)


def store_sensitivity(store: Path, cap: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    """How the bench error would move the stored pre-take impedances (descriptive; not the
    erratum, which the decision rule decides): per take the channels above the contact limit
    and the median |Z| as recorded, divided by the channel's scale k, and through the channel's
    calibration curve (true R interpolated from the mean |Z| of its a / b captures, piecewise
    linear, at the dataset's current). ``store``: a snapshot directory of the paper store."""
    from biomech.io.session import MUSCLES
    from biomech.results import ResultStore
    from jtsmm.emg.impedance import CONTACT_Z_MAX_KOHM

    chk = ResultStore(store).read_table("qc", "emg_check")
    t6 = tab[(tab["current_na"] == PRIMARY_NA) & (tab["channel"] != "all")]
    k = {int(c): float(v) for c, v in zip(t6["channel"], t6["k"])}
    c6 = (cap[(cap["current_na"] == PRIMARY_NA) & cap["rep"].isin(REPS)]
          .groupby(["channel", "nominal_kohm"])
          .agg(z=("z_kohm", "mean"), r=("r_true_kohm", "first")).reset_index())

    def curve(ch: int, z: float) -> float:
        g = c6[c6["channel"] == ch].sort_values("z")
        return float(np.interp(z, g["z"], g["r"])) if np.isfinite(z) and len(g) else np.nan

    rows = []
    for take, t in chk.groupby("take", sort=False):
        ch = t["muscle"].map(lambda m: MUSCLES.index(m) + 1)
        z = t["z_kohm"].to_numpy(float)
        zk = z / ch.map(k).to_numpy(float)
        zc = np.array([curve(c, v) for c, v in zip(ch, z)])
        row = {"take": take, "n": int(np.isfinite(z).sum()), "limit_kohm": CONTACT_Z_MAX_KOHM}
        for name, v in (("recorded", z), ("scale", zk), ("curve", zc)):
            row[f"above_{name}_n"] = int((v > CONTACT_Z_MAX_KOHM).sum())
            row[f"median_{name}_kohm"] = float(np.nanmedian(v))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EMG impedance scale bench check (B1)")
    ap.add_argument("folder", type=Path, help="folder with the JSON exports and resistors.csv")
    ap.add_argument("--tables", type=Path, default=TABLES,
                    help="where the macro tables go (default hardware/bench/tables)")
    ap.add_argument("--store", type=Path, default=None,
                    help="paper-store snapshot directory: also write how the bench error would "
                         "move its stored pre-take impedances (emg_impedance_sensitivity.csv)")
    ap.add_argument("--allow-nominal", action="store_true",
                    help="use the nominal value where resistors.csv has no multimeter value")
    args = ap.parse_args(argv)

    res = read_resistors(args.folder / "resistors.csv")
    cap = read_captures(args.folder, res, args.allow_nominal)
    if cap.empty:
        print(f"no captures in {args.folder}", file=sys.stderr)
        return 1
    # only the planned repetitions a / b enter the analysis; other captures (e.g. pre1, a
    # settings comparison before the series) are listed against them in the report
    extra = cap[~cap["rep"].isin(REPS)]
    tab = channel_table(cap[cap["rep"].isin(REPS)])
    cap.to_csv(args.folder / "captures.csv", index=False)
    args.tables.mkdir(parents=True, exist_ok=True)
    cap.drop(columns=["note"]).to_csv(args.tables / "emg_impedance_captures.csv", index=False)
    tab.to_csv(args.tables / "emg_impedance_channels.csv", index=False)
    if args.store is not None:
        sens = store_sensitivity(args.store, cap, tab)
        sens.to_csv(args.tables / "emg_impedance_sensitivity.csv", index=False)
        print(f"\nstored pre-take |Z| of {args.store.name} under the bench error:")
        print(sens.to_string(index=False))
    print(report(cap[cap["rep"].isin(REPS)], tab, completeness(cap)))
    if not extra.empty:
        ref = cap[cap["rep"].isin(REPS)].groupby(["nominal_kohm", "channel", "current_na"]).z_kohm
        print("\nother captures (not in the analysis), against the mean of a / b:")
        for _, r in extra.iterrows():
            m = ref.mean().get((r.nominal_kohm, r.channel, r.current_na), np.nan)
            print(f"  {r.file}: |Z| {r.z_kohm:.3f} kΩ vs {m:.3f} kΩ ({r.z_kohm - m:+.3f})")
    print(f"\nwritten: {args.folder / 'captures.csv'}, "
          f"{args.tables / 'emg_impedance_captures.csv'}, "
          f"{args.tables / 'emg_impedance_channels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
