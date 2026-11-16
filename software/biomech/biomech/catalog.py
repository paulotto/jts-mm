# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Metric/signal catalog (``biomech/etc/catalog.yaml``): labels, symbols, units, rounding.

Tables, the TikZ exporter and the viewer all format numbers through here, so a
metric is named and rounded identically in every output (an08 §3).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from biomech.config import PACKAGE_DIR

CATALOG_PATH = PACKAGE_DIR / "etc" / "catalog.yaml"


@dataclass(frozen=True)
class Entry:
    name: str
    label: str
    symbol: str
    unit: str
    decimals: int
    desc: str
    range: tuple[float, float] | None = None    # plausible values (biomech check-outputs)

    def fmt(self, value: float) -> str:
        return "--" if value != value else f"{value:.{self.decimals}f}"

    def latex_header(self) -> str:
        """``Label / \\si{unit}`` style column header for booktabs tables."""
        return f"{self.label} (\\si{{{self.unit}}})" if self.unit else self.label


@lru_cache(maxsize=4)
def load(path: str | Path = CATALOG_PATH) -> dict[str, Entry]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {
        name: Entry(name=name, label=str(d.get("label", name)), symbol=str(d.get("symbol", "")),
                    unit=str(d.get("unit", "")), decimals=int(d.get("decimals", 2)),
                    desc=str(d.get("desc", "")),
                    range=tuple(float(v) for v in d["range"]) if d.get("range") else None)
        for name, d in doc.items()
    }


# Unit by column-name suffix for names without an entry (longest suffix first).
SUFFIX_UNITS = (("_mm_per_deg2", "\\milli\\metre\\per\\degree\\squared"),
                ("_mm_per_deg", "\\milli\\metre\\per\\degree"),
                ("_deg_per_min", "\\degree\\per\\minute"), ("_mm_s", "\\milli\\metre\\per\\second"),
                ("_ms2", "\\metre\\per\\second\\squared"), ("_dps", "\\degree\\per\\second"),
                ("_per_min", "\\per\\minute"), ("_mm", "\\milli\\metre"), ("_deg", "\\degree"),
                ("_hz", "\\hertz"), ("_ms", "\\milli\\second"), ("_uv", "\\micro\\volt"),
                ("_db", "\\decibel"), ("_pct", "\\percent"), ("_s", "\\second"))
_WORDS = {"rms": "RMS", "sd": "SD", "ci": "CI", "lo": "low", "hi": "high", "tmj": "TMJ",
          "imu": "IMU", "emg": "EMG", "mvc": "MVC", "icc": "ICC", "mdc": "MDC", "cv": "CV",
          "ba": "BA", "fha": "FHA", "lf": "LF", "q1": "Q1", "q3": "Q3", "p50": "P50",
          "p95": "P95", "p99": "P99", "f95": "f95", "f99": "f99", "f50": "f50", "n": "n"}


def guess(name: str) -> Entry:
    """Fallback entry: unit from the name's suffix, label from its words."""
    unit, stem = "", name
    for suf, u in SUFFIX_UNITS:
        if name.endswith(suf) and len(name) > len(suf):
            unit, stem = u, name[: -len(suf)]
            break
    words = [_WORDS.get(w, w) for w in stem.split("_") if w]
    label = " ".join(words)
    label = label[:1].upper() + label[1:] if label and not label[0].isupper() else label
    return Entry(name, label or name, "", unit, 2 if unit else 3, "")


_MISSES: set[str] = set()


def get(name: str) -> Entry:
    """Catalog entry; names without one get :func:`guess` (unit from the suffix) and are
    recorded (:func:`misses`: ``check-outputs`` fails a paper item with a guessed label).
    A pivoted name ``<channel>|<feature>`` (decoder features) is composed: "<channel
    label>, <feature label>" with the feature's unit and rounding."""
    e = load().get(name)
    if e is None and "|" in name:
        e = _composed(name)
    if e is None:
        _MISSES.add(name)
        return guess(name)
    return e


def _composed(name: str) -> Entry | None:
    from biomech.viz.views import vocab as V

    left, _, right = name.partition("|")
    cat = load()
    f = cat.get(right)
    if f is None:
        return None
    if left in cat:
        lab = cat[left].label
    elif left in V.labels("muscle"):
        lab = V.term("muscle", left)
        lab = lab[:1].upper() + lab[1:]
    else:
        return None
    fl = f.label
    if len(fl) > 1 and fl[1].islower():          # "Mean" -> "mean"; "SD" stays
        fl = fl[:1].lower() + fl[1:]
    return Entry(name, f"{lab}, {fl}", "", f.unit, f.decimals, f"{f.desc} – {lab}")


def misses(reset: bool = False) -> set[str]:
    """Column names looked up without a catalog entry since the last reset."""
    out = set(_MISSES)
    if reset:
        _MISSES.clear()
    return out
