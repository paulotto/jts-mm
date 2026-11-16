# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Study configuration: package defaults overlaid with one study YAML.

The study file may override any key that exists in ``biomech/etc/biomech.yaml``
(package data); unknown keys and type changes raise, so a typo cannot silently
fall back to a default. Unknown *stage* blocks are allowed here (plugins bring
their own) and are checked against the stage registry by
:func:`biomech.pipeline.validate_config`. Relative paths in the study file
resolve against the study file's directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from biomech.paths import DATA_DIR_ENV, resolve_data_dir

PACKAGE_DIR = Path(__file__).resolve().parent          # biomech/
PACKAGE_ROOT = PACKAGE_DIR.parent                        # software/biomech (in a checkout)
DEFAULTS_PATH = PACKAGE_DIR / "etc" / "biomech.yaml"
REPO_ROOT = PACKAGE_ROOT.parents[1]


class ConfigError(ValueError):
    """The study configuration is inconsistent or incomplete."""


def _merge(base: dict[str, Any], over: dict[str, Any], where: str = "") -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in over.items():
        path = f"{where}.{key}" if where else key
        if key not in out:
            # New stages may bring their own parameter blocks; studies may add variants and
            # add keys to the package's variants (e.g. skip_stages).
            if where in ("stages", "sensitivity") or where.startswith("sensitivity."):
                out[key] = copy.deepcopy(val)
                continue
            raise ConfigError(f"unknown config key: {path}")
        if isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _merge(out[key], val, path)
        else:
            _check_type(out[key], val, path)
            out[key] = copy.deepcopy(val)
    return out


def _kind(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _check_type(default: Any, val: Any, path: str) -> None:
    """An override keeps the default's kind; ``None`` defaults accept anything and a
    scalar default may be switched off with ``null``."""
    kd, kv = _kind(default), _kind(val)
    if kd == "none" or kd == kv or (kv == "none" and kd != "dict"):
        return
    raise ConfigError(f"config key {path}: expected {kd} (default {default!r}), got {val!r}")


def dotted_overrides(sets: dict[str, Any]) -> dict[str, Any]:
    """{"a.b.c": v} -> {"a": {"b": {"c": v}}} (merged with the same type checks as a study file)."""
    out: dict[str, Any] = {}
    for key, val in sets.items():
        parts = str(key).split(".")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
            if not isinstance(d, dict):
                raise ConfigError(f"override {key}: {p} is set twice")
        d[parts[-1]] = val
    return out


_PACKAGE_DEFAULTS: dict[str, Any] | None = None


def package_default(*keys: str) -> Any:
    """One value of the package defaults (``etc/biomech.yaml``), for module constants that
    must agree with a stage parameter (e.g. ``package_default("stages", "qc", "imu",
    "gap_s")``). A study override does not reach these constants."""
    global _PACKAGE_DEFAULTS
    if _PACKAGE_DEFAULTS is None:
        _PACKAGE_DEFAULTS = yaml.safe_load(DEFAULTS_PATH.read_text())
    v: Any = _PACKAGE_DEFAULTS
    for k in keys:
        v = v[k]
    return copy.deepcopy(v)


def stable_hash(obj: Any, n: int = 16) -> str:
    """Order-independent hash of a JSON-able object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


@dataclass
class StudyConfig:
    study: str
    seed: int
    data_dir: Path
    out_dir: Path
    cohort_path: Path
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    # Stage blocks the study file itself sets (validated against the registry; the
    # package defaults may carry blocks of plugins that are not installed).
    study_stage_keys: tuple[str, ...] = ()
    # Read results from here instead of out_dir (a frozen snapshot, results/snapshot.py);
    # None = the live store. Outputs (exports, videos) still go under out_dir.
    store_dir: Path | None = None
    snapshot: str | None = None
    # The sensitivity variant whose store ``out_dir`` is (``<primary>/sensitivity/<name>``,
    # biomech.sensitivity.variant_config); None = the primary analysis.
    variant: str | None = None

    @property
    def read_dir(self) -> Path:
        """Root of the result store that readers (views, exports, checks) use."""
        return self.store_dir if self.store_dir is not None else self.out_dir

    @property
    def groups(self) -> tuple[str, ...]:
        """Stage groups this study runs (``pipeline.groups``; default ``dataset``): the
        dataset analysis and the system validation are separate studies on one code base."""
        g = (self.raw.get("pipeline") or {}).get("groups") or ["dataset"]
        return tuple(str(x) for x in g)

    def with_snapshot(self, name: str | None) -> "StudyConfig":
        """A copy that reads from snapshot ``name`` (``None`` or ``live``: the live store).

        The snapshot is self-contained: its frozen cohort manifest and the configuration it
        was computed with (``snapshot.json``) replace the live ones, so a frozen n = 12 store
        never shows participants added later and views read the parameters of that run.
        Session paths (``data_dir``) stay live; outputs still go under ``out_dir``."""
        import dataclasses
        import json

        from biomech.results import snapshot as SN

        if not name or name == SN.LIVE:
            return self
        root = SN.path(self.out_dir, name)
        out = dataclasses.replace(self, store_dir=root, snapshot=name)
        if (root / "cohort.yaml").is_file():
            out = dataclasses.replace(out, cohort_path=root / "cohort.yaml")
        meta = root / "snapshot.json"
        raw = json.loads(meta.read_text(encoding="utf-8")).get("config") if meta.is_file() \
            else None
        if isinstance(raw, dict) and raw:
            out = dataclasses.replace(out, raw=raw, stages=dict(raw.get("stages") or {}),
                                      signals=dict(raw.get("signals") or {}),
                                      seed=int(raw.get("seed", self.seed)))
        return out

    def params(self, stage: str) -> dict[str, Any]:
        return copy.deepcopy(self.stages.get(stage, {}))

    def params_hash(self, stage: str) -> str:
        """The stage's own block plus the shared ``signals`` block."""
        return stable_hash({"stage": self.stages.get(stage, {}), "signals": self.signals})

    @property
    def hash(self) -> str:
        return stable_hash(self.raw)


def _resolve(p: Any, base: Path) -> Path | None:
    if p in (None, ""):
        return None
    path = Path(os.path.expandvars(str(p))).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(
    study_file: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> StudyConfig:
    """Defaults <- study file <- ``overrides`` (dotted keys) <- explicit arguments (highest)."""
    doc = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8")) or {}
    base = Path.cwd()
    source = None
    study_keys: tuple[str, ...] = ()
    if study_file is not None:
        source = Path(study_file).resolve()
        base = source.parent
        over = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        study_keys = tuple(sorted((over.get("stages") or {}).keys()))
        doc = _merge(doc, over)
    if overrides:
        doc = _merge(doc, dotted_overrides(overrides))

    paths = doc.get("paths", {})
    # --data-dir > $JTSMM_DATA_DIR > study paths.data_dir > repo pilot data > ~/jtsmm_data
    if data_dir in (None, "") and not os.environ.get(DATA_DIR_ENV) and paths.get("data_dir"):
        dd = _resolve(paths.get("data_dir"), base)
    else:
        dd = resolve_data_dir(data_dir if data_dir not in (None, "") else None).resolve()
    study = str(doc.get("study") or "unnamed")
    od = (
        _resolve(out_dir, Path.cwd())
        or _resolve(paths.get("out_dir"), base)
        or REPO_ROOT / "data" / "analysis" / study
    )
    cohort = _resolve(paths.get("cohort"), base)
    if cohort is None:
        raise ConfigError("paths.cohort is not set")
    return StudyConfig(
        study=study,
        seed=int(doc.get("seed", 42)),
        data_dir=dd,
        out_dir=od,
        cohort_path=cohort,
        stages=dict(doc.get("stages") or {}),
        signals=dict(doc.get("signals") or {}),
        source=source,
        raw=doc,
        study_stage_keys=study_keys,
    )
