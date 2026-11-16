# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every jtsmm.yaml key takes effect or is rejected loudly (config-plumbing audit).

For each leaf of the schema (template keys and optional ones) the test sets a
schema-valid, non-default sentinel and requires one of:

- mcu.*: the MCU payload changes at that key's wire path AND config_json.c
  parses it (FIRMWARE_JSON_KEYS) AND it is not a single-value key;
- host.* / session.*: some registry accessor returns something different;
- the sentinel produces a new validation error (rejected loudly);
- the schema admits no other value (nothing to set).

Anything else must be in RESERVED with the reason; an unclassified key fails.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import pytest

from jtsmm.config import build_mcu_apply_payload, validate_strict
from jtsmm.config import registry as reg
from jtsmm.config.coverage import _resolve_ref, firmware_parses, schema_leaf_paths
from jtsmm.config.loader import load_json_schema, load_yaml, template_path
from jtsmm.config.validate import _INERT_KEYS

# Keys whose effect this test cannot observe, with where it is covered instead.
RESERVED: dict[str, str] = {
    "mcu.platform.profile": "label; the firmware only checks default|custom (schema enum)",
    "mcu.emg.profile": "label: the MCU treats explicit keys as custom (s08 'Submodule pattern')",
    "mcu.imu_mta.profile": "label: explicit reports win on the MCU (test_fix_H_profiles)",
    "mcu.imu_cra.profile": "label: explicit reports win on the MCU (test_fix_H_profiles)",
    "mcu.trigger.profile": "label: an explicit rate_hz makes the MCU use custom",
    "hat.serial_port": "used by find_hat_port / preflight (test_fix_H_ct_cmd)",
    "hat.firmware_version": "checked by `jtsmm doctor` against the MCU banner",
    "host.cameras.track": "read by doctor/sysfs.py and the Track-B ae_enable check",
    "host.cameras.list": "per-camera controls: capture_cam.cameras_from_config (package F)",
    "host.schema.relaxed_unknown_keys": "changes validation only (test_fix_H_strict)",
}

# Parsed by the firmware and ignored there: a payload diff is not an effect.
_SINGLE_VALUE = {path for path, _, _ in _INERT_KEYS}

_WIRE_RENAMES = {"emg.active_channels": "emg.pd_channels"}
_MISSING = object()


def _get(doc: Any, dotted: str) -> Any:
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _set(doc: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = doc
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _schema_node(dotted: str) -> dict:
    root = load_json_schema()
    node: Any = root
    for part in dotted.split("."):
        node = _resolve_ref(root, node)
        node = node["properties"][part]
    return _resolve_ref(root, node)


def _number_candidates(current: Any, lo: Any, hi: Any, integer: bool) -> list:
    base = current if isinstance(current, (int, float)) and not isinstance(current, bool) else None
    cands = []
    if base is not None:
        cands += [base + 1, base - 1, base * 2]
    cands += [lo, hi, 1, 0]
    out = []
    for c in cands:
        if c is None:
            continue
        c = int(c) if integer else float(c)
        if (lo is None or c >= lo) and (hi is None or c <= hi) and c != current:
            out.append(c)
    return out


def _sentinel(node: dict, current: Any, avoid: set) -> Any:
    """A schema-valid value that differs from ``current`` and from ``avoid``; _MISSING if none."""
    for branch in node.get("oneOf") or node.get("anyOf") or []:
        if branch.get("type") != "null":
            got = _sentinel(branch, current, avoid)
            if got is not _MISSING:
                return got
    if "const" in node:
        return _MISSING
    if "enum" in node:
        for value in node["enum"]:
            if value != current and value not in avoid:
                return value
        return _MISSING
    kind = node.get("type")
    if kind == "boolean":
        for value in (not current if isinstance(current, bool) else True, False):
            if value != current and value not in avoid:
                return value
        return _MISSING
    if kind in ("integer", "number"):
        for value in _number_candidates(
            current, node.get("minimum"), node.get("maximum"), kind == "integer"
        ):
            if value not in avoid:
                return value
        return _MISSING
    if kind == "string":
        return "sentinel/value.json"
    if kind == "array":
        items = node.get("items") or {}
        n = node.get("minItems", 1)
        if items.get("type") in ("integer", "number"):
            lo = items.get("minimum", 1)
            cand = [lo + i for i in range(max(n, 1))]
            return cand if cand != current else cand[:1] if len(cand) > 1 else [lo + 1]
        return _MISSING
    if kind == "object" or "additionalProperties" in node:
        return {"sentinel": "x"}
    return _MISSING


_SPECIAL = {
    "session.experiment": {"name": "sentinel_plan", "phases": ["open_close"]},
    "session.meta": {"operator": "sentinel"},
}


def _accessors() -> dict[str, Any]:
    out = {}
    for name, fn in inspect.getmembers(reg, inspect.isfunction):
        if name.startswith("_") or fn.__module__ != reg.__name__:
            continue
        params = list(inspect.signature(fn).parameters.values())
        if params and params[0].name == "resolved" and all(
            p.default is not inspect.Parameter.empty or p.kind is p.KEYWORD_ONLY
            for p in params[1:]
        ):
            out[name] = fn
    return out


def _observe(doc: dict) -> tuple[dict, dict]:
    try:
        payload = build_mcu_apply_payload(doc)
    except Exception as exc:  # noqa: BLE001 - a rejected value may not even build
        payload = {"error": repr(exc)}
    seen = {}
    for name, fn in _accessors().items():
        try:
            seen[name] = fn(doc)
        except Exception as exc:  # noqa: BLE001
            seen[name] = f"raised {type(exc).__name__}"
    return payload, seen


def _hard(doc: dict) -> set[str]:
    return {i for i in validate_strict(doc) if i.startswith("error:")}


def _enable_ancestors(doc: dict, dotted: str) -> None:
    """A leaf under a disabled block (report entry, test_signal) is inert by design:
    enable the block in the baseline and the variant alike."""
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        block = _get(doc, ".".join(parts[:i]))
        if isinstance(block, dict) and block.get("enabled") is False and parts[i] != "enabled":
            block["enabled"] = True


def _classify(path: str) -> str:
    base = load_yaml(template_path())
    _enable_ancestors(base, path)
    current = _get(base, path)
    avoid = {accepted for key, accepted, _ in _INERT_KEYS if key == path}
    value = _SPECIAL.get(path) or _sentinel(
        _schema_node(path), None if current is _MISSING else current, avoid
    )
    if value is _MISSING:
        return "fixed by schema"
    variant = copy.deepcopy(base)
    _set(variant, path, value)

    # Only this one change separates the two documents, so any new error is its.
    if _hard(variant) - _hard(base):
        return "rejected"

    (pay0, acc0), (pay1, acc1) = _observe(base), _observe(variant)
    if path.startswith("mcu."):
        wire = path[len("mcu."):]
        wire = _WIRE_RENAMES.get(wire, wire)
        if (
            _get(pay0, wire) != _get(pay1, wire)
            and firmware_parses(path)
            and path not in _SINGLE_VALUE
        ):
            return "effect"
        return "NO EFFECT"
    changed = [k for k in acc0 if acc0[k] != acc1[k]]
    return "effect" if changed else "NO EFFECT"


LEAVES = sorted(p for p in schema_leaf_paths() if p not in RESERVED)


@pytest.mark.parametrize("path", LEAVES)
def test_every_key_takes_effect_or_is_rejected(path):
    verdict = _classify(path)
    assert verdict != "NO EFFECT", (
        f"{path}: a non-default value neither changes the MCU payload / a registry "
        "accessor nor fails validation -- wire it, reject it, or add it to RESERVED "
        "with the reason"
    )


def test_reserved_keys_exist():
    leaves = schema_leaf_paths()
    assert set(RESERVED) <= leaves, sorted(set(RESERVED) - leaves)


def test_single_value_keys_are_rejected_not_passed_off_as_effects():
    for path in sorted(_SINGLE_VALUE):
        assert _classify(path) in ("rejected", "fixed by schema"), path


def test_detector_flags_a_dead_key(monkeypatch):
    """Without its single-value rule, host.cameras.marker_detector is a silent no-op."""
    import jtsmm.config.validate as validate

    rules = tuple(r for r in validate._INERT_KEYS if r[0] != "host.cameras.marker_detector")
    monkeypatch.setattr(validate, "_INERT_KEYS", rules)
    assert _classify("host.cameras.marker_detector") == "NO EFFECT"
