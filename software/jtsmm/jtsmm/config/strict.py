# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Strict structural validation — completeness, unknown keys, JSON Schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from jsonschema import Draft202012Validator

from .loader import dotted_path, load_json_schema, schema_dir
from .validate import validate_config

_RESERVED_SESSION_META_KEYS = frozenset(
    {
        "software_version",
        "schema_version",
        "jts_schema_version",
        "created_unix_utc",
        "recording_date",
        "config_hash",
        "mcu_config_json",
        "host_config_json",
        "firmware_version",
        "session_duration_s",
        "file_hashes_json",
        "subject_id",
        "protocol",
        "emg_check_json",
        "emg_check_passed",
    }
)


def _iter_schema_paths(schema: dict[str, Any], prefix: str = "") -> Iterator[str]:
    """Yield dot-paths for every leaf property in a closed JSON Schema object."""
    if "$ref" in schema:
        return
    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, sub in props.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from _iter_schema_paths(sub, path)
        if additional is False:
            return
        return
    if schema_type == "array" or schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            yield from _iter_schema_paths(items, f"{prefix}[]")
        return
    if prefix:
        yield prefix


def _resolve_ref(root: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if "$ref" not in schema:
        return schema
    ref = schema["$ref"]
    if not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    merged = dict(node)
    for key, value in schema.items():
        if key != "$ref":
            if key == "allOf" and key in merged:
                merged[key] = merged[key] + value
            else:
                merged[key] = value
    return merged


def _schema_node_at(root: dict[str, Any], path_parts: list[str]) -> dict[str, Any] | None:
    node: Any = root
    for part in path_parts:
        if part == "[]":
            items = node.get("items") if isinstance(node, dict) else None
            if not isinstance(items, dict):
                return None
            node = items
            continue
        if not isinstance(node, dict):
            return None
        if "$ref" in node:
            node = _resolve_ref(root, node)
        props = node.get("properties") or {}
        if part not in props:
            return None
        node = props[part]
    if isinstance(node, dict) and "$ref" in node:
        node = _resolve_ref(root, node)
    return node if isinstance(node, dict) else None


def _allowed_keys_at(root: dict[str, Any], path_parts: list[str]) -> set[str] | None:
    node = _schema_node_at(root, path_parts)
    if node is None:
        return None
    if node.get("type") == "object" or "properties" in node:
        if node.get("additionalProperties") is False:
            return set((node.get("properties") or {}).keys())
        return None
    return None


def _open_schema(node: Any, *, drop_required: bool) -> Any:
    """Copy of the schema with every ``additionalProperties: false`` removed.

    Unknown keys are reported by :func:`_walk_doc_paths` instead, with their full
    dotted path and as an error or a warning (``host.schema.relaxed_unknown_keys``);
    left in the schema they would always be a second, less readable, hard error.
    ``drop_required`` also removes ``required`` (non-strict mode: check the types
    and values of what is present, not completeness).
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "additionalProperties" and value is False:
                continue
            if key == "required" and drop_required and isinstance(value, list):
                continue
            out[key] = _open_schema(value, drop_required=drop_required)
        return out
    if isinstance(node, list):
        return [_open_schema(v, drop_required=drop_required) for v in node]
    return node


def schema_issues(doc: dict[str, Any], *, strict: bool = True) -> tuple[list[str], set[str]]:
    """JSON Schema check. Returns (messages, dotted paths reported missing)."""
    schema = _open_schema(load_json_schema(), drop_required=not strict)
    msgs: list[str] = []
    missing: set[str] = set()
    for err in Draft202012Validator(schema).iter_errors(doc):
        where = dotted_path(err.absolute_path)
        if err.validator == "required" and isinstance(err.instance, dict):
            for key in err.validator_value:
                if key not in err.instance:
                    dot = f"{where}.{key}" if where else key
                    if dot not in missing:
                        missing.add(dot)
                        msgs.append(f"missing required key: {dot}")
            continue
        msgs.append(f"{where}: {err.message}" if where else err.message)
    return msgs, missing


def _walk_doc_paths(
    doc: Any,
    *,
    root_schema: dict[str, Any],
    path_parts: list[str],
    issues: list[str],
    relaxed: bool,
) -> None:
    if not isinstance(doc, dict):
        return
    allowed = _allowed_keys_at(root_schema, path_parts)
    if allowed is not None:
        for key in doc:
            if key not in allowed:
                dot = ".".join(path_parts + [key]) if path_parts else key
                level = "warn" if relaxed else "error"
                issues.append(f"{level}: unknown key not in schema: {dot}")
    node_schema = _schema_node_at(root_schema, path_parts)
    if node_schema is None:
        return
    props = (node_schema.get("properties") or {}) if isinstance(node_schema, dict) else {}
    for key, value in doc.items():
        child_parts = path_parts + [key]
        child_schema = props.get(key)
        if child_schema is None:
            continue
        if isinstance(child_schema, dict) and "$ref" in child_schema:
            child_schema = _resolve_ref(root_schema, child_schema)
        if isinstance(value, dict):
            _walk_doc_paths(
                value,
                root_schema=root_schema,
                path_parts=child_parts,
                issues=issues,
                relaxed=relaxed,
            )
        elif isinstance(value, list):
            item_schema = child_schema.get("items") if isinstance(child_schema, dict) else None
            if isinstance(item_schema, dict):
                for item in value:
                    if isinstance(item, dict):
                        _walk_doc_paths(
                            item,
                            root_schema=root_schema,
                            path_parts=child_parts + ["[]"],
                            issues=issues,
                            relaxed=relaxed,
                        )


def _load_coverage_required() -> list[str]:
    import yaml

    path = schema_dir() / "config_coverage.yaml"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    out: list[str] = []
    for entry in data.get("parameters") or []:
        if entry.get("required"):
            out.append(str(entry["path"]))
    return out


def _has_path(doc: dict[str, Any], dot_path: str) -> bool:
    cur: Any = doc
    for part in dot_path.split("."):
        if part.endswith("[]"):
            part = part[:-2]
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _validate_session_meta(doc: dict[str, Any], issues: list[str]) -> None:
    session = doc.get("session") or {}
    meta = session.get("meta")
    if not isinstance(meta, dict):
        return
    for key in meta:
        if key in _RESERVED_SESSION_META_KEYS:
            issues.append(
                f"error: session.meta.{key} collides with reserved HDF5 /meta attribute"
            )


def _relaxed_unknown_keys(doc: dict[str, Any]) -> bool:
    schema_block = (doc.get("host") or {}).get("schema")
    return isinstance(schema_block, dict) and bool(schema_block.get("relaxed_unknown_keys"))


def validate_strict(
    doc: dict[str, Any], *, path: Path | None = None, strict: bool = True
) -> list[str]:
    """Return issues prefixed error: / warn: / info:

    ``strict`` (default) adds completeness: the schema's ``required`` lists and
    the coverage manifest. Unknown keys are errors unless
    ``host.schema.relaxed_unknown_keys`` is true (or ``strict`` is off), then
    warnings. Cross-subsystem checks (validate.py) only run on a document that
    passed the schema -- they assume its types.
    """
    issues: list[str] = []
    prefix = f"{path}: " if path else ""

    schema_msgs, missing = schema_issues(doc, strict=strict)
    issues.extend(f"error: {prefix}{msg}" for msg in schema_msgs)

    root_schema = load_json_schema()
    relaxed = _relaxed_unknown_keys(doc) or not strict
    _walk_doc_paths(doc, root_schema=root_schema, path_parts=[], issues=issues, relaxed=relaxed)

    if strict:
        for dot_path in _load_coverage_required():
            if dot_path not in missing and not _has_path(doc, dot_path):
                issues.append(f"error: {prefix}missing required key: {dot_path}")

    _validate_session_meta(doc, issues)

    if schema_msgs:
        issues.append("info: cross-subsystem checks skipped until the schema errors are fixed")
        return issues

    for msg in validate_config(doc, include_schema=False):
        if msg.startswith("warn:"):
            issues.append(msg)
        else:
            issues.append(f"error: {prefix}{msg}" if not msg.startswith("error:") else msg)

    return issues


def raise_on_errors(issues: list[str]) -> None:
    hard = [i for i in issues if i.startswith("error:")]
    if hard:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(hard))
