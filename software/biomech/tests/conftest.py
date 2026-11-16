# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared fixtures: a fake session tree and a study config pointing at it."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from biomech.config import load_config

SESSIONS = [
    "session_s01_f_31_20250302T101000Z_00000002",
    "session_s01_f_31_mvc_20250302T100000Z_00000003",
    "session_s02_f_44_20250303T101000Z_00000004",
    "session_s02_f_44_20250306T101000Z_00000005",
    "session_s02_f_44_mvc_20250303T100000Z_00000006",
    "session_s03_m_29_20250304T101000Z_00000007",
    "session_s03_m_29_mvc_20250304T100000Z_00000008",
]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "sessions"
    for name in SESSIONS:
        (root / name).mkdir(parents=True)
        (root / name / "session.h5").write_bytes(b"x")
    (root / "not_a_session").mkdir()
    return root


@pytest.fixture
def study(tmp_path: Path, data_dir: Path):
    """A study whose cohort excludes s03; returns a factory for overrides."""
    from biomech.cohort import draft_manifest

    doc = draft_manifest(data_dir, "test")
    for row in doc["participants"]:
        if row["pid"] == "s03":
            row["include"] = False
            row["exclusion_reason"] = "outside_protocol"
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

    def make(stages: dict | None = None):
        cfg = {"study": "test", "paths": {"data_dir": str(data_dir), "out_dir": "out",
                                          "cohort": "cohort.yaml"}}
        if stages:
            cfg["stages"] = stages
        path = tmp_path / "study.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return load_config(path)

    return make


@pytest.fixture(scope="session")
def synth_study(tmp_path_factory):
    """Four synthetic participants with a known truth, all stages run once per session.

    Returns ``(config, store, truths, run_summary)``; shared by the end-to-end, layout
    and data-view tests (biomech.testing.synth).
    """
    from dataclasses import replace

    from biomech.pipeline import run
    from biomech.results import ResultStore
    from biomech.testing.synth import DEFAULT_SPECS, write_study

    root = tmp_path_factory.mktemp("synth_study")
    specs = [replace(s, emg_fs=2000) for s in DEFAULT_SPECS]      # 2 kHz keeps it fast
    path, truths = write_study(root, specs)
    cfg = load_config(path)
    summary = run(cfg, None, n_jobs=4)
    return cfg, ResultStore(cfg.out_dir), truths, summary


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests on the session-wide synthetic study run the whole pipeline (ML stages included
    when the ml package is installed), so they are ``slow`` even when not marked."""
    for item in items:
        if "synth_study" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow)
