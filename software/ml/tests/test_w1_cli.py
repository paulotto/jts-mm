# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Independence: every module imports, every command's --help works, and a tiny
synthetic run goes label sidecar -> ml dataset -> ml train -> ml eval end to end."""

from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import ml
from ml.cli.main import main
from tests import synth

TESTS = Path(__file__).resolve().parent


def test_every_module_imports():
    failed = {}
    for m in pkgutil.walk_packages(ml.__path__, "ml."):
        if m.name == "ml.biomech_stages" and importlib.util.find_spec("biomech") is None:
            continue
        try:
            importlib.import_module(m.name)
        except Exception as exc:  # pragma: no cover - reported below
            failed[m.name] = repr(exc)
    assert not failed, failed


def test_no_optical_or_mock_dependencies():
    src = "\n".join(p.read_text() for p in Path(ml.__file__).parent.rglob("*.py"))
    assert "import optical" not in src and "from optical" not in src
    tests = "\n".join(p.read_text() for p in TESTS.glob("*.py") if p.name != Path(__file__).name)
    assert "data/mock_" not in tests and "pilot_study" not in tests


@pytest.mark.parametrize("args", [[], ["label"], ["labels"], ["labels", "from-biomech"],
                                  ["dataset"], ["train"], ["eval"]])
def test_help(args):
    r = CliRunner().invoke(main, [*args, "--help"])
    assert r.exit_code == 0, r.output
    assert "Usage" in r.output


def test_train_help_lists_registry_models():
    from ml.models.registry import names

    out = CliRunner().invoke(main, ["train", "--help"]).output
    assert all(n in out for n in names())


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """Three synthetic participants, sidecars in a separate label directory."""
    tmp = tmp_path_factory.mktemp("e2e")
    labels = tmp / "ml_labels"
    for i in range(3):
        h5 = synth.write_session(synth.folder(tmp / "data", f"s0{i + 1}"), seed=i)
        mvc = synth.write_mvc(synth.folder(tmp / "data", f"s0{i + 1}", mvc=True), seed=10 + i)
        synth.write_sidecar(h5, labels / f"{h5.parent.name}_labels.h5", mvc=mvc)
    cfg = tmp / "ml.yaml"
    cfg.write_text(f"dataset: {{window: {{warmup_s: 0.5}}}}\npaths: {{data_dir: {tmp / 'ml'}}}\n"
                   "train: {model: lda, feature_set: emg+imu, protocol: loso_subject}\n")
    return tmp, labels, cfg


def _ok(args):
    r = CliRunner().invoke(main, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    return r.output


def test_end_to_end_cli(pool):
    tmp, labels, cfg = pool
    out = _ok(["dataset", "--labels-dir", str(labels), "--config", str(cfg),
               "--output", str(tmp / "ds")])
    assert "config:" in out and "--config" in out and "subjects=['s01', 's02', 's03']" in out
    _ok(["train", "--dataset", str(tmp / "ds"), "--config", str(cfg),
         "--runs-dir", str(tmp / "runs")])
    run = next((tmp / "runs").iterdir())
    summ = json.loads((run / "summary.json").read_text())
    assert summ["protocol"] == "loso_subject" and summ["test_units"] == ["s01", "s02", "s03"]
    assert summ["aggregate"]["mean"]["balanced_accuracy"] > 0.5
    for k in range(3):
        z = np.load(run / "per_fold" / f"fold_{k}" / "predictions.npz")
        assert {"indices", "y_true", "y_pred", "y_proba", "t_end_us"} <= set(z.files)
    a = _ok(["eval", "--run", str(run)])
    b = _ok(["eval", "--run", str(run), "--dataset", str(tmp / "ds"), "--out-dir",
             str(run / "eval2")])
    assert a.splitlines()[-1] == b.splitlines()[-1]
    _ok(["eval", "--model-path", str(run / "per_fold" / "fold_0" / "model.joblib"),
         "--dataset", str(tmp / "ds"), "--feature-set", "emg+imu", "--out-dir",
         str(tmp / "one")])
    _ok(["train", "--compare", str(tmp / "runs")])
    assert (tmp / "runs" / "comparison.md").is_file()


def test_cli_other_protocols_models(pool):
    tmp, labels, cfg = pool
    if not (tmp / "ds").exists():
        _ok(["dataset", "--labels-dir", str(labels), "--config", str(cfg), "--output",
             str(tmp / "ds")])
    _ok(["train", "--dataset", str(tmp / "ds"), "--config", str(cfg), "--runs-dir",
         str(tmp / "runs2"), "--protocol", "blocked", "--model", "svm", "--feature-set", "emg"])
    r = CliRunner().invoke(main, ["train", "--dataset", str(tmp / "ds"), "--config", str(cfg),
                                  "--protocol", "loso"])
    assert r.exit_code != 0 and "loso_subject" in r.output
    from ml.models.registry import get

    if not get("gru").available():
        pytest.skip("torch not installed")
    cfg2 = tmp / "gru.yaml"
    cfg2.write_text(cfg.read_text().replace(
        "train: {model: lda, feature_set: emg+imu, protocol: loso_subject}",
        "train: {model: gru, feature_set: seq+seq_imu, protocol: loso_subject, search: false,\n"
        "        models: {gru: {params: {hidden_size: 8, max_epochs: 5, patience: 2}}}}"))
    _ok(["train", "--dataset", str(tmp / "ds"), "--config", str(cfg2), "--runs-dir",
         str(tmp / "runs3")])
