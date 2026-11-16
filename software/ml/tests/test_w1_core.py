# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Core building blocks: features, windows, labels, signals, dataset I/O, models, fusion."""

from __future__ import annotations

import numpy as np
import pytest

from ml.config.classes import CLASS_COUNT
from ml.core import features as F
from ml.core import signals as S
from ml.core.dataset import WindowConfig, WindowDataset, concat, feature_set_blocks
from ml.core.windows import gap_windows, label_runs, window_ends, window_labels
from ml.fusion.early import BlockScaler
from ml.fusion.late import LateFusionClassifier, late_fusion_posteriors
from ml.models import registry as reg


def test_emg_features_are_trailing():
    x = np.random.default_rng(0).normal(size=(4000, 2))
    ends = np.array([1999])
    f1, _ = F.emg_window_features(x, 4000.0, ends, 800, np.ones(2))
    y = x.copy()
    y[2000:] += 100.0                               # only the future changes
    assert np.allclose(f1, F.emg_window_features(y, 4000.0, ends, 800, np.ones(2))[0])
    y[1500] += 100.0                                # inside the window
    assert not np.allclose(f1, F.emg_window_features(y, 4000.0, ends, 800, np.ones(2))[0])


def test_feature_registry_dispatch_and_unknown():
    x = np.random.default_rng(1).normal(size=(500, 3))
    f, _ = F.emg_window_features(x, 1000.0, np.array([199, 499]), 200, np.zeros(3),
                                 names=("mav", "band_20_100", "mdf", "log_ttp"))
    assert f.shape == (2, 12) and np.isfinite(f).all()
    with pytest.raises(KeyError, match="known"):
        F.get_feature("ENV_MEAN")
    with pytest.raises(ValueError, match="unknown window settings"):
        WindowConfig.from_dict({"alignment": "center"})
    with pytest.raises(KeyError):
        WindowConfig.from_dict({"emg_features": ["nope"]})


def test_sequence_features_shape_and_order():
    x = np.zeros((400, 2))
    x[380:, 0] = 5.0                                # activity only in the newest sub-frame
    s = F.emg_sequence_features(x, 1000.0, np.array([399]), 200, np.zeros(2), 10, ("mav",))
    assert s.shape == (1, 10, 2)
    assert s[0, -1, 0] > 0 and np.all(s[0, :-1, 0] == 0) and np.all(s[0, :, 1] == 0)
    sig = np.random.default_rng(2).normal(size=(100, 12))
    assert F.imu_sequence_features(sig, np.array([99]), 40, 10).shape == (1, 10, 12)


def test_window_labels_end_majority_and_age():
    lt = np.arange(0, 1_000_000, 25_000)
    ly = np.where(lt < 500_000, 1, 2)
    te = np.array([510_000, 750_000, 1_200_000])
    y, pur = window_labels(te, lt, ly, 200_000, mode="end", max_age_us=100_000)
    assert y.tolist() == [2, 2, -1]                 # last one: label older than 0.1 s
    assert 0 < pur[0] < 0.2 and pur[1] == 1.0
    ym, _ = window_labels(te, lt, ly, 200_000, mode="majority", purity_min=0.6)
    assert ym.tolist() == [1, 2, -1]
    ly2 = ly.copy()
    ly2[25] = -1                                    # any ignored sample in the window
    assert window_labels(te[:2], lt, ly2, 200_000, mode="majority")[0][1] == -1


def test_gap_rejection_by_duration():
    t = np.arange(0, 1000) * 1000
    t[500:] += 300_000                              # 0.3 s recording gap
    ends = window_ends(t.size, 200, 50)
    gap = gap_windows(t, ends, 200, 200_000)
    spans = (t[ends] - t[ends - 199]) > 220_000
    assert np.array_equal(gap, spans) and gap.any() and not gap.all()


def test_label_runs():
    assert label_runs(np.array([0, 0, 1, 1, 0]), np.array(["a"] * 4 + ["b"])).tolist() == \
        [0, 0, 1, 1, 2]


def test_hold_fill_is_causal():
    x = np.arange(10.0)[:, None].repeat(2, axis=1)
    m = np.zeros((10, 2), bool)
    m[3:6, 0] = True
    m[0, 1] = True
    out = S.hold_fill(x, m)
    assert out[3:6, 0].tolist() == [2, 2, 2] and out[0, 1] == 0 and out[6, 0] == 6


def test_leadoff_mask_uses_recorded_channel_index():
    st = np.zeros((5, 2), np.uint8)
    st[:, 1] = 1 << 4                               # CH5 (index 4) = masseter_l in file order
    m = S.leadoff_mask(st, [4, 0], 5)
    assert m[:, 0].all() and not m[:, 1].any()


def test_rail_and_flatline_masks():
    x = np.random.default_rng(3).normal(size=(4000, 2))
    x[100, 0] = 1000.0
    x[2000:3500, 1] = 7.0
    assert S.rail_mask(x, np.array([1001.0, 1e9]))[100, 0]
    fl = S.flatline_mask(x, 1000.0, 1.0)
    assert fl[2500, 1] and not fl[:1000].any()


def test_dataset_roundtrip_concat_and_feature_sets(tmp_path):
    rng = np.random.default_rng(4)

    def part(pid):
        n = 20
        blocks = {"emg": rng.normal(size=(n, 4)), "imu": rng.normal(size=(n, 2))}
        return WindowDataset(blocks=blocks,
                             y=np.repeat([0, 1], 10), pid=np.full(n, pid), session=np.full(n, pid),
                             phase=np.full(n, "open_close"), t_end_us=np.arange(n) * 50_000,
                             groups=label_runs(np.repeat([0, 1], 10)),
                             names={"emg": ["masseter_r|mav", "digastric_l|mav",
                                            "digastric_r|mav", "masseter_l|mav"],
                                    "imu": ["a", "b"]})
    d = concat([part("s01"), part("s02")])
    assert np.unique(d.groups).size == 4            # runs are unique across takes
    X, sl = d.matrix("emg_imu")
    assert X.shape == (40, 6) and [s.stop for s in sl] == [4, 6]
    assert d.matrix("emg_nodig")[0].shape == (40, 2)
    assert feature_set_blocks("seq_emg_imu") == ["seq", "seq_imu"]
    back = WindowDataset.load(d.save(tmp_path / "ds"))
    assert np.array_equal(back.y, d.y) and back.names == d.names
    assert np.allclose(back.blocks["imu"], d.blocks["imu"], atol=1e-6)


def test_block_scaler_equalizes_block_variance():
    X = np.hstack([np.random.default_rng(5).normal(size=(500, 40)),
                   np.random.default_rng(6).normal(size=(500, 4))])
    Z = BlockScaler([slice(0, 40), slice(40, 44)]).fit_transform(X)
    assert np.isclose(Z[:, :40].var(0).sum(), Z[:, 40:].var(0).sum(), rtol=0.01)
    Z1 = BlockScaler().fit_transform(X)
    assert np.allclose(Z1.var(0), 1.0)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(3), n // 3)
    X = rng.normal(size=(n, 6)) + y[:, None] * 1.5
    g = np.repeat(np.arange(6), n // 6)
    return X, y, g


@pytest.mark.parametrize("name", ["lda", "svm", "mlp", "late_mean:lda", "late_weighted:lda",
                                  "late_stack:lda"])
def test_registry_models_fit_predict_proba(name):
    X, y, g = _toy()
    m = reg.get(name).make(0, slices=[slice(0, 3), slice(3, 6)], probability=True)
    m.fit(X, y, groups=g)
    p = m.predict_proba(X)
    assert p.shape == (len(y), CLASS_COUNT) and np.allclose(p.sum(1), 1.0)
    assert (m.predict(X) == y).mean() > 0.8


def test_gru_through_registry_or_skip():
    spec = reg.get("gru")
    if not spec.available():
        with pytest.raises(ImportError, match="torch"):
            spec.make(0)
        pytest.skip("torch not installed")
    import pickle

    rng = np.random.default_rng(0)
    y = np.tile(np.arange(3), 60)
    X = rng.normal(size=(180, 5, 4)) + y[:, None, None]
    m = spec.make(0, hidden_size=8, max_epochs=60, patience=20, learning_rate=1e-2)
    m.fit(X, y, groups=np.arange(180) % 4)
    assert (m.predict(X) == y).mean() > 0.8
    back = pickle.loads(pickle.dumps(m))
    assert np.allclose(back.predict_proba(X), m.predict_proba(X))


def test_late_fusion_posteriors():
    a, b = np.array([[0.8, 0.2]]), np.array([[0.2, 0.8]])
    assert np.allclose(late_fusion_posteriors([a, b]), [[0.5, 0.5]])
    assert np.allclose(late_fusion_posteriors([a, b], method="weighted", weights=[1, 0]), a)
    with pytest.raises(ValueError):
        LateFusionClassifier(slices=[slice(0, 2)]).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
