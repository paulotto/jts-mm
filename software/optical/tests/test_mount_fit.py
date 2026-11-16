# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tests for tag-frame orientation recovery in mounts_T_body_tag."""

from __future__ import annotations

import numpy as np
import pytest

from optical.config.loader import default_config_path
from optical.markers.config import load_marker_config
from optical.tracking.mount_fit import (
    MOUNT_CANDIDATES,
    fit_body_mount_corrections,
    measure_body_geometry,
)
from optical.tracking.pose import tag_object_points
from optical.tracking.triangulation import TriangulatedPoint


def _marker_config():
    return load_marker_config(default_config_path())


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _triangulated(mc, true_mounts, *, noise_mm=0.15, seed=3):
    """Corner cloud as the tracker would triangulate it, in world coordinates."""
    rng = np.random.default_rng(seed)
    obj = tag_object_points(mc.tag_size_mm)

    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = np.array([80.0, -20.0, 420.0])

    points = []
    for tag_id, T in true_mounts.items():
        for idx, p in enumerate(obj):
            body = (T @ np.append(p, 1.0))[:3]
            world = R @ body + t + rng.normal(0.0, noise_mm, 3)
            points.append(
                TriangulatedPoint(
                    marker_id=tag_id,
                    corner_index=idx,
                    position_mm=world,
                    n_views=2,
                    reprojection_rms_px=0.3,
                )
            )
    return points


def _ct_mounts(mc):
    lo, hi = mc.id_ranges["CT"]
    return {
        tid: T.copy()
        for tid, T in mc.mounts_T_body_tag.items()
        if lo <= tid <= hi
    }


def test_candidate_set_is_eight_proper_rotations():
    assert len(MOUNT_CANDIDATES) == 8
    names = [name for name, _ in MOUNT_CANDIDATES]
    assert names[0] == "as-configured"
    assert len(set(names)) == 8
    for _, R in MOUNT_CANDIDATES:
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_correct_config_needs_no_correction():
    mc = _marker_config()
    fit = fit_body_mount_corrections(
        _triangulated(mc, _ct_mounts(mc)), body="CT", marker_config=mc
    )
    assert fit is not None
    assert not fit.changed
    assert fit.rms_mm < 0.5


@pytest.mark.parametrize(
    "error",
    [
        _rot_z(90),
        _rot_z(180),
        _rot_z(270),
        _rot_x(180),
        _rot_x(180) @ _rot_z(90),
    ],
)
def test_recovers_single_tag_orientation_error(error):
    mc = _marker_config()
    mounts = _ct_mounts(mc)
    bad_tag = min(mounts)
    mounts[bad_tag][:3, :3] = mounts[bad_tag][:3, :3] @ error

    fit = fit_body_mount_corrections(
        _triangulated(mc, mounts), body="CT", marker_config=mc
    )
    assert fit is not None
    # A 90-degree tag error lands near the tag edge length before correction.
    assert fit.baseline_rms_mm > 10.0
    assert fit.rms_mm < 0.5
    assert fit.changed

    corrected = fit.corrected_T_body_tag[bad_tag]
    assert np.allclose(corrected[:3, :3], mounts[bad_tag][:3, :3], atol=1e-6)
    assert np.allclose(corrected[:3, 3], mounts[bad_tag][:3, 3], atol=1e-9)


def test_residual_is_invariant_to_body_origin():
    """A different body-frame origin (e.g. at the tool tip) must not matter."""
    mc = _marker_config()
    mounts = _ct_mounts(mc)
    tri = _triangulated(mc, mounts)
    base = fit_body_mount_corrections(tri, body="CT", marker_config=mc)

    shifted = {tid: T.copy() for tid, T in mounts.items()}
    for T in shifted.values():
        T[:3, 3] += np.array([0.0, 225.0, 0.0])
    moved = fit_body_mount_corrections(
        _triangulated(mc, shifted), body="CT", marker_config=mc
    )

    assert base is not None and moved is not None
    assert moved.baseline_rms_mm == pytest.approx(base.baseline_rms_mm, abs=0.2)


def test_single_tag_carries_no_orientation_information():
    mc = _marker_config()
    mounts = _ct_mounts(mc)
    one = {min(mounts): mounts[min(mounts)]}
    assert fit_body_mount_corrections(
        _triangulated(mc, one), body="CT", marker_config=mc
    ) is None


def test_geometry_readout_reports_edges_and_separation():
    mc = _marker_config()
    text = measure_body_geometry(
        _triangulated(mc, _ct_mounts(mc)), body="CT", marker_config=mc
    )
    assert "edge" in text
    assert "sep" in text


def test_yaml_snippet_round_trips_to_the_fitted_matrices():
    import yaml

    mc = _marker_config()
    mounts = _ct_mounts(mc)
    bad_tag = min(mounts)
    mounts[bad_tag][:3, :3] = mounts[bad_tag][:3, :3] @ _rot_z(90)
    fit = fit_body_mount_corrections(
        _triangulated(mc, mounts), body="CT", marker_config=mc
    )
    assert fit is not None

    parsed = yaml.safe_load(fit.yaml_snippet())["mounts_T_body_tag"]
    for tid in fit.tag_ids:
        got = np.asarray(parsed[str(tid)], dtype=float)
        assert got.shape == (4, 4)
        assert np.allclose(got, fit.corrected_T_body_tag[tid], atol=1e-6)
