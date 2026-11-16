# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The IMU mounting travels with every take: ``jtsmm imu calibrate`` merges accepted fits
into the stable mounting file, and each recorded session embeds it as
``/calibration/imu_*/mounting_transform`` (as the CT tip goes into ``/meta``)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jtsmm import native
from jtsmm.imu.mounting import (
    embed_mounting,
    load_mounting_file,
    load_mounting_rotation,
    update_mounting_file,
)
from tests.test_fix_F_supervisor import env  # noqa: F401 - pytest fixture


def _rec(R, accepted=True, rms=1.2):
    T = np.eye(4)
    T[:3, :3] = R
    return {"accepted": accepted, "rms_deg": rms, "n_pairs": 300, "n_samples": 40,
            "R_mount": R.tolist(), "T": T.tolist(), "phase": "imu_mount_mta"}


def test_stable_file_keeps_accepted_fits_only(tmp_path):
    f = tmp_path / "imu_mounting_calibration.json"
    Ra = Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix()
    Rb = Rotation.from_rotvec([-0.4, 0.0, 0.1]).as_matrix()
    assert update_mounting_file(f, {"imu_mta": _rec(Ra), "imu_cra": _rec(Rb, accepted=False)},
                                "calib_1.h5") == ["imu_mta"]
    # a later rejected mta fit does not replace the accepted one; an accepted cra is added
    assert update_mounting_file(f, {"imu_mta": _rec(Rb, accepted=False),
                                    "imu_cra": _rec(Rb)}, "calib_2.h5") == ["imu_cra"]
    e = load_mounting_file(f)
    np.testing.assert_allclose(e["imu_mta"]["R_mount"], Ra)
    assert e["imu_mta"]["source_session"] == "calib_1.h5"
    np.testing.assert_allclose(e["imu_cra"]["R_mount"], Rb)
    assert json.loads(f.read_text())["kind"] == "imu_mounting_calibration"


def test_embed_writes_what_the_loaders_read(tmp_path):
    f = tmp_path / "m.json"
    R = Rotation.from_rotvec([0.3, -0.1, 0.2]).as_matrix()
    update_mounting_file(f, {"imu_mta": _rec(R)}, "calib.h5")
    h5 = tmp_path / "session.h5"
    with h5py.File(h5, "w") as g:
        g.require_group("calibration/imu_mta").attrs["error"] = "stale"
    assert embed_mounting(h5, f) == ["imu_mta"]
    np.testing.assert_allclose(load_mounting_rotation(h5, "mta"), R)
    with h5py.File(h5, "r") as g:
        a = g["calibration/imu_mta"].attrs
        assert a["embedded"] and a["source_session"] == "calib.h5" and "error" not in a
    assert embed_mounting(h5, tmp_path / "missing.json") == []


@pytest.mark.skipif(not native.is_available(), reason="native extension not built")
@pytest.mark.parametrize("protocol", ["motion", "imu_mounting"])
def test_recorded_session_embeds_the_mounting(env, tmp_path, protocol):  # noqa: F811
    from jtsmm.capture.supervisor import SessionSupervisor

    f = tmp_path / "imu_mounting_calibration.json"
    R = Rotation.from_rotvec([0.2, 0.1, -0.2]).as_matrix()
    update_mounting_file(f, {"imu_mta": _rec(R)}, "calib.h5")
    env.resolved.setdefault("session", {})["imu_mounting_calibration"] = str(f)
    h5 = SessionSupervisor(env.resolved, port="/dev/null",
                           extra_meta={"protocol": protocol}).run(duration_s=0.3)
    got = load_mounting_rotation(h5, "mta")
    if protocol == "imu_mounting":            # the calibration take gets its own fresh fit
        assert got is None
    else:
        np.testing.assert_allclose(got, R)
