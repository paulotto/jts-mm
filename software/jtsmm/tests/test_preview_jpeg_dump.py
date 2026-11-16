# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Viewer JPEG sidecar during experiment-plan phases."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import h5py
import pytest

from jtsmm.capture.preview_dump import PreviewJpegDump, _slug
from jtsmm.capture.session import run_session
from jtsmm.io.session_layout import PREVIEW_INDEX_NAME, preview_dir, preview_index_path


def test_slug_phase_names():
    assert _slug("mand_point_1") == "mand_point_1"
    assert _slug("Open / Close") == "open_close"


def test_preview_jpeg_dump_writes_phase_folders(tmp_path: Path):
    dump = PreviewJpegDump(tmp_path)
    jpeg = b"\xff\xd8" + b"x" * 20 + b"\xff\xd9"
    assert dump.offer(
        cam_id="cam0",
        jpeg=jpeg,
        t_master_us=1000,
        phase_index=0,
        phase_name="mand_point_1",
        phase_kind="calibration_mandibular",
    )
    assert dump.offer(
        cam_id="cam1",
        jpeg=jpeg,
        t_master_us=1001,
        phase_index=0,
        phase_name="mand_point_1",
        phase_kind="calibration_mandibular",
    )
    assert dump.offer(
        cam_id="cam0",
        jpeg=jpeg,
        t_master_us=5000,
        phase_index=6,
        phase_name="open_close",
        phase_kind="experiment",
    )
    stats = dump.close()
    assert stats["written"] == 3
    assert stats["dropped"] == 0
    cam0 = tmp_path / "preview" / "00_mand_point_1" / "cam0_000000.jpg"
    cam1 = tmp_path / "preview" / "00_mand_point_1" / "cam1_000000.jpg"
    motion = tmp_path / "preview" / "06_open_close" / "cam0_000000.jpg"
    assert cam0.read_bytes() == jpeg
    assert cam1.is_file()
    assert motion.is_file()
    lines = preview_index_path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["phase"] for r in rows] == ["mand_point_1", "mand_point_1", "open_close"]
    assert rows[0]["kind"] == "calibration_mandibular"
    assert rows[2]["kind"] == "experiment"
    assert rows[0]["path"].endswith("cam0_000000.jpg")


def test_preview_dump_skips_empty_jpeg(tmp_path: Path):
    dump = PreviewJpegDump(tmp_path)
    assert dump.offer(
        cam_id="cam0",
        jpeg=b"",
        t_master_us=1,
        phase_index=0,
        phase_name="x",
        phase_kind="experiment",
    ) is False
    stats = dump.close()
    assert stats["written"] == 0


def test_session_layout_preview_paths(tmp_path: Path):
    assert preview_dir(tmp_path).name == "preview"
    assert preview_index_path(tmp_path).name == PREVIEW_INDEX_NAME


def test_mock_capture_attaches_rgb_when_requested(monkeypatch):
    import jtsmm.capture.capture_cam as cam_mod
    from jtsmm.capture.capture_cam import CameraCapture

    monkeypatch.setattr(cam_mod, "_try_open_picamera2", lambda _cam: False)
    cam = CameraCapture(
        cam_id="cam0",
        device="csi0",
        resolution=(8, 8),
        exposure_us=1000,
        gain=1.0,
        expect_rate_hz=60.0,
    )
    meta = cam.poll_frame(with_rgb=True)
    assert meta is not None and meta.mock is True
    assert meta.rgb is not None
    assert meta.rgb.shape == (16, 16)
    meta2 = cam._capture_one(blocking=False, with_rgb=False)
    assert meta2 is not None and meta2.rgb is None


class _ScriptedKeys:
    """SPACE / Q on a wall-clock schedule (replaces TerminalKeys in tests)."""

    def __init__(self) -> None:
        self.t0: float | None = None
        self.events: list[tuple[float, int]] = [
            (0.08, ord(" ")),  # start mand_point_1 (calibration)
            (0.28, ord(" ")),  # end — gap until next SPACE
            (0.50, ord(" ")),  # start mand_point_2
            (0.70, ord("q")),  # finish early
        ]
        self.i = 0
        self.fired: list[tuple[float, int]] = []

    def __enter__(self) -> _ScriptedKeys:
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *_args) -> None:
        return None

    def poll(self) -> int | None:
        if self.t0 is None or self.i >= len(self.events):
            return None
        delay, key = self.events[self.i]
        if time.monotonic() - self.t0 >= delay:
            self.i += 1
            self.fired.append((time.monotonic(), key))
            return key
        return None


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="needs OpenCV")
def test_full_session_writes_preview_jpegs_only_during_phases(
    tmp_config_path: Path, tmp_path: Path, monkeypatch
):
    offers: list[tuple[float, str, int]] = []
    keys = _ScriptedKeys()

    class _SpyDump(PreviewJpegDump):
        def offer(self, **kwargs):  # type: ignore[no-untyped-def]
            offers.append((time.monotonic(), str(kwargs["phase_name"]), int(kwargs["phase_index"])))
            return super().offer(**kwargs)

    monkeypatch.setattr("jtsmm.io.terminal_keys.TerminalKeys", lambda: keys)
    monkeypatch.setattr("jtsmm.capture.preview_dump.PreviewJpegDump", _SpyDump)
    monkeypatch.setattr("jtsmm.capture.capture_cam._try_open_picamera2", lambda _cam: False)
    monkeypatch.setenv("JTSMM_DATA_DIR", str(tmp_path / "data"))

    experiment = pytest.importorskip("optical.tracking.experiment")
    plan = experiment.ExperimentPlan(name="preview_dump_test", phases=[experiment.Phase("open_close")])
    h5 = run_session(
        config_path=tmp_config_path,
        duration_s=2.0,
        mock_only=True,
        skip_mcu_apply=True,
        experiment_plan=plan,
        write_jts_config=False,
        optical_track=False,
        viewer=False,
    )
    session_dir = h5.parent
    assert offers, "expected preview JPEGs during active phases"
    names = {n for _, n, _ in offers}
    assert "mand_point_1" in names
    assert "mand_point_2" in names
    assert "open_close" not in names  # aborted before motion

    # Nothing queued in the gap between phase 1 end and phase 2 start.
    assert len(keys.fired) >= 3
    t_end_p1 = keys.fired[1][0]
    t_start_p2 = keys.fired[2][0]
    gap_offers = [t for t, _, _ in offers if t_end_p1 < t < t_start_p2]
    assert gap_offers == []

    cal_dir = preview_dir(session_dir) / "00_mand_point_1"
    assert cal_dir.is_dir()
    assert list(cal_dir.glob("*.jpg"))
    assert (preview_dir(session_dir) / "01_mand_point_2").is_dir()
    assert not (preview_dir(session_dir) / "02_mand_point_3").exists()

    index = preview_index_path(session_dir)
    assert index.is_file()
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    assert {r["kind"] for r in rows} == {"calibration_mandibular"}
    for r in rows:
        jpeg_path = session_dir / r["path"]
        assert jpeg_path.is_file()
        assert jpeg_path.read_bytes()[:2] == b"\xff\xd8"

    with h5py.File(h5, "r") as f:
        report = json.loads(f["sync"].attrs["report_json"])
        assert int(report["preview_jpeg"]["written"]) == len(rows)
        hashes = json.loads(f["meta"].attrs["file_hashes_json"])
        assert PREVIEW_INDEX_NAME in hashes


def test_duration_only_session_does_not_dump_preview_jpegs(
    tmp_config_path: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr("jtsmm.capture.capture_cam._try_open_picamera2", lambda _cam: False)
    monkeypatch.setenv("JTSMM_DATA_DIR", str(tmp_path / "data"))
    h5 = run_session(
        config_path=tmp_config_path,
        duration_s=0.12,
        mock_only=True,
        skip_mcu_apply=True,
        use_experiment_plan=False,
        optical_track=False,
    )
    session_dir = h5.parent
    assert not preview_index_path(session_dir).exists()
    preview = preview_dir(session_dir)
    assert not preview.exists() or not any(preview.rglob("*.jpg"))
