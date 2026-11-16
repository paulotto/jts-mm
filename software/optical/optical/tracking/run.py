#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""
Live ArUco jaw tracker: detect markers, estimate HP/MP/CT poses, write HDF5.

The preview starts immediately. With ``experiment.use_experiment_plan: true``
(default for jaw-motion takes), landmark calibration is inserted before motion
phases and each phase is SPACE-gated. With ``use_experiment_plan: false``, a
single free-recording phase (named after ``experiment.condition``) is used —
no landmark calib, no JTS sidecar.

On completion the tool writes (default ``data/optical/tracking/``):

- the tracking session HDF5 (named after the experiment),
- a ready-to-run JTS config (``<experiment>_jts_config.json``) next to it.

With ``--persist-frames``, both ``session.h5`` and the frame dump go under::

    ~/experiments/{experiment.folder}/recordings/{name}_{condition}_{stamp}/

``experiment.name`` and ``experiment.condition`` come from ``optical.yaml``;
``--condition`` overrides the YAML condition.

Example::

    optical track --persist-frames
    optical track --persist-frames --condition near

Offline replay of saved frames: ``optical replay``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    from optical.calibration.multicam import (
        CamSpec,
        PreviewServer,
        _PREVIEW_PERIOD_S,
        _TerminalKeys,
        _as_bgr,
        _draw_status_bar,
        _fit_preview,
        _open_rig,
        _parse_cams,
        _stack_preview,
        enable_track_lores,
    )
    from optical.paths import (
        CALIBRATION_BUNDLE_SNAPSHOT_NAME,
        SESSION_H5_NAME,
        resolve_persist_recording,
        snapshot_persist_calibration,
        tracking_session_path,
    )
    from optical.sync.hat_sync import HatSyncBridge
    from optical.sync.hat_trigger import (
        active_trigger_rate_hz,
        arm_hat_trigger,
        configured_trigger_rate_hz,
        hat_trigger,
    )
    from optical.tracking.experiment import (
        CALIBRATION_PHASES,
        ExperimentPlan,
        RecordedPhase,
        build_jts_config,
        prompt_experiment_plan,
        write_jts_config,
    )
    from optical.tracking.io import TrackingSessionWriter
    from optical.tracking.frame_store import AsyncFrameWriter, FrameTimestampLog, write_session_meta
    from optical.tracking.preview_overlay import overlay_frames_bgr
    from optical.tracking.tracker import (
        ArucoJawTracker,
        TrackerConfig,
        apply_ct_tip,
        load_ct_tip_offset,
    )
except ModuleNotFoundError:
    _src_root = Path(__file__).resolve().parents[2]
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    from optical.calibration.multicam import (
        CamSpec,
        PreviewServer,
        _PREVIEW_PERIOD_S,
        _TerminalKeys,
        _as_bgr,
        _draw_status_bar,
        _fit_preview,
        _open_rig,
        _parse_cams,
        _stack_preview,
        enable_track_lores,
    )
    from optical.paths import (
        CALIBRATION_BUNDLE_SNAPSHOT_NAME,
        SESSION_H5_NAME,
        resolve_persist_recording,
        snapshot_persist_calibration,
        tracking_session_path,
    )
    from optical.sync.hat_sync import HatSyncBridge
    from optical.sync.hat_trigger import (
        active_trigger_rate_hz,
        arm_hat_trigger,
        configured_trigger_rate_hz,
        hat_trigger,
    )
    from optical.tracking.experiment import (
        CALIBRATION_PHASES,
        ExperimentPlan,
        RecordedPhase,
        build_jts_config,
        prompt_experiment_plan,
        write_jts_config,
    )
    from optical.tracking.io import TrackingSessionWriter
    from optical.tracking.frame_store import AsyncFrameWriter, FrameTimestampLog, write_session_meta
    from optical.tracking.preview_overlay import overlay_frames_bgr
    from optical.tracking.tracker import (
        ArucoJawTracker,
        TrackerConfig,
        apply_ct_tip,
        load_ct_tip_offset,
    )



def _load_json(path: str | Path, *, n_cameras: int | None = None) -> dict:
    from optical.config.loader import calibration_section
    from optical.config.resolve import load_resolved_config

    path = Path(path)
    doc = load_resolved_config(path, n_cameras=n_cameras)
    try:
        return calibration_section(doc, config_path=path)
    except KeyError:
        return doc


def _resolve_plan(plan_path: Optional[Path], tracker_config_path: Path) -> ExperimentPlan:
    """Load a plan from file, from optical.yaml experiment:, or prompt."""
    if plan_path is not None:
        plan = ExperimentPlan.from_json(plan_path)
        print(f"Loaded experiment plan from {plan_path} ({len(plan.phases)} phases).")
        return plan
    # Same optical.yaml often already has experiment: — use it when present.
    try:
        from optical.config.loader import experiment_section, is_monolithic, load_config_file

        doc = load_config_file(tracker_config_path)
        if is_monolithic(doc):
            plan = ExperimentPlan.from_dict(experiment_section(doc))
            print(
                f"Loaded experiment plan from {tracker_config_path} "
                f"({len(plan.phases)} motion phases)."
            )
            return plan
    except Exception:
        pass
    return prompt_experiment_plan()


def _use_experiment_plan(tracker_config_path: Path) -> bool:
    """Read ``experiment.use_experiment_plan`` (default True for jaw-motion takes)."""
    try:
        from optical.config.loader import experiment_section, load_config_file

        exp = experiment_section(load_config_file(tracker_config_path))
        if "use_experiment_plan" in exp:
            return bool(exp.get("use_experiment_plan"))
    except Exception:
        pass
    return True


def _resolve_session_plan(
    *,
    tracker_config_path: Path,
    experiment_plan_path: Optional[Path],
    condition: str | None,
    use_experiment_plan: bool | None = None,
) -> tuple[ExperimentPlan, list, bool]:
    """Return ``(plan, optical_phases, use_plan)``.

    When ``use_experiment_plan`` is false: one free-recording phase (no landmark
    calib). Name/condition still come from optical.yaml ``experiment:``.
    """
    from optical.config.loader import experiment_section, load_config_file
    from optical.tracking.experiment import free_recording_plan

    if use_experiment_plan is None:
        use_experiment_plan = _use_experiment_plan(tracker_config_path)

    if not use_experiment_plan:
        if experiment_plan_path is not None:
            raise ValueError(
                "--experiment-plan requires experiment.use_experiment_plan: true"
            )
        name = "experiment"
        description = ""
        cond = (condition or "").strip()
        try:
            exp = experiment_section(load_config_file(tracker_config_path))
            name = str(exp.get("name") or name).strip() or name
            description = str(exp.get("description") or "").strip()
            if not cond:
                cond = str(exp.get("condition") or "").strip()
        except Exception:
            pass
        plan = free_recording_plan(name=name, condition=cond or None, description=description)
        # Use plan.phases (not all_phases) so landmark calibration is skipped.
        phases = list(plan.phases)
        print(
            f"Free recording (use_experiment_plan=false): "
            f"name={plan.name!r} phase={phases[0].name!r} — no landmark calib"
        )
        return plan, phases, False

    plan = _resolve_plan(experiment_plan_path, tracker_config_path)
    skipped_mvc = [p.name for p in plan.all_phases if not p.needs_optical]
    phases = [p for p in plan.all_phases if p.needs_optical]
    if not phases:
        raise RuntimeError(
            "experiment plan is MVC-only (EMG). Use `jtsmm record --mvc` "
            "for opening/closing MVC; `optical track` has no EMG path."
        )
    if skipped_mvc:
        print(
            "Skipping EMG-only MVC phases (use jtsmm record): "
            + ", ".join(skipped_mvc)
        )
    return plan, phases, True


def _resolve_persist_outputs(
    *,
    tracker_config_path: Path,
    plan: ExperimentPlan,
    persist_frames: bool,
    condition: str | None,
    output_path: Path | None,
    record_frames_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    """Return (output_path, record_frames_dir), applying --persist-frames layout.

    ``condition`` comes from CLI ``--condition``, else ``experiment.condition``
    in optical.yaml. ``experiment.name`` (via ``plan.name``) is the leaf name.
    """
    if not persist_frames:
        return output_path, record_frames_dir
    if record_frames_dir is not None:
        raise ValueError("use either --persist-frames or --record-frames, not both")

    from optical.config.loader import experiment_section, load_config_file

    doc = load_config_file(tracker_config_path)
    cond = (condition or "").strip()
    if not cond:
        try:
            exp = experiment_section(doc)
            cond = str(exp.get("condition") or "").strip()
        except Exception:
            cond = ""
    if not cond:
        raise ValueError(
            "--persist-frames needs a condition: set experiment.condition in "
            "optical.yaml or pass --condition"
        )

    leaf = resolve_persist_recording(doc, experiment_name=plan.name, condition=cond)
    leaf.mkdir(parents=True, exist_ok=True)
    print(f"Persist leaf: {leaf}")
    print(f"  experiment.name={plan.name!r}  condition={cond!r}")
    frames_dir = leaf
    h5_path = output_path or (leaf / SESSION_H5_NAME)
    return h5_path, frames_dir


def run_live(
    *,
    tracker_config_path: Path,
    experiment_plan_path: Optional[Path],
    camera_backend: str,
    preview_host: str,
    preview_port: int,
    output_path: Path | None,
    jts_config_path: Path | None,
    record_frames_dir: Path | None,
    persist_frames: bool = False,
    condition: str | None = None,
    use_experiment_plan: bool | None = None,
    hat_port: str | None = None,
    trigger_rate_hz: int | None = None,
) -> Path:
    """Configure phases, capture frames, then write session + JTS config."""
    tcfg = TrackerConfig.from_json(tracker_config_path)
    tracker = ArucoJawTracker(tcfg)
    # Fail before any camera opens when a configured tip file is missing.
    ct_tip = load_ct_tip_offset(tcfg)
    if ct_tip is not None:
        print(f"CT tip offset: {ct_tip[1]['ct_tip_body_mm']} mm ({ct_tip[1]['ct_tip_source']})")

    plan, phases, use_plan = _resolve_session_plan(
        tracker_config_path=tracker_config_path,
        experiment_plan_path=experiment_plan_path,
        condition=condition,
        use_experiment_plan=use_experiment_plan,
    )

    output_path, record_frames_dir = _resolve_persist_outputs(
        tracker_config_path=tracker_config_path,
        plan=plan,
        persist_frames=persist_frames,
        condition=condition,
        output_path=output_path,
        record_frames_dir=record_frames_dir,
    )

    calib_snapshot_name: str | None = None
    calib_snapshot_source: str | None = None
    if record_frames_dir is not None:
        arts = snapshot_persist_calibration(
            record_frames_dir,
            bundle_path=tcfg.calibration_bundle_path,
            config_path=tracker_config_path,
        )
        calib_snapshot_name = CALIBRATION_BUNDLE_SNAPSHOT_NAME
        calib_snapshot_source = str(tcfg.calibration_bundle_path.resolve())
        print(f"  snapshotted calibration → {arts['calibration_bundle'].name}")
        if "optical_config" in arts:
            print(f"  snapshotted config → {arts['optical_config'].name}")

    print(f"\nExperiment: {plan.name}")
    motion = any(p.kind == "experiment" for p in plan.phases)
    if use_plan and motion and plan.no_landmark_calibration:
        print("Landmark calibration: off (experiment.no_landmark_calibration)")
    elif use_plan and motion:
        print("Calibration (always included before motion):")
        for ph in CALIBRATION_PHASES:
            print(f"  • {ph.name}")
    print("Optical phases:")
    for ph in phases:
        print(f"  • {ph.name} ({ph.kind})")

    # Cameras for standalone capture come from optical.yaml calibration: (same file).
    # Tracking uses half-res lores Y (same as jtsmm record --optical-track).
    try:
        cam_cfg = _load_json(tracker_config_path)
        all_cams = _parse_cams(cam_cfg)
        cams = [c for c in all_cams if c.cam_id in tracker.cam_ids]
        if not cams:
            cams = all_cams
    except Exception:
        cams = [CamSpec(cam_id=cid, camera_num=i) for i, cid in enumerate(tracker.cam_ids)]
    enable_track_lores(cams)
    lores = cams[0].lores_size
    assert lores is not None
    tracker.scale_intrinsics_to(lores)
    print(f"Track stream: lores_y {lores[0]}x{lores[1]} (intrinsics scaled)")

    preview = PreviewServer(preview_host, preview_port)
    preview.start()
    print(f"\nPreview: {preview.url}")

    with _trigger_context(camera_backend, hat_port=hat_port, rate_hz=trigger_rate_hz):
        return _run_live_triggered(
            tracker=tracker,
            tcfg=tcfg,
            plan=plan,
            phases=phases,
            cams=cams,
            preview=preview,
            camera_backend=camera_backend,
            output_path=output_path,
            jts_config_path=jts_config_path,
            record_frames_dir=record_frames_dir,
            write_jts_config_file=use_plan,
            hat_port=hat_port,
            calibration_bundle=calib_snapshot_name,
            calibration_bundle_source=calib_snapshot_source,
            ct_tip=ct_tip,
        )


def _trigger_context(camera_backend: str, *, hat_port: str | None, rate_hz: int | None):
    """HAT trigger for the take: picamera2 only, on ``hat_port``, pulse deferred.

    The reference is taken now but the pulse held back until
    :func:`_run_live_triggered` arms it, immediately before picamera2 starts.
    An IMX296 in external-trigger mode delivers no frame until XTRIG pulses,
    and libcamera drops the frontend after 1 s with no buffer. PicamRig's own
    ``hat_trigger()`` nests inside and does not pulse early. ``rate_hz=None``
    uses mcu.trigger.rate_hz from the jtsmm config.
    """
    from contextlib import nullcontext

    if camera_backend != "picamera2":
        return nullcontext()
    return hat_trigger(rate_hz=rate_hz, defer=True, port=hat_port)


class _LiveOutputs:
    """Everything a live take writes, opened up front and closed in ``finally``.

    Poses go to the HDF5 incrementally (TrackingSessionWriter), replay PNGs
    through a bounded background writer, per-frame stamps to timestamps.jsonl.
    Whatever ends the loop — SPACE/Q, Ctrl+C or an exception — ``close()``
    still writes the session with every frame captured so far.
    """

    def __init__(
        self,
        *,
        out_path: Path,
        tracker: ArucoJawTracker,
        tcfg: TrackerConfig,
        plan: ExperimentPlan,
        cams: List[CamSpec],
        record_frames_dir: Path | None,
        frame_rate_hz: float | None,
        calibration_bundle: str | None,
        calibration_bundle_source: str | None,
    ) -> None:
        self.out_path = out_path
        self._cams = cams
        self._plan = plan
        self._record_frames_dir = record_frames_dir
        self._frame_rate_hz = frame_rate_hz
        self._calib = (calibration_bundle, calibration_bundle_source)
        self.writer = TrackingSessionWriter(
            out_path,
            rig_id=tracker.rig_id,
            experiment_name=plan.name,
            method=tcfg.pose_method,
            notes="live capture (phase-controlled)",
        )
        self.dump: AsyncFrameWriter | None = None
        self.stamps: FrameTimestampLog | None = None

    def record(self, frame_idx: int, raw: dict, record, t_cam_ns, t_master_us) -> None:
        if self._record_frames_dir is not None:
            if self.dump is None:
                lores = self._cams[0].lores_size if self._cams else None
                write_session_meta(
                    self._record_frames_dir,
                    cam_ids=[c.cam_id for c in self._cams],
                    frame_rate_hz=self._frame_rate_hz,
                    notes=f"live capture: {self._plan.name} (lores)",
                    image_size=lores,
                    stream="lores_y" if lores else None,
                    calibration_bundle=self._calib[0],
                    calibration_bundle_source=self._calib[1],
                )
                self.dump = AsyncFrameWriter(self._record_frames_dir)
                self.stamps = FrameTimestampLog(self._record_frames_dir)
                print(f"Recording replay frames to {self._record_frames_dir}")
            self.dump.submit(frame_idx, raw)
            self.stamps.append(frame_idx, t_cam_ns=t_cam_ns, t_master_us=t_master_us)
        self.writer.append(record)

    def close(self, *, notes: str, meta: dict) -> int:
        """Finish the dump and the HDF5; returns the number of frames written.

        The poses are the take, the dump and stamps are replay aids: a failure
        closing those is reported but never keeps the HDF5 from being closed.
        """
        try:
            if self.stamps is not None:
                self.stamps.close()
            if self.dump is not None:
                self.dump.close()
                meta["frame_dump"] = self.dump.stats()
                if self.dump.dropped or self.dump.failed:
                    print(
                        f"warning: replay dump incomplete: {self.dump.dropped} frame set(s) "
                        f"dropped (writer too slow), {self.dump.failed} failed"
                        + (f" (first: {self.dump.first_error})" if self.dump.first_error else ""),
                        file=sys.stderr,
                    )
        except Exception as exc:  # noqa: BLE001 - reported, the HDF5 still closes below
            print(f"warning: closing the replay dump failed: {exc}", file=sys.stderr)
            meta["frame_dump_error"] = str(exc)
        finally:
            self.writer.set_meta(**meta)
            self.writer.close(notes=notes)
        return self.writer.n_frames


def _close_phase(recorded: List[RecordedPhase], phase, start: int, next_idx: int) -> None:
    """Record frames ``[start, next_idx - 1]`` (inclusive, the JTS convention).

    A phase that captured no frame (SPACE pressed twice before a bundle came
    in) has no valid interval and is skipped with a note.
    """
    n = next_idx - start
    if n <= 0:
        print(f"  ■ '{phase.name}' captured no frames — not written to the JTS config")
        return
    recorded.append(RecordedPhase(phase.name, phase.kind, start, next_idx - 1))
    print(f"  ■ '{phase.name}' done: frames [{start}, {next_idx - 1}] ({n} frames)")


def _run_live_triggered(
    *,
    tracker: ArucoJawTracker,
    tcfg: TrackerConfig,
    plan: ExperimentPlan,
    phases,
    cams: List[CamSpec],
    preview: PreviewServer,
    camera_backend: str,
    output_path: Path | None,
    jts_config_path: Path | None,
    record_frames_dir: Path | None,
    write_jts_config_file: bool = True,
    hat_port: str | None = None,
    calibration_bundle: str | None = None,
    calibration_bundle_source: str | None = None,
    ct_tip: tuple | None = None,
) -> Path:
    picam = camera_backend == "picamera2"
    rate_hz: float | None = None
    if picam:
        rate_hz = float(active_trigger_rate_hz() or configured_trigger_rate_hz())
    tip_body_mm = ct_tip[0] if ct_tip is not None else None

    # IMX296 stays in external-trigger mode. Ingest opens first, then the
    # pulse, then picamera2. The frontend times out if the cameras are already
    # waiting when the (slow) trig_start CDC round-trip finally returns.
    hat_sync: HatSyncBridge | None = None
    if picam:
        hat_sync = HatSyncBridge(hat_port)
        port = hat_sync.start()
        print(f"HAT sync on {port} (MCU t_master_us via TRIGGER_EVENT)")

    rig = None
    try:
        if picam:
            dropped = hat_sync.reset_trigger_backlog() if hat_sync is not None else 0
            arm_hat_trigger()
            print(
                f"Trigger armed at {rate_hz:g} Hz before the cameras started "
                f"(dropped {dropped} early trigger event(s))"
            )
        rig = _open_rig(cams, camera_backend)
    finally:
        if rig is None and hat_sync is not None:
            hat_sync.stop()
    preview_order = [c.cam_id for c in cams]
    # capture_all_with_meta pairs cameras by read order; with two triggered
    # cameras use the SensorTimestamp-checked read instead.
    synced = picam and len(cams) > 1

    out_path = output_path or tracking_session_path(plan.slug)
    outputs: _LiveOutputs | None = None
    recorded: List[RecordedPhase] = []
    frame_idx = 0
    last_preview = 0.0

    print("=" * 70)
    print(f"{len(cams)} camera(s) ready (method={tcfg.pose_method})")
    print("SPACE: start/stop the current phase | Q or ESC: finish early | Ctrl+C: abort")
    print("=" * 70)
    print(f"Phase 1/{len(phases)}: '{phases[0].name}' ({phases[0].kind}) — SPACE to start")

    phase_i = 0
    phase_active = False  # frames are saved only while a phase is active
    phase_start_frame = 0
    aborted = False
    failure: BaseException | None = None
    try:
        outputs = _LiveOutputs(
            out_path=out_path,
            tracker=tracker,
            tcfg=tcfg,
            plan=plan,
            cams=cams,
            record_frames_dir=record_frames_dir,
            frame_rate_hz=rate_hz,
            calibration_bundle=calibration_bundle,
            calibration_bundle_source=calibration_bundle_source,
        )
        with _TerminalKeys() as keys:
            while phase_i < len(phases):
                if synced:
                    # Re-reads a lagging camera until the SensorTimestamps agree
                    # (one trigger); ({}, {}) when no aligned set was found.
                    frames, frame_meta = rig.capture_all_synced()
                else:
                    frames, frame_meta = rig.capture_all_with_meta()
                if not frames:
                    time.sleep(0.001)
                else:
                    # Always feed HAT sync while streaming so trigger sequence
                    # stays aligned during idle gaps between phases.
                    t_master_us = None
                    t_cam_ns = {cid: int(m.sensor_timestamp_ns) for cid, m in frame_meta.items()}
                    if hat_sync is not None:
                        t_cam_ns, t_master_us = hat_sync.align_frame_bundle(
                            frame_meta,
                            frame_idx=frame_idx if phase_active else -1,
                            cam_order=preview_order,
                        )

                    detections = None
                    if phase_active:
                        record = tracker.process_frame(
                            frames,
                            frame_idx=frame_idx,
                            t_cam_ns=t_cam_ns,
                            t_master_us=t_master_us,
                        )
                        apply_ct_tip(record, tip_body_mm)
                        outputs.record(frame_idx, frames, record, t_cam_ns, t_master_us)
                        q = record.quality
                        valid = " ".join(f"{b}={q.get(b, {}).get('valid')}" for b in ("HP", "MP", "CT"))
                        if frame_idx % 30 == 0:
                            print(f"  frame {frame_idx}: {valid}")
                        frame_idx += 1
                        detections = tracker.last_detections
                        status = f"REC {phases[phase_i].name}  f={frame_idx}  {valid}"
                        status_color = (0, 0, 220)

                    # The preview publishes at _PREVIEW_PERIOD_S; overlay, stack
                    # and status bar are only built when a frame will be sent.
                    now = time.monotonic()
                    if now - last_preview >= _PREVIEW_PERIOD_S:
                        last_preview = now
                        if detections is None:
                            detections = tracker.detect_all(frames)
                            n_markers = sum(len(d.markers) for d in detections.values())
                            status = (
                                f"IDLE  next={phases[phase_i].name}  "
                                f"markers={n_markers}  SPACE=start"
                            )
                            status_color = (0, 220, 0)
                        overlaid = overlay_frames_bgr(frames, detections, as_bgr=_as_bgr)
                        stacked = _stack_preview(overlaid, preview_order)
                        if stacked is not None:
                            preview.update(
                                _draw_status_bar(_fit_preview(stacked), status, status_color)
                            )

                key = keys.poll()
                if key in (27, ord("q")):  # ESC / Q finish early
                    if phase_active:
                        _close_phase(recorded, phases[phase_i], phase_start_frame, frame_idx)
                        print(f"  ended '{phases[phase_i].name}' (finishing early)")
                    print("Finishing — remaining phases skipped.")
                    break
                if key == ord(" "):
                    cur = phases[phase_i]
                    if not phase_active:
                        phase_active = True
                        phase_start_frame = frame_idx
                        print(
                            f"  ▶ '{cur.name}' RECORDING (frame {frame_idx}) — SPACE to end"
                        )
                    else:
                        _close_phase(recorded, cur, phase_start_frame, frame_idx)
                        phase_active = False
                        phase_i += 1
                        if phase_i < len(phases):
                            nxt = phases[phase_i]
                            print(
                                f"Phase {phase_i + 1}/{len(phases)}: "
                                f"'{nxt.name}' ({nxt.kind}) — SPACE to start"
                            )
    except KeyboardInterrupt:
        aborted = True
        print("\nAborted (Ctrl+C) — saving captured frames so far.")
    except Exception as exc:  # noqa: BLE001 - save what was captured, then re-raise
        aborted = True
        failure = exc
        print(f"\nERROR during capture: {exc!r} — saving captured frames so far.", file=sys.stderr)
    finally:
        # A phase interrupted by Ctrl+C / an error still covers its frames.
        if aborted and phase_active:
            _close_phase(recorded, phases[phase_i], phase_start_frame, frame_idx)
        try:
            rig.close()
        except Exception as exc:  # noqa: BLE001
            print(f"warning: camera close failed: {exc}", file=sys.stderr)
        meta: dict = {"trigger_after_cameras": False}
        if rate_hz is not None:
            meta["trigger_rate_hz"] = float(rate_hz)
        if synced:
            st = dict(getattr(rig, "sync_stats", None) or {})
            meta["pair_mode"] = "sensor_time"
            meta["pairing"] = st
            if st.get("resynced") or st.get("dropped"):
                print(
                    f"Camera pairing: {st.get('resynced', 0)} set(s) re-aligned, "
                    f"{st.get('dropped', 0)} dropped (SensorTimestamps disagreed)"
                )
        if ct_tip is not None:
            meta.update(ct_tip[1])
        if hat_sync is not None:
            try:
                report = hat_sync.sync_report()
                meta["sync_report"] = report
                drops = report.get("drops", {})
                slips = report.get("slip_count", {})
                if drops or slips:
                    print(f"HAT sync report: drops={drops} slip_count={slips}")
            finally:
                hat_sync.stop()
        if failure is not None:
            meta["capture_error"] = repr(failure)
        if outputs is not None:
            notes = "live capture (phase-controlled)"
            if failure is not None:
                notes += f" [aborted: {type(failure).__name__}]"
            elif aborted:
                notes += " [aborted]"
            n_written = outputs.close(notes=notes, meta=meta)
            print(f"\nWrote {n_written} frames to {out_path}")

    # Generate a ready-to-run JTS config from the captured intervals (jaw-motion plans only).
    if write_jts_config_file and recorded:
        jts_path = jts_config_path or out_path.parent / f"{plan.slug}_jts_config.json"
        cfg = build_jts_config(
            recorded, session_path=out_path, description=plan.description
        )
        write_jts_config(jts_path, cfg)
        print(f"Wrote JTS config to {jts_path}")
    elif not write_jts_config_file:
        print("Free recording — skipped JTS config generation.")
    else:
        print("No phases captured — skipped JTS config generation.")

    if failure is not None:
        raise failure
    return out_path


def main() -> None:
    """Argparse entry for ``python -m optical.tracking.run``."""
    import argparse

    ap = argparse.ArgumentParser(description="Live ArUco jaw tracker (HP/MP/CT poses).")
    ap.add_argument(
        "--config",
        required=True,
        help="optical.yaml (default: software/optical/etc/optical.yaml)",
    )
    ap.add_argument(
        "--experiment-plan",
        default=None,
        help="Experiment plan JSON (phase list). If omitted, you'll be prompted.",
    )
    ap.add_argument(
        "--use-experiment-plan",
        dest="use_experiment_plan",
        action="store_true",
        default=None,
        help="Force motion+landmark phase walk",
    )
    ap.add_argument(
        "--no-use-experiment-plan",
        dest="use_experiment_plan",
        action="store_false",
        help="Single free take (no landmark calib); default from optical.yaml",
    )
    ap.add_argument(
        "--backend",
        default="picamera2",
        choices=["picamera2", "v4l2"],
        help="Camera backend",
    )
    ap.add_argument("--output", default=None, help="Output HDF5 path (default: next to calibration bundle)")
    ap.add_argument(
        "--jts-config",
        default=None,
        help="Where to write the generated JTS config (default: jts_config.json next to session)",
    )
    ap.add_argument(
        "--record-frames",
        default=None,
        type=Path,
        help="Save synchronized camera PNGs for offline replay (explicit dir)",
    )
    ap.add_argument(
        "--persist-frames",
        action="store_true",
        help="Write session.h5 + frames under paths.recordings_dir "
        "(uses experiment.name/condition from YAML; --condition overrides)",
    )
    ap.add_argument(
        "--condition",
        default=None,
        help="Override experiment.condition from optical.yaml",
    )
    ap.add_argument("--preview-host", default="0.0.0.0")
    ap.add_argument("--preview-port", type=int, default=5000)
    ap.add_argument(
        "--hat-port",
        default=None,
        help="Serial port for HAT CDC (default: auto-detect /dev/jtsmm_hat or ttyACM*)",
    )
    ap.add_argument(
        "--trigger-rate",
        type=int,
        default=None,
        help="Camera trigger rate in Hz (default: mcu.trigger.rate_hz from jtsmm.yaml)",
    )
    args = ap.parse_args()

    run_live(
        tracker_config_path=Path(args.config),
        experiment_plan_path=Path(args.experiment_plan) if args.experiment_plan else None,
        camera_backend=args.backend,
        preview_host=args.preview_host,
        preview_port=args.preview_port,
        output_path=Path(args.output) if args.output else None,
        jts_config_path=Path(args.jts_config) if args.jts_config else None,
        record_frames_dir=Path(args.record_frames) if args.record_frames else None,
        persist_frames=bool(args.persist_frames),
        condition=args.condition,
        use_experiment_plan=args.use_experiment_plan,
        hat_port=args.hat_port,
        trigger_rate_hz=args.trigger_rate,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(130)
