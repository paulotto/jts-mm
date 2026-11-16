# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""
Experiment plans and JTS config generation for the live tracker.

Each motion recording walks the experiment plan in order. If the plan
includes motion phases, the landmark-calibration phases are inserted
immediately before the first motion phase. Each phase is bounded live by
SPACE (start/stop).

EMG MVC is a separate ``jtsmm record --mvc`` take (``mvc_opening`` then
``mvc_closing``). It is not part of the motion plan.

Motion phase names come from the experiment plan JSON or manual entry.
Calibration is built in — it is not part of the plan file.

Recorded intervals are merged into a JTS config via :func:`build_jts_config`.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def slugify(text: str) -> str:
    """Filesystem-safe slug for naming generated files."""
    s = re.sub(r"[^\w\-]+", "_", (text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "experiment"


_KIND_ALIASES = {
    "experiment": "experiment",
    "calibration_mandibular": "calibration_mandibular",
    "calibration_maxillary": "calibration_maxillary",
    "calibration_tmj": "calibration_tmj",
    "mvc": "mvc",
}

# Names that mark EMG-only MVC (no cameras / ArUco). Any ``mvc_*`` slug also matches.
_MVC_NAMES = frozenset(
    {
        "mvc_opening",
        "mvc_closing",
        "mvc_open",
        "mvc_close",
        "open_mvc",
        "close_mvc",
        "mvc_bite",
        "mvc_clench",
    }
)


def infer_kind_from_name(name: str) -> str | None:
    """Return ``mvc`` for opening/closing MVC slugs, else ``None``."""
    key = (name or "").strip().lower()
    if key in _MVC_NAMES or key.startswith("mvc_"):
        return "mvc"
    return None


def normalize_kind(kind: str) -> str:
    """Map a kind string to a canonical phase kind (internal use)."""
    key = (kind or "experiment").strip().lower()
    if key in ("mand", "cal_mand", "mandibular"):
        key = "calibration_mandibular"
    elif key in ("max", "cal_max", "maxillary"):
        key = "calibration_maxillary"
    elif key in ("tmj", "cal_tmj"):
        key = "calibration_tmj"
    elif key in ("mvc", "emg_mvc", "mvc_only"):
        key = "mvc"
    elif key in ("", "exp"):
        key = "experiment"
    if key not in _KIND_ALIASES:
        raise ValueError(f"Unknown phase kind {kind!r}")
    return _KIND_ALIASES[key]


@dataclass
class Phase:
    """A single recording phase (calibration, MVC, or experiment motion)."""

    name: str
    kind: str = "experiment"

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Phase name must be non-empty")
        raw = (self.kind or "experiment").strip().lower()
        inferred = infer_kind_from_name(self.name)
        if inferred == "mvc" and raw in ("", "experiment", "exp"):
            self.kind = "mvc"
        else:
            self.kind = normalize_kind(self.kind)

    @property
    def needs_optical(self) -> bool:
        """False for EMG-only MVC (opening/closing clench against resistance)."""
        return self.kind != "mvc"

    @property
    def hint(self) -> Optional[str]:
        """What the operator touches with the CT tip (landmark phases only)."""
        return LANDMARK_HINTS.get(self.name)


MVC_PHASES: List[Phase] = [
    Phase("mvc_opening", "mvc"),
    Phase("mvc_closing", "mvc"),
]


def mvc_experiment_plan(*, name: str = "mvc") -> ExperimentPlan:
    """Dedicated EMG-only MVC take used by ``jtsmm record --mvc``."""
    return ExperimentPlan(
        phases=[Phase(p.name, p.kind) for p in MVC_PHASES],
        name=name,
        description="EMG MVC (opening + closing)",
    )


def free_recording_plan(
    *,
    name: str,
    condition: str | None = None,
    description: str = "",
) -> ExperimentPlan:
    """Single SPACE-gated take without landmark calibration (study / persist mode)."""
    phase = (condition or "recording").strip() or "recording"
    return ExperimentPlan(
        phases=[Phase(phase, "experiment")],
        name=name,
        description=description or name,
    )


# Fixed calibration sequence (always run before experiment motion phases).
# tmj_* are digitizing intervals only. They stay out of the mandibular and
# maxillary point lists: those three-point sets define the landmark frames.
CALIBRATION_PHASES: List[Phase] = [
    Phase("mand_point_1", "calibration_mandibular"),
    Phase("mand_point_2", "calibration_mandibular"),
    Phase("mand_point_3", "calibration_mandibular"),
    Phase("max_point_1", "calibration_maxillary"),
    Phase("max_point_2", "calibration_maxillary"),
    Phase("max_point_3", "calibration_maxillary"),
    Phase("tmj_left", "calibration_tmj"),
    Phase("tmj_right", "calibration_tmj"),
]

# Tooth touched with the CT tip in each landmark phase. Left/right are the
# participant's. Point 1 is the incisor because JTS puts the landmark-frame
# origin at coordinate_origin_index=0, and the template model_points below
# list the incisor first (x≈0) and the posterior teeth at ±24 mm. The model
# frame is right-handed with +x the participant's left, +y up and +z anterior,
# so point 2 (x = +23.6) is the left molar (tests/test_model_points_convention.py).
LANDMARK_TEETH: Dict[str, str] = {
    "mand_point_1": "lower central incisor (incisal edge, midline)",
    "mand_point_2": "lower left first molar",
    "mand_point_3": "lower right first molar",
    "max_point_1": "upper central incisor (incisal edge, midline)",
    "max_point_2": "upper left first molar",
    "max_point_3": "upper right first molar",
}

# Operator prompt per calibration phase: the teeth plus the palpated condyles.
LANDMARK_HINTS: Dict[str, str] = {
    **LANDMARK_TEETH,
    "tmj_left": "left condyle (skin-palpated, participant's left)",
    "tmj_right": "right condyle (skin-palpated, participant's right)",
}


@dataclass
class ExperimentPlan:
    """
    Motion phases for a recording session.

        Landmark calibration is inserted before the first motion phase at runtime
        (not stored in the plan), unless ``no_landmark_calibration`` is set.
        MVC is a separate ``jtsmm record --mvc`` take
        (see :func:`mvc_experiment_plan`). Plans whose motion phases are all
        ``imu_mount_*`` skip landmarks.
    """

    phases: List[Phase]
    name: str = "experiment"
    description: str = ""
    no_landmark_calibration: bool = False

    def __post_init__(self) -> None:
        self.name = self.name.strip() or "experiment"
        if not self.description:
            self.description = self.name
        for ph in self.phases:
            if ph.kind not in ("experiment", "mvc"):
                raise ValueError(
                    f"Experiment plan must contain motion or MVC phases only, "
                    f"got {ph.name!r} ({ph.kind})"
                )

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def all_phases(self) -> List[Phase]:
        """Plan order, with landmark calibration inserted before the first motion phase.

        MVC-only plans (``jtsmm record --mvc``) skip landmarks. IMU mounting
        figure-eights (``imu_mount_*``) also skip landmarks: the arrays are
        already tracked as rigid bodies. ``no_landmark_calibration`` skips the
        eight landmark phases and walks the configured motion phases only.
        """
        has_motion = any(p.kind == "experiment" for p in self.phases)
        if not has_motion or self.no_landmark_calibration:
            return list(self.phases)
        motion = [p for p in self.phases if p.kind == "experiment"]
        if motion and all(p.name.startswith("imu_mount") for p in motion):
            return list(self.phases)
        out: List[Phase] = []
        inserted = False
        for ph in self.phases:
            if ph.kind == "experiment" and not inserted:
                out.extend(CALIBRATION_PHASES)
                inserted = True
            out.append(ph)
        if not inserted:
            return list(CALIBRATION_PHASES) + list(self.phases)
        return out

    @property
    def needs_optical(self) -> bool:
        """True when any walked phase needs cameras / ArUco (not MVC-only)."""
        return any(p.needs_optical for p in self.all_phases)

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentPlan":
        raw = data.get("phases", [])
        if not raw:
            raise ValueError("Experiment plan has no 'phases'")
        phases: List[Phase] = []
        skipped_mvc = False
        for entry in raw:
            if isinstance(entry, str):
                name = entry.strip()
                if not name:
                    continue
                if infer_kind_from_name(name) == "mvc":
                    skipped_mvc = True
                    continue
                phases.append(Phase(name=name))
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            kind = normalize_kind(entry.get("kind", "experiment"))
            if kind in ("calibration_mandibular", "calibration_maxillary", "calibration_tmj"):
                # Ignore legacy cal entries — calibration is always built in.
                continue
            if kind == "mvc" or infer_kind_from_name(name) == "mvc":
                skipped_mvc = True
                continue
            phases.append(Phase(name=name, kind=kind))
        if not phases:
            hint = (
                " MVC names belong on `jtsmm record --mvc`, not the motion plan."
                if skipped_mvc
                else ""
            )
            raise ValueError("Experiment plan has no motion phases." + hint)
        return cls(
            phases=phases,
            name=data.get("name", "experiment"),
            description=data.get("description", ""),
            no_landmark_calibration=bool(data.get("no_landmark_calibration", False)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentPlan":
        """Load from optical.yaml ``experiment:`` or a legacy plan JSON."""
        from optical.config.loader import experiment_section, load_config_file

        doc = load_config_file(path)
        return cls.from_dict(experiment_section(doc))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "no_landmark_calibration": self.no_landmark_calibration,
            "phases": [p.name for p in self.phases],
        }


@dataclass
class RecordedPhase:
    """A phase with the frame interval captured live."""

    name: str
    kind: str
    start_frame: int
    end_frame: int  # inclusive: [start_frame, end_frame] is end - start + 1 frames (JTS)

    @property
    def interval(self) -> List[int]:
        return [int(self.start_frame), int(self.end_frame)]


# --------------------------------------------------------------------------- #
# Interactive configuration
# --------------------------------------------------------------------------- #
def prompt_experiment_plan() -> ExperimentPlan:
    """
    Ask the operator to either type motion phases or load a plan file.

    Runs in cooked terminal mode (plain ``input()``), so call this *before*
    entering the live cbreak loop.
    """
    print("=" * 70)
    print("Configure experiment")
    print("  Landmark calibration (8 points) is always included first:")
    for name, hint in LANDMARK_HINTS.items():
        print(f"    {name:13s} {hint}")
    print("  [m] enter motion phases manually")
    print("  [l] load an experiment plan JSON")
    print("=" * 70)
    while True:
        choice = input("Choose [m/l]: ").strip().lower()
        if choice in ("m", "manual"):
            return _prompt_manual_phases()
        if choice in ("l", "load"):
            path = input("Path to experiment plan JSON: ").strip()
            try:
                return ExperimentPlan.from_json(path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"  could not load plan: {exc}")
                continue
        print("  please type 'm' or 'l'.")


def _prompt_manual_phases() -> ExperimentPlan:
    print("-" * 70)
    exp_name = input("Experiment name: ").strip() or "experiment"
    print("-" * 70)
    print("Enter motion phases (one name per line). Empty line finishes.")
    print("  e.g.  open_close")
    print("        left_right")
    print("        chewing")
    print("  EMG MVC is a separate take: jtsmm record --mvc")
    print("-" * 70)
    phases: List[Phase] = []
    while True:
        line = input(f"motion {len(phases) + 1}> ").strip()
        if not line:
            if phases:
                break
            print("  add at least one motion phase.")
            continue
        name = line.split()[0]
        try:
            phases.append(Phase(name=name))
        except ValueError as exc:
            print(f"  {exc}")
    return ExperimentPlan(phases=phases, name=exp_name)


# --------------------------------------------------------------------------- #
# JTS config generation
# --------------------------------------------------------------------------- #
def _default_jts_config() -> Dict:
    """Full JTS config template with sane defaults (intervals filled later)."""
    return {
        "data_source": {
            "type": "aruco_camera",
            "mode": "offline",
            "filename": "",
        },
        "analysis": {
            "description": "Jaw motion analysis experiment",
            "calibration": {
                "mandibular": {"rigid_bodies": ["MP", "CT"], "points": []},
                "maxillary": {"rigid_bodies": ["HP", "CT"], "points": []},
            },
            "experiment": {
                "frame_interval": [0, 0],
                "use_sub_experiments": False,
                "combine_sub_experiments": [],
                "sub_experiments": {},
                "interpolation": {
                    "method": "cubic",
                    "transition_frames": 10,
                    "connect_intervals": True,
                },
            },
            "relative_motion": {"reference_body": "HP", "moving_body": "MP"},
            "coordinate_transform": {
                "enabled": True,
                "calibration_type": "maxillary",
                "model_points": [
                    [-0.244678, -109.416, 101.356],
                    [23.6189, -106.201, 73.9031],
                    [-24.0445, -106.078, 74.1795],
                ],
            },
            "smoothing": {"enabled": True, "window_length": 11, "poly_order": 3},
            "coordinate_origin_index": 0,
        },
        "visualization": {
            "raw_data": True,
            "raw_data_3d": False,
            "calibration_transforms": True,
            "relative_marker_motion": True,
            "landmark_motion": True,
            "final_trajectory": True,
            "plot_2d_components": True,
            "plot_export_only": True,
            "plot_rot_3d": False,
            "show_plots": False,
            "use_tex_font": False,
            "plot_style": {
                "sample_rate": 10,
                "linewidth": 1.5,
                "axes_label_fontsize": 12,
                "axes_tick_fontsize": 10,
                "title_fontsize": 14,
                "legend_fontsize": 10,
                "figure_size": [12, 8],
                "labelpad_scale": 0.7,
                "view_3d": {"elev": 30, "azim": 0, "roll": 0, "vertical_axis": "y"},
                "colors": [
                    "tab:blue",
                    "tab:orange",
                    "tab:green",
                    "tab:red",
                    "tab:purple",
                    "tab:brown",
                ],
                "line_styles": ["-", "--", "-.", ":"],
                "grid": {"enabled": True, "alpha": 0.3},
            },
        },
        "output": {
            "directory": "./analysis_results",
            "save_csv": True,
            "save_hdf5": True,
            "save_plots": False,
            "save_tikz": False,
            "csv_filename": "jaw_motion",
            # When set, JTS appends kinematics into data_source.filename under
            # this group (e.g. session.h5:/jts) instead of writing jaw_motion.h5.
            "hdf5_append_group": "jts",
            "hdf5_filename": "jaw_motion.h5",
            "store_quaternions": True,
            "derivative_order": 2,
            "scale_factor": 0.001,
            "unit": "m",
        },
    }


def build_jts_config(
    recorded: Sequence[RecordedPhase],
    *,
    session_path: str | Path,
    description: str = "Jaw motion analysis experiment",
    base: Optional[Dict] = None,
) -> Dict:
    """
    Merge recorded phase intervals into a full JTS config dict.

    Args:
        recorded: Phases captured live (with frame intervals).
        session_path: Path to the tracker output the config should point at.
        description: Experiment description.
        base: Optional template to fill (defaults to :func:`_default_jts_config`).
    """
    cfg = copy.deepcopy(base) if base is not None else _default_jts_config()

    cfg["data_source"]["type"] = "aruco_camera"
    cfg["data_source"]["filename"] = str(session_path)
    cfg["analysis"]["description"] = description

    mand = [r for r in recorded if r.kind == "calibration_mandibular"]
    maxl = [r for r in recorded if r.kind == "calibration_maxillary"]
    exp = [r for r in recorded if r.kind == "experiment"]

    calib = cfg["analysis"]["calibration"]
    calib["mandibular"]["points"] = [
        {"name": r.name, "frame_interval": r.interval} for r in mand
    ]
    calib["maxillary"]["points"] = [
        {"name": r.name, "frame_interval": r.interval} for r in maxl
    ]

    experiment = cfg["analysis"]["experiment"]
    experiment["sub_experiments"] = {r.name: r.interval for r in exp}
    experiment["use_sub_experiments"] = len(exp) > 1
    experiment["combine_sub_experiments"] = [r.name for r in exp]

    span_src = exp or list(recorded)
    if span_src:
        experiment["frame_interval"] = [
            min(r.start_frame for r in span_src),
            max(r.end_frame for r in span_src),
        ]

    return cfg


def write_jts_config(path: str | Path, cfg: Dict) -> Path:
    """Write a JTS config dict to ``path`` as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path
