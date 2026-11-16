# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Load, validate, and render ArUco markers for JTS rigid bodies (HP, MP, CT).

Marker IDs and mount transforms live under ``markers:`` in
``software/optical/etc/optical.yaml`` (auto 1- or 2-camera profile).
See ``docs/optical/01_marker_setup.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from optical.config.loader import load_config_file, markers_section

if TYPE_CHECKING:
    import cv2

# ArUco corner refinement used by the tracker (``markers.detector``).
DEFAULT_CORNER_REFINEMENT = "subpix"
DEFAULT_CORNER_REFINE_WIN_PX = 3


@dataclass(frozen=True)
class MarkerConfig:
    """
    Parsed marker layout for jaw tracking.

    Attributes:
        dictionary: OpenCV ArUco dictionary name (e.g. ``DICT_4X4_50``).
        tag_size_mm: Printed marker edge length in millimeters (used for PnP).
        id_ranges: Inclusive ID ranges per rigid body (``HP``, ``MP``, ``CT``).
        mounts_T_body_tag: Fixed 4×4 transforms ``T_body←tag`` per marker ID.
        corner_refinement: ArUco corner refinement (``subpix`` / ``contour`` / ``none``).
        corner_refine_win_px: Sub-pixel search half-window in pixels.
    """

    dictionary: str
    tag_size_mm: float
    id_ranges: Dict[str, Tuple[int, int]]
    mounts_T_body_tag: Dict[int, np.ndarray]
    corner_refinement: str = DEFAULT_CORNER_REFINEMENT
    corner_refine_win_px: int = DEFAULT_CORNER_REFINE_WIN_PX


def load_marker_config(path: str | Path) -> MarkerConfig:
    """
    Load markers from monolithic ``optical.yaml`` or a legacy marker JSON.

    Args:
        path: optical.yaml (uses ``markers:``) or legacy marker_config.json.

    Returns:
        Parsed :class:`MarkerConfig`.
    """
    path = Path(path)
    doc = load_config_file(path)
    cfg = markers_section(doc)

    det = cfg["detector"]
    dictionary = str(det["dictionary"])
    tag_size_mm = float(det["tag_size_mm"])

    id_ranges = {
        k: (int(v[0]), int(v[1]))
        for k, v in cfg["id_ranges"].items()
    }

    mounts = {}
    for k, T in cfg.get("mounts_T_body_tag", {}).items():
        mounts[int(k)] = np.asarray(T, dtype=np.float64)

    return MarkerConfig(
        dictionary=dictionary,
        tag_size_mm=tag_size_mm,
        id_ranges=id_ranges,
        mounts_T_body_tag=mounts,
        corner_refinement=str(det.get("corner_refinement", DEFAULT_CORNER_REFINEMENT)),
        corner_refine_win_px=int(det.get("corner_refine_win_px", DEFAULT_CORNER_REFINE_WIN_PX)),
    )


def validate_marker_config(cfg: MarkerConfig) -> List[str]:
    """
    Check marker config before printing or tracking.

    Args:
        cfg: Loaded marker configuration.

    Returns:
        Error messages; empty list if valid.
    """
    errors: List[str] = []

    if cfg.tag_size_mm <= 0:
        errors.append("detector.tag_size_mm must be > 0")

    for body, (lo, hi) in cfg.id_ranges.items():
        if lo > hi:
            errors.append(f"id_ranges.{body} has lo > hi ({lo} > {hi})")
        if lo < 0:
            errors.append(f"id_ranges.{body} has negative start ({lo})")

    # A tag ID in two ranges would be assigned to whichever body is checked first.
    ranges = sorted(cfg.id_ranges.items(), key=lambda kv: kv[1][0])
    for i, (body_a, (lo_a, hi_a)) in enumerate(ranges):
        for body_b, (lo_b, hi_b) in ranges[i + 1 :]:
            if lo_b <= hi_a and lo_a <= hi_b:
                errors.append(
                    f"id_ranges.{body_a} [{lo_a}, {hi_a}] overlaps "
                    f"id_ranges.{body_b} [{lo_b}, {hi_b}]"
                )

    for mid, T in cfg.mounts_T_body_tag.items():
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            errors.append(f"mounts_T_body_tag[{mid}] must be 4x4, got {T.shape}")
            continue
        if not np.all(np.isfinite(T)):
            errors.append(f"mounts_T_body_tag[{mid}] has non-finite entries")
            continue
        if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-6):
            errors.append(f"mounts_T_body_tag[{mid}] last row must be [0,0,0,1]")
        R = T[:3, :3]
        # CAD values are usually rounded to ~7 digits; 1e-4 still catches typos.
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0.0:
            errors.append(
                f"mounts_T_body_tag[{mid}] rotation is not orthonormal with det +1 "
                f"(det={np.linalg.det(R):.6f})"
            )
        if not any(lo <= int(mid) <= hi for lo, hi in cfg.id_ranges.values()):
            errors.append(f"mounts_T_body_tag[{mid}] is outside every id_ranges entry")

    if str(cfg.corner_refinement).strip().lower() not in ("none", "subpix", "contour"):
        errors.append(
            f"detector.corner_refinement must be none, subpix or contour "
            f"(got {cfg.corner_refinement!r})"
        )
    if int(cfg.corner_refine_win_px) < 1:
        errors.append("detector.corner_refine_win_px must be >= 1")

    return errors


def get_aruco_dictionary(name: str) -> "cv2.aruco.Dictionary":
    """
    Resolve an OpenCV predefined dictionary by constant name.

    Args:
        name: e.g. ``DICT_4X4_50``.

    Raises:
        ValueError: If ``name`` is not a valid ``cv2.aruco`` attribute.
    """
    import cv2

    try:
        dict_id = getattr(cv2.aruco, name)
    except AttributeError as e:
        raise ValueError(f"Unknown OpenCV aruco dictionary: {name}") from e
    return cv2.aruco.getPredefinedDictionary(dict_id)


def generate_marker_png(
    *,
    dictionary_name: str,
    marker_id: int,
    side_px: int,
    out_path: str | Path,
) -> Path:
    """
    Render one ArUco marker to a grayscale PNG for printing.

    Args:
        dictionary_name: OpenCV dictionary constant name.
        marker_id: Marker ID within that dictionary.
        side_px: Output image width/height in pixels.
        out_path: Destination PNG path.

    Returns:
        Resolved output path.
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = get_aruco_dictionary(dictionary_name)
    img = np.zeros((side_px, side_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(d, marker_id, side_px, img, 1)
    cv2.imwrite(str(out_path), img)
    return out_path


def make_charuco_board(
    *,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary_name: str,
) -> "cv2.aruco.CharucoBoard":
    """
    Build a ``cv2.aruco.CharucoBoard`` with geometry expressed in millimeters.

    The board's chessboard corners (``board.getChessboardCorners()``) come out
    in the same mm units, so calibration object points share the scale used by
    the plain-checkerboard path.

    Args:
        squares_x: Number of squares along the board width.
        squares_y: Number of squares along the board height.
        square_length_mm: Chessboard square edge length in millimeters.
        marker_length_mm: ArUco marker edge length in millimeters (< square).
        dictionary_name: OpenCV dictionary constant (e.g. ``DICT_4X4_50``).

    Returns:
        Configured ChArUco board.
    """
    import cv2

    if marker_length_mm >= square_length_mm:
        raise ValueError("marker_length_mm must be smaller than square_length_mm")

    d = get_aruco_dictionary(dictionary_name)
    return cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_length_mm),
        float(marker_length_mm),
        d,
    )


def generate_charuco_board_png(
    *,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dictionary_name: str,
    out_path: str | Path,
    dpi: int = 300,
    margin_mm: float = 5.0,
) -> Path:
    """
    Render a printable ChArUco board PNG at an exact physical scale.

    The output is sized so that one chessboard square measures
    ``square_length_mm`` when printed at ``dpi`` with scaling disabled.

    Args:
        squares_x: Squares along the board width.
        squares_y: Squares along the board height.
        square_length_mm: Square edge length in millimeters.
        marker_length_mm: Marker edge length in millimeters (< square).
        dictionary_name: OpenCV dictionary constant name.
        out_path: Destination PNG path.
        dpi: Print resolution used to convert millimeters to pixels.
        margin_mm: White quiet-zone border around the board, in millimeters.

    Returns:
        Resolved output path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    board = make_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
        dictionary_name=dictionary_name,
    )

    import cv2

    px_per_mm = dpi / 25.4
    width_px = int(round(squares_x * square_length_mm * px_per_mm))
    height_px = int(round(squares_y * square_length_mm * px_per_mm))
    margin_px = int(round(margin_mm * px_per_mm))

    # generateImage draws the board *inside* outSize minus the margin, so the
    # margin has to be added to the canvas; otherwise every square shrinks by
    # (W - 2·margin)/W (≈ 4 % for a 7×5 board at 5 mm) and calibration runs
    # with the wrong square length.
    img = board.generateImage(
        (width_px + 2 * margin_px, height_px + 2 * margin_px),
        marginSize=margin_px,
        borderBits=1,
    )
    cv2.imwrite(str(out_path), img)
    return out_path
