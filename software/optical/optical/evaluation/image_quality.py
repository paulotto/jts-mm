# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Image-quality profiles for offline size × resolution tradeoff evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import cv2
import numpy as np

from optical.calibration.geometry import scale_K


@dataclass(frozen=True)
class QualityProfile:
    """One offline degradation setting applied to full-resolution RGB frames.

    ``width`` / ``height`` are the target canvas size after resize. Optional
    ``blur_sigma``, ``jpeg_quality``, and ``noise_std`` simulate ISP / compression
    loss on top of the resolution change.
    """

    name: str
    width: int
    height: int
    blur_sigma: float = 0.0
    jpeg_quality: int | None = None
    noise_std: float = 0.0

    @property
    def size(self) -> Tuple[int, int]:
        return (int(self.width), int(self.height))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "blur_sigma": self.blur_sigma,
            "jpeg_quality": self.jpeg_quality,
            "noise_std": self.noise_std,
        }


def parse_quality_profiles(entries: Sequence[dict[str, Any]]) -> list[QualityProfile]:
    """Parse quality ladder entries from an eval config JSON."""
    out: list[QualityProfile] = []
    for entry in entries:
        if "width" in entry and "height" in entry:
            width = int(entry["width"])
            height = int(entry["height"])
        elif "scale" in entry:
            raise ValueError(
                "quality entry uses 'scale' without absolute size; "
                "provide width/height (scales are applied relative to captured frames at runtime)"
            )
        else:
            raise ValueError(f"quality entry needs width+height: {entry!r}")
        name = str(entry.get("name", f"{width}x{height}"))
        jpeg = entry.get("jpeg_quality")
        out.append(
            QualityProfile(
                name=name,
                width=width,
                height=height,
                blur_sigma=float(entry.get("blur_sigma", 0.0)),
                jpeg_quality=int(jpeg) if jpeg is not None else None,
                noise_std=float(entry.get("noise_std", 0.0)),
            )
        )
    return out


def scale_intrinsics(
    K: np.ndarray,
    *,
    src_wh: Tuple[int, int],
    dst_wh: Tuple[int, int],
) -> np.ndarray:
    """Scale a pinhole ``K`` when the image is resized from ``src_wh`` to ``dst_wh``.

    Thin wrapper over :func:`optical.calibration.geometry.scale_K` (pixel-centre
    aligned principal point; raises on > 0.5 % aspect change). Distortion
    coefficients are left unchanged.
    """
    return scale_K(K, src_wh, dst_wh)


def degrade_rgb(rgb: np.ndarray, profile: QualityProfile) -> np.ndarray:
    """Apply a quality profile to an RGB uint8 frame.

    Order: resize (AREA) → optional Gaussian blur → optional JPEG round-trip →
    optional additive Gaussian noise.
    """
    if rgb is None or rgb.size == 0:
        raise ValueError("empty rgb frame")
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {arr.shape}")
    arr = arr[:, :, :3]
    h, w = arr.shape[:2]
    dst_w, dst_h = profile.size
    if (w, h) != (dst_w, dst_h):
        arr = cv2.resize(arr, (dst_w, dst_h), interpolation=cv2.INTER_AREA)

    if profile.blur_sigma > 0:
        # Kernel size must be odd and large enough for the sigma.
        k = max(3, int(round(profile.blur_sigma * 4)) | 1)
        arr = cv2.GaussianBlur(arr, (k, k), profile.blur_sigma)

    if profile.jpeg_quality is not None:
        q = int(np.clip(profile.jpeg_quality, 1, 100))
        # OpenCV imencode expects BGR channel order.
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("JPEG encode failed during quality degrade")
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("JPEG decode failed during quality degrade")
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if profile.noise_std > 0:
        noise = np.random.normal(0.0, profile.noise_std, size=arr.shape)
        arr = np.clip(arr.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    else:
        arr = np.ascontiguousarray(arr, dtype=np.uint8)

    return arr


def resolve_profile_for_frame(
    entry: dict[str, Any],
    *,
    src_wh: Tuple[int, int],
) -> QualityProfile:
    """Build a :class:`QualityProfile` from config, supporting relative ``scale``."""
    src_w, src_h = (int(src_wh[0]), int(src_wh[1]))
    if "width" in entry and "height" in entry:
        width = int(entry["width"])
        height = int(entry["height"])
        name = str(entry.get("name", f"{width}x{height}"))
    elif "scale" in entry:
        scale = float(entry["scale"])
        if scale <= 0:
            raise ValueError(f"quality scale must be > 0, got {scale}")
        width = max(1, int(round(src_w * scale)))
        height = max(1, int(round(src_h * scale)))
        name = str(entry.get("name", f"scale_{scale:g}"))
    else:
        raise ValueError(f"quality entry needs width+height or scale: {entry!r}")
    jpeg = entry.get("jpeg_quality")
    return QualityProfile(
        name=name,
        width=width,
        height=height,
        blur_sigma=float(entry.get("blur_sigma", 0.0)),
        jpeg_quality=int(jpeg) if jpeg is not None else None,
        noise_std=float(entry.get("noise_std", 0.0)),
    )
