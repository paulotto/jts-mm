# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Minimal RGB→JPEG encode for viewer camera previews."""

from __future__ import annotations

import io

import numpy as np

PREVIEW_JPEG_QUALITY = 80

__all__ = [
    "PLACEHOLDER_JPEG",
    "PREVIEW_JPEG_QUALITY",
    "encode_jpeg",
    "encode_jpeg_rgb",
    "synthetic_preview_rgb",
]


def encode_jpeg(image: np.ndarray, *, quality: int = PREVIEW_JPEG_QUALITY) -> bytes:
    """Encode HxW gray or HxWx3 RGB to JPEG."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        try:
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=int(quality))
            return buf.getvalue()
        except Exception:
            pass
        try:
            import cv2  # type: ignore

            ok, buf = cv2.imencode(
                ".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            )
            if ok:
                return buf.tobytes()
        except Exception:
            pass
        return PLACEHOLDER_JPEG
    return encode_jpeg_rgb(arr, quality=quality)


def encode_jpeg_rgb(rgb: np.ndarray, *, quality: int = PREVIEW_JPEG_QUALITY) -> bytes:
    """Encode HxWx3 uint8 RGB to JPEG (Pillow → OpenCV → tiny placeholder)."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return PLACEHOLDER_JPEG
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=int(quality))
        return buf.getvalue()
    except Exception:
        pass
    try:
        import cv2  # type: ignore

        ok, buf = cv2.imencode(
            ".jpg", arr[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    return PLACEHOLDER_JPEG


def synthetic_preview_rgb(
    cam_id: str, *, size: tuple[int, int] = (320, 240), t: float = 0.0
) -> np.ndarray:
    """Cheap animated RGB for mock cams (no hardware)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    phase = (hash(cam_id) % 97) / 97.0
    r = ((xx / max(w - 1, 1) + 0.2 * np.sin(t + phase)) * 180 + 40).astype(np.uint8)
    g = ((yy / max(h - 1, 1) + 0.2 * np.cos(t * 0.7)) * 120 + 30).astype(np.uint8)
    b = np.full((h, w), 40 + (hash(cam_id) % 80), dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


def _build_placeholder() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (320, 240), (18, 21, 31)).save(buf, format="JPEG", quality=40)
        return buf.getvalue()
    except Exception:
        # Valid 1×1 gray JPEG (no third-party deps).
        return bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300"
            + ("08" * 64)
            + "ffc0000b080001000101011100"
            "ffc400140001000000000000000000000000000009"
            "ffc400141001000000000000000000000000000000"
            "ffda0008010100003f007fff d9".replace(" ", "")
        )


PLACEHOLDER_JPEG = _build_placeholder()
