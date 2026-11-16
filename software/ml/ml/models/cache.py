# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Content-addressed cache of fitted models (speed only; invisible to results).

``ml_eval`` fits the LOSO fold models; ``ml_online`` needs the very same models (same
training rows, hyper-parameters, seed and subsample) and used to refit them. With a cache
directory (``<out>/model_cache``, next to the result store's ``cache``), :func:`fit` stores
every model it fits under a key that covers *everything* that determines the fit:

* the exact training inputs (bytes, dtype and shape of X, y, groups, sample weights);
* the estimator class and all its parameters (``get_params``: hyper-parameters, seed,
  block slices, probability);
* the code: hashes of the model modules' source files and the versions of Python, numpy,
  scipy, scikit-learn and the BLAS library.

A later :func:`fit` with an equal key loads the stored model instead of fitting (the
fitted object is the one an identical fit produced, so its predictions are bit-identical);
any difference in data, parameters or code is a different key and a fresh fit. Entries are
written atomically; unreadable entries are ignored (refit). The directory is pruned to
:data:`MAX_BYTES` (least recently used first).
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

MAX_BYTES = 4 << 30
_FINGERPRINT: str | None = None
_SOURCES = ("ml/models/registry.py", "ml/models/base.py", "ml/models/gru.py",
            "ml/fusion/early.py", "ml/fusion/late.py", "ml/evaluation/metrics.py",
            "ml/config/classes.py")


def code_fingerprint() -> str:
    global _FINGERPRINT
    if _FINGERPRINT is None:
        import scipy
        import sklearn

        h = hashlib.blake2b(digest_size=16)
        root = Path(__file__).resolve().parents[2]
        for rel in _SOURCES:
            p = root / rel
            h.update(rel.encode())
            h.update(p.read_bytes() if p.is_file() else b"-")
        blas = ""
        try:
            from threadpoolctl import threadpool_info

            blas = repr(sorted((d.get("internal_api"), d.get("version"), d.get("architecture"))
                               for d in threadpool_info() if d.get("user_api") == "blas"))
        except Exception:                                   # pragma: no cover
            pass
        h.update(repr((sys.version, platform.machine(), np.__version__, scipy.__version__,
                       sklearn.__version__, blas)).encode())
        _FINGERPRINT = h.hexdigest()
    return _FINGERPRINT


def _arr(h: Any, name: str, a: Any) -> None:
    h.update(name.encode())
    if a is None:
        h.update(b"None")
        return
    a = np.asarray(a)
    if a.dtype == object:
        a = a.astype(str)
    a = np.ascontiguousarray(a)
    h.update(repr((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())


def fit_key(est: Any, X: Any, y: Any, groups: Any = None, sample_weight: Any = None) -> str:
    h = hashlib.blake2b(digest_size=20)
    h.update(code_fingerprint().encode())
    h.update(f"{type(est).__module__}.{type(est).__qualname__}".encode())
    h.update(repr(sorted(est.get_params(deep=False).items())).encode())
    for name, a in (("X", X), ("y", y), ("groups", groups), ("w", sample_weight)):
        _arr(h, name, a)
    return h.hexdigest()


def _prune(d: Path, max_bytes: int) -> None:
    try:
        files = [(p.stat().st_mtime, p.stat().st_size, p) for p in d.glob("*.joblib")]
    except OSError:
        return
    total = sum(s for _, s, _ in files)
    for _, s, p in sorted(files, key=lambda r: r[0]):
        if total <= max_bytes:
            break
        try:
            p.unlink()
            total -= s
        except OSError:
            pass


def fit(est: Any, X: Any, y: Any, *, groups: Any = None, sample_weight: Any = None,
        cache_dir: str | Path | None = None) -> tuple[Any, float, bool]:
    """``est.fit(X, y, groups=groups[, sample_weight=...])`` through the cache.

    Returns ``(fitted estimator, fit time in s, hit)``; on a hit the fit time is that of
    the original fit."""
    kw: dict[str, Any] = {"groups": groups}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    if cache_dir is None:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X, y, **kw)
        return est, time.perf_counter() - t0, False
    import joblib

    d = Path(cache_dir)
    key = fit_key(est, X, y, groups, sample_weight)
    path = d / f"{key}.joblib"
    if path.is_file():
        try:
            entry = joblib.load(path)
            got = entry.get("est") if isinstance(entry, dict) else None
            if entry.get("key") == key and type(got) is type(est) and \
                    got.get_params(deep=False).keys() == est.get_params(deep=False).keys():
                try:
                    os.utime(path)
                except OSError:
                    pass
                return got, float(entry.get("fit_s", np.nan)), True
        except Exception:                       # unreadable / foreign entry: refit
            pass
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X, y, **kw)
    dt = time.perf_counter() - t0
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.{os.getpid()}.{time.monotonic_ns()}.tmp"
        joblib.dump({"key": key, "est": est, "fit_s": dt}, tmp)
        os.replace(tmp, path)
        _prune(d, MAX_BYTES)
    except OSError:                             # a read-only or full disk only costs speed
        pass
    return est, dt, False


__all__ = ["MAX_BYTES", "code_fingerprint", "fit", "fit_key"]
