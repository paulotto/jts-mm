# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pseudo-online (simulated real-time) evaluation: X4 (change log), docs/ml/ml08_online.md.

A recorded take is replayed chunk by chunk (default 20 ms of EMG samples) through a strictly
causal chain; nothing reads a sample after the current chunk.

* :mod:`ml.online.stream`   chunked replay driver: stateful causal EMG chain, causal masks,
  IMU chain, ring buffers and window features from the same functions as the offline windows
* :mod:`ml.online.decoder`  per-fold models, calibration modes, decision smoothing, onsets
* :mod:`ml.online.release`  the X3 closer-release detector, streaming, on the causal chain
* :mod:`ml.online.metrics`  stream scores, latency, false activations, X3 metrics, compute time
* :mod:`ml.online.stage`    the cohort stage ``ml_online`` of the biomech pipeline
"""
