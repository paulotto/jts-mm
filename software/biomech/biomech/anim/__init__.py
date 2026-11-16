# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Animated, synchronized views of one take or of the cohort (an11).

Entry points:

* :func:`biomech.anim.layouts.make` -- load a clip and return an :class:`Animation`;
* :func:`biomech.anim.render_video.render_video` -- MP4/WebM/GIF, parallel chunks,
  progress callback and cancel (the viewer's *Render video* button);
* :func:`biomech.anim.render_plotly.preview_figure` -- Plotly frames + slider (the
  viewer's interactive preview);
* :mod:`biomech.anim.render_latex` -- vector filmstrip (pgfplots) and ``animate``-package
  frames for LaTeX;
* :mod:`biomech.anim.supplementary` -- the D5 supplementary-video layouts (``v1_hinge``,
  ``v2_intent``, ``v3_release``), registered on import;
* :mod:`biomech.anim.validation_video` -- the systems-paper layout (``validation``: JTS-MM
  against Qualisys from the stored validation results), registered on import;
* :mod:`biomech.anim.r8_videos` -- J2.7 layouts (``workspace3d``: the cohort's
  incisal workspace rotating in 3-D; ``emg_spectrogram``: EMG with live spectrum strips;
  ``onset_rules``: the C2 onset rules explained on real cycles), registered on import;
* :mod:`biomech.anim.clipfile` -- ``<stem>.clip.npz/.json``: everything a render needs,
  re-rendered without the result store or the session (``biomech animate --from-clip``).

The package is presentation code (``pipeline.PRESENTATION``): editing it never marks
analysis results stale.
"""

from biomech.anim.layouts import (LAYOUTS, SIZES, Animation, Scene, default_phases,
                                  default_speed, make, video_stem)
from biomech.anim import supplementary  # noqa: F401  (registers the D5 layouts)
from biomech.anim import validation_video  # noqa: F401  (registers the systems-paper layout)
from biomech.anim import r8_videos  # noqa: F401  (registers the J2.7 layouts)
from biomech.anim.timecode import read_timecode
from biomech.anim.timeline import Timeline

__all__ = ["LAYOUTS", "SIZES", "Animation", "Scene", "Timeline", "default_phases",
           "default_speed", "make", "read_timecode", "video_stem"]
