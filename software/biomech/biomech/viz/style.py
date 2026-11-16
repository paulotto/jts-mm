# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared style tokens for every renderer (pgf, matplotlib, Plotly, video).

Roles name *what* a series is ("primary", "model:hinge_fit", "sex:f", ...);
all renderers map a role to the same colour, dash and marker, so the paper,
the viewer and the videos look alike.

Colour jobs (dataviz rules):
* categorical: Okabe-Ito in one fixed order (:data:`CATEGORICAL`), assigned by
  entity, never cycled; at most 6 hues on one chart, more series fold into
  small multiples or the emphasis form;
* emphasis: one accent (``accent``) over de-emphasized context (``context``);
* sequential: one hue light -> dark (:data:`SEQUENTIAL`) for magnitudes
  (heatmaps, speed or activation along a path);
* diverging: two poles around a neutral grey (:data:`DIVERGING`).
Text never wears a series colour; references are the only dashed/dotted lines (besides the
left-side muscles and one feature set, where the dash is a secondary encoding).

Venues (:data:`VENUES`, :func:`use_venue`): IEEE (default), Springer and Nature profiles map
``FigureSpec.width`` to inches and give absolute font sizes and the figure font family; every
renderer lays out with the active venue. :func:`unicode_minus` gives text a real minus.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

# Okabe-Ito (colour-blind safe) plus neutrals, hex without '#'.
PALETTE = {
    "black": "000000", "orange": "E69F00", "sky": "56B4E9", "green": "009E73",
    "yellow": "F0E442", "blue": "0072B2", "vermillion": "D55E00", "purple": "CC79A7",
    "grey": "8C8C8C", "lightgrey": "D0D0D0", "ink": "1F2328", "ink2": "57606A",
}
# Fixed categorical order. Validated with the dataviz palette validator (2026-10-15):
# light on #FFFFFF passes (adjacent CVD dE >= 9.6, normal >= 16.4; orange and sky are below
# 3:1 contrast, so identity is always also carried by legends / direct labels / markers);
# dark uses its own steps (PALETTE_DARK) inside the dark lightness band 0.48-0.67 on #0D1117
# (adjacent CVD dE >= 8.6, normal >= 17.2, contrast >= 3:1). All-pairs forms add marker shape.
CATEGORICAL = ("blue", "orange", "green", "vermillion", "purple", "sky")
PALETTE_DARK = {"blue": "1F80C4", "orange": "C28400", "green": "00A27A", "vermillion": "DA6A12",
                "purple": "B46399", "sky": "3E9BD2", "yellow": "B8A92A", "black": "E6EDF3"}

# Light -> dark single-hue ramp (blues) and a blue-grey-vermillion diverging ramp.
# Ramps anchored to the categorical template (colour harmonisation 2026-11-05): sequential through
# the palette's blue (0072B2), diverging blue (0072B2) - grey - vermillion (D55E00); steps
# interpolated in OKLab (even lightness). Before: ColorBrewer Blues / a brown-orange pole.
SEQUENTIAL = ("F4F8FB", "BFD6EA", "8AB5D8", "5493C6", "0072B2", "056097", "084F7E", "093F65",
              "082F4D")
DIVERGING = ("0072B2", "5F95C1", "9DB7CE", "D9D9D9", "DDB29E", "DB8A61", "D55E00")
# Dark-theme ramps (viewer, videos): on the #161B22 panel the low end must not glare and the
# high end must stay visible, so lightness runs the other way -- sequential from a mid-dark
# blue (still visible as a line) to near white; diverging from light poles to a dark neutral
# centre (a large |value| is a large contrast with the background in both themes).
SEQUENTIAL_DARK = ("0E3A5C", "14507D", "1968A0", "1F80C4", "6AA5D6", "A6C9E7", "E2EEF7")
DIVERGING_DARK = ("9CCBEA", "5099D1", "316B9A", "3A424B", "A45F38", "E48346", "F4B48A")

# Surfaces and ink per theme (viewer / video). The paper is always light.
THEMES = {
    "light": {"surface": "FFFFFF", "panel": "FFFFFF", "ink": "1F2328", "ink2": "57606A",
              "muted": "8C959F", "grid": "E6E8EB", "axis": "8C959F", "context": "B8BEC6",
              "faint": "D8DCE0"},
    "dark": {"surface": "0D1117", "panel": "161B22", "ink": "E6EDF3", "ink2": "A8B3BD",
             "muted": "7D8590", "grid": "262C34", "axis": "6E7781", "context": "5C6670",
             "faint": "3A424B"},
}

# role -> (colour, line style, marker)
# One meaning per colour (D2):
# * ``accent`` (and its spread ``band``) is the cohort summary: the ink colour, never a hue,
#   so it can sit next to any categorical series without claiming its meaning;
# * muscles always wear their entity colour (ENTITY); the opener / closer groups take the
#   hue of their lead muscle (digastric R vermillion, masseter R blue);
# * protocols and feature sets use hues distinct from the opener / closer hues (orange, sky,
#   green, purple; validated pairwise in both themes for the protocol pair, adjacently for
#   the feature sets, which also differ by marker and dash);
# * limits and references are grey dashed lines, labelled directly at the line;
# * no two roles that can share a panel use the same marker and colour ('v' is a downward
#   triangle in every renderer, '^' an upward one).
ROLES: dict[str, tuple[str, str, str]] = {
    "primary": ("blue", "solid", "o"),
    "secondary": ("orange", "solid", "s"),
    "tertiary": ("green", "solid", "^"),
    "quaternary": ("vermillion", "solid", "d"),
    "accent": ("ink", "solid", "o"),
    # The panel's result over grey participants (P95 marker, median, design box; JNER revision
    # 3.0, 2026-11-10): one hue so the eye finds the result first. Blue: only in figures where
    # blue has no other meaning (closers, the compromise axis, JTS-MM); there the meaning's own
    # hue carries the summary.
    "summary": ("blue", "solid", "d"),
    "summary:fill": ("blue", "solid", "none"),
    "summary:out": ("vermillion", "solid", "none"),
    "context": ("grey", "solid", "none"),
    "muted": ("grey", "solid", "none"),
    "faint": ("lightgrey", "solid", "none"),
    # flagged or below-minimum units, shown but not summarised: a mid-grey down triangle at
    # dot size, readable in print next to the grey dots
    "flagged": ("ink2", "solid", "v"),
    "reference": ("ink2", "dashed", "none"),
    "threshold": ("ink2", "dashed", "none"),
    "band": ("ink", "solid", "none"),
    "model:hinge_fit": ("blue", "solid", "o"),
    "model:hinge_tmj": ("vermillion", "solid", "s"),
    "model:rot_trans": ("green", "solid", "^"),
    "noise": ("grey", "dashed", "none"),
    "sex:f": ("purple", "solid", "o"),
    "sex:m": ("green", "solid", "s"),
    "side:left": ("blue", "solid", "o"),
    "side:right": ("orange", "solid", "s"),
    "group:openers": ("vermillion", "solid", "none"),
    "group:closers": ("blue", "solid", "none"),
    "set:emg": ("orange", "solid", "o"),
    "set:imu": ("sky", "solid", "s"),
    "set:emg_imu": ("green", "solid", "^"),
    "set:emg_nodig": ("purple", "solid", "d"),
    "set:emg_nodig_imu": ("purple", "dashed", "v"),
    # The two comparison takes where one panel overlays them (validation, 2026-11-05): colour
    # and line style both carry the take; marker shape = participant as everywhere (X1).
    # Vermillion / sky: OKLab dE 31 normal, >= 25 under protan / deutan / tritan simulation,
    # different lightness (orange / purple was 8.5 for tritan); never blue (= JTS-MM).
    "participant:0": ("vermillion", "solid", "o"),
    "participant:1": ("sky", "dashed", "s"),
    "protocol:loso": ("orange", "solid", "o"),
    "protocol:blocked": ("sky", "solid", "s"),
    # Measurement systems (validation study): the system under test solid blue, the
    # reference systems in ink (black / near-white), Qualisys dashed, Delsys dotted -- never
    # the grey of reference lines, so a system and a limit line are not confused.
    "system:jtsmm": ("blue", "solid", "o"),
    "system:qualisys": ("ink", "dashed", "s"),
    "system:delsys": ("ink", "dotted", "d"),
    "axis:x": ("vermillion", "solid", "none"),
    "axis:y": ("green", "solid", "none"),
    "axis:z": ("blue", "solid", "none"),
}
for _i, _c in enumerate(CATEGORICAL):
    ROLES[f"cat:{_i}"] = (_c, "solid", "osd^v"[_i % 5])
# a variant of categorical series i (same colour, dashed): e.g. a fusion estimator run with
# the hand-eye IMU mounting next to the same estimator with its in-take mounting (VD14)
for _i, _c in enumerate(CATEGORICAL):
    ROLES[f"cat:{_i}:alt"] = (_c, "dashed", "osd^v"[_i % 5])
# a causal (filter-only) variant of categorical series i (same colour, dotted): e.g. the Kalman
# filter next to its smoother (colour harmonisation 2026-11-05)
for _i, _c in enumerate(CATEGORICAL):
    ROLES[f"cat:{_i}:dot"] = (_c, "dotted", "osd^v"[_i % 5])

# Muscles and landmarks keep one colour each everywhere (colour follows the entity).
# Muscles that share panels (the four closers; the two openers) get well-separated hues:
# right/left pairs in neighbouring hues (blue/sky, vermillion/orange) merged in dark mode
# (closers: normal-vision dE >= 17, CVD >= 5.4; openers >= 24).
ENTITY = {
    "masseter_r": "blue", "masseter_l": "orange", "temporalis_r": "green",
    "temporalis_l": "purple", "digastric_r": "vermillion", "digastric_l": "sky",
    "incisor": "blue", "left": "orange", "right": "green", "tmj_left": "vermillion",
    "tmj_right": "purple",
}
# Four closers cannot all reach the CVD separation target (best dE 7.3 in dark mode), so
# the side is a secondary encoding: left-side muscles are dashed.
for _k, _c in ENTITY.items():
    _dash = "dashed" if _k.endswith("_l") and _k.split("_")[0] in (
        "masseter", "temporalis", "digastric") else "solid"
    ROLES.setdefault(f"entity:{_k}", (_c, _dash, "none"))

# Typography and marks (points). The default venue is IEEE: 8-pt labels/titles, 7-pt ticks
# and legends, 6-pt notes (as \\footnotesize / \\scriptsize in a 10-pt IEEE document, but
# written as absolute sizes, so an 11- or 12-pt document class does not change them).
IEEE_COLUMN_IN = 3.5
IEEE_TEXT_IN = 7.16
FONT_PT = {"label": 8.0, "tick": 7.0, "title": 8.0, "legend": 7.0, "note": 6.0, "tag": 8.0}


@dataclass(frozen=True)
class Venue:
    """A journal / conference figure profile: widths, absolute font sizes, font family.

    ``FigureSpec.width`` stays "column" / "text"; the active venue maps it to inches. All
    three renderers lay out with the venue's sizes (``layout.compute``), so the paper, the
    previews and the size check agree. ``family`` is the figure font: "document" uses the
    paper's own text font (IEEE: Times), "sans" switches figure text to the document's sans
    family. ``glyph`` scales the layout's text-width estimate for that family (Helvetica-like
    sans faces are wider than Computer Modern). ``upload_fonts`` is the preamble of the
    standalone per-figure PDF (``export --standalone``), which has no document to inherit
    the font from.
    """

    name: str
    column_in: float
    text_in: float
    font_pt: dict[str, float]
    family: str = "document"           # "document" | "sans"
    glyph: float = 1.0
    min_pt: float = 6.0
    upload_fonts: str = ""
    max_height_in: float = 8.5         # a figure (without caption) must fit the page
    marker_pt: float = 1.6             # scatter marker radius (pt); Springer larger (JNER 3.0)
    # Result markers (the diamond of a cohort median, P95 or summary over grey participant
    # dots): size factor on top of the series' own, and the width (pt) of their surface-
    # coloured outline. A diamond of size 1.5 has the area of a dot, so without these it does
    # not stand out, and it disappears on a line of its own colour or a reference line.
    result_marker_scale: float = 1.0
    result_halo_pt: float = 0.3

    def width_in(self, width: str) -> float:
        return self.column_in if width == "column" else self.text_in


_MM = 1 / 25.4
# Sans venues: TeX Gyre Heros (Helvetica-like) with a glyph fallback (DejaVu Sans) for the
# Greek letters and math symbols Heros lacks, so ω, θ, ≤, ≈ are sans like the text (E32).
# \jtsmmunicodesym tells the pgf style to typeset those symbols as Unicode characters;
# without it (a paper preamble of its own) they fall back to the document's math.
# The document (serif) font of our own previews and captioned PDFs gets the same kind of
# fallback (DejaVu Serif), so symbols in captions and table cells (✓, ρ, √, ...) never drop
# out.
_SERIF_FB = ("\\directlua{luaotfload.add_fallback('jtsmmfbs',{'DejaVu Serif:mode=node;',"
             "'DejaVu Sans:mode=node;'})}"
             "\\setmainfont{TeX Gyre Termes}[RawFeature={fallback=jtsmmfbs}]\n")
_SANS_UPLOAD = ("\\usepackage{fontspec}\\directlua{luaotfload.add_fallback('jtsmmfb',"
                "{'DejaVu Sans:mode=node;'})}\n"
                "\\setsansfont{TeX Gyre Heros}[RawFeature={fallback=jtsmmfb}]"
                "\\def\\jtsmmunicodesym{}\n" + _SERIF_FB)
VENUES: dict[str, Venue] = {
    "ieee": Venue("ieee", IEEE_COLUMN_IN, IEEE_TEXT_IN, dict(FONT_PT), "document", 1.0, 6.0,
                  "\\usepackage{fontspec}" + _SERIF_FB),
    # IEEE with an 8-pt floor (2026-11-12, the ICORR camera-free paper: 88 / 181 mm, no figure
    # text below 8 pt at that width); otherwise as "ieee".
    "ieee8": Venue("ieee8", IEEE_COLUMN_IN, IEEE_TEXT_IN,
                   {"label": 8.0, "tick": 8.0, "title": 8.0, "legend": 8.0, "note": 8.0,
                    "tag": 8.0}, "document", 1.0, 8.0, "\\usepackage{fontspec}" + _SERIF_FB),
    # Springer: 85 / 170 mm, sans-serif lettering, nothing below 7 pt (notes included).
    "springer": Venue("springer", 85 * _MM, 170 * _MM,
                      {"label": 8.0, "tick": 7.0, "title": 8.0, "legend": 7.0, "note": 7.0,
                       "tag": 8.0}, "sans", 1.08, 7.0,
                      _SANS_UPLOAD, marker_pt=2.1,    # JNER revision 3.0: dots read at 85 mm
                      result_marker_scale=1.3, result_halo_pt=0.7),   # 2026-11-13: the result
    #                 diamonds were hard to find at print size (requirements e, f and others)
    # Nature: 89 / 183 mm, sans-serif, 5-7 pt text, 8-pt bold panel letters.
    "nature": Venue("nature", 89 * _MM, 183 * _MM,
                    {"label": 7.0, "tick": 6.0, "title": 7.0, "legend": 6.0, "note": 5.0,
                     "tag": 8.0}, "sans", 1.08, 5.0,
                    _SANS_UPLOAD),
}
DEFAULT_VENUE = "ieee"
_ACTIVE: ContextVar[str] = ContextVar("biomech_venue", default=DEFAULT_VENUE)


def venue(name: str | Venue | None = None) -> Venue:
    """The named venue (``None``: the active one, see :func:`use_venue`)."""
    if isinstance(name, Venue):
        return name
    key = (name or _ACTIVE.get()).lower()
    if key not in VENUES:
        raise ValueError(f"unknown venue {name!r}; known: {', '.join(VENUES)}")
    return VENUES[key]


@contextmanager
def use_venue(name: str | Venue | None) -> Iterator[Venue]:
    """Make ``name`` the active venue inside the block (``None``: keep the current one).
    Context-local, so concurrent renders (viewer threads, workers) do not interfere."""
    if name is None:
        yield venue()
        return
    v = venue(name)
    tok = _ACTIVE.set(v.name)
    try:
        yield v
    finally:
        _ACTIVE.reset(tok)


LINE_PT = 0.9          # data lines
THIN_PT = 0.45         # context lines, grid
MARKER_PT = 1.6        # scatter marker radius (pt) of the default venue; see marker_pt()


def marker_pt() -> float:
    """Scatter marker radius (pt) of the active venue (Springer: larger, JNER revision 3.0)."""
    return venue().marker_pt


_CONTEXT_ROLES = ("context", "muted", "faint")


def is_result_marker(series: Any) -> bool:
    """A filled diamond that is not a context series: the marker of a cohort median, P95 or
    summary in every view."""
    st = getattr(series, "style", None) or {}
    return (getattr(series, "kind", "") == "scatter"
            and getattr(series, "c", None) is None
            and getattr(series, "role", "") not in _CONTEXT_ROLES
            and str(st.get("ms", resolve(series)[2])) == "d")


def marker_factor(series: Any) -> float:
    """Marker size factor of a scatter series: its ``style["size"]``, times the venue's
    ``result_marker_scale`` for a result marker."""
    f = float((getattr(series, "style", None) or {}).get("size", 1.0))
    return f * venue().result_marker_scale if is_result_marker(series) else f


def halo_pt(series: Any) -> float | None:
    """Width (pt) of the surface-coloured outline of a scatter marker: ``style["halo"]``,
    else the venue's ``result_halo_pt`` for a result marker, else None (the role default)."""
    st = getattr(series, "style", None) or {}
    if "halo" in st:
        return float(st["halo"])
    return venue().result_halo_pt if is_result_marker(series) else None
BAND_ALPHA = 0.18

_UNIT_PLAIN = {
    "\\milli\\metre": "mm", "\\degree": "°", "\\degree\\per\\second": "°/s",
    "\\degree\\squared": "°²", "\\degree\\squared\\per\\hertz": "°²/Hz",
    "\\degree\\per\\second\\squared": "°/s²", "\\milli\\metre\\per\\second": "mm/s",
    "\\milli\\metre\\per\\second\\squared": "mm/s²",
    "\\milli\\second": "ms", "\\second": "s", "\\percent": "%", "\\hertz": "Hz",
    "\\micro\\volt": "µV", "\\metre\\per\\second\\squared": "m/s²", "\\decibel": "dB",
    "\\micro\\volt\\squared\\per\\hertz": "µV²/Hz", "\\pixel": "px", "\\per\\minute": "/min",
    "\\ppm": "ppm", "\\milli\\metre\\squared": "mm²", "\\micro\\second": "µs",
    "\\milli\\metre\\per\\degree": "mm/°", "\\milli\\metre\\per\\degree\\squared": "mm/°²",
    "\\degree\\per\\minute": "°/min", "\\minute": "min", "\\kilo\\ohm": "kΩ",
    "\\degree\\squared\\per\\second\\squared\\per\\hertz": "(°/s)²/Hz",
}


def plain_unit(u: str) -> str:
    return _UNIT_PLAIN.get(u, u.replace("\\", ""))


def role(name: str) -> tuple[str, str, str]:
    return ROLES.get(name, ROLES["primary"])


def resolve(series: Any) -> tuple[str, str, str]:
    """(colour, dash, marker) of a series: its role, overridden by ``style["color"]`` (a
    palette name), ``style["dash"]`` (solid / dashed / dotted) and ``style["marker"]``. Every
    renderer and the legend check resolve through here."""
    col, dash, mark = role(getattr(series, "role", "primary"))
    st = getattr(series, "style", None) or {}
    return (str(st.get("color", col)), str(st.get("dash", dash)),
            str(st.get("marker", mark)))


def hexcolor(name: str, theme: str = "light") -> str:
    """'#rrggbb' of a palette name, a theme token or a bare hex string."""
    if name in THEMES.get(theme, {}):
        return "#" + THEMES[theme][name]
    if theme == "dark" and name in ("ink", "black"):
        return "#" + THEMES["dark"]["ink"]
    if theme == "dark" and name == "ink2":
        return "#" + THEMES["dark"]["ink2"]
    if theme == "dark" and name == "lightgrey":
        return "#" + THEMES["dark"]["faint"]
    if theme == "dark" and name == "grey":
        return "#" + THEMES["dark"]["context"]
    if theme == "dark" and name in PALETTE_DARK:
        return "#" + PALETTE_DARK[name]
    return "#" + PALETTE.get(name, name)


def rgba(name: str, alpha: float, theme: str = "light") -> str:
    h = hexcolor(name, theme).lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def ramp(values: tuple[str, ...] = SEQUENTIAL) -> list[tuple[float, str]]:
    """Evenly spaced (position, '#hex') stops of a ramp, 0..1."""
    n = len(values)
    return [(i / (n - 1), "#" + v) for i, v in enumerate(values)]


def ramp_values(kind: str = "sequential", theme: str = "light") -> tuple[str, ...]:
    """The token ramp of ``kind`` (sequential / diverging) for ``theme`` (hex, no '#')."""
    if kind == "diverging":
        return DIVERGING_DARK if theme == "dark" else DIVERGING
    return SEQUENTIAL_DARK if theme == "dark" else SEQUENTIAL


def ramp_color(f: float, kind: str = "sequential", theme: str = "light") -> str:
    """'#RRGGBB' at position ``f`` (0..1, clipped) of the theme's ramp ``kind``."""
    vals = ramp_values(kind, theme)
    pos = [i / (len(vals) - 1) for i in range(len(vals))]
    f = min(max(float(f), 0.0), 1.0)
    out = ""
    for i in range(3):
        chan = [int(v[2 * i:2 * i + 2], 16) for v in vals]
        lo = max(j for j in range(len(pos)) if pos[j] <= f)
        hi = min(lo + 1, len(pos) - 1)
        t = 0.0 if hi == lo else (f - pos[lo]) / (pos[hi] - pos[lo])
        out += f"{int(round(chan[lo] + t * (chan[hi] - chan[lo]))):02X}"
    return "#" + out


# -- text conventions ---------------------------------------------------------------------------
MINUS = "\u2212"
# A hyphen-minus that is a sign: at the start or after a space / bracket / operator, and
# followed by a digit (or .digit). Ranges ("10-20"), words ("open-close") and exponents
# ("1e-3") keep their hyphen.
_SIGN_RE = re.compile(r"(?:(?<=^)|(?<=[\s(\[{=,;:/<>~±+]))-(?=\.?\d)")


def unicode_minus(s: str) -> str:
    """Replace sign hyphens with U+2212 (tick labels, cell values, labels, notes)."""
    return _SIGN_RE.sub(MINUS, s) if s and "-" in s else s
