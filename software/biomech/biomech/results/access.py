# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Record which stored tables and arrays a piece of code reads (view coverage guard).

``with track() as log:`` makes every :class:`~biomech.results.ResultStore` read inside it
(this thread or task) land in ``log``: tables as ``(stage, table)``, arrays as
``(stage, key)`` -- arrays are returned as a :class:`TrackingDict`, so only the keys a view
actually looks up count, not the whole file. Tables read inside ``track()`` come back as a
:class:`TrackingFrame`, which notes the columns a view uses as ``(stage, table, column)``
(every stored result column reaches some artefact). :func:`family` maps an array key to its
pattern (``open_close__norm_pct`` -> ``{task}__norm_pct``) so that "the view shows this
kind of output" can be compared with "the pipeline stores this kind of output".
"""

from __future__ import annotations

import contextvars
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

_LOG: contextvars.ContextVar["AccessLog | None"] = contextvars.ContextVar("biomech_access",
                                                                        default=None)


@dataclass
class AccessLog:
    tables: set[tuple[str, str]] = field(default_factory=set)
    arrays: set[tuple[str, str]] = field(default_factory=set)
    columns: set[tuple[str, str, str]] = field(default_factory=set)
    # (stage, table, column) -> the artefacts (view ids) that used it: the coverage matrix
    consumers: dict[tuple[str, str, str], set[str]] = field(default_factory=dict)

    def families(self) -> set[tuple[str, str]]:
        return {(st, family(k)) for st, k in self.arrays}


@contextmanager
def track() -> Iterator[AccessLog]:
    log = AccessLog()
    tok = _LOG.set(log)
    try:
        yield log
    finally:
        _LOG.reset(tok)


def note_table(stage: str, name: str) -> None:
    log = _LOG.get()
    if log is not None:
        log.tables.add((stage, name))


def active() -> bool:
    return _LOG.get() is not None


def _note_columns(src: tuple[str, str] | None, cols: Any) -> None:
    log = _LOG.get()
    if log is None or src is None:
        return
    if isinstance(cols, str):
        cols = [cols]
    for c in cols:
        if isinstance(c, str):
            log.columns.add((src[0], src[1], c))


def _column_keys(key: Any) -> list[str]:
    """Column names named by an indexing key (a name, a list of names, a boolean frame no)."""
    if isinstance(key, str):
        return [key]
    if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
        return list(key)
    try:                                  # pd.Index of names
        import pandas as pd

        if isinstance(key, pd.Index):
            return [k for k in key if isinstance(k, str)]
    except Exception:
        pass
    return []


class _TrackedIndexer:
    """``.loc`` / ``.at`` of a TrackingFrame: notes the column part of a 2-D key."""

    def __init__(self, frame: Any, inner: Any) -> None:
        self._frame, self._inner = frame, inner

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            _note_columns(self._frame._bm_src, _column_keys(key[1]) or (
                list(self._frame.columns) if isinstance(key[1], slice) else []))
        else:                             # rows only: every column comes along
            _note_columns(self._frame._bm_src, list(self._frame.columns))
        return self._inner[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._inner[key] = value

    def __call__(self, *a: Any, **k: Any) -> "_TrackedIndexer":      # df.loc(axis=...)
        return _TrackedIndexer(self._frame, self._inner(*a, **k))

    def __getattr__(self, name: str) -> Any:            # pandas internals of the indexer
        return getattr(self._inner, name)


def tracking_frame(df: Any, stage: str, name: str) -> Any:
    """``df`` as a :class:`TrackingFrame` reporting to the active log (or ``df`` itself
    outside ``track()``)."""
    if not active():
        return df
    out = TrackingFrame(df)
    out._bm_src = (stage, name)
    return out


def _make_tracking_frame():
    import pandas as pd

    class TrackingFrame(pd.DataFrame):
        """A stage table that notes the columns a view reads (indexing, attributes,
        ``.loc``/``.at``, ``get``); whole-frame reads (iteration, ``to_dict``, ``values``)
        count as reading every column. Derived frames keep tracking."""

        _metadata = ["_bm_src"]
        _bm_src: tuple[str, str] | None = None

        @property
        def _constructor(self):
            return TrackingFrame

        def __getitem__(self, key):
            _note_columns(self._bm_src, _column_keys(key))
            return super().__getitem__(key)

        def __getattr__(self, name):
            if not name.startswith("_") and name in self.columns:
                _note_columns(self._bm_src, name)
            return super().__getattr__(name)

        def get(self, key, default=None):
            _note_columns(self._bm_src, _column_keys(key))
            return super().get(key, default)

        @property
        def loc(self):
            return _TrackedIndexer(self, super().loc)

        @property
        def at(self):
            return _TrackedIndexer(self, super().at)

        def _all(self):
            _note_columns(self._bm_src, list(self.columns))

        def itertuples(self, *a, **k):
            self._all()
            return super().itertuples(*a, **k)

        def iterrows(self):
            self._all()
            return super().iterrows()

        def to_dict(self, *a, **k):
            self._all()
            return super().to_dict(*a, **k)

        def to_numpy(self, *a, **k):
            self._all()
            return super().to_numpy(*a, **k)

        def to_records(self, *a, **k):
            self._all()
            return super().to_records(*a, **k)

        @property
        def values(self):
            self._all()
            return super().values

    return TrackingFrame


TrackingFrame = _make_tracking_frame()


class TrackingDict(dict):
    """A dict of arrays that reports looked-up keys to the active :class:`AccessLog`."""

    stage: str = ""

    def _note(self, key: Any) -> None:
        log = _LOG.get()
        if log is not None and isinstance(key, str):
            log.arrays.add((self.stage, key))

    def __getitem__(self, key: Any) -> Any:
        self._note(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._note(key)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            self._note(key)
            return True
        return False

    def _note_all(self) -> None:
        log = _LOG.get()
        if log is not None:
            log.arrays.update((self.stage, k) for k in super().keys() if isinstance(k, str))

    def items(self):  # iteration over everything = the view reads everything it filters
        self._note_all()
        return super().items()

    def values(self):
        self._note_all()
        return super().values()

    def __iter__(self):
        # Defining __iter__ takes CPython's dict-merge fast path away: ``dict(d)``,
        # ``{**d}`` and ``x.update(d)`` then read every value through __getitem__ (noted).
        # Iterating the keys alone reads no array and notes nothing.
        return super().__iter__()

    def copy(self) -> "TrackingDict":
        """A copy that keeps tracking (``dict.copy`` would return an untracked dict)."""
        return tracking(self.stage, dict(super().items()))

    def __copy__(self) -> "TrackingDict":
        return self.copy()

    def __or__(self, other):
        self._note_all()
        return tracking(self.stage, {**dict(super().items()), **dict(other)})


def tracking(stage: str, d: dict[str, Any]) -> TrackingDict:
    t = TrackingDict(d)
    t.stage = stage
    return t


# Tokens replaced by placeholders in array keys (task/phase, muscle, direction, ...).
_TASKS = ("rest", "open_close", "protrusion_retrusion", "left_right", "chewing",
          "free_motion", "mvc_opening", "mvc_closing", "mvc")
_MUSCLES = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l", "digastric_r",
            "digastric_l")
# A token is a whole ``__``-separated field (or the part before a ``|``), so "rest" in
# "rest_level" or a muscle name inside a longer word is never replaced.
def _field(tok: str) -> re.Pattern:
    return re.compile(rf"(?:(?<=^)|(?<=__)){re.escape(tok)}(?=__|\||$)")


_TOKENS = [(re.compile(r"(?:(?<=^)|(?<=__))s\d{2}(?=__|\||$)"), "{pid}")]
_TOKENS += [(_field(t), "{task}") for t in sorted(_TASKS, key=len, reverse=True)]
_TOKENS += [(_field(m), "{muscle}") for m in _MUSCLES]
_TOKENS += [(re.compile(r"(?<=\|)[a-z_]+$"), "{model}")]


def family(key: str) -> str:
    """Pattern of an array key; keys of one family differ only in task, muscle, pid, model."""
    out = key
    for rx, rep in _TOKENS:
        out = rx.sub(rep, out)
    return out
