# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Session I/O — HDF5 writer and reader."""

from .session_reader import EmgStream, QuatStream, Session, TriggerStream

__all__ = ["Hdf5Writer", "Session", "EmgStream", "QuatStream", "TriggerStream"]


def __getattr__(name: str):
    if name == "Hdf5Writer":
        from .hdf5_writer import Hdf5Writer

        return Hdf5Writer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
