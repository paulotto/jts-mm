# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""G4: the dataset release carries pseudonyms only and no per-person demographics."""

from __future__ import annotations

import pandas as pd

from biomech.results.release import pseudonymize


def test_pseudonymize_rows_ids_and_private_columns():
    pm = {"s01": "P1", "s03": "P2"}
    df = pd.DataFrame({"unit": ["s01", "s03", "s09"], "pid": ["s01", "s03", "s09"],
                       "age": [30, 40, 50], "note": ["vs s03", "s09 low", "x"],
                       "lead_s01": [1.0, 2.0, 3.0]})
    out, dropped = pseudonymize(df, pm)
    assert dropped == ["age"] and len(out) == 2                     # s09 (excluded) dropped
    assert out.pid.tolist() == ["P1", "P2"] and out.unit.tolist() == ["P1", "P2"]
    assert out.note.tolist() == ["vs P2", "excluded low"] and "lead_P1" in out.columns
    agg = pd.DataFrame({"unit": ["_cohort"], "age_mean": [41.0], "n": [12]})
    out2, dropped2 = pseudonymize(agg, pm)
    assert dropped2 == [] and out2.age_mean.tolist() == [41.0]      # aggregates released


def test_session_names_never_released():
    """Session folder names carry study id, sex, age and recording time."""
    from biomech.results.release import SESSION_RE

    pm = {"s01": "P1"}
    df = pd.DataFrame({"pid": ["s01"], "session": ["session_s01_f_31_20250302T101000Z_0000000c"],
                       "note": ["from session_s01_f_31_mvc_20250302T100000Z_00000003"]})
    out, dropped = pseudonymize(df, pm)
    assert "session" in dropped and "session" not in out.columns
    assert not SESSION_RE.search(out.note.iloc[0]) and "withheld" in out.note.iloc[0]
