# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Viewer operator notes: host-clock stamp on arrival, drainable by the recorder."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from jtsmm.viewer.notes import MAX_NOTE_CHARS, NoteLog


def test_notelog_stamps_with_host_clock_and_keeps_browser_time():
    ticks = iter([1_000, 2_000, 3_000])
    log = NoteLog(clock_ns=lambda: next(ticks))
    a = log.add("jaw click", 1.5)
    b = log.add("  second\nnote \x00 ", None)
    assert (a["t_host_ns"], a["t_s"], a["text"]) == (1_000, 1.5, "jaw click")
    assert (b["t_host_ns"], b["t_s"], b["text"]) == (2_000, None, "second note")
    assert log.drain() == [a, b]
    assert log.drain() == []  # each note is handed out once
    c = log.add("third", float("nan"))
    assert c["t_s"] is None  # /api/session is JSON: no NaN
    assert log.drain() == [c]
    assert len(log) == 3 and [n["text"] for n in log.recent(2)] == ["second note", "third"]


def test_notelog_rejects_empty_and_bad_time_and_caps_length():
    log = NoteLog()
    with pytest.raises(ValueError):
        log.add("   ")
    with pytest.raises(ValueError):
        log.add("x", "not a number")
    assert len(log.add("y" * (MAX_NOTE_CHARS + 50))["text"]) == MAX_NOTE_CHARS
    assert len(log) == 1


def test_notelog_is_thread_safe():
    log = NoteLog()
    got: list[dict] = []
    stop = threading.Event()

    def drainer():
        while not stop.is_set():
            got.extend(log.drain())

    th = threading.Thread(target=drainer)
    th.start()
    for i in range(2000):
        log.add(f"n{i}")
    stop.set()
    th.join()
    got.extend(log.drain())
    assert [g["text"] for g in got] == [f"n{i}" for i in range(2000)]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _post(port: int, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/note", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_post_note_is_collected_with_host_time():
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from jtsmm.viewer.server import ViewerServer
    from jtsmm.viewer.sources import MockSource

    port = _free_port()
    server = ViewerServer(MockSource(), host="127.0.0.1", port=port)
    server.start_in_thread()
    try:
        deadline = time.monotonic() + 5.0
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    pytest.fail("viewer server did not start")
                time.sleep(0.05)

        # Standalone: accepted, kept in memory, reported as not saved.
        t0 = time.monotonic_ns()
        code, r = _post(port, {"t_s": 12.25, "text": "bite"})
        t1 = time.monotonic_ns()
        assert code == 200 and r["ok"] and r["saved"] is False
        assert t0 <= r["t_host_ns"] <= t1  # same clock as the supervisor's stamps

        server.notes.saved_to = "session.h5"  # what the recorder does
        code, r = _post(port, {"t_s": 13.0, "text": "open wide"})
        assert code == 200 and r["saved"] is True and r["saved_to"] == "session.h5"
        code, r = _post(port, {"t_s": 1.0, "text": "  "})
        assert code == 400 and not r["ok"]

        notes = server.notes.drain()
        assert [n["text"] for n in notes] == ["bite", "open wide"]
        assert [n["t_s"] for n in notes] == [12.25, 13.0]
        assert notes[0]["t_host_ns"] < notes[1]["t_host_ns"]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/session", timeout=3) as resp:
            meta = json.loads(resp.read())
        assert [n["text"] for n in meta["notes"]] == ["bite", "open wide"]
        assert meta["notes_saved_to"] == "session.h5"
    finally:
        server.shutdown()
