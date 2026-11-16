# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Remote / LAN viewer integration tests (mock source, no hardware)."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from jtsmm.net.lan_urls import format_viewer_urls, print_viewer_startup
from jtsmm.viewer.sources import MockSource


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def viewer_server():
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from jtsmm.viewer.server import ViewerServer

    port = _free_port()
    server = ViewerServer(MockSource(), host="0.0.0.0", port=port)
    server.start_in_thread()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        server.shutdown()
        pytest.fail("viewer server did not start")
    yield server
    server.shutdown()


def test_viewer_lan_urls_all_interfaces():
    urls = format_viewer_urls("0.0.0.0", 8770, mdns_hostname="jtsmm")
    assert urls[0] == "http://127.0.0.1:8770/"
    assert any("jtsmm.local" in u for u in urls)


def test_viewer_server_lan_urls_property():
    pytest.importorskip("fastapi")
    from jtsmm.viewer.server import ViewerServer

    server = ViewerServer(MockSource(), host="0.0.0.0", port=8770)
    urls = server.lan_urls
    assert urls[0] == "http://127.0.0.1:8770/"
    assert any(".local" in u for u in urls)


def test_viewer_http_api_session(viewer_server):
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    from jtsmm.viewer.server import build_app

    app = build_app(viewer_server.source, viewer_server.hub)
    client = TestClient(app)
    r = client.get("/api/session")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "mock"
    assert body["ws_rate_hz"] == 30.0


def test_viewer_http_index_and_static(viewer_server):
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    from jtsmm.viewer.server import build_app

    app = build_app(viewer_server.source, viewer_server.hub)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_viewer_websocket_streams_frames(viewer_server):
    pytest.importorskip("msgpack")
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    from jtsmm.viewer.protocol import decode
    from jtsmm.viewer.server import build_app

    app = build_app(viewer_server.source, viewer_server.hub)
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        frames = [decode(ws.receive_bytes()) for _ in range(3)]
    assert all(f["type"] == "frame" for f in frames)
    assert frames[-1]["seq"] >= frames[0]["seq"]
    assert frames[0]["emg"]["c"] == 7


def test_viewer_websocket_multiple_clients(viewer_server):
    pytest.importorskip("msgpack")
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    from jtsmm.viewer.protocol import decode
    from jtsmm.viewer.server import build_app

    app = build_app(viewer_server.source, viewer_server.hub)
    client = TestClient(app)
    results: dict[int, list] = {0: [], 1: []}
    barrier = threading.Barrier(2)

    def read_frames(idx: int):
        with client.websocket_connect("/ws") as ws:
            barrier.wait(timeout=5)
            for _ in range(20):
                results[idx].append(decode(ws.receive_bytes()))

    t1 = threading.Thread(target=read_frames, args=(0,))
    t2 = threading.Thread(target=read_frames, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert len(results[0]) == 20 and len(results[1]) == 20
    assert all(r["type"] == "frame" for r in results[0] + results[1])
    # Hub.snapshot() drains rows: with one snapshot per client each would get
    # disjoint frames (half the rows). One broadcaster -> the same frames.
    by_seq = [{f["seq"]: f for f in results[i]} for i in (0, 1)]
    common = sorted(set(by_seq[0]) & set(by_seq[1]))
    assert len(common) >= 10, (sorted(by_seq[0]), sorted(by_seq[1]))
    for seq in common:
        a, b = by_seq[0][seq]["emg"], by_seq[1][seq]["emg"]
        assert a["n"] == b["n"]
        assert (a["t"] == b["t"]).all()
    assert sum(by_seq[0][s]["emg"]["n"] for s in common) > 0


def test_viewer_note_api(viewer_server):
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    from jtsmm.viewer.server import build_app

    app = build_app(viewer_server.source, viewer_server.hub)
    client = TestClient(app)
    r = client.post("/api/note", json={"t_s": 1.5, "text": "jaw click"})
    assert r.status_code == 200
    assert r.json()["count"] == 1
    sess = client.get("/api/session").json()
    assert sess["notes"][-1]["text"] == "jaw click"


def test_viewer_tcp_listen_all_interfaces(viewer_server):
    """Simulate laptop connecting to Pi bind address 0.0.0.0."""
    port = viewer_server.port
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as sock:
        req = f"GET /api/session HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        sock.sendall(req)
        data = sock.recv(4096)
    assert b"HTTP/1.1 200" in data or b"200 OK" in data


def test_print_viewer_startup_includes_lan(capsys):
    print_viewer_startup("0.0.0.0", 8770, stream=None)
    # print_viewer_startup writes to stderr; capsys may not capture without redirect
    import io
    import sys

    buf = io.StringIO()
    print_viewer_startup("0.0.0.0", 8770, stream=buf)
    out = buf.getvalue()
    assert "bound 0.0.0.0:8770" in out
    assert "open on this machine" in out
    assert "SSH tunnel" in out
