# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""FastAPI + WebSocket server for the viewer (standalone or in-recording).

The same server backs all three modes; only the `Source` differs. It serves the
static SPA, exposes session metadata + replay controls over REST, and streams
coalesced msgpack snapshots over a WebSocket at a fixed rate with frame-drop
semantics (the hub drops rows if a client lags; the writer is never blocked).

``Hub.snapshot()`` *drains* the pending rows, so exactly one broadcaster calls
it per tick and hands the same encoded frame to every client (a per-client
size-1 queue that drops the oldest frame for a client that cannot keep up).
Each client used to call it itself, so two clients received half the rows each.

Operator notes (``POST /api/note``) are stamped with the host clock on arrival
(`jtsmm.viewer.notes.NoteLog`); an in-recording server's notes are drained by
the recorder into ``/events`` (kind ``note``), standalone they stay in memory.

Camera previews use optical-style MJPEG (``multipart/x-mixed-replace``) at
``/stream/{cam_id}``, fed from JPEG slots on the Hub — not the WebSocket.

FastAPI/uvicorn are optional deps (``jtsmm[viewer]``) imported lazily.
"""

# NOTE: no ``from __future__ import annotations`` here — FastAPI resolves the
# ``WebSocket`` parameter annotation against module globals, and it is imported
# locally inside build_app(), so stringized annotations would misfire.

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from jtsmm.viewer.hub import Hub
from jtsmm.viewer.jpeg_util import PLACEHOLDER_JPEG
from jtsmm.viewer.notes import NoteLog
from jtsmm.viewer.protocol import encode
from jtsmm.viewer.sources import Source

__all__ = ["Broadcaster", "build_app", "ViewerServer", "static_dir"]

WS_RATE_HZ = 30.0
MJPEG_PERIOD_S = 0.1  # poll latest JPEG ~10 Hz (source pushes ~5 Hz)


def static_dir() -> Path:
    override = None
    import os

    env = os.environ.get("JTSMM_VIEWER_STATIC")
    if env:
        override = Path(env)
    if override and override.is_dir():
        return override
    base = Path(__file__).resolve().parents[2] / "viewer_web"
    dist = base / "dist"
    return dist if dist.is_dir() else base


class Broadcaster:
    """One snapshot per tick, fanned out to every connected WebSocket client.

    Runs in its own daemon thread while at least one client is registered, so
    it does not depend on which event loop a client lives in, and encoding a
    frame never blocks the server loop. Each client owns an ``asyncio.Queue``
    of size 1 in its own loop; the newest frame replaces an unsent one.
    """

    def __init__(self, hub: Hub, rate_hz: float = WS_RATE_HZ) -> None:
        self._hub = hub
        self._period = 1.0 / rate_hz if rate_hz > 0 else 1.0 / WS_RATE_HZ
        self._lock = threading.Lock()
        self._clients: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._next_key = 0
        self._thread: threading.Thread | None = None

    @property
    def n_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def register(self) -> tuple[int, asyncio.Queue]:
        """Call from the client's coroutine; returns (key, its frame queue)."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        with self._lock:
            key = self._next_key
            self._next_key += 1
            self._clients[key] = (loop, q)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="viewer-broadcast", daemon=True
                )
                self._thread.start()
        return key, q

    def unregister(self, key: int) -> None:
        with self._lock:
            self._clients.pop(key, None)

    @staticmethod
    def _offer(q: asyncio.Queue, data: bytes) -> None:
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(data)

    def _run(self) -> None:
        import logging

        log = logging.getLogger("jtsmm.viewer")
        while True:
            t0 = time.monotonic()
            with self._lock:
                targets = list(self._clients.items())
                if not targets:
                    self._thread = None
                    return
            try:
                data = encode(self._hub.snapshot())
            except Exception:  # noqa: BLE001 - keep serving the other frames
                log.exception("viewer snapshot/encode failed")
                data = None
            if data is not None:
                for key, (loop, q) in targets:
                    try:
                        loop.call_soon_threadsafe(self._offer, q, data)
                    except RuntimeError:          # client's loop already closed
                        self.unregister(key)
            dt = time.monotonic() - t0
            time.sleep(max(0.0, self._period - dt))


def build_app(source: Source, hub: Hub, notes: NoteLog | None = None) -> Any:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="jtsmm viewer", docs_url=None, redoc_url=None)
    if notes is None:
        notes = NoteLog()
    app.state.notes = notes
    broadcaster = Broadcaster(hub)
    app.state.broadcaster = broadcaster
    sdir = static_dir()

    @app.get("/api/session")
    def api_session() -> Any:
        meta = dict(source.meta())
        meta["display_fs"] = hub.display_fs
        meta["ws_rate_hz"] = WS_RATE_HZ
        meta["camera_ids"] = hub.camera_ids()
        meta["notes"] = notes.recent(100)
        meta["notes_saved_to"] = notes.saved_to
        return JSONResponse(meta)

    @app.get("/stream/{cam_id}")
    def mjpeg_stream(cam_id: str) -> Any:
        """Optical-style MJPEG: multipart/x-mixed-replace of latest Hub JPEG."""

        def generate():
            while True:
                frame = hub.get_camera_jpeg(cam_id) or PLACEHOLDER_JPEG
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                time.sleep(MJPEG_PERIOD_S)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/position")
    def api_position() -> Any:
        return JSONResponse(source.position())

    @app.post("/api/note")
    async def api_note(payload: dict[str, Any]) -> Any:
        # Stamped with the host clock here; `saved` says whether a recorder
        # collects it into the session file (else it stays in memory only).
        try:
            note = notes.add(payload.get("text", ""), payload.get("t_s"))
        except (TypeError, ValueError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({
            "ok": True, "count": len(notes), "t_host_ns": note["t_host_ns"],
            "saved": notes.saved_to is not None, "saved_to": notes.saved_to,
        })

    @app.post("/api/seek")
    async def api_seek(payload: dict[str, Any]) -> Any:
        try:
            source.seek(float(payload.get("t_s", 0.0)))
        except (TypeError, ValueError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True})

    @app.post("/api/speed")
    async def api_speed(payload: dict[str, Any]) -> Any:
        from jtsmm.viewer.sources import validate_replay_speed

        try:
            speed = validate_replay_speed(payload.get("speed", 1.0))
        except (TypeError, ValueError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        source.set_speed(speed)
        return JSONResponse({"ok": True, "speed": speed})

    @app.post("/api/pause")
    async def api_pause(payload: dict[str, Any]) -> Any:
        source.set_paused(bool(payload.get("paused", True)))
        return JSONResponse({"ok": True})

    @app.post("/api/impedance")
    async def api_impedance(payload: dict[str, Any]) -> Any:
        enabled = bool(payload.get("enabled", True))
        hub.set_impedance_enabled(enabled)
        return JSONResponse({"ok": True, "enabled": enabled})

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        key, frames = broadcaster.register()
        try:
            while True:
                await sock.send_bytes(await frames.get())
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 - log then drop the client cleanly
            import logging

            logging.getLogger("jtsmm.viewer").exception("viewer WS closed on error")
            return
        finally:
            broadcaster.unregister(key)

    if sdir.is_dir():
        @app.get("/")
        def index() -> Any:
            idx = sdir / "index.html"
            if idx.is_file():
                return FileResponse(str(idx))
            return JSONResponse({"error": "viewer_web not built"}, status_code=404)

        app.mount("/", StaticFiles(directory=str(sdir), html=True), name="static")

    return app


class ViewerServer:
    """Owns a `Source`, its `Hub`, and a uvicorn server (blocking or threaded)."""

    def __init__(self, source: Source, *, host: str = "0.0.0.0", port: int = 8770) -> None:
        self.source = source
        self.host = host
        self.port = port
        self.hub = source.make_hub()
        # Operator notes; a recorder drains them into the session file.
        self.notes = NoteLog()
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/"

    @property
    def lan_urls(self) -> list[str]:
        from jtsmm.net.lan_urls import format_viewer_urls

        return format_viewer_urls(self.host, self.port, mdns_hostname="jtsmm")

    def print_startup(self, *, prefix: str = "viewer", extra: str = "") -> None:
        from jtsmm.net.lan_urls import print_viewer_startup

        print_viewer_startup(self.host, self.port, prefix=prefix)
        if extra:
            import sys

            print(f"{prefix}: {extra}", file=sys.stderr)

    def _make_server(self) -> Any:
        import uvicorn

        app = build_app(self.source, self.hub, self.notes)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning",
                                ws_ping_interval=None)
        return uvicorn.Server(config)

    def run(self) -> None:
        """Blocking run (standalone `jtsmm view`/`jtsmm replay`)."""
        self.source.start(self.hub)
        self._server = self._make_server()
        try:
            self._server.run()
        finally:
            self.source.stop()

    def start_in_thread(self) -> None:
        """Non-blocking run in a daemon thread (in-recording viewing)."""
        self.source.start(self.hub)
        self._server = self._make_server()

        def _serve() -> None:
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=_serve, name="viewer-server", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.source.stop()
