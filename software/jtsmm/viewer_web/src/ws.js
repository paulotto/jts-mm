// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

// Reconnecting WebSocket that decodes msgpack frames and forwards them, plus a
// simple "dropping frames" heuristic based on the server-provided drop counter.

import { decode, asF32 } from "./msgpack.js";

export class FrameSocket {
  constructor(onFrame, onStatus) {
    this.onFrame = onFrame;
    this.onStatus = onStatus;
    this.ws = null;
    this.retry = 0;
    this.closed = false;
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws`;
    this.onStatus("connecting");
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      this.onStatus("live");
    };
    ws.onmessage = (ev) => {
      try {
        const frame = decode(ev.data);
        // Materialize EMG float blocks once.
        const e = frame.emg;
        if (e) {
          e.tF = asF32(e.t);
          e.loF = asF32(e.lo);
          e.hiF = asF32(e.hi);
          e.filtF = asF32(e.filt);
          e.envF = asF32(e.env);
        }
        this.onFrame(frame);
      } catch (err) {
        console.error("frame decode failed", err);
      }
    };
    ws.onclose = () => {
      if (this.closed) return;
      this.onStatus("reconnecting");
      const delay = Math.min(2000, 250 * Math.pow(2, this.retry++));
      setTimeout(() => this.connect(), delay);
    };
    ws.onerror = () => ws.close();
  }

  close() {
    this.closed = true;
    if (this.ws) this.ws.close();
  }
}
