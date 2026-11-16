// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-FileCopyrightText: 2026 David Stebich
// SPDX-License-Identifier: MIT

// Viewer SPA entry point: fetch session metadata, build the EMG + IMU panels,
// stream frames over the reconnecting WebSocket, and render on rAF.

import { EmgPanel } from "./emg.js";
import { ImuPanel } from "./imu.js";
import { CamPanel } from "./cam.js";
import { DockLayout } from "./layout.js";
import { FrameSocket } from "./ws.js";

const $ = (id) => document.getElementById(id);

class App {
  constructor() {
    this.meta = null;
    this.emg = null;
    this.imu = null;
    this.sock = null;
    this.lastSeq = 0;
    this._impOn = false;
  }

  async init() {
    this.meta = await (await fetch("/api/session")).json();
    this._startedAt = performance.now();
    $("session-mode").textContent = this.meta.mode || this.meta.kind || "live";
    $("session-mode").className = "badge badge-" + (this.meta.mode || "live");
    document.title = `jtsmm viewer — ${this.meta.mode || "live"}`;

    this.emg = new EmgPanel($("emg-panel"), this.meta);
    this.imu = new ImuPanel($("imu-panel"));
    this.cam = new CamPanel($("cam-grid"));
    this.cam.setCameras(this.meta.camera_ids || []);
    this.layout = new DockLayout($("layout-root"), [
      { id: "emg", el: $("panel-emg") },
      { id: "imu", el: $("panel-imu") },
      { id: "cam", el: $("panel-cam") },
    ]);
    this.layout.onResize(() => {
      this.emg.resize();
      this.imu.resize();
    });
    this._wireLayout();
    this._wireControls();
    if (this.meta.mode === "replay") this._wireReplay();

    this.sock = new FrameSocket((f) => this._onFrame(f), (s) => this._status(s));
    this.sock.connect();
    this._loop();
  }

  _wireLayout() {
    document.querySelectorAll(".preset-btn[data-focus]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.focus || null;
        if (id) this.layout.setFocus(id);
        else this.layout.clearFocus();
      });
    });
    $("layout-reset").addEventListener("click", () => this.layout.resetLayout());
  }

  _wireControls() {
    document.querySelectorAll("input[name=mode]").forEach((r) => {
      r.addEventListener("change", () => { if (r.checked) this.emg.setMode(r.value); });
    });
    $("window-sel").addEventListener("change", (e) => this.emg.setWindow(+e.target.value));
    $("y-auto").addEventListener("change", (e) => this.emg.setYAuto(e.target.checked));
    $("y-span").addEventListener("change", (e) => this.emg.setYSpan(+e.target.value));
    $("imp-toggle").addEventListener("click", () => this._toggleImpedance());
    $("note-send").addEventListener("click", () => this._sendNote());
    $("note-text").addEventListener("keydown", (e) => { if (e.key === "Enter") this._sendNote(); });
  }

  async _sendNote() {
    const text = $("note-text").value.trim();
    if (!text) return;
    let r = null;
    try {
      r = await (await fetch("/api/note", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ t_s: this._elapsed ?? null, text }),
      })).json();
    } catch { /* network error: keep the text so it can be re-sent */ }
    if (!r || !r.ok) { this._flash("note NOT stored" + (r && r.error ? `: ${r.error}` : "")); return; }
    $("note-text").value = "";
    // Only `jtsmm record --viewer` writes notes into the session file.
    this._flash(r.saved ? "note saved to session file" : "note kept in viewer memory only (not recording)");
  }

  _wireReplay() {
    $("transport").classList.remove("hidden");
    const dur = this.meta.duration_s || 0;
    const seek = $("seek");
    seek.max = dur.toFixed(2);
    $("dur-label").textContent = fmt(dur);
    let seeking = false;
    seek.addEventListener("input", () => { seeking = true; $("pos-label").textContent = fmt(+seek.value); });
    seek.addEventListener("change", async () => {
      await fetch("/api/seek", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ t_s: +seek.value }) });
      seeking = false;
    });
    $("play-pause").addEventListener("click", async () => {
      this._paused = !this._paused;
      $("play-pause").textContent = this._paused ? "▶" : "⏸";
      await fetch("/api/pause", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paused: this._paused }) });
    });
    $("speed-sel").addEventListener("change", async (e) => {
      await fetch("/api/speed", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed: +e.target.value }) });
    });
    setInterval(async () => {
      if (seeking) return;
      try {
        const p = await (await fetch("/api/position")).json();
        if (p.t_s != null) { seek.value = p.t_s; $("pos-label").textContent = fmt(p.t_s); }
      } catch { /* ignore */ }
    }, 500);
    // Event markers
    (this.meta.events_s || []).forEach((t) => {
      const m = document.createElement("div");
      m.className = "seek-marker";
      m.style.left = dur ? (100 * t / dur) + "%" : "0";
      $("seek-track").append(m);
    });
  }

  async _toggleImpedance() {
    this._impOn = !this._impOn;
    $("imp-toggle").classList.toggle("active", this._impOn);
    this.emg.toggleImpedance(this._impOn);
    try {
      await fetch("/api/impedance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: this._impOn }),
      });
    } catch { /* ignore */ }
  }

  _onFrame(f) {
    if (f.emg) {
      this.emg.push(f.emg);
      this._emgHz = f.emg.hz || 0;
    }
    if (f.leadoff) {
      const active = f.leadoff.active === true;
      this.emg.setLeadoff(
        f.leadoff.p, f.leadoff.n, f.sat, active, f.leadoff.drl, f.leadoff.cmp,
      );
    }
    if (f.impedance) this.emg.setImpedance(f.impedance);
    if (f.imu) this.imu.update(f.imu);
    this._camHz = f.camera_hz || 0;
    this._emgFs = f.emg?.fs || this.meta?.emg_fs || 0;
    this._elapsed = (f.t_ref_us - f.t0_us) / 1e6;
    $("elapsed").textContent = fmt(this._elapsed);
    this._health(f.health || {}, f.stats || {});
    if (f.dropped > 0) this._flash(`dropping ${f.dropped}`);
  }

  _emgChipLabel() {
    if (!this._emgHz) return "EMG";
    const target = this._emgFs || 0;
    const hz = this._emgHz;
    // Show configured target when measured is within ~3% (tap-window noise).
    if (target > 0 && Math.abs(hz - target) / target < 0.03) {
      return `EMG ${Math.round(target)}`;
    }
    return `EMG ${Math.round(hz)}`;
  }

  _health(h, s) {
    const emgLabel = this._emgChipLabel();
    setChip($("chip-emg"), h.emg_ok, emgLabel);
    const hdr = this.imu ? this.imu.headerRates() : { mta: 0, cra: 0 };
    // Green only while samples arrive: a quiet slot used to look green because
    // it had no drops. Grey = disabled in the config (not started).
    const enabled = this.meta?.imu_enabled || {};
    const settled = (performance.now() - (this._startedAt ?? performance.now())) > 3000;
    imuChip($("chip-imu-mta"), "MTA", enabled.mta, hdr.mta, s.imu_mta_drops, settled);
    imuChip($("chip-imu-cra"), "CRA", enabled.cra, hdr.cra, s.imu_cra_drops, settled);
    const camLabel = this._camHz ? `CAM ${Math.round(this._camHz)}` : "CAM";
    setChip($("chip-cam"), true, camLabel);
    setChip($("chip-tx"), h.tx_ok, "TX");
    const bits = [];
    if (s.missed_drdy != null) {
      const missed = s.missed_drdy;
      const drdy = s.drdy_count || 0;
      const total = drdy + missed;
      if (total > 0 && missed > 0) {
        const pct = (100 * missed / total).toFixed(2);
        bits.push(`missed ${missed} (${pct}% lifetime)`);
      } else if (missed > 0) {
        bits.push(`missed ${missed} (lifetime)`);
      }
    }
    if (s.emg_drops != null) bits.push(`emg_drops ${s.emg_drops}`);
    if (h.max_drdy_gap_us != null) bits.push(`gap ${h.max_drdy_gap_us}µs`);
    // Framing slip first and unabbreviated: it is the one counter that means the
    // EMG trace on screen is not the signal. Without it, a corrupted stream is
    // indistinguishable from a noisy subject.
    if (s.frame_misalign) {
      bits.unshift(
        `FRAMING SLIP ${s.frame_misalign}` +
          (s.emg_misalign_drops ? ` (${s.emg_misalign_drops} dropped)` : "") +
          (s.emg_resyncs ? ` resync ${s.emg_resyncs}` : ""),
      );
    }
    if (s.spi_wedges) bits.push(`spi_wedges ${s.spi_wedges}`);
    if (s.tx_drops) bits.push(`tx_drops ${s.tx_drops}`);
    const line = $("stats-line");
    line.textContent = bits.join("  ·  ");
    line.classList.toggle("bad", Boolean(s.frame_misalign));
  }

  _status(s) {
    const el = $("conn");
    el.textContent = s;
    el.className = "conn conn-" + s;
  }

  _flash(msg) {
    const el = $("flash");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(this._flashT);
    this._flashT = setTimeout(() => el.classList.remove("show"), 1200);
  }

  _loop() {
    const tick = () => {
      if (this.emg) this.emg.render();
      if (this.imu) this.imu.render();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

function setChip(el, ok, label) {
  el.textContent = label;
  el.className = "chip " + (ok ? "chip-ok" : "chip-bad");
}

// IMU slot chip: off (disabled in the config), no data (enabled but silent,
// after a short start-up grace), or the measured rate; drops turn it red.
function imuChip(el, name, enabled, rateHz, drops, settled) {
  if (enabled === false) {
    el.textContent = `${name} off`;
    el.className = "chip chip-off";
    el.title = `disabled in the config (mcu.imu_${name.toLowerCase()}.enabled: false)`;
    return;
  }
  if (rateHz > 0) {
    setChip(el, (drops || 0) === 0, `${name} ${Math.round(rateHz)}`);
    el.title = (drops || 0) ? `${drops} samples dropped on the MCU` : "streaming";
    return;
  }
  el.textContent = settled ? `${name} no data` : name;
  el.className = settled ? "chip chip-bad" : "chip";
  el.title = settled ? "enabled but no samples arrive: IMU plugged in? see `mcu:` lines in the terminal" : "";
}

function fmt(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60), r = (s % 60);
  return `${String(m).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`;
}

new App().init().catch((e) => {
  document.getElementById("emg-panel").innerHTML =
    `<div class="fatal">Failed to start viewer: ${e}</div>`;
  console.error(e);
});
