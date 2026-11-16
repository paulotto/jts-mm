// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

// IMU panel: a shared dual-body three.js scene (Cranium/CRA + Jaw/MTA) driven by
// fusion quaternions with an optional *estimated* relative-translation trail,
// plus per-slot cards that tab between the orientation (Euler) strip and the
// accelerometer / gyroscope / magnetometer / linear-acceleration x-y-z strips.
//
// Quaternions are carried (w, x, y, z); three.js wants (x, y, z, w). Motion
// windows arrive as little-endian Float32 blobs: t is (n,) seconds since t0 and
// v is (n, 3) interleaved SI values.

import * as THREE from "../vendor/three.module.js";
import { asF32 } from "./msgpack.js";

const SLOTS = [
  { key: "cra", title: "Cranium (CRA)", accent: "#5ec8f0" },
  { key: "mta", title: "Jaw (MTA)", accent: "#f0a35e" },
];

// SH-2 fusion report ids -> label (matches records.FUSION_REPORT_IDS).
const FUSIONS = [
  { id: 5, name: "rotation_vector" },
  { id: 8, name: "game_rotation_vector" },
  { id: 9, name: "geomagnetic_rotation_vector" },
];
const DEFAULT_FUSION = 8;

// Motion streams (tab key, label, SI unit) shown as x/y/z strips.
const STREAMS = [
  { key: "accel", label: "Accel", unit: "m/s²" },
  { key: "gyro", label: "Gyro", unit: "rad/s" },
  { key: "mag", label: "Mag", unit: "µT" },
  { key: "linear_accel", label: "Linear", unit: "m/s²" },
];
const TABS = [{ key: "orient", label: "Orient", unit: "°" }, ...STREAMS];

// Fixed per-series (channel) colors shared by every strip; index 0..2 = the
// three plotted channels (x/y/z or r/p/y).
const SERIES_COLORS = ["#f05e8e", "#8ef05e", "#5ec8f0"];

// Which hub rate stream backs each tab (the fusion quaternion for orientation).
const TAB_RATE_STREAM = {
  orient: "quat", accel: "accel", gyro: "gyro", mag: "mag", linear_accel: "linear_accel",
};

const CAL_LABELS = ["unrel", "low", "med", "high"];
const CAL_STREAMS = ["quat", "accel", "gyro", "mag", "linear_accel"];
const CAL_SHORT = { quat: "Q", accel: "A", gyro: "G", mag: "M", linear_accel: "L" };

function quatToEuler(w, x, y, z) {
  const sinr = 2 * (w * x + y * z);
  const cosr = 1 - 2 * (x * x + y * y);
  const roll = Math.atan2(sinr, cosr);
  let sinp = 2 * (w * y - z * x);
  sinp = Math.max(-1, Math.min(1, sinp));
  const pitch = Math.asin(sinp);
  const siny = 2 * (w * z + x * y);
  const cosy = 1 - 2 * (y * y + z * z);
  const yaw = Math.atan2(siny, cosy);
  const d = 180 / Math.PI;
  return [roll * d, pitch * d, yaw * d];
}

// Rolling 10 s uPlot strip with three (x/y/z) series. `push` accepts a Float32
// timestamp array and interleaved (n,3) values.
class TripleStrip {
  constructor(el, labels) {
    this.windowS = 10;
    this.x = []; this.a = []; this.b = []; this.c = [];
    this.dirty = false;
    const opts = {
      width: el.clientWidth || 260, height: el.clientHeight || 110,
      legend: { show: false },
      cursor: { drag: { x: false, y: false } },
      scales: { x: { time: false }, y: { auto: true } },
      axes: [{ show: false }, {
        stroke: "#8a93a6", size: 42, grid: { stroke: "#232838" },
        ticks: { stroke: "#232838" }, font: "10px system-ui",
      }],
      series: [
        {},
        { stroke: SERIES_COLORS[0], width: 1, label: labels[0] },
        { stroke: SERIES_COLORS[1], width: 1, label: labels[1] },
        { stroke: SERIES_COLORS[2], width: 1, label: labels[2] },
      ],
    };
    this.u = new window.uPlot(opts, [[0], [0], [0], [0]], el);
  }

  pushInterleaved(tArr, vArr) {
    const n = tArr.length;
    if (n && this.x.length && tArr[0] < this.x[this.x.length - 1]) this._clear();  // seek
    for (let i = 0; i < n; i++) {
      this.x.push(tArr[i]);
      this.a.push(vArr[i * 3]); this.b.push(vArr[i * 3 + 1]); this.c.push(vArr[i * 3 + 2]);
    }
    if (n) this._trim();
  }

  pushScalar(t, a, b, c) {
    if (this.x.length && t < this.x[this.x.length - 1]) this._clear();  // seek
    this.x.push(t); this.a.push(a); this.b.push(b); this.c.push(c);
    this._trim();
  }

  _clear() {
    this.x.length = 0; this.a.length = 0; this.b.length = 0; this.c.length = 0;
  }

  _trim() {
    const tMax = this.x[this.x.length - 1], tMin = tMax - this.windowS;
    let cut = 0;
    while (cut < this.x.length && this.x[cut] < tMin) cut++;
    if (cut > 0) {
      this.x.splice(0, cut); this.a.splice(0, cut); this.b.splice(0, cut); this.c.splice(0, cut);
    }
    this.dirty = true;
  }

  render() {
    if (this.dirty && this.x.length) {
      this.u.setData([this.x, this.a, this.b, this.c]);
      this.dirty = false;
    }
  }

  setSize(w, h) { if (w > 0 && h > 0) this.u.setSize({ width: w, height: h }); }
}

// One shared scene with both bodies; the mandible is optionally offset by the
// integrated (drift-prone) relative translation, with a fading position trail.
class DualBodyScene {
  constructor(canvas) {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    this.camera.position.set(3.0, 2.2, 3.4);
    this.camera.lookAt(0, -0.3, 0);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

    this.bodies = {};
    this.bodies.cra = this._makeBody("#5ec8f0", new THREE.Vector3(0, 0.55, 0));
    this.bodies.mta = this._makeBody("#f0a35e", new THREE.Vector3(0, -0.55, 0));
    this._mtaBase = new THREE.Vector3(0, -0.55, 0);

    const grid = new THREE.GridHelper(4, 8, 0x2a3040, 0x1c2130);
    grid.position.y = -1.2;
    this.scene.add(grid);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(3, 5, 4);
    this.scene.add(dir);

    // Position trail for the mandible estimate (uniform faint line).
    this.trailMax = 300;
    this.trailPts = [];
    const tgeo = new THREE.BufferGeometry();
    tgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.trailMax * 3), 3));
    tgeo.setDrawRange(0, 0);
    this.trail = new THREE.Line(tgeo, new THREE.LineBasicMaterial({ color: 0xf0a35e, transparent: true, opacity: 0.5 }));
    this.scene.add(this.trail);

    this.showTranslation = false;
    this.posScale = 3.0; // metres -> scene units (visual only)
    this._resize();
  }

  _makeBody(color, pos) {
    const geo = new THREE.BoxGeometry(1.3, 0.32, 0.95);
    const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.55 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    mesh.add(new THREE.AxesHelper(0.9));
    mesh.userData.q = new THREE.Quaternion();
    this.scene.add(mesh);
    return mesh;
  }

  setQuat(slot, w, x, y, z) {
    const b = this.bodies[slot];
    if (b) b.userData.q.set(x, y, z, w).normalize();
  }

  setRelPos(pos, valid) {
    this._relPos = pos && valid ? pos : null;
  }

  setShowTranslation(on) {
    this.showTranslation = on;
    if (!on) this.resetTrail();
  }

  resetTrail() {
    this.trailPts = [];
    this.trail.geometry.setDrawRange(0, 0);
  }

  _resize() {
    const c = this.renderer.domElement;
    const w = c.clientWidth || 320, h = c.clientHeight || 220;
    if (w > 0 && h > 0 && (c.width !== w || c.height !== h)) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  _updateTrail(vec) {
    this.trailPts.push(vec.x, vec.y, vec.z);
    if (this.trailPts.length > this.trailMax * 3) this.trailPts.splice(0, 3);
    const arr = this.trail.geometry.attributes.position.array;
    arr.set(this.trailPts);
    this.trail.geometry.attributes.position.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, this.trailPts.length / 3);
  }

  draw() {
    this._resize();
    for (const slot of ["cra", "mta"]) {
      const b = this.bodies[slot];
      b.quaternion.slerp(b.userData.q, 0.35);
    }
    const mta = this.bodies.mta;
    if (this.showTranslation && this._relPos) {
      const target = this._mtaBase.clone().add(
        new THREE.Vector3(this._relPos[0], this._relPos[1], this._relPos[2]).multiplyScalar(this.posScale)
      );
      mta.position.lerp(target, 0.3);
      this._updateTrail(mta.position);
    } else {
      mta.position.lerp(this._mtaBase, 0.3);
    }
    this.renderer.render(this.scene, this.camera);
  }
}

class SlotCard {
  constructor(root, slot) {
    this.slot = slot.key;
    this.activeTab = "orient";
    this.lastQuatT = 0;
    this.lastVal = {};
    this.rates = {};
    this.cal = {};
    this.fusionSeen = new Set();

    const card = document.createElement("div");
    card.className = "imu-card";

    const head = document.createElement("div");
    head.className = "imu-card-head";
    const title = document.createElement("span");
    title.className = "imu-title";
    title.textContent = slot.title;
    const tabs = document.createElement("div");
    tabs.className = "imu-tabs";
    const rate = document.createElement("span");
    rate.className = "imu-rate";
    rate.title = "Measured (true) sample rate of the active stream";
    rate.textContent = "– Hz";
    head.append(title, tabs, rate);

    const cal = document.createElement("div");
    cal.className = "imu-cal";
    cal.title = "BNO085 on-chip calibration (0=unreliable..3=high)";

    // Channel legend: colored ch0/ch1/ch2 labels matching the plotted series.
    const legend = document.createElement("div");
    legend.className = "imu-legend";

    const readout = document.createElement("div");
    readout.className = "imu-readout";

    const stripHost = document.createElement("div");
    stripHost.className = "imu-strip-host";

    const offBanner = document.createElement("div");
    offBanner.className = "imu-off hidden";
    offBanner.textContent = "not streaming (not configured in IMU profile)";
    stripHost.append(offBanner);

    card.append(head, cal, legend, readout, stripHost);
    this.rateEl = rate;
    this.calEl = cal;
    this.legendEl = legend;
    this.offBanner = offBanner;
    this.stripHost = stripHost;

    this.tabBtns = {};
    for (const t of TABS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "imu-tab";
      btn.textContent = t.label;
      btn.addEventListener("click", () => this.setTab(t.key));
      tabs.append(btn);
      this.tabBtns[t.key] = btn;
    }

    // Strip containers: one Euler strip + one triple strip per motion stream.
    this.strips = {};
    this.stripEls = {};
    for (const t of TABS) {
      const el = document.createElement("div");
      el.className = "imu-strip";
      el.style.display = "none";
      stripHost.append(el);
      this.stripEls[t.key] = el;
      const labels = t.key === "orient" ? ["r", "p", "y"] : ["x", "y", "z"];
      this.strips[t.key] = new TripleStrip(el, labels);
    }
    this.readoutEl = readout;
    root.append(card);
    this.setTab("orient");
  }

  _streamActive(key) {
    const stream = TAB_RATE_STREAM[key];
    const hz = this.rates[stream] || 0;
    const hasCal = this.cal[stream] !== undefined;
    const strip = this.strips[key];
    const hasData = strip && strip.x.length > 0;
    return hz > 0 || hasCal || hasData;
  }

  _updateOffBanner() {
    const off = !this._streamActive(this.activeTab);
    this.offBanner.classList.toggle("hidden", !off);
    for (const t of TABS) {
      this.tabBtns[t.key].classList.toggle("inactive", !this._streamActive(t.key));
    }
  }

  setTab(key) {
    this.activeTab = key;
    for (const t of TABS) {
      const on = t.key === key;
      this.tabBtns[t.key].classList.toggle("active", on);
      this.stripEls[t.key].style.display = on ? "block" : "none";
    }
    const el = this.stripEls[key];
    this.strips[key].setSize(el.clientWidth, el.clientHeight);
    this._renderReadout();
    this._renderLegend();
    this._renderRate();
    this._renderCal();
    this._updateOffBanner();
  }

  _renderCal() {
    const active = TAB_RATE_STREAM[this.activeTab];
    const parts = CAL_STREAMS
      .filter((k) => this.cal[k] !== undefined)
      .map((k) => {
        const v = this.cal[k];
        const name = CAL_LABELS[v] || "?";
        const short = CAL_SHORT[k] || k;
        const on = k === active;
        return `<span class="imu-cal-badge cal-${v}${on ? " cal-active" : ""}"` +
          ` title="${k}: ${name}">${short} ${name}</span>`;
      });
    this.calEl.innerHTML = parts.length
      ? parts.join("")
      : '<span class="imu-cal-badge cal-na">cal n/a</span>';
  }

  setCal(cal) {
    if (!cal) return;
    this.cal = cal;
    this._renderCal();
  }

  _channelLabels() {
    return this.activeTab === "orient" ? ["roll", "pitch", "yaw"] : ["x", "y", "z"];
  }

  _renderLegend() {
    const labels = this._channelLabels();
    this.legendEl.innerHTML = labels
      .map((l, i) => `<span class="imu-leg" style="color:${SERIES_COLORS[i]}">ch${i} ${l}</span>`)
      .join("");
  }

  _renderRate() {
    const hz = this.rates[TAB_RATE_STREAM[this.activeTab]];
    this.rateEl.textContent = hz ? `${hz.toFixed(0)} Hz` : "– Hz";
  }

  setRates(rates) {
    if (!rates) return;
    this.rates = rates;
    this._renderRate();
    this._updateOffBanner();
  }

  _renderReadout() {
    const key = this.activeTab;
    if (key === "orient") {
      const [r, p, y] = this._euler || [0, 0, 0];
      this.readoutEl.innerHTML =
        `<span>r <b>${r.toFixed(0)}</b></span><span>p <b>${p.toFixed(0)}</b></span><span>y <b>${y.toFixed(0)}</b></span>`;
    } else {
      const v = this.lastVal[key] || [0, 0, 0];
      const u = (STREAMS.find((s) => s.key === key) || {}).unit || "";
      this.readoutEl.innerHTML =
        `<span>x <b>${v[0].toFixed(2)}</b></span><span>y <b>${v[1].toFixed(2)}</b></span>` +
        `<span>z <b>${v[2].toFixed(2)}</b></span><span class="imu-unit">${u}</span>`;
    }
  }

  applyQuat(q, t_s) {
    this.lastQuatT = t_s;
    this._euler = quatToEuler(q[0], q[1], q[2], q[3]);
    this.strips.orient.pushScalar(t_s, this._euler[0], this._euler[1], this._euler[2]);
    if (this.activeTab === "orient") this._renderReadout();
    this._updateOffBanner();
  }

  applyMotion(key, tArr, vArr) {
    this.strips[key].pushInterleaved(tArr, vArr);
    if (vArr.length >= 3) {
      const n = vArr.length;
      this.lastVal[key] = [vArr[n - 3], vArr[n - 2], vArr[n - 1]];
      if (this.activeTab === key) this._renderReadout();
    }
    this._updateOffBanner();
  }

  renderActive() { this.strips[this.activeTab].render(); }

  resize() {
    const el = this.stripEls[this.activeTab];
    if (el) this.strips[this.activeTab].setSize(el.clientWidth, el.clientHeight);
  }
}

const RATE_STREAMS = [
  { key: "quat", label: "QUAT" },
  ...STREAMS,
];
const STREAM_CHIP_LABELS = {
  quat: "Q", accel: "A", gyro: "G", mag: "M", linear_accel: "L",
};

export class ImuPanel {
  constructor(root) {
    this.root = root;
    this.root.innerHTML = "";
    this._dirty = false;
    this._seen = {}; // slot:stream -> last-seen ms
    this._rates = { cra: {}, mta: {} }; // slot -> {stream: measured Hz}
    this.quatByFusion = { cra: {}, mta: {} };

    this._buildToolbar();

    const sceneWrap = document.createElement("div");
    sceneWrap.className = "imu-scene-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "imu-scene";
    const disclaimer = document.createElement("div");
    disclaimer.className = "imu-disclaimer";
    disclaimer.textContent = "Translation is a drift-prone IMU estimate, not optical ground truth. "
      + "The two game rotation vectors do not share a heading, so relative yaw carries an arbitrary offset.";
    sceneWrap.append(canvas, disclaimer);
    this.root.append(sceneWrap);
    this.scene = new DualBodyScene(canvas);

    const cards = document.createElement("div");
    cards.className = "imu-cards";
    this.root.append(cards);
    this.cards = {};
    for (const s of SLOTS) this.cards[s.key] = new SlotCard(cards, s);
  }

  _buildToolbar() {
    const bar = document.createElement("div");
    bar.className = "imu-toolbar";

    this.slotChips = { mta: {}, cra: {} };
    this.chipLabels = {};
    const chipWrap = document.createElement("div");
    chipWrap.className = "imu-chips-split";
    for (const slot of SLOTS) {
      const row = document.createElement("div");
      row.className = "imu-chip-row";
      const title = document.createElement("span");
      title.className = "imu-chip-row-title";
      title.textContent = slot.key.toUpperCase();
      title.style.color = slot.accent;
      row.append(title);
      for (const t of RATE_STREAMS) {
        const chip = document.createElement("span");
        chip.className = "imu-chip chip-bad";
        chip.title = `${slot.title} — ${t.label} measured rate`;
        const label = t.label || t.key;
        chip.textContent = label;
        row.append(chip);
        this.slotChips[slot.key][t.key] = chip;
        this.chipLabels[`${slot.key}:${t.key}`] = label;
      }
      chipWrap.append(row);
    }

    const fusion = document.createElement("select");
    fusion.className = "imu-fusion";
    fusion.title = "Fusion source driving the 3D orientation";
    for (const f of FUSIONS) {
      const o = document.createElement("option");
      o.value = String(f.id);
      o.textContent = f.name;
      if (f.id === DEFAULT_FUSION) o.selected = true;
      fusion.append(o);
    }
    this.fusionSel = fusion;

    const transLabel = document.createElement("label");
    transLabel.className = "imu-toggle";
    const trans = document.createElement("input");
    trans.type = "checkbox";
    trans.addEventListener("change", () => this.scene.setShowTranslation(trans.checked));
    transLabel.append(trans, document.createTextNode(" est. translation"));

    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "imu-reset";
    reset.textContent = "Reset trail";
    reset.addEventListener("click", () => this.scene.resetTrail());

    bar.append(chipWrap, fusion, transLabel, reset);
    this.root.append(bar);
  }

  /** Mean measured Hz across active streams for one slot (header chips). */
  headerRates() {
    return { mta: this._slotAvgHz("mta"), cra: this._slotAvgHz("cra") };
  }

  _slotAvgHz(slot) {
    const r = this._rates[slot] || {};
    let n = 0, sum = 0;
    for (const t of RATE_STREAMS) {
      const hz = r[t.key] || 0;
      if (hz > 0) { sum += hz; n++; }
    }
    return n ? sum / n : 0;
  }

  _renderSlotChips() {
    const now = performance.now();
    for (const slot of ["mta", "cra"]) {
      for (const t of RATE_STREAMS) {
        const chip = this.slotChips[slot][t.key];
        if (!chip) continue;
        const hz = this._rates[slot][t.key] || 0;
        const seenKey = `${slot}:${t.key}`;
        const fresh = this._seen[seenKey] != null && now - this._seen[seenKey] < 1000;
        const active = fresh || hz > 0;
        chip.className = "imu-chip " + (active ? "chip-ok" : "chip-bad");
        const label = this.chipLabels[seenKey];
        chip.textContent = hz > 0 ? `${label} ${hz.toFixed(0)}` : label;
      }
    }
  }

  _markSeen(slot, stream) {
    this._seen[`${slot}:${stream}`] = performance.now();
  }

  update(imu) {
    if (!imu) return;
    const fusion = +this.fusionSel.value;
    this._applySlot("cra", imu.cra, fusion);
    this._applySlot("mta", imu.mta, fusion);
    if (imu.derived) {
      this.scene.setRelPos(imu.derived.pos || null, !!imu.derived.pos_valid);
    }
    this._dirty = true;
  }

  _applySlot(slot, s, fusion) {
    if (!s) return;
    const card = this.cards[slot];
    if (s.rates) {
      card.setRates(s.rates);
      this._rates[slot] = s.rates;
    }
    if (s.cal) card.setCal(s.cal);
    if (s.q) {
      const qid = s.qid != null ? s.qid : DEFAULT_FUSION;
      this.quatByFusion[slot][qid] = s.q;
      card.fusionSeen.add(qid);
      this._markSeen(slot, "quat");
      // Drive display with the selected fusion, falling back to whatever exists.
      const q = this.quatByFusion[slot][fusion] || s.q;
      card.applyQuat(q, s.t || 0);
      this.scene.setQuat(slot, q[0], q[1], q[2], q[3]);
    }
    if (s.motion) {
      for (const st of STREAMS) {
        const m = s.motion[st.key];
        if (m && m.t && m.v) {
          const tArr = asF32(m.t), vArr = asF32(m.v);
          card.applyMotion(st.key, tArr, vArr);
          this._markSeen(slot, st.key);
        }
      }
    }
  }

  resize() {
    for (const key of Object.keys(this.cards)) this.cards[key].resize();
  }

  render() {
    this.scene.draw();
    this._renderSlotChips();
    if (this._dirty) {
      for (const key of Object.keys(this.cards)) this.cards[key].renderActive();
      this._dirty = false;
    }
  }
}
