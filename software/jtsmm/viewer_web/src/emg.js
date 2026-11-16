// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-FileCopyrightText: 2026 David Stebich
// SPDX-License-Identifier: MIT

// EMG panel: one compact uPlot per channel with rolling buffers and a
// raw(min/max band) / filtered / envelope view toggle, plus lead-off and
// saturation indicators. uPlot is loaded globally via the vendored IIFE build.

const COLORS = ["#5ec8f0", "#f0a35e", "#8ef05e", "#f05e8e",
                "#b98ef0", "#f0e15e", "#9aa3b6", "#f0745e"];

const Y_PAD = 1.08;
const Y_ENV_MIN = 10; // µV — keep a readable 0…N floor so rest noise stays small

export function yRangeFor(mode, yAuto, ySpan, dataMin, dataMax) {
  if (!yAuto) {
    return mode === "envelope" ? [0, ySpan] : [-ySpan, ySpan];
  }
  if (mode === "envelope") {
    const hi = Number.isFinite(dataMax) ? dataMax : 0;
    return [0, Math.max(Y_ENV_MIN, hi * Y_PAD)];
  }
  let lo = Number.isFinite(dataMin) ? dataMin : 0;
  let hi = Number.isFinite(dataMax) ? dataMax : 0;
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  lo = Math.min(lo, 0);
  hi = Math.max(hi, 0);
  const extra = (hi - lo) * (Y_PAD - 1);
  return [lo - extra, hi + extra];
}

function fmtUv(_u, splits) {
  return splits.map((v) => {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 20) return String(Math.round(v));
    if (a >= 1) return v.toFixed(1);
    return v.toFixed(2);
  });
}

export class EmgPanel {
  constructor(root, meta) {
    this.root = root;
    this.meta = meta;
    this.C = meta.n_channels || 7;
    this.muscleC = meta.muscle_channels || Math.max(0, this.C - 1);
    this.names = meta.channel_names || [];
    this.windowS = 10;
    this.mode = "filtered"; // raw | filtered | envelope
    this.yAuto = true;
    this.ySpan = 100; // µV half-range (raw/filtered ±N, envelope 0…N)
    this.enabled = new Array(this.C).fill(true);
    this.charts = [];
    this.data = []; // per channel: {x:[], lo:[], hi:[], filt:[], env:[]}
    this.leds = [];
    this._dirty = false;
    this._minPlotH = 28;
    this._build();
  }

  _build() {
    this.root.innerHTML = "";
    for (let ch = 0; ch < this.C; ch++) {
      this.data.push({ x: [], lo: [], hi: [], filt: [], env: [] });

      const card = document.createElement("div");
      card.className = "emg-ch";
      const head = document.createElement("div");
      head.className = "emg-ch-head";
      const name = document.createElement("span");
      name.className = "emg-ch-name";
      const isDrl = ch === this.muscleC;
      if (isDrl) {
        name.textContent = "CH8 · DRL · mastoid reference";
      } else {
        name.textContent = this.names[ch] ? `CH${ch + 1} · ${this.names[ch]}` : `CH${ch + 1}`;
      }
      name.style.color = COLORS[ch % COLORS.length];
      const led = document.createElement("span");
      led.className = "led led-na";
      led.title = isDrl ? "DRL lead-off n/a" : "lead-off / saturation";
      this.leds.push(led);
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.className = "emg-ch-cb";
      cb.addEventListener("change", () => {
        this.enabled[ch] = cb.checked;
        card.classList.toggle("disabled", !cb.checked);
        this._relayout();
      });
      head.append(cb, name, led);

      const plot = document.createElement("div");
      plot.className = "emg-ch-plot";
      card.append(head, plot);
      this.root.append(card);

      const opts = {
        width: plot.clientWidth || 600,
        height: this._minPlotH,
        cursor: { drag: { x: false, y: false } },
        legend: { show: false },
        scales: {
          x: { time: false },
          y: { auto: true, range: (_u, lo, hi) => this._yRange(lo, hi) },
        },
        axes: [
          { show: false },
          { stroke: "#8a93a6", size: 44, grid: { stroke: "#232838" },
            ticks: { stroke: "#232838" }, font: "10px system-ui",
            values: fmtUv },
        ],
        series: [
          {},
          { stroke: "transparent", fill: COLORS[ch % COLORS.length] + "33", points: { show: false } },
          { stroke: "transparent", fill: COLORS[ch % COLORS.length] + "33", points: { show: false } },
          { stroke: COLORS[ch % COLORS.length], width: 1, points: { show: false } },
          { stroke: COLORS[ch % COLORS.length], width: 1.2, points: { show: false } },
        ],
        bands: [{ series: [2, 1], fill: COLORS[ch % COLORS.length] + "33" }],
      };
      const u = new window.uPlot(opts, [[0], [0], [0], [0], [0]], plot);
      this.charts.push(u);
    }
    this.setMode(this.mode);
    this._syncYSpanUi();
    this._relayout();
    this._ro = new ResizeObserver(() => this._relayout());
    this._ro.observe(this.root);
    window.addEventListener("resize", () => this._relayout());
  }

  _relayout() {
    const n = this.charts.length;
    if (!n) return;
    const rootH = this.root.clientHeight;
    if (rootH < 40) return;
    // Two passes: first commit every card's flex/display so the browser's
    // flex distribution is final, *then* measure. Sizing each uPlot canvas
    // from a hand-rolled height formula (assumed header height / gap) instead
    // of the plot box's real, laid-out height let the canvas end up taller
    // than its container and bleed into the channel below it -- interleaving
    // reads and writes here would reintroduce that via layout thrashing.
    for (let ch = 0; ch < n; ch++) {
      const u = this.charts[ch];
      // `u.root` is uPlot's own wrapper div; its parent is the `.emg-ch-plot`
      // flex box whose laid-out size we actually need to measure below --
      // reading `u.root`'s own size would just be self-referential (it's
      // exactly whatever the previous `setSize()` call set it to).
      const plotBox = u.root.parentElement;
      if (!this.enabled[ch]) {
        u.root.style.display = "none";
        plotBox.style.flex = "0 0 auto";
      } else {
        u.root.style.display = "";
        plotBox.style.flex = "1 1 0";
      }
    }
    for (let ch = 0; ch < n; ch++) {
      if (!this.enabled[ch]) continue;
      const u = this.charts[ch];
      const plotBox = u.root.parentElement;
      const w = plotBox.clientWidth || 600;
      const h = Math.max(this._minPlotH, plotBox.clientHeight);
      if (w > 0 && h > 0) u.setSize({ width: w, height: h });
    }
  }

  resize() {
    this._relayout();
  }

  setWindow(seconds) {
    this.windowS = Math.max(1, seconds);
  }

  setYAuto(on) {
    this.yAuto = !!on;
    this._applyYScale();
  }

  setYSpan(uv) {
    const n = Number(uv);
    this.ySpan = Number.isFinite(n) && n > 0 ? n : 100;
    this._applyYScale();
  }

  setMode(mode) {
    this.mode = mode;
    for (const u of this.charts) {
      const raw = mode === "raw";
      u.setSeries(1, { show: raw }, false);
      u.setSeries(2, { show: raw }, false);
      u.setSeries(3, { show: mode === "filtered" }, false);
      u.setSeries(4, { show: mode === "envelope" }, false);
    }
    this._applyYScale();
  }

  _yRange(dataMin, dataMax) {
    return yRangeFor(this.mode, this.yAuto, this.ySpan, dataMin, dataMax);
  }

  _syncYSpanUi() {
    const sel = document.getElementById("y-span");
    const auto = document.getElementById("y-auto");
    if (auto) auto.checked = this.yAuto;
    if (!sel) return;
    sel.disabled = this.yAuto;
    for (const opt of sel.options) {
      const n = +opt.value;
      opt.textContent = this.mode === "envelope" ? `0–${n} µV` : `±${n} µV`;
    }
  }

  _applyYScale() {
    this._syncYSpanUi();
    this._dirty = true;
    this.render();
  }

  push(emg) {
    const n = emg.n | 0;
    if (!n) return;
    const C = emg.c;
    const t = emg.tF, lo = emg.loF, hi = emg.hiF, filt = emg.filtF, env = emg.envF;
    for (let ch = 0; ch < this.C; ch++) {
      const d = this.data[ch];
      // Time went backwards (replay seek): the old window no longer applies.
      if (d.x.length && t[0] < d.x[d.x.length - 1]) this._clear(d);
      for (let i = 0; i < n; i++) {
        const k = i * C + ch;
        d.x.push(t[i]);
        d.lo.push(lo[k]);
        d.hi.push(hi[k]);
        d.filt.push(filt[k]);
        d.env.push(env[k]);
      }
      // Trim here, not only in render(): requestAnimationFrame does not run in
      // a hidden tab, and the buffers then grew without bound.
      this._trim(d);
    }
    this._dirty = true;
  }

  _clear(d) {
    d.x.length = 0; d.lo.length = 0; d.hi.length = 0; d.filt.length = 0; d.env.length = 0;
  }

  _trim(d) {
    if (d.x.length === 0) return;
    const tMin = d.x[d.x.length - 1] - this.windowS;
    let cut = 0;
    while (cut < d.x.length && d.x[cut] < tMin) cut++;
    if (cut > 0) {
      d.x.splice(0, cut); d.lo.splice(0, cut); d.hi.splice(0, cut);
      d.filt.splice(0, cut); d.env.splice(0, cut);
    }
  }

  render() {
    if (!this._dirty) return;
    this._dirty = false;
    for (let ch = 0; ch < this.C; ch++) {
      const d = this.data[ch];
      if (d.x.length === 0) continue;
      this._trim(d);                 // the window length may have changed
      this.charts[ch].setData([d.x, d.lo, d.hi, d.filt, d.env]);
    }
  }

  _setLed(led, off, sat, active, label, cmp) {
    if (!led) return;
    if (!active) {
      led.classList.remove("led-off", "led-sat", "led-ok");
      led.classList.add("led-na");
      led.title = `${label} lead-off n/a (disabled during recording)`;
      return;
    }
    // Comparators not confirmed running (old firmware powered them down):
    // all-zero bits mean nothing, so never show a green "connected" then.
    if (cmp !== undefined && cmp !== "on" && !off && !sat) {
      led.classList.remove("led-off", "led-sat", "led-ok");
      led.classList.add("led-na");
      led.title = `${label} contact unknown (lead-off comparators ${cmp}); a railed input still shows red`;
      return;
    }
    led.classList.remove("led-na");
    led.classList.toggle("led-off", !!off);
    led.classList.toggle("led-sat", !!sat && !off);
    led.classList.toggle("led-ok", !off && !sat);
    led.title = off ? `${label} lead-off` : (sat ? `${label} saturation` : `${label} ok`);
  }

  setLeadoff(p, n, sat, active, drlOff, cmp) {
    for (let ch = 0; ch < this.muscleC; ch++) {
      const off = (p && p[ch]) || (n && n[ch]);
      const s = sat && sat[ch];
      this._setLed(this.leds[ch], off, s, active, `CH${ch + 1}`, cmp);
    }
    if (this.C > this.muscleC) {
      // The IN8N comparator bit is masked on this board (IN8P is unwired, and
      // it never tripped on the bench). Always show n/a here; run
      // `jtsmm emg check` for the isolated DRL probe.
      void drlOff;
      this._setLed(this.leds[this.muscleC], false, false, false, "DRL");
      this.leds[this.muscleC].title = "DRL lead-off n/a in live view — run `jtsmm emg check` for a real result";
    }
  }

  setImpedance(rows) {
    const panel = this._impPanel;
    if (!panel) return;
    const body = panel.querySelector(".imp-body");
    if (!rows || !rows.length) {
      body.innerHTML = "<p class='imp-hint'>Collecting |Z| (AC lead-off tone; use jtsmm emg impedance)…</p>";
      return;
    }
    body.innerHTML = rows.map((r) => {
      const z = r.z_kohm != null ? `${r.z_kohm.toFixed(1)} kΩ` : "n/a";
      return `<div class="imp-row qual-${r.quality}"><span>CH${r.ch}</span><span>${z}</span><span>${r.quality}</span></div>`;
    }).join("");
  }

  toggleImpedance(show) {
    if (!this._impPanel) {
      const panel = document.createElement("div");
      panel.className = "imp-panel hidden";
      panel.innerHTML = "<header>Electrode |Z|</header><div class='imp-body'></div>";
      this.root.parentElement.append(panel);
      this._impPanel = panel;
    }
    this._impPanel.classList.toggle("hidden", !show);
  }
}
