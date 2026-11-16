// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

// Resizable, reorderable dock layout for EMG / IMU / camera panels.
//
// - Splitters use absolute pointer position (no delta accumulation) so resizing
//   is stable and predictable.
// - Panels are reordered by dragging the ⠿ handle onto another panel (custom
//   pointer-based drag with a floating ghost — reliable across form controls).
// - Any panel can be "focused" (auto-enlarged) via its ⛶ button, the top-bar
//   presets, a double-click on the header, or number keys 1/2/3.
// - Layout (split ratios, panel order, focus) persists in localStorage.

const STORAGE_KEY = "jtsmm-viewer-layout-v2";

const DEFAULTS = {
  colSplit: 0.66,   // main column fraction (0–1)
  rowSplit: 0.55,   // top-of-stack fraction (0–1)
  focus: null,      // null | "emg" | "imu" | "cam"
  slots: ["emg", "imu", "cam"], // main, stack-top, stack-bottom
};

// How the layout skews when a modality is focused: [colSplit, rowSplit].
// Slot 0 is the wide main column; slots 1 and 2 stack on the right.
const FOCUS_SKEW = {
  main: { col: 0.86, row: 0.55 },
  top: { col: 0.26, row: 0.82 },
  bottom: { col: 0.26, row: 0.18 },
};

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const s = JSON.parse(raw);
    const merged = { ...DEFAULTS, ...s };
    if (!Array.isArray(merged.slots) || merged.slots.length !== 3) {
      merged.slots = [...DEFAULTS.slots];
    }
    return merged;
  } catch {
    return { ...DEFAULTS };
  }
}

function saveState(state) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch { /* private mode / quota */ }
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const isNarrow = () => window.matchMedia("(max-width: 900px)").matches;

/**
 * @param {HTMLElement} root  #layout-root
 * @param {{ id: string, el: HTMLElement }[]} panels
 */
export class DockLayout {
  constructor(root, panels) {
    this.root = root;
    this.panels = new Map(panels.map((p) => [p.id, p.el]));
    this.state = loadState();
    this._onResize = null;

    this._colSplit = root.querySelector("#split-col");
    this._rowSplit = root.querySelector("#split-row");
    this._stack = root.querySelector("#dock-stack");

    this._installSplitters();
    this._installFocusButtons();
    this._installReorder();
    this._installHeaderDblClick();
    this._installKeyboard();
    this.apply();

    this._ro = new ResizeObserver(() => this.apply());
    this._ro.observe(root);
  }

  onResize(fn) { this._onResize = fn; }
  _notify() { if (this._onResize) this._onResize(); }

  // ---- resizing -----------------------------------------------------------

  _installSplitters() {
    this._dragSplit(this._colSplit, "col");
    this._dragSplit(this._rowSplit, "row");
    this._colSplit.addEventListener("dblclick", () => this._resetSplit("col"));
    this._rowSplit.addEventListener("dblclick", () => this._resetSplit("row"));
  }

  _resetSplit(which) {
    if (which === "col") this.state.colSplit = DEFAULTS.colSplit;
    else this.state.rowSplit = DEFAULTS.rowSplit;
    this.state.focus = null;
    this.apply();
    saveState(this.state);
    this._notify();
  }

  _dragSplit(el, axis) {
    el.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      el.classList.add("dragging");
      try { el.setPointerCapture(e.pointerId); } catch { /* noop */ }
      this.state.focus = null;

      const move = (ev) => {
        // The narrow (stacked) layout turns the column splitter horizontal.
        if (axis === "col" && !isNarrow()) {
          const r = this.root.getBoundingClientRect();
          this.state.colSplit = clamp((ev.clientX - r.left) / r.width, 0.2, 0.85);
        } else if (axis === "col") {
          const r = this.root.getBoundingClientRect();
          this.state.colSplit = clamp((ev.clientY - r.top) / r.height, 0.2, 0.85);
        } else {
          const r = this._stack.getBoundingClientRect();
          this.state.rowSplit = clamp((ev.clientY - r.top) / r.height, 0.15, 0.85);
        }
        this.apply();
      };
      const up = () => {
        el.classList.remove("dragging");
        try { el.releasePointerCapture(e.pointerId); } catch { /* noop */ }
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
        el.removeEventListener("pointercancel", up);
        saveState(this.state);
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
      el.addEventListener("pointercancel", up);
    });
  }

  // ---- focus --------------------------------------------------------------

  _installFocusButtons() {
    this.root.querySelectorAll(".dock-focus").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const id = btn.closest(".dock-panel")?.dataset.panel;
        if (id) this.toggleFocus(id);
      });
    });
  }

  _installHeaderDblClick() {
    this.panels.forEach((el, id) => {
      const head = el.querySelector(".dock-head");
      if (head) {
        head.addEventListener("dblclick", (e) => {
          if (e.target.closest("input, select, button, label")) return;
          this.toggleFocus(id);
        });
      }
    });
  }

  // Keyboard: 1/2/3 focus the panel in each slot, 0 or Esc balances, R resets.
  _installKeyboard() {
    window.addEventListener("keydown", (e) => {
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case "1": this.toggleFocus(this.state.slots[0]); break;
        case "2": this.toggleFocus(this.state.slots[1]); break;
        case "3": this.toggleFocus(this.state.slots[2]); break;
        case "0": case "Escape": this.clearFocus(); break;
        case "r": case "R": this.resetLayout(); break;
        default: return;
      }
      e.preventDefault();
    });
  }

  toggleFocus(id) {
    this.state.focus = this.state.focus === id ? null : id;
    this._commit();
  }

  setFocus(id) { this.state.focus = id; this._commit(); }
  clearFocus() { this.state.focus = null; this._commit(); }

  resetLayout() {
    this.state = { ...DEFAULTS, slots: [...DEFAULTS.slots] };
    this._commit();
  }

  _commit() {
    this.apply();
    saveState(this.state);
  }

  // ---- reorder (custom pointer drag) -------------------------------------

  _installReorder() {
    this.root.querySelectorAll(".dock-drag").forEach((handle) => {
      handle.addEventListener("pointerdown", (e) => this._startReorder(e, handle));
    });
  }

  _startReorder(e, handle) {
    if (e.button !== 0) return;
    e.preventDefault();
    const panel = handle.closest(".dock-panel");
    if (!panel) return;
    const fromId = panel.dataset.panel;
    try { handle.setPointerCapture(e.pointerId); } catch { /* noop */ }
    panel.classList.add("dragging");

    const ghost = document.createElement("div");
    ghost.className = "dock-ghost";
    ghost.textContent = panel.querySelector("h2")?.textContent || fromId;
    document.body.appendChild(ghost);
    const place = (x, y) => { ghost.style.left = `${x + 12}px`; ghost.style.top = `${y + 12}px`; };
    place(e.clientX, e.clientY);

    let overId = null;
    const setOver = (id) => {
      if (id === overId) return;
      for (const [, el] of this.panels) el.classList.remove("drop-target");
      overId = id;
      if (overId) this.panels.get(overId).classList.add("drop-target");
    };

    const move = (ev) => {
      place(ev.clientX, ev.clientY);
      const under = document.elementFromPoint(ev.clientX, ev.clientY);
      const overPanel = under && under.closest(".dock-panel");
      setOver(overPanel && overPanel.dataset.panel !== fromId ? overPanel.dataset.panel : null);
    };
    const up = () => {
      try { handle.releasePointerCapture(e.pointerId); } catch { /* noop */ }
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
      ghost.remove();
      panel.classList.remove("dragging");
      const target = overId;
      for (const [, el] of this.panels) el.classList.remove("drop-target");
      if (target && target !== fromId) {
        this._swapSlots(fromId, target);
        this.state.focus = null;
        this._commit();
      }
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  }

  _swapSlots(a, b) {
    const slots = [...this.state.slots];
    const ia = slots.indexOf(a);
    const ib = slots.indexOf(b);
    if (ia < 0 || ib < 0) return;
    [slots[ia], slots[ib]] = [slots[ib], slots[ia]];
    this.state.slots = slots;
  }

  // ---- apply --------------------------------------------------------------

  apply() {
    const { colSplit, rowSplit, focus, slots } = this.state;
    const [mainId, topId, botId] = slots;

    const slotEls = {
      tl: this.root.querySelector('[data-slot="tl"]'),
      tr: this.root.querySelector('[data-slot="tr"]'),
      br: this.root.querySelector('[data-slot="br"]'),
    };
    const mount = (id, key) => {
      const el = this.panels.get(id);
      const host = slotEls[key];
      if (el && host && el.parentElement !== host) host.appendChild(el);
    };
    mount(mainId, "tl");
    mount(topId, "tr");
    mount(botId, "br");

    let cs = colSplit;
    let rs = rowSplit;
    if (focus === mainId) { cs = FOCUS_SKEW.main.col; rs = FOCUS_SKEW.main.row; }
    else if (focus === topId) { cs = FOCUS_SKEW.top.col; rs = FOCUS_SKEW.top.row; }
    else if (focus === botId) { cs = FOCUS_SKEW.bottom.col; rs = FOCUS_SKEW.bottom.row; }

    const mainFr = Math.max(0.15, cs) * 100;
    const sideFr = Math.max(0.12, 1 - cs) * 100;
    const topFr = Math.max(0.1, rs) * 100;
    const botFr = Math.max(0.1, 1 - rs) * 100;

    if (isNarrow()) {
      this.root.style.gridTemplateColumns = "";
      this.root.style.gridTemplateRows =
        `minmax(160px, ${mainFr}fr) 8px minmax(200px, ${sideFr}fr)`;
      if (this._stack) this._stack.style.gridTemplateRows = "";
    } else {
      this.root.style.gridTemplateRows = "";
      this.root.style.gridTemplateColumns =
        `minmax(200px, ${mainFr}fr) 8px minmax(180px, ${sideFr}fr)`;
      if (this._stack) {
        this._stack.style.gridTemplateRows =
          `minmax(90px, ${topFr}fr) 8px minmax(90px, ${botFr}fr)`;
      }
    }

    this.root.dataset.focus = focus || "";
    for (const [id, el] of this.panels) {
      el.classList.toggle("is-focused", focus === id);
      el.classList.toggle("is-dimmed", focus != null && focus !== id);
      const btn = el.querySelector(".dock-focus");
      if (btn) btn.classList.toggle("active", focus === id);
    }

    // Let plot widgets re-fit to their new box (deferred to next frame so the
    // grid has actually reflowed before we measure client sizes).
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => { this._raf = null; this._notify(); });
  }
}
