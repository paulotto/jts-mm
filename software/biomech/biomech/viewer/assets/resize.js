// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* biomech viewer: responsive figures (loaded by Dash from assets/ on every page).
 *
 * 1. A ResizeObserver on every Plotly graph container (.graph-box, .dash-graph,
 *    .js-plotly-plot), including graphs added later (MutationObserver), calls
 *    Plotly.Plots.resize (debounced) when the container changes size: window resize,
 *    a card dragged taller, the explorer's column divider.
 * 2. The window width goes into the `viewport` store (debounced), so the server lays the
 *    figures out again for the new width (render_plotly.to_plotly(fit=True)).
 * 3. The explorer's split divider (.split-handle): drag to resize the columns,
 *    double-click resets; the width is remembered per browser (localStorage, optional).
 * 4. Theme "auto" follows the OS while the page is open: a prefers-color-scheme listener
 *    pushes the new theme into the `theme` store (figures re-render with the template).
 */
(function () {
  'use strict';
  var RESIZE_MS = 120, VIEWPORT_MS = 400, VIEWPORT_STEP_PX = 24;
  var SPLIT_KEY = 'biomech.split-left';
  var api = window.biomechViewer = window.biomechViewer || {};
  api.lastWidth = window.innerWidth;

  function debounce(fn, ms) {
    var t = null;
    return function () {
      var args = arguments;
      clearTimeout(t);
      t = setTimeout(function () { fn.apply(null, args); }, ms);
    };
  }

  // -- 1. Plotly graphs follow their containers -------------------------------------------
  var pending = new Set();
  var flush = debounce(function () {
    pending.forEach(function (el) {
      if (el.isConnected && window.Plotly && el._fullLayout) {
        try { window.Plotly.Plots.resize(el); } catch (e) { /* graph being replaced */ }
      }
    });
    pending.clear();
  }, RESIZE_MS);

  function plotsIn(el) {
    if (el.classList && el.classList.contains('js-plotly-plot')) { return [el]; }
    return el.querySelectorAll ? Array.prototype.slice.call(
      el.querySelectorAll('.js-plotly-plot')) : [];
  }

  var observer = window.ResizeObserver ? new ResizeObserver(function (entries) {
    entries.forEach(function (en) { plotsIn(en.target).forEach(function (p) { pending.add(p); }); });
    flush();
  }) : null;
  var watched = new WeakSet();
  var SELECTOR = '.graph-box, .dash-graph, .js-plotly-plot';

  function attach(root) {
    if (!observer || !root || root.nodeType !== 1) { return; }
    if (root.namespaceURI === 'http://www.w3.org/2000/svg') { return; }   // Plotly internals
    var els = Array.prototype.slice.call(root.querySelectorAll(SELECTOR));
    if (root.matches && root.matches(SELECTOR)) { els.push(root); }
    els.forEach(function (el) {
      if (!watched.has(el)) { watched.add(el); observer.observe(el); }
    });
    if (root.matches && root.matches('.split')) { restoreSplit(root); }
    Array.prototype.forEach.call(root.querySelectorAll('.split'), restoreSplit);
  }
  api.attach = attach;

  // -- 2. viewport store ------------------------------------------------------------------
  var pushViewport = debounce(function () {
    var w = window.innerWidth;
    if (Math.abs(w - api.lastWidth) < VIEWPORT_STEP_PX) { return; }
    api.lastWidth = w;
    var dc = window.dash_clientside;
    if (dc && typeof dc.set_props === 'function') {
      try { dc.set_props('viewport', { data: w }); } catch (e) { /* store not mounted */ }
    }
  }, VIEWPORT_MS);
  window.addEventListener('resize', pushViewport);

  // -- 3. explorer column divider ---------------------------------------------------------
  function readSplit() {
    try { return window.localStorage.getItem(SPLIT_KEY); } catch (e) { return null; }
  }
  function writeSplit(v) {
    try {
      if (v === null) { window.localStorage.removeItem(SPLIT_KEY); }
      else { window.localStorage.setItem(SPLIT_KEY, v); }
    } catch (e) { /* storage blocked: the divider still works for this page */ }
  }
  function restoreSplit(split) {
    var v = readSplit();
    if (v && !split.style.getPropertyValue('--split-left')) {
      split.style.setProperty('--split-left', v);
    }
  }

  document.addEventListener('pointerdown', function (ev) {
    var handle = ev.target.closest ? ev.target.closest('.split-handle') : null;
    if (!handle || ev.button !== 0) { return; }
    var split = handle.parentElement;
    ev.preventDefault();
    handle.setPointerCapture && handle.setPointerCapture(ev.pointerId);
    split.classList.add('dragging');
    function move(e) {
      var r = split.getBoundingClientRect();
      var f = Math.min(0.85, Math.max(0.3, (e.clientX - r.left) / r.width));
      split.style.setProperty('--split-left', (f * 100).toFixed(1) + '%');
    }
    function up() {
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      handle.removeEventListener('pointercancel', up);
      split.classList.remove('dragging');
      writeSplit(split.style.getPropertyValue('--split-left') || null);
    }
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
    handle.addEventListener('pointercancel', up);
  });
  document.addEventListener('dblclick', function (ev) {
    var handle = ev.target.closest ? ev.target.closest('.split-handle') : null;
    if (!handle) { return; }
    handle.parentElement.style.removeProperty('--split-left');
    writeSplit(null);
  });

  // -- 4. theme "auto" follows OS changes -------------------------------------------------
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onScheme = function (e) {
      var choice = api.themeChoice || 'auto';
      if (choice !== 'auto') { return; }
      var t = e.matches ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', t);
      var dc = window.dash_clientside;
      if (dc && typeof dc.set_props === 'function') {
        try { dc.set_props('theme', { data: t }); } catch (err) { /* store not mounted */ }
      }
    };
    if (mq.addEventListener) { mq.addEventListener('change', onScheme); }
    else if (mq.addListener) { mq.addListener(onScheme); }
  }

  // -- start ------------------------------------------------------------------------------
  function start() {
    attach(document.body);
    new MutationObserver(function (muts) {
      muts.forEach(function (m) { m.addedNodes.forEach(attach); });
    }).observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', start); }
  else { start(); }
})();
