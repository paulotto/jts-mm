// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

// JTS-MM parts viewer: parts.yaml variants (via manifest.json, written by tools/viewer_export.py),
// the molar-clamp concepts on the teeth scan, a compare overlay, section plane, measuring and the
// CAD-check overlay (tag seats, IMU seat, pivot-calibrated tip). Served from the repo root.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const COLORS = { variant: 0x8fb3d9, printed: 0x8fb3d9, used: 0x6fae8c, superseded: 0xb8bcc2,
  compare: 0xe8742a, tooth: 0xf3e6d2, printedC: 0x3f7cc9, rigid: 0x35507a, spring: 0xe8742a,
  wire: 0x8a96a3, tpu: 0x4caf50 };
const dark = matchMedia('(prefers-color-scheme: dark)').matches;

// -- scene --------------------------------------------------------------------------------------
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.localClippingEnabled = true;
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(dark ? 0x1d1f21 : 0xf7f7f5);
const camera = new THREE.PerspectiveCamera(35, 1, 0.5, 5000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8c8c8c, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
camera.add(sun);
sun.position.set(0.5, 1, 0.8);
scene.add(camera);
const groups = { part: new THREE.Group(), compare: new THREE.Group(), concepts: new THREE.Group(),
  overlay: new THREE.Group(), measure: new THREE.Group() };
Object.values(groups).forEach((g) => scene.add(g));
const clip = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
let clipPlanes = [];

function resize() {
  const r = view.getBoundingClientRect();
  renderer.setSize(r.width, r.height);
  camera.aspect = r.width / Math.max(1, r.height);
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

// -- STL (binary or ASCII) ----------------------------------------------------------------------
const cache = new Map();
async function loadSTL(url) {
  if (cache.has(url)) return cache.get(url);
  const p = fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.arrayBuffer();
  }).then(parseSTL);
  cache.set(url, p);
  return p;
}
function parseSTL(buf) {
  const dv = new DataView(buf);
  const n = buf.byteLength >= 84 ? dv.getUint32(80, true) : 0;
  let pos;
  if (84 + n * 50 === buf.byteLength) {
    pos = new Float32Array(n * 9);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 9; k++) pos[i * 9 + k] = dv.getFloat32(84 + i * 50 + 12 + k * 4, true);
    }
  } else {
    const txt = new TextDecoder().decode(buf);
    const v = [...txt.matchAll(/vertex\s+(\S+)\s+(\S+)\s+(\S+)/g)].flatMap((m) => m.slice(1, 4).map(Number));
    pos = new Float32Array(v);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.computeVertexNormals();
  g.computeBoundingBox();
  return g;
}
function mesh(geom, color, opts = {}) {
  const m = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
    color, roughness: 0.6, metalness: 0.05, side: THREE.DoubleSide, flatShading: true,
    clippingPlanes: clipPlanes, transparent: !!opts.opacity, opacity: opts.opacity ?? 1,
    depthWrite: !opts.opacity }));
  m.userData.pickable = !opts.opacity;
  return m;
}
function edgesOf(geom) {
  return new THREE.LineSegments(new THREE.EdgesGeometry(geom, 25),
    new THREE.LineBasicMaterial({ color: dark ? 0xdddddd : 0x333333, clippingPlanes: clipPlanes }));
}

// -- state --------------------------------------------------------------------------------------
let M = null;
const state = { part: null, variant: null, compare: '', concepts: new Set(), scan: false };

function part() { return M.parts.find((p) => p.id === state.part); }
function variant() { return part().variants.find((v) => v.id === state.variant); }
function clear(g) { while (g.children.length) g.remove(g.children[0]); }
function fmt(x, d = 1) { return x == null ? '–' : Number(x).toFixed(d); }

async function showPart(fit = true) {
  clear(groups.part); clear(groups.compare); clear(groups.overlay); clear(groups.measure);
  measurePts = [];
  $('measured').textContent = '';
  const v = variant();
  if (state.concepts.size || state.scan) { renderInfo(); await showConcepts(fit); return; }
  status(`loading ${v.url}`);
  const g = await loadSTL(v.url);
  const used = Object.values(part().used).includes(v.id);
  groups.part.add(mesh(g, used ? COLORS.used : (COLORS[v.status] ?? COLORS.variant)));
  if ($('edges').checked) groups.part.add(edgesOf(g));
  if (state.compare) {
    const c = part().variants.find((x) => x.id === state.compare);
    groups.compare.add(mesh(await loadSTL(c.url), COLORS.compare, { opacity: 0.35 }));
  }
  if ($('overlay').checked && v.overlay) drawOverlay(v.overlay);
  renderInfo();
  if (fit) fitTo(groups.part);
  status('');
  writeHash();
}

async function showConcepts(fit) {
  clear(groups.concepts);
  if (state.scan) groups.concepts.add(mesh(await loadSTL(M.scan.url), COLORS.tooth));
  for (const c of M.concepts.filter((x) => state.concepts.has(x.concept))) {
    const col = { printed: COLORS.printedC, rigid: COLORS.rigid, spring: COLORS.spring,
      wire: COLORS.wire, tpu: COLORS.tpu }[c.kind];
    groups.concepts.add(mesh(await loadSTL(c.url), col));
  }
  if (fit) fitTo(groups.concepts, OCCLUSAL);
  writeHash();
}

function drawOverlay(o) {
  const line = (pts, color, dashed = false) => {
    const g = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)));
    const mat = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 1.5, gapSize: 1 })
      : new THREE.LineBasicMaterial({ color });
    const l = new THREE.Line(g, mat);
    if (dashed) l.computeLineDistances();
    groups.overlay.add(l);
  };
  const square = (c, side, ang, z) => {
    const r = ang * Math.PI / 180;
    const R = (x, y) => [c[0] + x * Math.cos(r) - y * Math.sin(r), c[1] + x * Math.sin(r) + y * Math.cos(r), z];
    const h = side / 2;
    return [R(-h, -h), R(h, -h), R(h, h), R(-h, h), R(-h, -h)];
  };
  o.seats.forEach((c, i) => {
    line(square(c, o.seat_side[i], o.angles[i], c[2] + 0.2), dark ? 0xeeeeee : 0x111111);
    line(square(c, 30, o.angles[i], c[2] + 0.25), 0xc0392b, true);
  });
  if (o.imu_seat) {
    const [w, h] = o.imu_seat_size, c = o.imu_seat;
    line([[c[0] - w / 2, c[1] - h / 2, c[2] + 0.2], [c[0] + w / 2, c[1] - h / 2, c[2] + 0.2],
      [c[0] + w / 2, c[1] + h / 2, c[2] + 0.2], [c[0] - w / 2, c[1] + h / 2, c[2] + 0.2],
      [c[0] - w / 2, c[1] - h / 2, c[2] + 0.2]], 0xe8742a);
  }
  if (o.tip_cad) {
    line([o.rod_end, o.tip_cad], 0xc0392b);
    const s = new THREE.Mesh(new THREE.SphereGeometry(1.2, 16, 12), new THREE.MeshBasicMaterial({ color: 0xc0392b }));
    s.position.set(...o.tip_cad);
    groups.overlay.add(s);
  }
}

function renderInfo() {
  const p = part(), v = variant();
  if (state.concepts.size || state.scan) {
    $('info').innerHTML = `<p class="muted small">Showing the concept study: ${[...state.concepts].join(', ') || 'scan only'}.
      Clear the concept boxes to return to the part.</p>`;
    return;
  }
  const usage = Object.entries(p.used).map(([s, u]) => `${s}: ${u === v.id ? '<b>this variant</b>' : (u ?? '<i>not yet specified</i>')}`).join('<br>');
  const cad = v.cad === 'pending' ? '<span class="badge pending">CAD pending</span>' : (v.cad === 'yes' ? 'FreeCAD body' : 'STL only');
  $('info').innerHTML = `
    <p class="small">${p.role}</p>
    <table>
      <tr><td>status</td><td><span class="badge ${v.status}">${v.status}</span> ${cad}</td></tr>
      <tr><td>size</td><td>${v.size_mm.map((x) => fmt(x)).join(' × ')} mm</td></tr>
      <tr><td>volume</td><td>${fmt(v.volume_mm3 / 1000, 2)} cm³</td></tr>
      <tr><td>mass</td><td>${fmt(v.mass_solid_g)} g solid ${p.material}, ≈ ${fmt(v.mass_printed_g)} g printed</td></tr>
      <tr><td>mesh</td><td>${v.triangles} triangles</td></tr>
      ${v.date ? `<tr><td>date</td><td>${v.date}</td></tr>` : ''}
      <tr><td>used in</td><td>${usage || '–'}</td></tr>
    </table>
    ${v.note ? `<p class="muted small">${v.note}</p>` : ''}`;
}

// -- camera, section plane, measuring -----------------------------------------------------------
const box = new THREE.Box3();
// dir: view direction (camera offset); the scan frame has the cusp tips at -z (occlusal = -z)
const ISO = [-0.9, -1.4, 1.0], OCCLUSAL = [0.35, -1.0, -1.3];
function fitTo(g, dir = ISO) {
  box.setFromObject(g);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2;
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(...dir).normalize().multiplyScalar(r / Math.tan(camera.fov * Math.PI / 360) * 1.1));
  camera.near = r / 100; camera.far = r * 50;
  camera.updateProjectionMatrix();
  updateClip();
}
function updateClip() {
  const ax = $('clipaxis').value;
  $('clippos').disabled = !ax;
  clipPlanes.length = 0;
  if (ax) {
    const i = 'xyz'.indexOf(ax), t = $('clippos').value / 1000;
    const lo = box.min.getComponent(i), hi = box.max.getComponent(i);
    clip.normal.set(...[0, 1, 2].map((k) => (k === i ? -1 : 0)));
    clip.constant = lo + t * (hi - lo);
    clipPlanes.push(clip);
  }
  scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
}
let measurePts = [];
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
renderer.domElement.addEventListener('click', (e) => {
  if (!$('measure').checked) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hits = ray.intersectObjects([...groups.part.children, ...groups.concepts.children].filter((o) => o.userData.pickable));
  const hit = hits.find((h) => !clipPlanes.length || clip.distanceToPoint(h.point) >= 0);
  if (!hit) return;
  const p = hit.point;
  if (measurePts.length === 2) { measurePts = []; clear(groups.measure); }
  measurePts.push(p.clone());
  const dot = new THREE.Mesh(new THREE.SphereGeometry(box.getSize(new THREE.Vector3()).length() / 120, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0xc0392b, depthTest: false }));
  dot.position.copy(p); dot.renderOrder = 2;
  groups.measure.add(dot);
  if (measurePts.length === 2) {
    groups.measure.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(measurePts),
      new THREE.LineBasicMaterial({ color: 0xc0392b, depthTest: false })));
    const d = measurePts[0].distanceTo(measurePts[1]);
    const dv = measurePts[1].clone().sub(measurePts[0]);
    $('measured').textContent = `distance ${d.toFixed(2)} mm (dx ${dv.x.toFixed(2)}, dy ${dv.y.toFixed(2)}, dz ${dv.z.toFixed(2)})`;
  } else $('measured').textContent = 'pick the second point';
});

// -- UI ------------------------------------------------------------------------------------------
function status(t) { $('status').textContent = t; }
function fillVariants() {
  const p = part();
  const usedIds = new Set(Object.values(p.used).filter(Boolean));
  $('variant').innerHTML = p.variants.map((v) => `<option value="${v.id}">${v.title}${usedIds.has(v.id) ? ' (used)' : ''} · ${v.status}</option>`).join('');
  $('compare').innerHTML = '<option value="">none</option>' + p.variants.map((v) => `<option value="${v.id}">${v.title}</option>`).join('');
  if (!p.variants.some((v) => v.id === state.variant)) {
    const pick = p.variants.find((v) => usedIds.has(v.id)) ?? [...p.variants].reverse().find((v) => v.status !== 'superseded') ?? p.variants[0];
    state.variant = pick.id;
  }
  $('variant').value = state.variant;
  state.compare = '';
}
function writeHash() {
  const h = new URLSearchParams({ part: state.part, v: state.variant });
  if (state.compare) h.set('cmp', state.compare);
  if (state.concepts.size) h.set('concepts', [...state.concepts].join(','));
  if (state.scan) h.set('scan', '1');
  history.replaceState(null, '', '#' + h);
}

async function init() {
  M = await (await fetch('manifest.json')).json();
  const h = new URLSearchParams(location.hash.slice(1));
  $('part').innerHTML = M.parts.filter((p) => p.variants.length).map((p) => `<option value="${p.id}">${p.title}</option>`).join('');
  state.part = M.parts.some((p) => p.id === h.get('part')) ? h.get('part') : 'teeth_attachment';
  state.variant = h.get('v');
  $('part').value = state.part;
  fillVariants();
  if (h.get('cmp')) { state.compare = h.get('cmp'); $('compare').value = state.compare; }
  const ids = [...new Set(M.concepts.map((c) => c.concept))];
  $('concepts').innerHTML = ids.map((c) => `<label class="row"><input type="checkbox" data-c="${c}"> ${M.concepts.find((x) => x.concept === c).title.split(':')[0]}</label>`).join('');
  (h.get('concepts') || '').split(',').filter(Boolean).forEach((c) => state.concepts.add(c));
  state.scan = h.get('scan') === '1';
  $('scan').checked = state.scan;
  document.querySelectorAll('#concepts input').forEach((el) => {
    el.checked = state.concepts.has(el.dataset.c);
    el.addEventListener('change', () => {
      el.checked ? state.concepts.add(el.dataset.c) : state.concepts.delete(el.dataset.c);
      if (state.concepts.size) state.scan = $('scan').checked = true;
      clear(groups.concepts); showPart();
    });
  });
  $('part').addEventListener('change', () => { state.part = $('part').value; state.variant = null; fillVariants(); clearConcepts(); showPart(); });
  $('variant').addEventListener('change', () => { state.variant = $('variant').value; showPart(false); });
  $('compare').addEventListener('change', () => { state.compare = $('compare').value; showPart(false); });
  $('scan').addEventListener('change', () => { state.scan = $('scan').checked; clear(groups.concepts); showPart(); });
  ['overlay', 'edges'].forEach((id) => $(id).addEventListener('change', () => showPart(false)));
  $('clipaxis').addEventListener('change', updateClip);
  $('clippos').addEventListener('input', updateClip);
  $('reset').addEventListener('click', () => (state.concepts.size || state.scan ? fitTo(groups.concepts, OCCLUSAL) : fitTo(groups.part)));
  resize();
  await showPart();
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
}
function clearConcepts() {
  state.concepts.clear(); state.scan = false; $('scan').checked = false;
  document.querySelectorAll('#concepts input').forEach((el) => { el.checked = false; });
  clear(groups.concepts);
}
init().catch((e) => { status(String(e)); console.error(e); });
