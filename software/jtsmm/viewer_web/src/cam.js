// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

// Camera dock: MJPEG <img> tiles for connected cams only (0 / 1 / N).

export class CamPanel {
  constructor(root) {
    this.root = root;
    this._ids = [];
  }

  /** Build tiles from session camera_ids (MJPEG via /stream/{id}). */
  setCameras(ids) {
    const list = Array.isArray(ids) ? ids.map(String) : [];
    if (sameIds(this._ids, list)) return;
    this._ids = list;
    this.root.innerHTML = "";
    this.root.classList.toggle("cam-grid-1", list.length === 1);
    this.root.classList.toggle("cam-grid-empty", list.length === 0);
    if (!list.length) {
      const empty = document.createElement("div");
      empty.className = "cam-tile cam-empty";
      empty.textContent = "No cameras connected";
      this.root.append(empty);
      return;
    }
    for (const id of list) {
      const tile = document.createElement("div");
      tile.className = "cam-tile cam-live";
      const img = document.createElement("img");
      img.src = `/stream/${encodeURIComponent(id)}`;
      img.alt = id;
      const label = document.createElement("span");
      label.textContent = id;
      tile.append(img, label);
      this.root.append(tile);
    }
  }
}

function sameIds(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
