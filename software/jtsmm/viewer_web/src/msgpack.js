// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

// Minimal MessagePack decoder covering exactly what jtsmm.viewer.protocol emits:
// maps, arrays, str, bin (raw Float32 blocks), ints, floats, bool, nil.
// `bin` values are returned as standalone (copied) Uint8Arrays so callers can
// wrap them in a Float32Array without worrying about byte alignment.

export function decode(buf) {
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let pos = 0;

  function readStr(len) {
    const s = new TextDecoder("utf-8").decode(u8.subarray(pos, pos + len));
    pos += len;
    return s;
  }
  function readBin(len) {
    const out = u8.slice(pos, pos + len); // copy -> own aligned buffer
    pos += len;
    return out;
  }
  function readArray(len) {
    const a = new Array(len);
    for (let i = 0; i < len; i++) a[i] = read();
    return a;
  }
  function readMap(len) {
    const m = {};
    for (let i = 0; i < len; i++) {
      const k = read();
      m[k] = read();
    }
    return m;
  }

  function read() {
    const b = view.getUint8(pos++);
    if (b < 0x80) return b; // positive fixint
    if (b >= 0xe0) return b - 0x100; // negative fixint
    if (b >= 0x80 && b <= 0x8f) return readMap(b & 0x0f); // fixmap
    if (b >= 0x90 && b <= 0x9f) return readArray(b & 0x0f); // fixarray
    if (b >= 0xa0 && b <= 0xbf) return readStr(b & 0x1f); // fixstr
    switch (b) {
      case 0xc0: return null;
      case 0xc2: return false;
      case 0xc3: return true;
      case 0xc4: { const n = view.getUint8(pos); pos += 1; return readBin(n); }
      case 0xc5: { const n = view.getUint16(pos); pos += 2; return readBin(n); }
      case 0xc6: { const n = view.getUint32(pos); pos += 4; return readBin(n); }
      case 0xca: { const v = view.getFloat32(pos); pos += 4; return v; }
      case 0xcb: { const v = view.getFloat64(pos); pos += 8; return v; }
      case 0xcc: { const v = view.getUint8(pos); pos += 1; return v; }
      case 0xcd: { const v = view.getUint16(pos); pos += 2; return v; }
      case 0xce: { const v = view.getUint32(pos); pos += 4; return v; }
      case 0xcf: { const hi = view.getUint32(pos); const lo = view.getUint32(pos + 4); pos += 8; return hi * 4294967296 + lo; }
      case 0xd0: { const v = view.getInt8(pos); pos += 1; return v; }
      case 0xd1: { const v = view.getInt16(pos); pos += 2; return v; }
      case 0xd2: { const v = view.getInt32(pos); pos += 4; return v; }
      case 0xd3: { const hi = view.getInt32(pos); const lo = view.getUint32(pos + 4); pos += 8; return hi * 4294967296 + lo; }
      case 0xd9: { const n = view.getUint8(pos); pos += 1; return readStr(n); }
      case 0xda: { const n = view.getUint16(pos); pos += 2; return readStr(n); }
      case 0xdb: { const n = view.getUint32(pos); pos += 4; return readStr(n); }
      case 0xdc: { const n = view.getUint16(pos); pos += 2; return readArray(n); }
      case 0xdd: { const n = view.getUint32(pos); pos += 4; return readArray(n); }
      case 0xde: { const n = view.getUint16(pos); pos += 2; return readMap(n); }
      case 0xdf: { const n = view.getUint32(pos); pos += 4; return readMap(n); }
      default:
        throw new Error("msgpack: unsupported byte 0x" + b.toString(16));
    }
  }
  return read();
}

// Wrap a copied Uint8Array of little-endian float32 into a Float32Array.
export function asF32(bin) {
  if (!bin || bin.length === 0) return new Float32Array(0);
  return new Float32Array(bin.buffer, bin.byteOffset, bin.length >> 2);
}
