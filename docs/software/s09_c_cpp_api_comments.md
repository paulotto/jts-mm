<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# s09 — C/C++ API comments and layout

Conventions for first-party C and C++ in this repository: where headers live, and how APIs are documented so IDEs and Doxygen render them consistently.

**Trees covered**

| Tree | Headers | Sources |
|------|---------|---------|
| Production firmware | [`firmware/jtsmm/include/`](../../firmware/jtsmm/include/) | [`firmware/jtsmm/src/`](../../firmware/jtsmm/src/) |
| Host native (C++) | [`software/jtsmm/native/include/`](../../software/jtsmm/native/include/) | [`software/jtsmm/native/src/`](../../software/jtsmm/native/src/) |

**Out of scope:** `firmware/jtsmm/third_party/`, `firmware/jtsmm/tests/stubs/`.

---

## 1. Layout rule (`include/` vs `src/`)

| Kind | Location |
|------|----------|
| Module / public API headers | `include/jtsmm/<layer>/…` (firmware) or `include/jtsmm_native/…` (native) |
| Implementations (`.c` / `.cpp`) | `src/<layer>/…` |
| TinyUSB board config | `firmware/jtsmm/include/tusb_config.h` (bare name required by TinyUSB) |
| Generated PIO headers | build tree via `pico_generate_pio_header` — not hand-edited |

**Do not** place module `.h` files next to `.c` under `src/`. Include them as `"jtsmm/<layer>/<name>.h"` (firmware) or `"jtsmm_native/<name>.hpp"` (native).

Firmware layers: `app`, `runtime`, `platform`, `domain`, `drivers`.

---

## 2. Comment format: Doxygen with `@` commands

Use Javadoc-style blocks that Doxygen and clangd understand:

```c
/**
 * @brief One-sentence summary.
 *
 * @details Longer context, invariants, ordering, cross-refs to docs/.
 *
 * @param name Description with units / NULL rules / ranges.
 * @return What success and failure mean.
 *
 * @note Threading, ISR, or side-effect constraints.
 * @see related_symbol, docs/hardware/h05_….md
 */
```

Prefer `@` tags over Qt `\` commands. Convert legacy `//` / `/* */` module notes into `@file` / `@brief` / `@details` when touching a file.

### Placement

1. **Full docs on the declaration** in the header (or on `static` helpers at their definition in `.c`/`.cpp`).
2. Definitions in `.c`/`.cpp` that merely implement a documented header API need no duplicate block.
3. Every translation unit gets a `@file` brief at the top.
4. Enums: document the type; document non-obvious enumerators with `/**< … */`.
5. Struct/class fields that are part of the API get a trailing or preceding Doxygen comment when the name is not self-explanatory.

### Content bar

Each documented symbol should answer, without opening other files when possible:

- What it does (`@brief`)
- Why / when to call it (`@details` if non-obvious)
- Parameters: units, ownership, NULL-ness, valid ranges
- Return / out-params: success vs error meaning
- Side effects (rings, USB, GPIO, flash, globals)
- Threading / core / ISR constraints
- Ordering / preconditions
- `@see` to related APIs and Markdown specs

Use project vocabulary (`t_master_us`, XTRIG, DRDY, Track B) rather than vague “handles data”.

---

## 3. Generating HTML (optional)

From the tree root of interest:

```bash
# Firmware
cd firmware/jtsmm && doxygen Doxyfile
# HTML under firmware/jtsmm/doxygen/html/

# Native
cd software/jtsmm/native && doxygen Doxyfile
# HTML under software/jtsmm/native/doxygen/html/
```

CI does **not** fail on undocumented symbols yet; treat Doxygen warnings as a local checklist when adding APIs.

<!-- docs-reviewed: 2026-11-06 -->
