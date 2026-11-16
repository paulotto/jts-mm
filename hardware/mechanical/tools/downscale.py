# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shrink the PNGs of a folder in place (common.shrink_png): at most MAX px on the long side,
and the files matching ``--quantize GLOB`` (3D renders) to a 256-colour palette.

    ../.venv/bin/python tools/downscale.py <dir> [MAX] [--quantize 'render_*']
"""

import sys
from fnmatch import fnmatch
from pathlib import Path

from common import shrink_png


def main(argv) -> int:
    pattern = None
    if "--quantize" in argv:
        i = argv.index("--quantize")
        pattern = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    folder, mx = Path(argv[0]), int(argv[1]) if len(argv) > 1 else 2000
    n = sum(shrink_png(p, mx, bool(pattern) and fnmatch(p.name, pattern))
            for p in sorted(folder.glob("*.png")))
    print(f"{folder}: {n} PNGs shrunk (<= {mx} px{', ' + pattern + ' quantized' if pattern else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
