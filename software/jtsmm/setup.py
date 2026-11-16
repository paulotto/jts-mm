# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Build hook: bundle schema/v1 and etc/jtsmm.yaml into jtsmm/_data.

All metadata lives in pyproject.toml; this file only extends build_py so a
regular (non-editable) install can load and validate its config without the
repository checkout. Editable installs read the repo files directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _bundle_data import copy_config_assets  # noqa: E402


class BuildPyWithConfigAssets(build_py):
    def run(self) -> None:
        super().run()
        if getattr(self, "editable_mode", False):
            return
        copy_config_assets(HERE.parents[1], Path(self.build_lib) / "jtsmm" / "_data")


setup(cmdclass={"build_py": BuildPyWithConfigAssets})
