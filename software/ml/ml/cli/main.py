# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""CLI entry point: label, labels, dataset, train, eval."""

from __future__ import annotations

import click

from .dataset import dataset_cmd
from .eval import eval_cmd
from .label import label_cmd
from .labels import labels_cmd
from .train import train_cmd


@click.group()
@click.version_option(package_name="jtsmm-ml")
def main() -> None:
    """Jaw-motion ML: labeling, datasets, training and evaluation."""


main.add_command(label_cmd)
main.add_command(labels_cmd)
main.add_command(dataset_cmd)
main.add_command(train_cmd)
main.add_command(eval_cmd)


if __name__ == "__main__":
    main()
