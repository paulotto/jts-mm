# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""State shared by all viewer pages (one study, one local user)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biomech.cohort import Cohort, load_cohort
from biomech.config import REPO_ROOT, StudyConfig, load_config
from biomech.io.context import DataContext
from biomech.results import ResultStore
from biomech.viewer.jobs import Job
from biomech.viewer.runs import RunManager
from biomech.viewer.videos import VideoJob


@dataclass
class ViewerContext:
    config: StudyConfig
    cohort: Cohort
    store: ResultStore
    export_dir: Path
    data: DataContext
    paper_items: list[Any] = field(default_factory=list)
    paper_sets: list[Any] = field(default_factory=list)   # biomech.papers.PaperSet
    venue: str | None = None                          # the paper yaml's venue profile
    runs: RunManager = field(default_factory=RunManager)
    videos: VideoJob = field(default_factory=VideoJob)
    exports: Job = field(default_factory=Job)        # long exports (paper board)

    @classmethod
    def from_study(cls, study_yaml: str | Path, data_dir: str | None = None,
                   snapshot: str | None = None) -> "ViewerContext":
        cfg = load_config(study_yaml, data_dir=data_dir).with_snapshot(snapshot)
        cohort = load_cohort(cfg.cohort_path, cfg.data_dir)
        # The paper board shows the study's active paper set (paper_dataset.yaml
        # for the dataset study, paper_systems.yaml for the validation study), default target.
        from biomech import papers

        sets = papers.study_paper_sets(study_yaml)
        export = cfg.out_dir / "paper"
        items: list[Any] = []
        venue = None
        if sets:
            tgt = sets[0].target()
            items = [i.entry() for i in sets[0].items_for(tgt.name)]
            venue, export = tgt.venue, tgt.out_dir
        return cls(cfg, cohort, ResultStore(cfg.read_dir), export, DataContext(cfg, cohort),
                   items, venue=venue, paper_sets=sets)

    def reload(self) -> None:
        """Re-read the cohort and drop cached data (after a run or a manifest edit)."""
        self.cohort = load_cohort(self.config.cohort_path, self.config.data_dir)
        self.data.close()
        self.data = DataContext(self.config, self.cohort)

    def plan_tag(self) -> str:
        """The frozen analysis plan of *this* study (git tag ``…-plan-frozen`` + commits
        since): the validation study has its own plan and tag."""
        tag = ("validation-plan-frozen" if "validation" in tuple(self.config.groups)
               else "icorr-plan-frozen")
        try:
            r = subprocess.run(["git", "describe", "--tags", "--match", tag, "--always"],
                               cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
            return r.stdout.strip() or "untagged"
        except (OSError, subprocess.SubprocessError):
            return "unknown"
