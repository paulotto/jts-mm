# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Participant report (W6 §6): every data view of one participant plus its key tables.

``build_report(cfg, "s03")`` renders all registered ``data`` views (TikZ/pgfplots,
the same specs as the viewer) and the participant's key result tables
(booktabs) into ``<out_dir>/reports/<pid>/`` and writes ``report_<pid>.tex``,
which compiles with LuaLaTeX. The output is subject data: it stays under the
study's (git-ignored) ``data/analysis/...`` tree. Figures name the participant
only by pseudonym; the landmark view includes the camera frames of the holds
(identifiable), so the report must not be shared outside the study team.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from biomech.viz import render_pgf
from biomech.viz.spec import Column, FigureSpec, TableSpec

SECTIONS = (  # (heading, view ids in order); views not listed go to "Other views"
    ("Kinematics in space", ("data_orthographic", "data_jaw3d", "data_speed_path",
                             "data_condylar_paths", "data_activation_path",
                             "data_cohort_atlas")),
    ("Kinematics in time", ("data_kinematic_stack", "data_cycle_profiles",
                            "data_phase_portrait", "data_demand_map")),
    ("EMG", ("data_emg_chain", "data_emg_spectra", "data_activation_raster", "data_emg_cycle")),
    ("IMU", ("data_imu_estimators", "data_imu_accel", "data_imu_quality")),
    ("Tracking and synchronization", ("data_timeline", "data_tracking_scene", "data_landmarks")),
)
VIEW_OPTIONS: dict[str, dict[str, Any]] = {"data_landmarks": {"filmstrip": True}}
TASK_LABEL = {"open_close": "open-close", "protrusion_retrusion": "protrusion",
              "left_right": "laterotrusion", "chewing": "chewing", "free_motion": "free motion",
              "rest": "rest"}


# -- tables ------------------------------------------------------------------------------------
def _kinematics_table(ctx, pid: str, who: str) -> TableSpec | None:
    s = ctx.table("kinematics", "summary", [pid])
    if s.empty:
        return None
    metrics = (("opening_mm", "Vertical opening", "\\milli\\metre", 1),
               ("protrusion_mm", "Protrusion", "\\milli\\metre", 1),
               ("lateral_mm", "Lateral excursion", "\\milli\\metre", 1),
               ("deviation_mm", "Frontal deviation", "\\milli\\metre", 1),
               ("theta_y_deg", "Opening angle", "\\degree", 1),
               ("peak_open_vel_dps", "Peak opening velocity", "\\degree\\per\\second", 0),
               ("peak_close_vel_dps", "Peak closing velocity", "\\degree\\per\\second", 0),
               ("peak_incisal_speed_mm_s", "Peak incisal speed", "\\milli\\metre\\per\\second",
                0),
               ("duration_s", "Cycle duration", "\\second", 2))
    tasks = [t for t in ("open_close", "protrusion_retrusion", "left_right", "chewing")
             if t in set(s.task)]
    rows = []
    for key, lab, unit, dec in metrics:
        r: dict[str, Any] = {"metric": f"{lab} ({unit_plain(unit)})"}
        for t in tasks:
            v = s[(s.task == t) & (s.metric == key)]
            r[t] = float(v["median"].median()) if not v.empty else None
        rows.append(r)
    n = {t: int(s[(s.task == t)].n_cycles.max()) for t in tasks}
    cols = [Column("metric", "Metric", align="l")] + [
        Column(t, f"{TASK_LABEL[t]} (n={n[t]})", decimals=1) for t in tasks]
    return TableSpec("tab_kinematics", f"Clinical kinematics of {who}: median over ok cycles",
                     cols, rows, notes="Lateral tasks pool both sides (median of the side "
                                       "medians). Maxillary frame.",
                     sources=["kinematics:summary"], width="text")


def unit_plain(u: str) -> str:
    from biomech.viz.style import plain_unit

    return plain_unit(u)


def _landmark_table(ctx, pid: str, who: str) -> TableSpec | None:
    lm = ctx.table("kinematics", "landmarks", [pid])
    if lm.empty:
        return None
    lm = lm[~lm.name.str.endswith("_hp")]
    from biomech.viz.views.data import LM_SHORT

    rows = [{"name": LM_SHORT.get(r.landmark, r.landmark), "what": r.description,
             "mad": float(r.mad_mm), "n": int(r.n)} for r in lm.itertuples()]
    return TableSpec("tab_landmarks", f"Landmark digitization of {who}",
                     [Column("name", "Landmark", align="l"),
                      Column("what", "Description", align="X", decimals=6),
                      Column("mad", "MAD", "\\milli\\metre", 2), Column("n", "n", decimals=0)],
                     rows, notes="MAD: 3-D median absolute deviation of the CT-pointer tip "
                                 "during the central part of the hold.",
                     sources=["kinematics:landmarks"], width="text")


def _emg_table(ctx, pid: str, who: str) -> TableSpec | None:
    q = ctx.table("qc", "emg", [pid])
    if q.empty:
        return None
    rows = []
    for r in q.itertuples():
        rows.append({"muscle": str(r.muscle).replace("_", " "), "ref": str(r.ref_phase).replace(
            "mvc_", ""), "p95": float(r.ref_p95_uv), "snr": float(r.snr_db),
            "rest": float(r.rest_rms_uv), "mask": 100 * float(r.mask_frac_task),
            "use": ("shape+ampl." if r.usable_amplitude else "shape") if r.usable_shape
            else "no"})
    return TableSpec("tab_emg", f"EMG references and quality of {who}",
                     [Column("muscle", "Muscle", align="l"), Column("ref", "Ref.", align="l"),
                      Column("p95", "MVC/RVC p95", "\\micro\\volt", 0),
                      Column("rest", "Rest RMS", "\\micro\\volt", 1),
                      Column("snr", "SNR", "\\decibel", 1),
                      Column("mask", "Masked task", "\\percent", 1),
                      Column("use", "Usable", align="l")],
                     rows, sources=["qc:emg"], width="text")


def _imu_table(ctx, pid: str, who: str) -> TableSpec | None:
    a = ctx.table("imu", "agreement", [pid])
    if a.empty:
        return None
    rows = [{"task": TASK_LABEL.get(r.task, r.task) + (" (cal.)" if r.is_calibration else ""),
             "q": float(r.rmse_quat_deg), "t": float(r.rmse_tilt_deg),
             "c": float(r.rmse_comp_deg), "lag": float(r.lag_ms)} for r in a.itertuples()]
    return TableSpec("tab_imu", f"IMU opening angle vs optical, {who}",
                     [Column("task", "Task", align="l"),
                      Column("q", "RMSE quat.", "\\degree", 2),
                      Column("t", "RMSE tilt", "\\degree", 2),
                      Column("c", "RMSE compl.", "\\degree", 2),
                      Column("lag", "Lag", "\\milli\\second", 0)],
                     rows, sources=["imu:agreement"])


def _timing_table(ctx, pid: str, who: str) -> TableSpec | None:
    t = ctx.table("timing", "summary", [pid])
    if t.empty:
        return None
    if "rule" in t:
        t = t[t.rule == ctx.onset_rule()]
    group = {"opening": ("digastric_l", "digastric_r"),
             "closing": ("masseter_l", "masseter_r", "temporalis_l", "temporalis_r")}
    t = t[t.apply(lambda r: r.muscle in group.get(r.direction, ()), axis=1)]
    if t.empty:
        return None
    pv = t.pivot_table(index=["task", "direction", "muscle"], columns="detector",
                       values="lead_ms", aggfunc="median")
    dets = list(pv.columns)
    rows = []
    for (task, direction, muscle), r in pv.iterrows():
        rows.append({"what": f"{TASK_LABEL.get(task, task)}, {direction}",
                     "muscle": muscle.replace("_", " "),
                     **{d: (float(r[d]) if np.isfinite(r[d]) else None) for d in dets}})
    return TableSpec("tab_timing", f"EMG onset lead before motion onset, {who} (median, ms)",
                     [Column("what", "Task, direction", align="l"),
                      Column("muscle", "Muscle", align="l")] +
                     [Column(d, d.replace("_", "-"), "\\milli\\second", 0) for d in dets],
                     rows, notes="Positive: EMG onset precedes the mechanical onset.",
                     sources=["timing:summary"], width="text")


def report_tables(ctx, pid: str) -> list[TableSpec]:
    who = f"P{ctx.pseudonym(pid)}"
    out = []
    for fn in (_kinematics_table, _landmark_table, _emg_table, _imu_table, _timing_table):
        try:
            t = fn(ctx, pid, who)
        except Exception:                   # a table from an older store must not stop it
            t = None
        if t is not None:
            out.append(t)
    return out


# -- document ----------------------------------------------------------------------------------
def _section_of(view_id: str) -> str:
    for head, ids in SECTIONS:
        if view_id in ids:
            return head
    return "Other views"


def _tex(who: str, figures: list[FigureSpec], tables: list[TableSpec], errors: list[str],
         study: str) -> str:
    esc = render_pgf._esc
    body = [f"\\section*{{Participant report: {esc(who)}}}",
            "{\\small Study " + esc(study) + ". Generated by \\texttt{biomech report} from the "
            "result store; every figure is a registered data view (W6) rendered through the "
            "same spec as the viewer. Subject data: do not share outside the study team "
            "(the landmark view shows camera frames). " + esc(
                "Coordinates in the maxillary frame (x anterior, y left, z up; "
                "origin upper incisor).") + "}"]
    if tables:
        body.append("\\subsection*{Key tables}")
        for t in tables:
            body.append(f"\\begin{{table}}[!htbp]\\centering\\footnotesize\\caption{{{esc(t.caption)}}}"
                        f"\\input{{tables/{t.id}.tex}}\\end{{table}}")
        body.append("\\clearpage")
    current = None
    for f in figures:
        sec = _section_of(f.id.split("--")[0])
        if sec != current:
            if current is not None:
                body.append("\\clearpage")          # floats of a section stay in it
            body.append(f"\\subsection*{{{esc(sec)}}}")
            current = sec
        cap = f"\\textbf{{{esc(f.title)}.}} {esc(f.caption)}"
        body.append(f"\\begin{{figure}}[!htb]\\centering\\input{{figures/{f.id}/fig.tex}}"
                    f"\\caption{{{cap}}}\\end{{figure}}")
    if errors:
        body.append("\\subsection*{Views that could not be built}\\begin{itemize}")
        body += [f"\\item {esc(e)}" for e in errors]
        body.append("\\end{itemize}")
    return ("\\documentclass[10pt]{article}\n"
            "\\usepackage[letterpaper,top=0.7in,bottom=0.8in,left=0.67in,right=0.67in]"
            "{geometry}\n" + render_pgf.PREAMBLE +
            "\\usepackage{caption}\\captionsetup{font=footnotesize,labelfont=bf}\n"
            "\\setlength{\\parindent}{0pt}\n"
            "\\renewcommand{\\topfraction}{0.95}\\renewcommand{\\bottomfraction}{0.9}"
            "\\renewcommand{\\textfraction}{0.02}\\renewcommand{\\floatpagefraction}{0.8}"
            "\\setcounter{topnumber}{4}\\setcounter{totalnumber}{6}\n\\begin{document}\n" + "\n".join(body) +
            "\n\\end{document}\n")


def build_report(cfg, pid: str, out_dir: str | Path | None = None, *,
                 compile_pdf: bool = False, views: list[str] | None = None,
                 log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Render the participant report; returns {tex, pdf, figures, tables, errors}."""
    from biomech.io.context import DataContext
    from biomech.viz.views import registry

    log = log or (lambda m: None)
    ctx = DataContext(cfg)
    try:
        if pid not in ctx.pids():
            raise ValueError(f"{pid!r} is not an included participant ({', '.join(ctx.pids())})")
        out = Path(out_dir) if out_dir else Path(cfg.out_dir) / "reports" / pid
        out.mkdir(parents=True, exist_ok=True)
        render_pgf.write_style(out)
        who = f"P{ctx.pseudonym(pid)}"
        order = [v for _, ids in SECTIONS for v in ids]
        ids = views or sorted((v.id for v in registry.views("data", cfg.groups)),
                              key=lambda i: (order.index(i) if i in order else len(order), i))
        figures, errors = [], []
        for vid in ids:
            try:
                req = registry.request(ctx, vid, pid=pid, **VIEW_OPTIONS.get(vid, {}))
                spec = registry.build(vid, req)
                render_pgf.render_figure(spec, out)
                figures.append(spec)
                log(f"  {vid:<24} -> figures/{spec.id}")
            except Exception as exc:          # one broken view must not stop the report
                errors.append(f"{vid}: {type(exc).__name__}: {exc}")
                log(f"  {vid:<24} ERROR {type(exc).__name__}: {exc}")
        tables = report_tables(ctx, pid)
        for t in tables:
            render_pgf.render_table(t, out)
        tex = out / f"report_{pid}.tex"
        tex.write_text(_tex(who, figures, tables, errors, cfg.study), encoding="utf-8")
        res: dict[str, Any] = {"tex": tex, "pdf": None, "figures": [f.id for f in figures],
                               "tables": [t.id for t in tables], "errors": errors}
        if compile_pdf:
            res["pdf"] = compile_tex(tex)
            if res["pdf"] is None:
                log_path = tex.with_suffix(".log")
                tail = log_path.read_text(errors="replace")[-2500:] if log_path.exists() else ""
                res["errors"].append(f"LuaLaTeX failed:\n{tail}")
        return res
    finally:
        ctx.close()


def compile_tex(tex: Path, timeout_s: int = 900) -> Path | None:
    """Compile with LuaLaTeX (dynamic memory for the 3-D and colour-line figures)."""
    exe = shutil.which("lualatex")
    if exe is None:
        raise RuntimeError("lualatex not found")
    r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                       cwd=tex.parent, capture_output=True, text=True, timeout=timeout_s)
    pdf = tex.with_suffix(".pdf")
    return pdf if r.returncode == 0 and pdf.is_file() else None

