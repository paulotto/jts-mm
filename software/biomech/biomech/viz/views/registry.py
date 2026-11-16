# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The one registry of everything that can be shown or exported.

Each view is a builder ``(ViewRequest) -> FigureSpec | TableSpec`` registered
with a *kind* (``paper``, ``dashboard``, ``data``, ``animation``) and the
parameters it understands (participant, phase/task, perspective, alignment,
statistic, time window). The viewer pages, ``biomech export``, ``biomech report``
and the output check (``biomech check-outputs``) all enumerate this registry;
nothing keeps its own list.

Perspectives (W6a): ``trials`` · ``trials+summary`` · ``participant`` ·
``cohort`` · ``compare``; ``align``: ``cycle`` · ``angle`` · ``time`` ·
``event`` · ``path``; ``stat``: ``mean_sd`` · ``median_iqr`` · ``mean_ci``.
Exported ids carry non-default parameters (``fig_x--participant-s03``), so a
perspective explored in the viewer never overwrites the paper item.

Views registered with ``multi_phase=True`` take a list of phases (or ``"all"``) and
overlay them; the id then carries one stable token: ``phase-all`` or the phases in task
order joined by ``+`` (``--phase-open_close+chewing``). Other views get the first phase.

Each view belongs to one study *group* (``dataset`` | ``validation``, as the pipeline stages,
``StudyConfig.groups``): export, check-outputs, the coverage check and the viewer list only
the views of the active study's groups, so the dataset analysis and the system validation
never export or check each other's views.

Adding an analysis mode (an08 §8): a stage stores its tables/arrays; one builder function
in any module of this package registers the view with ``domain`` (where the Results page
lists it) and, if it takes a phase, ``phases`` (the tasks export and checks expand). Modules
are discovered automatically; the shared building blocks are in ``views.blocks``, labels
and units in ``views.vocab``. The coverage guard (tests/test_view_coverage.py) fails when a
stored output family is read by no view.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

from biomech.viz.spec import FigureSpec, TableSpec

Kind = Literal["paper", "dashboard", "data", "animation"]
# Results-page sections, in display order.
DOMAINS = {"paper": "Paper", "kinematics": "Kinematics", "emg_time": "EMG — time domain",
           "emg_freq": "EMG — frequency domain", "imu": "IMU", "timing": "Timing and intent cues",
           "uncertainty": "Uncertainty", "variability": "Variability and sensitivity",
           "qc": "Quality control", "ml": "Intent decoding", "data": "Recorded data",
           "validation": "System validation", "tables": "Result tables"}
# Study groups a view can belong to (the pipeline's stage groups, StudyConfig.groups).
GROUPS = ("dataset", "validation")
NON_VIEW_MODULES = {"registry", "blocks", "vocab", "forms"}
PERSPECTIVES = ("trials", "trials+summary", "participant", "cohort", "compare")
ALIGNS = ("cycle", "angle", "time", "event", "path")
STATS = ("mean_sd", "median_iqr", "mean_ci")


@dataclass
class ViewRequest:
    ctx: Any                                   # biomech.io.context.DataContext
    pid: str | None = None
    phase: str | list[str] | None = None       # a phase/task name; a list or "all" (multi_phase)
    perspective: str = "cohort"
    align: str = "cycle"
    stat: str = "mean_sd"
    window_s: tuple[float, float] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def store(self):
        return self.ctx.store


@dataclass(frozen=True)
class ViewDef:
    id: str
    kind: Kind
    title: str
    builder: Callable[[ViewRequest], FigureSpec | TableSpec]
    params: tuple[str, ...] = ()               # subset of pid, phase, perspective, align, stat, window
    perspectives: tuple[str, ...] = ()
    aligns: tuple[str, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    multi_phase: bool = False                  # phase may be a list / "all" (overlaid)
    domain: str = ""                           # DOMAINS key (Results page section)
    phases: tuple[str, ...] = ()               # phases export/check expand ("phase" views)
    # Pickers: option name -> allowed values, or a function ctx -> values (from the data).
    # The viewer shows a dropdown per picker; the value reaches the builder in
    # ``req.options[name]`` (default: ``defaults[name]`` or the first value).
    choices: dict[str, Any] = field(default_factory=dict)
    # A raw-data browser that reads the recorded session (not reproducible from the result
    # store or a snapshot alone, G2); without the session it shows a card saying so.
    needs_session: bool = False
    group: str = "dataset"                     # study group (GROUPS) whose outputs it shows

    def choice_values(self, name: str, ctx: Any = None) -> list[Any]:
        c = self.choices.get(name)
        if c is None:
            return []
        if callable(c):
            try:
                return list(c(ctx)) if ctx is not None else []
            except Exception:                  # a picker must never break the page
                return []
        return list(c)

    @property
    def scope(self) -> str:
        """"participant" when the view shows one participant, else "cohort"."""
        return "participant" if "pid" in self.params else "cohort"


REGISTRY: dict[str, ViewDef] = {}
ALL_PHASES = "all"


def register(view_id: str, kind: Kind, title: str, *, params: tuple[str, ...] = (),
             perspectives: tuple[str, ...] = (), aligns: tuple[str, ...] = (),
             description: str = "", multi_phase: bool = False, domain: str = "",
             phases: tuple[str, ...] = (), choices: dict[str, Any] | None = None,
             needs_session: bool = False, group: str = "dataset", **defaults: Any):
    """Register a view (decorator); ``group``: the study group whose results it shows."""
    if domain and domain not in DOMAINS:
        raise ValueError(f"view {view_id}: unknown domain {domain!r}; known: {list(DOMAINS)}")
    if group not in GROUPS:
        raise ValueError(f"view {view_id}: unknown group {group!r}; known: {list(GROUPS)}")

    def deco(fn: Callable[[ViewRequest], FigureSpec | TableSpec]):
        dom = domain or ("paper" if kind == "paper" else "data" if kind == "data" else "")
        REGISTRY[view_id] = ViewDef(view_id, kind, title, fn, params, perspectives, aligns,
                                    defaults, description or (fn.__doc__ or "").strip()
                                    .split("\n")[0], multi_phase, dom, tuple(phases),
                                    dict(choices or {}), needs_session, group)
        return fn

    return deco


def _task_order() -> tuple[str, ...]:
    try:
        from biomech.viz.views.data import TASKS
    except ImportError:                          # pragma: no cover - data views missing
        return ()
    return TASKS


def normalize_phase(phase: Any, multi: bool = True) -> str | list[str] | None:
    """Canonical phase argument: a string stays; a list is de-duplicated and put in task
    order, collapses to ``"all"`` when it holds ``"all"`` or every task, to a string when it
    holds one phase; a single-phase view (``multi=False``) gets the first phase."""
    if phase is None or isinstance(phase, str):
        return phase
    vals = [str(v) for v in dict.fromkeys(phase) if v]
    if not vals:
        return None
    if not multi:
        return next((v for v in vals if v != ALL_PHASES), vals[0])
    order = _task_order()
    if ALL_PHASES in vals or (order and set(order) <= set(vals)):
        return ALL_PHASES
    pos = {v: i for i, v in enumerate(vals)}
    vals.sort(key=lambda v: (order.index(v) if v in order else len(order), pos[v]))
    return vals[0] if len(vals) == 1 else vals


def phase_token(phase: Any) -> str | None:
    """Id token of a phase argument (``all``, ``chewing``, ``open_close+chewing``)."""
    p = normalize_phase(phase)
    if p is None or isinstance(p, str):
        return p
    return "+".join(p)


_LOADED = False


def view_modules() -> list[str]:
    """Every module of this package that defines views (discovered, no hand list)."""
    import pkgutil
    from pathlib import Path

    here = Path(__file__).parent
    return sorted(m.name for m in pkgutil.iter_modules([str(here)])
                  if not m.name.startswith("_") and m.name not in NON_VIEW_MODULES)


def _load() -> None:
    # Importing the view modules registers their builders.
    global _LOADED
    if _LOADED:
        return
    import importlib

    for mod in view_modules():
        importlib.import_module(f"biomech.viz.views.{mod}")
    _LOADED = True


def by_domain(kinds: tuple[str, ...] = ("paper", "dashboard"),
              groups: Any = None) -> dict[str, list[ViewDef]]:
    """Views of ``kinds`` (and ``groups``, see :func:`views`) grouped by domain, in DOMAINS
    order (unassigned last)."""
    out: dict[str, list[ViewDef]] = {d: [] for d in DOMAINS}
    for v in views(groups=groups):
        if v.kind in kinds:
            out.setdefault(v.domain or "other", []).append(v)
    return {d: sorted(vs, key=lambda v: v.title) for d, vs in out.items() if vs}


def expand_phases(v: ViewDef, ctx: Any = None) -> list[str | None]:
    """The phases export and check-outputs build a view for: its declared ``phases``, or
    its default phase; [None] for views without a phase parameter."""
    if "phase" not in v.params:
        return [None]
    if v.phases:
        return list(v.phases)
    return [v.defaults.get("phase")]


def study_groups(ctx: Any) -> tuple[str, ...] | None:
    """The view groups of the study behind ``ctx`` (a DataContext, a StudyConfig or a group
    tuple); None (every group) when it names none."""
    if ctx is None:
        return None
    if isinstance(ctx, str):
        return (ctx,)
    if isinstance(ctx, (tuple, list, set, frozenset)):
        return tuple(ctx)
    cfg = getattr(ctx, "config", ctx)
    g = getattr(cfg, "groups", None)
    return tuple(g) if g else None


def views(kind: Kind | None = None, groups: Any = None) -> list[ViewDef]:
    """Registered views of ``kind`` (None: every kind) whose group is in ``groups`` (group
    names, or a context/config whose study groups count, :func:`study_groups`; None: every
    group)."""
    _load()
    g = study_groups(groups)
    return [v for v in REGISTRY.values()
            if (kind is None or v.kind == kind) and (g is None or v.group in g)]


def get(view_id: str) -> ViewDef:
    _load()
    if view_id not in REGISTRY:
        raise KeyError(f"unknown view {view_id!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[view_id]


def request(ctx: Any, view_id: str, **kw: Any) -> ViewRequest:
    """A request with the view's defaults filled in (explicit arguments win)."""
    v = get(view_id)
    base = {k: v.defaults[k] for k in ("perspective", "align", "stat", "phase", "pid")
            if k in v.defaults}
    base.update({k: val for k, val in kw.items() if val is not None})
    if "phase" in base:
        base["phase"] = normalize_phase(base["phase"], v.multi_phase)
        if base["phase"] is None:
            base.pop("phase")
    opts = {k: val for k, val in base.items()
            if k not in ("perspective", "align", "stat", "phase", "pid", "window_s")}
    for name in v.choices:                      # pickers: default = declared or first value
        if name not in opts:
            vals = v.choice_values(name, ctx)
            if name in v.defaults or vals:
                opts[name] = default_choice(v, name, ctx, vals)
    core = {k: val for k, val in base.items() if k not in opts}
    return ViewRequest(ctx, options=opts, **core)


def session_available(ctx: Any, pid: str | None) -> bool:
    """True when the recorded session of ``pid`` (or of the first participant) can be read."""
    try:
        pids = [pid] if pid else list(ctx.pids())[:1]
        paths = [ctx.session_path(x) for x in pids]
        return bool(paths) and all(p is not None and p.is_dir() for p in paths)
    except Exception:
        return False


def _session_card(v: ViewDef, req: ViewRequest) -> TableSpec:
    from biomech.viz.spec import Column

    return TableSpec(v.id, f"{v.title}: needs the recorded session.",
                     [Column("note", "Note", align="l")],
                     [{"note": "This view reads the recorded session files, which are not "
                               "part of the result store or snapshot; point --data-dir at the "
                               "session root to show it."}], width="text")


def build(view_id: str, req: ViewRequest) -> FigureSpec | TableSpec:
    """Build a view; the spec id carries every non-default parameter.

    A participant view asked for someone who is not an included participant shows the
    default participant: that is said explicitly (caption, first panel's note, ``meta``),
    never silently, and the id carries no participant (it is the default view)."""
    v = get(view_id)
    if v.needs_session and not session_available(req.ctx, req.pid):
        return _session_card(v, req)
    spec = v.builder(req)
    substituted = bool(getattr(spec, "meta", None) and spec.meta.get("substituted"))
    if "pid" in v.params and req.pid is not None and not substituted:
        included = _included(req.ctx)
        if included is not None and req.pid not in included:
            spec = _note_substitute(spec, req.ctx, included)
            substituted = True
    suffix = []
    for k in ("pid", "phase", "perspective", "align", "stat"):
        val = getattr(req, k)
        if k == "pid" and substituted:
            continue
        if k in v.params and val is not None and val != v.defaults.get(k):
            if k == "phase" and not isinstance(val, str):
                suffix.append(f"phase-{phase_token(val)}")    # '+' joins the phases
            elif k == "pid":
                suffix.append(f"pid-{_pseudo(req.ctx, val)}")  # never the study id
            else:
                suffix.append(f"{k[:5]}-{val}".replace("+", "plus"))
    for name in v.choices:                      # non-default picker values name the export
        val = req.options.get(name)
        if val is not None and val != default_choice(v, name, req.ctx):
            suffix.append(f"{name}-{_token(val)}")
    if suffix:
        spec = replace(spec, id=spec.id + "--" + "--".join(suffix))
    return spec


def _included(ctx: Any) -> list[str] | None:
    try:
        return list(ctx.pids())
    except AttributeError:
        return None


def _note_substitute(spec: FigureSpec | TableSpec, ctx: Any, included: list[str]
                     ) -> FigureSpec | TableSpec:
    meta = getattr(spec, "meta", None) or {}
    shown_pid = meta.get("pid") if meta.get("pid") in included else (
        included[0] if included else None)
    shown = f"P{ctx.pseudonym(shown_pid)}" if shown_pid is not None else "nobody"
    msg = f"The requested participant is not included (no data); showing {shown}"
    if isinstance(spec, FigureSpec):
        panels = list(spec.panels)
        if panels and not panels[0].note:
            panels[0] = replace(panels[0], note=msg)
        return replace(spec, panels=panels, caption=f"{msg}. {spec.caption}".strip(),
                       meta={**spec.meta, "substituted": True})
    return replace(spec, caption=f"{msg}. {spec.caption}".strip())


def _pseudo(ctx: Any, pid: str) -> str:
    """The pseudonym of ``pid`` for ids; never the study id (raises when there is none)."""
    try:
        return f"P{ctx.pseudonym(pid)}"
    except (AttributeError, ValueError) as exc:
        raise ValueError("a participant without a pseudonym (not included) cannot name an "
                         "exported view") from exc


def default_choice(v: ViewDef, name: str, ctx: Any, vals: list[Any] | None = None) -> Any:
    """The picker's default: the declared one when the data offer it, else the first value
    (a declared default the data do not offer would build an invalid view)."""
    vals = v.choice_values(name, ctx) if vals is None else vals
    d = v.defaults.get(name)
    if vals:
        return d if name in v.defaults and d in vals else vals[0]
    return d


def _token(val: Any) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(val)).strip("-")[:48]
