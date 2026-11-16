# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Animation: preview a layout as Plotly frames, then render the full-quality video.

The same entry points as ``biomech animate`` (:func:`biomech.anim.make`,
``render_video``), so a video rendered here is identical to one from the CLI -- including
the clip files ``<stem>.clip.npz/.json`` (re-render with ``biomech animate --from-clip``)
and, for the D5 supplementary layouts (``v1_hinge``, ``v2_intent``, ``v3_release``), the PNG
filmstrip. Choosing a layout sets its default phases and speed (v2_intent: 1x, simulated
real time). Camera frames show the participant: they are off unless explicitly ticked
(an11 §privacy).
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from dash import Input, Output, State, dcc, html, no_update
from dash import ctx as dctx

from biomech.anim import LAYOUTS, SIZES, default_phases, default_speed, video_stem
from biomech.anim.layouts import COHORT_LAYOUTS
from biomech.viewer.components import env_id, env_of, env_store, graph_box
from biomech.viewer.context import ViewerContext
from biomech.viewer.data import pid_label
from biomech.viz.views import vocab as V

DEFAULT_PHASE = "chewing"


def _phase_names(ctx: ViewerContext, pid: str | None) -> list[str]:
    if not pid:
        return []
    try:
        return [getattr(p, "name", str(p)) for p in ctx.data.phases(pid)]
    except (OSError, KeyError, ValueError):          # unreadable session: no phases offered
        return []


def _phase_opts(names: Any) -> list[dict[str, str]]:
    return [{"label": V.phase_label(n), "value": n} for n in names]


WINDOW_S = (-3.0, 1.0)      # scrolling panels: seconds before / after the cursor (default)
TRAIL_S = 1.0


def _make_kw(layout, pid, phases, t0, t1, speed, fps, camera, theme, window=None,
             trail=None) -> dict:
    ph = list(phases or default_phases(layout))
    if layout in COHORT_LAYOUTS:
        return {"pid": None, "layout": layout, "speed": float(speed or default_speed(layout)),
                "fps": float(fps or 30), "theme": theme, "task": ph[0]}
    kw = {"pid": pid, "layout": layout, "phase": ph, "t0_s": t0, "t1_s": t1,
          "speed": float(speed or 0.5), "fps": float(fps or 30), "theme": theme,
          "with_camera": bool(camera)}
    if window is not None and len(window) == 2 and tuple(map(float, window)) != WINDOW_S:
        kw["window_s"] = (-abs(float(window[0])), abs(float(window[1])))
    if trail is not None and float(trail) != TRAIL_S:
        kw["trail_s"] = float(trail)
    return kw


def video_path(ctx: ViewerContext, kw: dict) -> Path:
    phases = [kw["task"]] if kw["layout"] in COHORT_LAYOUTS else kw["phase"]
    pid = kw.get("pid")
    stem = video_stem(kw["layout"], pid, phases, kw.get("with_camera", False),
                      pseudonym=ctx.data.pseudonym(pid) if pid else None,
                      t0_s=kw.get("t0_s"), t1_s=kw.get("t1_s"), speed=kw.get("speed"),
                      theme=kw.get("theme"), size=kw.get("size"),
                      window_s=kw.get("window_s"), trail_s=kw.get("trail_s"),
                      fps=kw.get("fps"))
    return ctx.config.out_dir / "videos" / f"{stem}.mp4"


def layout(ctx: ViewerContext) -> html.Div:
    pids = ctx.data.pids()
    pid0 = pids[0] if pids else None
    phases = _phase_names(ctx, pid0)
    return html.Div(className="page", children=[
        env_store("videos"),
        html.Div(className="card controls", children=[
            dcc.Dropdown(id="av-layout", options=[{"label": V.term("video_layout", k),
                                                   "value": k} for k in LAYOUTS],
                         value="multimodal", clearable=False, className="dd-wide"),
            dcc.Dropdown(id="av-pid", options=[{"label": pid_label(ctx.data, p), "value": p}
                                               for p in pids],
                         value=pid0, clearable=False, className="dd-mid"),
            dcc.Dropdown(id="av-phases", options=_phase_opts(phases), multi=True,
                         value=[DEFAULT_PHASE] if DEFAULT_PHASE in phases else phases[:1],
                         placeholder="phase(s), played in order", className="dd-wide"),
            html.Span([
                dcc.Input(id="av-t0", type="number", placeholder="t0 [s]", min=0, step=0.5,
                          style={"width": "80px"}),
                dcc.Input(id="av-t1", type="number", placeholder="t1 [s]", min=0, step=0.5,
                          style={"width": "80px"})],
                title="Window in seconds from the start of each selected phase (applied to "
                      "every phase, then played in order)"),
            html.Label(["speed ", dcc.Input(id="av-speed", type="number", value=0.5, min=0.05,
                                            max=8, step=0.05, style={"width": "70px"})]),
            html.Label(["fps ", dcc.Input(id="av-fps", type="number", value=30, min=5, max=60,
                                          step=1, style={"width": "60px"})]),
            html.Label(["window before/after [s] ",
                        dcc.Input(id="av-win0", type="number", value=-WINDOW_S[0], min=0.5,
                                  max=20, step=0.5, style={"width": "60px"}),
                        dcc.Input(id="av-win1", type="number", value=WINDOW_S[1], min=0,
                                  max=10, step=0.5, style={"width": "60px"})],
                       title="Scrolling panels show this much signal before and after the "
                             "cursor"),
            html.Label(["trail [s] ", dcc.Input(id="av-trail", type="number", value=TRAIL_S,
                                                min=0, max=10, step=0.25,
                                                style={"width": "60px"})],
                       title="Length of the fading path behind the moving landmarks"),
        ]),
        html.Div(className="card controls", children=[
            dcc.Dropdown(id="av-size", options=list(SIZES) + ["640"], value="720p",
                         clearable=False, style={"width": "120px"}),
            # The video's own theme, not the viewer's: a video rendered here equals the
            # CLI's (dark unless asked), whatever theme the page is shown in.
            dcc.RadioItems(id="av-theme", value="dark", inline=True, className="seg",
                           options=[{"label": f" {V.term('video_theme', k)} ", "value": k}
                                    for k in ("dark", "light")]),
            dcc.Checklist(id="av-camera", options=[{"label": " camera frames (shows the "
                                                    "participant; not for sharing)",
                                                    "value": "cam"}], value=[]),
            html.Button("Preview", id="av-preview", className="btn",
                        title="Fast Plotly preview (downsampled, cursor instead of scrolling)"),
            html.Label(["still at [s] ", dcc.Input(id="av-still-s", type="number", value=0,
                                                   min=0, step=0.5, style={"width": "70px"})]),
            html.Button("Still frame", id="av-still", className="btn",
                        title="Render one frame exactly as in the video (matplotlib, 720p)"),
            html.Button("Render video", id="av-render", className="btn primary"),
            html.Button("Cancel", id="av-cancel", className="btn"),
            html.Progress(id="av-progress", value="0", max="1", style={"width": "200px"}),
            html.Span(id="av-state", className="muted"),
            dcc.Interval(id="av-tick", interval=1000,
                         disabled=not ctx.videos.snapshot().running),
        ]),
        html.Div(className="card", children=dcc.Loading(html.Div(id="av-still-body"))),
        html.Div(className="card", children=dcc.Loading(html.Div(id="av-body", children=html.P(
            "Preview plays a downsampled version in the browser; Render video writes the "
            "full-quality MP4 (same as `biomech animate`) to the study's videos folder.",
            className="muted")))),
    ])


def register(app, ctx: ViewerContext) -> None:
    @app.callback(Output("av-phases", "options"), Output("av-phases", "value"),
                  Output("av-pid", "disabled"), Output("av-speed", "value"),
                  Input("av-pid", "value"), Input("av-layout", "value"),
                  State("av-phases", "value"))
    def _phases(pid, layout_, current):
        # A layout change sets that layout's default phases and speed; a participant change
        # keeps the chosen phases where the participant has them.
        new_layout = dctx.triggered_id == "av-layout"
        speed = default_speed(layout_) if new_layout else no_update
        if layout_ in COHORT_LAYOUTS:
            from biomech.kinematics.clinical import TASKS

            keep = [p for p in (current or []) if p in TASKS]
            return _phase_opts(TASKS), keep[:1] or ["open_close"], True, speed
        names = _phase_names(ctx, pid)
        dflt = [p for p in default_phases(layout_) if p in names]
        if new_layout and dflt:
            return _phase_opts(names), dflt, False, speed
        keep = [p for p in (current or []) if p in names]
        return _phase_opts(names), keep or dflt or names[:1], False, speed

    @app.callback(Output("av-body", "children"), Input("av-preview", "n_clicks"),
                  State("av-layout", "value"), State("av-pid", "value"),
                  State("av-phases", "value"), State("av-t0", "value"), State("av-t1", "value"),
                  State("av-speed", "value"), State("av-fps", "value"),
                  State("av-camera", "value"), State("av-theme", "value"),
                  State(env_id("videos"), "data"), State("av-win0", "value"),
                  State("av-win1", "value"), State("av-trail", "value"),
                  prevent_initial_call=True)
    def _preview(_, layout_, pid, phases, t0, t1, speed, fps, cam, theme, env, w0, w1,
                 trail):
        from biomech.anim import make
        from biomech.anim.render_plotly import preview_figure
        from biomech.viz.render_plotly import PX_PER_IN, container_width_in

        width_in = container_width_in(env_of(env)[1])
        try:
            anim = make(ctx.data, **_make_kw(layout_, pid, phases, t0, t1, speed, fps,
                                             "cam" in (cam or []), theme or "dark",
                                             (w0, w1), trail))
            fig = preview_figure(anim, theme=theme or "dark", width_in=width_in)
        except Exception as exc:
            return html.Pre(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                            className="log")
        tl = anim.timeline
        # The animation layout fixes its own height (play button, slider); the box keeps
        # that aspect as the card width changes.
        h = float(fig.layout.height or 0) or 0.56 * width_in * PX_PER_IN
        return html.Div([graph_box("av-graph", fig, aspect=width_in * PX_PER_IN / h),
                         html.P(f"{tl.duration_s:.1f} s of video ({tl.n_frames} frames at "
                                f"{anim.fps:g} fps, {anim.speed:g}× real time); the preview "
                                f"shows {len(fig.frames)} frames.", className="caption")])

    @app.callback(Output("av-state", "children", allow_duplicate=True),
                  Output("av-tick", "disabled", allow_duplicate=True),
                  Input("av-render", "n_clicks"), State("av-layout", "value"),
                  State("av-pid", "value"), State("av-phases", "value"),
                  State("av-t0", "value"), State("av-t1", "value"), State("av-speed", "value"),
                  State("av-fps", "value"), State("av-camera", "value"),
                  State("av-size", "value"), State("av-theme", "value"),
                  State("av-win0", "value"), State("av-win1", "value"),
                  State("av-trail", "value"), prevent_initial_call=True)
    def _render(_, layout_, pid, phases, t0, t1, speed, fps, cam, size, theme, w0, w1, trail):
        kw = _make_kw(layout_, pid, phases, t0, t1, speed, fps, "cam" in (cam or []),
                      theme or "dark", (w0, w1), trail)
        out = video_path(ctx, {**kw, "size": size})
        ok = ctx.videos.start(ctx.data, out, kw, size=int(size) if str(size).isdigit()
                              else size)
        return (f"rendering -> {out.name}" if ok else "a render is already in progress"), False

    @app.callback(Output("av-still-body", "children"), Input("av-still", "n_clicks"),
                  State("av-layout", "value"), State("av-pid", "value"),
                  State("av-phases", "value"), State("av-t0", "value"), State("av-t1", "value"),
                  State("av-speed", "value"), State("av-fps", "value"),
                  State("av-camera", "value"), State("av-theme", "value"),
                  State("av-win0", "value"), State("av-win1", "value"),
                  State("av-trail", "value"), State("av-still-s", "value"),
                  prevent_initial_call=True)
    def _still(_, layout_, pid, phases, t0, t1, speed, fps, cam, theme, w0, w1, trail, at_s):
        """One frame rendered exactly as in the video (same Scene as render_video)."""
        import base64
        import io

        from matplotlib.image import imsave

        from biomech.anim import make

        try:
            anim = make(ctx.data, **_make_kw(layout_, pid, phases, t0, t1, speed, fps,
                                             "cam" in (cam or []), theme or "dark",
                                             (w0, w1), trail))
            n = anim.timeline.n_frames
            i = int(min(max(round(float(at_s or 0) * anim.fps), 0), max(n - 1, 0)))
            scene = anim.scene("720p")
            scene.prepare([i])
            buf = io.BytesIO()
            imsave(buf, scene.frame(0), format="png")
        except Exception as exc:
            return html.Pre(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                            className="log")
        src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return html.Figure([html.Img(src=src, style={"width": "100%"}),
                            html.Figcaption(f"video frame {i + 1} of {n} (t = {i / anim.fps:.2f} "
                                            "s of video), rendered exactly as in the file",
                                            className="caption")])

    @app.callback(Output("av-state", "children", allow_duplicate=True),
                  Input("av-cancel", "n_clicks"), prevent_initial_call=True)
    def _cancel(_):
        ctx.videos.cancel()
        return "cancel requested"

    @app.callback(Output("av-progress", "value"), Output("av-progress", "max"),
                  Output("av-state", "children"), Output("av-tick", "disabled"),
                  Input("av-tick", "n_intervals"))
    def _tick(_):
        s = ctx.videos.snapshot()
        if s.started is None:
            return "0", "1", no_update, True
        if s.running:
            msg = f"rendering {Path(s.path).name}: {s.done}/{s.total} frames"
        elif s.error:
            msg = f"failed: {s.error.strip().splitlines()[-1]}"
        elif s.cancelled:
            msg = "cancelled; no file written"
        else:
            p = Path(s.path)
            rel = p.name
            film = p.with_name(f"{p.stem}_filmstrip.png")
            extra = [", ", html.A("filmstrip", href=f"/video-files/{film.name}", target="_blank")] \
                if film.is_file() else []
            msg = html.Span(["done: ", html.A(rel, href=f"/video-files/{rel}", target="_blank"),
                             *extra, f" ({s.finished - s.started:.0f} s; clip files "
                             f"{p.stem}.clip.npz/.json for biomech animate --from-clip)"])
        return str(s.done), str(max(s.total, 1)), msg, not s.running
