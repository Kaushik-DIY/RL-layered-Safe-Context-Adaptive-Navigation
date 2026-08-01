"""Render the industrial showcase as a self-contained video, entirely offline.

No Gazebo needed: this replays the SAME scene the offline gate verifies, through the
real MpcController + CbfFilter + the exported ONNX policy, and draws it top-down with
both runs stacked -- RL-supervised above, fixed-parameter baseline below.

The thing that carries the argument is the speed-dependent PROTECTIVE FIELD drawn around
each robot: radius = d_stop(sigma*v) + d_hard, i.e. how much room that robot needs to
stop from the speed it is currently doing. At 1.5 m/s that is 2.8 m; at 0.5 m/s it is
0.8 m. It turns RED the instant a worker is inside it -- a stopping-distance violation,
which is a certification failure, not just a near miss. Watching the baseline's field
swell to fill the aisle and flash red, while the supervised one shrinks before each
hazard, is the whole result in one image.

"Protective field" is the AMR-industry term (IEC 61496 / ISO 3691-4) for the zone that
must stay clear, and real machines switch fields by speed -- which is exactly what this
shows, continuously. Earlier drafts called it a "stopping envelope"; "envelope" is
standardised only for ARM REACH VOLUMES (ISO 10218 / R15.06), so it was the wrong word.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_demo.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_demo.py --gif
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_showcase_demo.py --still 14.0

Honest labelling: this is the 2D simulation the policy was trained and evaluated in --
the same `core/` control stack the Gazebo demo runs, but not a 3D render. Say so.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from core.cbf.cbf_filter import d_stop
from core.common.platform import load_platform
from core.demo.showcase_scene import (AISLE_TOP, CUES, EVENT_X, GOAL, HALF_W, MOUTH,
                                      WALLS, X_MAX, X_MIN)

OUT = Path("experiments/results")
STATIONS = [("A", "blind corner\nnobody there"), ("B", "worker crosses\nthe intersection"),
            ("C", "occluded worker\nsteps out")]
C_SAFE, C_WARN, C_BAD = "#2e8b57", "#e8a317", "#cc2b1d"


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_showcase", Path(__file__).resolve().parent / "verify_showcase.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def envelope_radius(v: float, plat) -> float:
    """How much room this robot needs to stop from its CURRENT speed (ISO 13855 chain)."""
    return d_stop(plat.cbf.sigma * max(0.0, v), plat.cbf.tau, plat.cbf.a_brake) + plat.cbf.d_hard


def draw_static(ax):
    # anchor E: with aspect="equal" the box is HEIGHT-limited (data is 29 m x 8.8 m in a
    # much wider subplot), so matplotlib shrinks it and centres the slack. Pushing the box
    # right instead collects all of that slack into one left gutter -- free space, because
    # the map cannot get any wider anyway -- which is where the instruments now live.
    ax.set_anchor("E")
    for x1, y1, x2, y2 in WALLS:
        ax.plot([x1, x2], [y1, y2], color="#4a4d55", lw=3.0, solid_capstyle="butt", zorder=2)
    for i, x in enumerate(EVENT_X):                       # hazard hatching at each mouth
        ax.add_patch(Rectangle((x - MOUTH / 2 - 0.25, HALF_W - 1.85), MOUTH + 0.5, 1.8,
                               color="#f0c419", alpha=0.20, lw=0, zorder=1))
        ax.axvspan(x - MOUTH / 2 - 0.25, x + MOUTH / 2 + 0.25, color="#f0c419",
                   alpha=0.07, lw=0, zorder=0)
    ax.plot(GOAL[0], GOAL[1], marker="*", ms=16, color="#1b6ca8", zorder=3)
    ax.set_xlim(X_MIN - 0.5, X_MAX - 0.5)
    ax.set_ylim(-4.5, AISLE_TOP + 0.3)
    ax.set_aspect("equal")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


class Panel:
    """One run: map artists in the axes, instruments in the figure's left gutter.

    The speed gauge and the readout used to be drawn in axes coordinates, which put them
    on top of the aisle -- the supervised robot drove underneath its own gauge for the
    first few metres. Everything that is text now lives in `build_sidebar`, outside the
    axes entirely, so the map stays clean and the numbers can be much larger.
    """

    def __init__(self, ax, traj, plat, title, accent, fps=20):
        self.ax, self.traj, self.plat, self.accent = ax, traj, plat, accent
        self.fps = fps
        self.title_str = title
        draw_static(ax)
        self.envelope = Circle((0, 0), 0.1, fc=C_SAFE, alpha=0.22, ec=C_SAFE,
                               lw=1.6, zorder=4)
        ax.add_patch(self.envelope)
        self.body = FancyBboxPatch((0, 0), 0.75, 0.5, boxstyle="round,pad=0.02",
                                   fc="#1b6ca8", ec="#0d3c5e", lw=1.2, zorder=6)
        ax.add_patch(self.body)
        self.nose, = ax.plot([], [], color="#f5c518", lw=2.5, zorder=7)
        self.trail, = ax.plot([], [], color=accent, lw=1.4, alpha=0.55, zorder=3)
        self.workers = [Circle((0, 0), 0.28, fc="#ff7a1a", ec="#7a3400", lw=1.2, zorder=6)
                        for _ in CUES]
        for w in self.workers:
            ax.add_patch(w)
        self.flag = ax.text(0.5, 0.05, "", transform=ax.transAxes, fontsize=15,
                            weight="bold", color="white", ha="center", zorder=9,
                            bbox=dict(facecolor=C_BAD, edgecolor="none", pad=5))
        # A breach lasts only a few control steps (0.15 s here), which is invisible at
        # 20 fps. Hold the red state for ~0.9 s and leave a permanent X where it happened,
        # so a real but momentary safety failure is actually legible on screen.
        self.marks, = ax.plot([], [], "x", color=C_BAD, ms=11, mew=2.6, zorder=8)
        self.mark_xy = []
        self._was_breach = False
        self.hold = 0
        # seconds spent in violation, NOT a frame count: the sim runs at 10 Hz and
        # the video at `fps`, so counting frames would scale with the render rate.
        self.viol_s = 0.0
        self._last = -1

    def build_sidebar(self, fig, rect, legend=False):
        """Instruments, in FIGURE coordinates, inside the gutter left of this panel.

        `rect` is (x, y, w, h) in figure fractions; u/v below are fractions of it, so the
        whole cluster rescales with the gutter instead of carrying magic numbers.
        `legend` draws the key -- once, under the lower panel, where the gutter is free.
        """
        sx, sy, sw, sh = rect
        T = fig.transFigure
        X = lambda u: sx + u * sw
        Y = lambda v: sy + v * sh

        # accent spine: ties the numbers to the run they belong to
        fig.add_artist(Rectangle((X(0.0), Y(0.34)), 0.0035, 0.66 * sh, transform=T,
                                 fc=self.accent, ec="none"))
        fig.text(X(0.05), Y(1.0), self.title_str, transform=T, fontsize=13,
                 weight="bold", color=self.accent, va="top", wrap=True)

        # --- SPEED GAUGE ------------------------------------------------------
        # Speed is the headline supporting metric, but as a line of small monospace text
        # it was easy to miss. A big number plus a bar filled to v / v_max makes the
        # difference between the two runs readable at a glance, without pausing. Sized to
        # about two thirds of the gutter: legible from the back of a room, not a billboard.
        cy0, cy1, cw = 0.595, 0.905, 0.68
        fig.add_artist(Rectangle((X(0.05), Y(cy0)), cw * sw, (cy1 - cy0) * sh,
                                 transform=T, fc="#f7f8fa", ec="#c9ccd2", lw=1.1))
        fig.text(X(0.09), Y(cy1 - 0.020), "SPEED", transform=T, fontsize=8.5,
                 color="#6b7079", weight="bold", va="top")
        self.speed_txt = fig.text(X(0.09), Y(cy1 - 0.058), "", transform=T, fontsize=25,
                                  weight="bold", color=self.accent, va="top")
        fig.text(X(0.40), Y(cy1 - 0.095), "m/s", transform=T, fontsize=11,
                 color="#6b7079", va="top")
        # bar: full width = the platform maximum, so the two panels are directly comparable
        self.bar_x, self.bar_y = X(0.09), Y(cy0 + 0.064)
        self.bar_w, self.bar_h = (cw - 0.08) * sw, 0.042 * sh
        fig.add_artist(Rectangle((self.bar_x, self.bar_y), self.bar_w, self.bar_h,
                                 transform=T, fc="#e4e7ea", ec="#c9ccd2", lw=0.8))
        self.bar = Rectangle((self.bar_x, self.bar_y), 0.0, self.bar_h, transform=T,
                             fc=self.accent, ec="none")
        fig.add_artist(self.bar)
        # the commanded cap, as a tick on the same bar
        self.cap_tick = Line2D([], [], transform=T, color="#22252a", lw=2.0)
        fig.add_artist(self.cap_tick)
        self.cap_txt = fig.text(X(0.09), Y(cy0 + 0.016), "", transform=T, fontsize=8.5,
                                family="monospace", color="#3c4048", va="bottom")

        self.readout = fig.text(X(0.05), Y(cy0 - 0.055), "", transform=T, fontsize=9.5,
                                family="monospace", color="#3c4048", va="top",
                                linespacing=1.55)
        if legend:
            self._legend(fig, rect)

    def _legend(self, fig, rect):
        """Key for the three quantities, in the gutter under the lower panel.

        Drawn once, not per panel: the terms mean the same thing in both runs. It goes
        under the LOWER readout, which is the last thing read and clear of the footer.
        """
        sx, sy, sw, sh = rect
        T = fig.transFigure
        fig.add_artist(Rectangle((sx + 0.05 * sw, sy + 0.020 * sh), 0.95 * sw, 0.270 * sh,
                                 transform=T, fc="#fbfbfc", ec="#d5d8dd", lw=1.0))
        fig.text(sx + 0.09 * sw, sy + 0.272 * sh, "WHAT THE NUMBERS MEAN", transform=T,
                 fontsize=8, weight="bold", color="#6b7079", va="top")
        fig.text(sx + 0.09 * sw, sy + 0.234 * sh,
                 "cap               the supervisor's live speed limit\n"
                 "protective field  room needed to stop at this speed\n"
                 "barrier h         gap from the worker to the\n"
                 "                  protective field; below 0 = breach",
                 transform=T, fontsize=7.5, family="monospace", color="#4a4d55",
                 va="top", linespacing=1.60)

    def draw(self, k):
        k = min(k, len(self.traj) - 1)
        if k < self._last:                       # rewound (loop): reset
            self.viol_s, self.mark_xy, self.hold = 0.0, [], 0
            self._was_breach = False
        f = self.traj[k]
        x, y, yaw, v, h = f["x"], f["y"], f["yaw"], f["v"], f["h"]
        r = envelope_radius(v, self.plat)
        breach = h is not None and h < 0.0
        if k > self._last:
            self.viol_s += (1.0 / self.fps) if breach else 0.0
            if breach:
                self.hold = int(0.9 * self.fps)
                # one mark per violation EVENT. Marking every breaching frame put two
                # crosses 14 cm apart on the crossing breach, which is a single 0.2 s
                # event spanning two control steps -- it read as two separate failures.
                if not self._was_breach:
                    self.mark_xy.append((x, y))
            else:
                self.hold = max(0, self.hold - 1)
            self._was_breach = breach
        self._last = k
        showing = breach or self.hold > 0

        self.envelope.center = (x, y)
        self.envelope.set_radius(r)
        col = C_BAD if showing else (C_WARN if (h is not None and h < 0.35) else C_SAFE)
        self.envelope.set_facecolor(col)
        self.envelope.set_edgecolor(col)
        self.envelope.set_alpha(0.34 if showing else 0.20)
        if self.mark_xy:
            self.marks.set_data([p[0] for p in self.mark_xy],
                                [p[1] for p in self.mark_xy])

        self.body.set_x(x - 0.375)
        self.body.set_y(y - 0.25)
        self.nose.set_data([x + 0.30, x + 0.52], [y, y])
        xs = [p["x"] for p in self.traj[:k + 1]]
        ys = [p["y"] for p in self.traj[:k + 1]]
        self.trail.set_data(xs, ys)
        for c, (wx, wy, _, seen) in zip(self.workers, f["workers"]):
            c.center = (wx, wy)
            c.set_alpha(1.0 if seen else 0.22)          # greyed while occluded
            c.set_facecolor("#ff7a1a" if seen else "#9aa0a6")

        vmax = self.plat.robot.v_max
        self.speed_txt.set_text(f"{v:.2f}")
        self.bar.set_width(self.bar_w * min(1.0, max(0.0, v / vmax)))
        cx = self.bar_x + self.bar_w * min(1.0, max(0.0, f["cap"] / vmax))
        over = 0.32 * self.bar_h        # tick must not reach up into the big number
        self.cap_tick.set_data([cx, cx],
                               [self.bar_y - over, self.bar_y + self.bar_h + over])
        self.cap_txt.set_text(f"cap {f['cap']:.2f}   of {vmax:.1f} m/s max")

        # narrow column: one label/value pair per line, values in a fixed column
        hs = "--" if h is None else f"{h:+.2f} m"
        self.readout.set_text(
            f"{'time':<18}{f['t']:5.1f} s\n"
            f"{'protective field':<18}{r:5.2f} m\n"
            f"{'barrier h':<18}{hs:>7s}\n"
            f"{'in violation':<18}{self.viol_s:5.2f} s")
        self.flag.set_text("STOPPING-DISTANCE VIOLATION" if showing else "")
        return [self.envelope, self.body, self.nose, self.trail, *self.workers,
                self.marks, self.readout, self.flag, self.speed_txt, self.bar,
                self.cap_tick, self.cap_txt]


def build(fps: int):
    v = _verifier()
    plat = load_platform("industrial")
    runs = []
    # "B" = rated speed, no commissioning -- the baseline this (older) video compares
    # against. The A-vs-C presentation video is scripts/render_ac_video.py.
    for arm in ("C", "B"):
        rec = []
        res = v.run(arm, record=rec)
        runs.append((rec, res))
        print(f"  {'RL-supervised' if arm == 'C' else 'baseline     '}: "
              f"{len(rec)} steps, mission {res['t']:.1f} s")

    # resample both onto a common wall clock; a finished run holds at its last frame
    dt = plat.robot.dt
    n = int(max(len(r) for r, _ in runs) * dt * fps) + 1
    out = []
    for rec, _ in runs:
        idx = np.clip((np.arange(n) / fps / dt).astype(int), 0, len(rec) - 1)
        out.append([rec[i] for i in idx])
    return out, [r for _, r in runs], plat, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gif", action="store_true", help="also write a GIF")
    ap.add_argument("--still", type=float, default=None,
                    help="write a single PNG at this mission time instead")
    ap.add_argument("--name", default="showcase_demo")
    args = ap.parse_args()

    print("replaying both runs through the real MPC + CBF + ONNX policy ...")
    (rl_t, bl_t), (rl_r, bl_r), plat, n = build(args.fps)

    fig, axes = plt.subplots(2, 1, figsize=(16.0, 9.0),
                             gridspec_kw=dict(hspace=0.06))
    fig.patch.set_facecolor("white")
    p_rl = Panel(axes[0], rl_t, plat, "MPC + CBF + RL supervisor", "#1b6ca8", args.fps)
    p_bl = Panel(axes[1], bl_t, plat, "MPC + CBF, fixed parameters (no RL)",
                 "#c1443c", args.fps)
    axes[0].tick_params(labelbottom=False)          # one x-axis, on the bottom panel
    axes[1].set_xlabel("position along the aisle  x  (m)", fontsize=10)
    for k, (code, name) in enumerate(STATIONS):
        axes[0].text(EVENT_X[k], AISLE_TOP + 0.15, f"{code}  {name}", ha="center",
                     va="bottom", fontsize=8.5, color="#7a5c00", linespacing=1.2)
    fig.suptitle("Same warehouse, same workers, same goal - identical MPC + CBF stack, "
                 "with and without the RL supervisor",
                 fontsize=13, weight="bold", x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.018,
             "Shaded disc = protective field: the ISO stopping distance at the robot's "
             "current speed (d_stop(sigma*v) + d_hard).  RED = a worker is inside it.\n"
             "Over 12 randomised runs:  ISO violations   blind corner 8/12 -> 0/12,   "
             "crossing 6/12 -> 0/12.        "
             "2D simulation (same control stack as the Gazebo build).",
             fontsize=8.5, family="monospace", va="bottom", color="#333333")
    fig.subplots_adjust(left=0.035, right=0.982, top=0.90, bottom=0.115)

    # The aspect-equal boxes are only sized once the figure is drawn, so settle the layout
    # first and then read each panel's REAL extent -- the gutter is whatever sits left of
    # it, and hard-coding that width would break the moment the aisle or figure changes.
    fig.canvas.draw()
    for panel in (p_rl, p_bl):
        bb = panel.ax.get_window_extent().transformed(fig.transFigure.inverted())
        panel.build_sidebar(fig, (0.012, bb.y0, bb.x0 - 0.030, bb.height),
                            legend=panel is p_bl)

    if args.still is not None:
        # replay every frame up to k: the violation marks and the time-in-violation
        # readout are ACCUMULATED state, so drawing frame k alone shows a blank history
        k = int(args.still * args.fps)
        for j in range(k + 1):
            p_rl.draw(j)
            p_bl.draw(j)
        path = OUT / f"{args.name}_t{args.still:g}.png"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")
        return

    def frame(k):
        return p_rl.draw(k) + p_bl.draw(k)

    anim = animation.FuncAnimation(fig, frame, frames=n, blit=False, interval=1000 / args.fps)
    mp4 = OUT / f"{args.name}.mp4"
    anim.save(mp4, writer=animation.FFMpegWriter(fps=args.fps, bitrate=3600), dpi=120)
    print(f"wrote {mp4}  ({n} frames, {n/args.fps:.1f} s)")
    if args.gif:
        gif = OUT / f"{args.name}.gif"
        anim.save(gif, writer=animation.PillowWriter(fps=min(args.fps, 12)), dpi=80)
        print(f"wrote {gif}")

    print(f"\nRL-supervised : {rl_r['t']:.1f} s, "
          f"{sum(e['viol'] for e in rl_r['events'])} violation steps")
    print(f"baseline      : {bl_r['t']:.1f} s, "
          f"{sum(e['viol'] for e in bl_r['events'])} violation steps")


if __name__ == "__main__":
    main()
