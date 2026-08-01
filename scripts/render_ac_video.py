"""The presentation video: a commissioned industrial AMR against the RL supervisor.

Both machines are ISO 3691-4 compliant on this route -- neither breaches the
stopping-distance barrier once. That is the whole point, and it is why this video does NOT
look like the older one: there is no red-flash violation to watch. The argument is the
clock, and the way each machine sizes its protective field.

    A   commissioned AMR      0.50 m/s everywhere, two fixed scanner fields
    C   MPC + CBF + RL        speed chosen per context, field scales with speed

A real AMR is compliant because an integrator hand-capped its speed so the stopping
distance fits inside what the scanner can see: 0.76 m at 0.50 m/s, against a 1.2 m reveal
at a blind corner. At the platform's rated 1.5 m/s it would need 2.83 m and the same corner
is unsurvivable. So the commissioned machine crawls the entire aisle to be safe at the two
places that need it. The supervisor slows only where the geometry demands it -- measured on
this route: 54.2 s -> 27.4 s, 1.98x the throughput, with the same zero violations and the
same barrier margin at the occluded corner (+0.65 m vs +0.68 m).

READ THE FIELDS CORRECTLY. A smaller field is not a safer robot: the field is sized to the
stopping distance at the current speed, so a small field means a slow robot. The
commissioned machine's rings never change size because its speed never changes.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_ac_video.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_ac_video.py --still 12
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_ac_video.py --gif
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
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D

from core.cbf.cbf_filter import d_stop
from core.common.platform import load_platform
from core.demo import scanner_amr
from core.demo.showcase_scene import (AISLE_TOP, EVENT_X, GOAL, HALF_W, WALLS,
                                      X_MAX, X_MIN)

OUT = Path("experiments/results")

# --- palette: a fleet-management console, which is how AMR fleets are actually viewed ---
C_FLOOR, C_AISLE = "#e9ecef", "#f7f8fa"
C_RACK, C_RACK_EDGE, C_PALLET = "#aeb7c2", "#79828f", "#c8a06a"
C_WALL = "#5a626d"
C_HAZARD = "#f2c200"
C_BOT, C_BOT_EDGE, C_LOAD = "#333d47", "#1d242b", "#c8a06a"
C_VEST, C_HEAD = "#f5a623", "#2f2f2f"
ACCENT = {"A": "#2e8b57", "C": "#1b6ca8"}
TITLE = {"A": "COMMISSIONED AMR",
         "C": "MPC + CBF + RL SUPERVISOR"}
SUBTITLE = {"A": "0.50 m/s site speed limit  |  fixed scanner fields",
            "C": "speed chosen per context  |  field scales with speed"}
STATE_COL = {scanner_amr.NORMAL: "#2e8b57", scanner_amr.WARNING: "#e8a317",
             scanner_amr.STOPPED: "#cc2b1d"}

RACK_DEPTH = 1.15
# The aspect-equal panel is height-limited, so the map's WIDTH is set by (x-range /
# y-range): a wider map eats the left gutter the instruments live in. +/-4.8 keeps the
# ratio near 3.2, which leaves a workable sidebar AND shows the side aisles in full.
Y_LO, Y_HI = -4.8, 4.8


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_showcase", Path(__file__).resolve().parent / "verify_showcase.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def field_radius(v: float, plat) -> float:
    """Room this machine needs to stop from its CURRENT speed (the ISO 13855 chain)."""
    return d_stop(plat.cbf.sigma * max(0.0, v), plat.cbf.tau, plat.cbf.a_brake) \
        + plat.cbf.d_hard


# --------------------------------------------------------------------- warehouse
def _rack_spans():
    """Solid wall runs along each side of the main aisle, taken FROM the collision
    geometry so the racking drawn is exactly the racking the planner avoids."""
    north, south = [], []
    for x1, y1, x2, y2 in WALLS:
        if abs(y1 - y2) < 1e-9 and abs(abs(y1) - HALF_W) < 1e-9:
            (north if y1 > 0 else south).append((min(x1, x2), max(x1, x2)))
    return north, south


def _draw_rack_run(ax, x0, x1, y_face, outward, label_row):
    """One run of pallet racking: bays, uprights, pallets, bay labels."""
    depth = RACK_DEPTH * outward
    ax.add_patch(Rectangle((x0, min(y_face, y_face + depth)), x1 - x0, RACK_DEPTH,
                           fc=C_RACK, ec=C_RACK_EDGE, lw=0.8, zorder=2))
    n_bay = max(1, int(round((x1 - x0) / 2.4)))
    w = (x1 - x0) / n_bay
    for b in range(n_bay):
        bx = x0 + b * w
        ax.plot([bx, bx], [y_face, y_face + depth], color=C_RACK_EDGE, lw=1.4, zorder=3)
        for slot in (0.30, 0.62):                      # two pallet positions per bay
            px = bx + slot * w
            ax.add_patch(Rectangle((px, y_face + 0.30 * depth), 0.26 * w,
                                   0.42 * depth * np.sign(outward) * outward,
                                   fc=C_PALLET, ec="#8a6c46", lw=0.5, zorder=4))
        if w > 1.6 and label_row is not None:
            ax.text(bx + w / 2, y_face + depth * 0.90, f"{label_row}-{b + 1:02d}",
                    ha="center", va="center", fontsize=4.6, color="#5c6470", zorder=5)
    ax.plot([x0, x1], [y_face, y_face], color=C_WALL, lw=2.2, solid_capstyle="butt",
            zorder=6)


def draw_warehouse(ax):
    """The static scene. Nothing decorative is ever drawn where the robot can drive."""
    ax.add_patch(Rectangle((X_MIN - 1, Y_LO), X_MAX - X_MIN + 2, Y_HI - Y_LO,
                           fc=C_FLOOR, ec="none", zorder=0))
    ax.add_patch(Rectangle((X_MIN - 1, -HALF_W), X_MAX - X_MIN + 2, 2 * HALF_W,
                           fc=C_AISLE, ec="none", zorder=1))

    north, south = _rack_spans()
    for x0, x1 in north:
        _draw_rack_run(ax, x0, x1, HALF_W, +1, "N")
    for x0, x1 in south:
        _draw_rack_run(ax, x0, x1, -HALF_W, -1, "S")

    # side-aisle walls: rack faces, drawn from the same segment list
    for x1, y1, x2, y2 in WALLS:
        if abs(x1 - x2) < 1e-9:
            ax.plot([x1, x2], [y1, y2], color=C_WALL, lw=2.2, zorder=6)

    # aisle centre line + hazard zones at the intersections
    ax.plot([X_MIN, X_MAX], [0, 0], color="#c6cbd2", lw=0.9, ls=(0, (9, 9)), zorder=2)
    for x in EVENT_X:
        ax.add_patch(Rectangle((x - 1.05, -HALF_W), 2.10, 2 * HALF_W, fc=C_HAZARD,
                               alpha=0.16, ec="none", zorder=2))
        for edge in (x - 1.05, x + 1.05):
            ax.plot([edge, edge], [-HALF_W, HALF_W], color=C_HAZARD, lw=2.0,
                    alpha=0.85, zorder=3)

    # Charging dock behind the start and the pick station BEYOND the goal: the AMR pulls
    # up to the station at x = GOAL, it does not drive through it. Nothing decorative is
    # placed anywhere the robot's path actually goes.
    ax.add_patch(Rectangle((X_MIN - 1.05, -0.75), 0.80, 1.5, fc="#5a626d",
                           ec="#3b424b", lw=0.8, zorder=4))
    ax.text(X_MIN - 0.65, 1.02, "CHARGE", fontsize=5.8, color="#5a626d", ha="center",
            zorder=5)
    sx0 = GOAL[0] + 0.55
    ax.add_patch(Rectangle((sx0, -1.30), 1.35, 2.6, fc="#d8dde3", ec="#79828f",
                           lw=1.0, zorder=4))
    for i in range(4):
        ax.plot([sx0 + 0.22 + i * 0.30] * 2, [-1.15, 1.15], color="#a8b0ba",
                lw=1.4, zorder=5)
    ax.text(sx0 + 0.68, -1.72, "PICK STATION P-12", fontsize=6.0, color="#5a626d",
            ha="center", zorder=5)

    ax.text(1.0, -HALF_W + 0.30, "AISLE 04", fontsize=6.5, color="#98a1ad",
            weight="bold", zorder=5)
    ax.set_xlim(X_MIN - 1.3, X_MAX + 1.2)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_aspect("equal")
    ax.set_anchor("E")          # collect the slack into one left gutter for the sidebar
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------------ panel
class Panel:
    """One machine: warehouse layer in the axes, instruments in the figure gutter."""

    def __init__(self, ax, traj, plat, arm, fps, arrive_s):
        self.ax, self.traj, self.plat, self.arm = ax, traj, plat, arm
        self.accent, self.fps, self.arrive_s = ACCENT[arm], fps, arrive_s
        draw_warehouse(ax)

        # safety field(s). Arm A gets the two FIXED rings a scanner datasheet shows;
        # arm C gets one disc that breathes with speed. The difference in KIND is the
        # argument, so they are deliberately drawn in different visual languages.
        self.field = Circle((0, 0), 0.1, fc=self.accent, alpha=0.16, ec=self.accent,
                            lw=1.4, zorder=7)
        ax.add_patch(self.field)
        self.rings = []
        if arm == "A":
            sc = scanner_amr.ScannerAMR(plat)
            for r, ls in ((sc.r_warn, (0, (4, 3))), (sc.r_prot, "solid")):
                ring = Circle((0, 0), r, fc="none", ec=self.accent, lw=1.5, ls=ls,
                              alpha=0.85, zorder=8)
                ax.add_patch(ring)
                self.rings.append(ring)

        self.trail, = ax.plot([], [], color=self.accent, lw=1.3, alpha=0.5, zorder=6)
        self.body = FancyBboxPatch((-0.45, -0.31), 0.90, 0.62,
                                   boxstyle="round,pad=0.015,rounding_size=0.10",
                                   fc=C_BOT, ec=C_BOT_EDGE, lw=1.0, zorder=10)
        self.load = Rectangle((-0.30, -0.22), 0.60, 0.44, fc=C_LOAD, ec="#8a6c46",
                              lw=0.7, zorder=11)
        self.strip = Rectangle((0.36, -0.26), 0.09, 0.52, fc="#2e8b57", ec="none",
                               zorder=12)
        for p in (self.body, self.load, self.strip):
            ax.add_patch(p)
        self.tag = ax.text(0, 0, "", fontsize=5.2, color="#3b424b", ha="center",
                           va="center", zorder=13)

        self.w_vest, self.w_head, self.w_arrow = [], [], []
        for _ in range(4):
            v = Ellipse((0, 0), 0.62, 0.46, fc=C_VEST, ec="#8a5a10", lw=0.9, zorder=10)
            hd = Circle((0, 0), 0.115, fc=C_HEAD, ec="none", zorder=11)
            ar, = ax.plot([], [], color="#8a5a10", lw=1.1, zorder=10)
            ax.add_patch(v)
            ax.add_patch(hd)
            self.w_vest.append(v)
            self.w_head.append(hd)
            self.w_arrow.append(ar)

        self.badge = ax.text(0.5, 0.055, "", transform=ax.transAxes, fontsize=13,
                             weight="bold", color="white", ha="center", zorder=15,
                             bbox=dict(facecolor=self.accent, edgecolor="none", pad=5))
        self._last = -1

    # ---------------------------------------------------------------- sidebar
    def build_sidebar(self, fig, rect):
        sx, sy, sw, sh = rect
        T = fig.transFigure
        X = lambda u: sx + u * sw
        Y = lambda v: sy + v * sh
        fig.add_artist(Rectangle((X(0.0), Y(0.30)), 0.0035, 0.70 * sh, transform=T,
                                 fc=self.accent, ec="none"))
        fig.text(X(0.05), Y(1.0), TITLE[self.arm], transform=T, fontsize=13,
                 weight="bold", color=self.accent, va="top")
        fig.text(X(0.05), Y(0.925), SUBTITLE[self.arm], transform=T, fontsize=8,
                 color="#6b7079", va="top")

        cy0, cy1, cw = 0.575, 0.865, 0.68
        fig.add_artist(Rectangle((X(0.05), Y(cy0)), cw * sw, (cy1 - cy0) * sh,
                                 transform=T, fc="#f7f8fa", ec="#c9ccd2", lw=1.1))
        fig.text(X(0.09), Y(cy1 - 0.020), "SPEED", transform=T, fontsize=8.5,
                 color="#6b7079", weight="bold", va="top")
        self.speed_txt = fig.text(X(0.09), Y(cy1 - 0.058), "", transform=T, fontsize=25,
                                  weight="bold", color=self.accent, va="top")
        fig.text(X(0.46), Y(cy1 - 0.095), "m/s", transform=T, fontsize=11,
                 color="#6b7079", va="top")
        self.bar_x, self.bar_y = X(0.09), Y(cy0 + 0.064)
        self.bar_w, self.bar_h = (cw - 0.08) * sw, 0.040 * sh
        fig.add_artist(Rectangle((self.bar_x, self.bar_y), self.bar_w, self.bar_h,
                                 transform=T, fc="#e4e7ea", ec="#c9ccd2", lw=0.8))
        self.bar = Rectangle((self.bar_x, self.bar_y), 0.0, self.bar_h, transform=T,
                             fc=self.accent, ec="none")
        fig.add_artist(self.bar)
        self.cap_txt = fig.text(X(0.09), Y(cy0 + 0.016), "", transform=T, fontsize=8.5,
                                family="monospace", color="#3c4048", va="bottom")

        # status lamp: the commissioned machine's actual field state, ours the cap source
        self.lamp = Rectangle((X(0.05), Y(0.495)), 0.030 * sw, 0.052 * sh, transform=T,
                              fc="#2e8b57", ec="none")
        fig.add_artist(self.lamp)
        self.lamp_txt = fig.text(X(0.14), Y(0.521), "", transform=T, fontsize=9,
                                 weight="bold", color="#3c4048", va="center")

        self.readout = fig.text(X(0.05), Y(0.445), "", transform=T, fontsize=9.5,
                                family="monospace", color="#3c4048", va="top",
                                linespacing=1.55)

        # route progress
        self.pb_x, self.pb_y = X(0.05), Y(0.175)
        self.pb_w, self.pb_h = 0.90 * sw, 0.030 * sh
        fig.add_artist(Rectangle((self.pb_x, self.pb_y), self.pb_w, self.pb_h,
                                 transform=T, fc="#e4e7ea", ec="#c9ccd2", lw=0.8))
        self.pbar = Rectangle((self.pb_x, self.pb_y), 0.0, self.pb_h, transform=T,
                              fc=self.accent, ec="none")
        fig.add_artist(self.pbar)
        self.pb_txt = fig.text(X(0.05), Y(0.130), "", transform=T, fontsize=8.5,
                               family="monospace", color="#3c4048", va="top")

    # ------------------------------------------------------------------- draw
    def draw(self, k):
        f = self.traj[min(k, len(self.traj) - 1)]
        t_now = k / self.fps
        x, y, yaw, v = f["x"], f["y"], f["yaw"], f["v"]
        # epsilon: arrive_s is accumulated as k*dt, so an exact >= misses its own frame
        arrived = self.arrive_s is not None and t_now >= self.arrive_s - 1e-6

        self.field.center = (x, y)
        self.field.set_radius(field_radius(v, self.plat))
        for ring in self.rings:
            ring.center = (x, y)

        rot = Affine2D().rotate_around(0, 0, yaw).translate(x, y) + self.ax.transData
        for p in (self.body, self.load, self.strip):
            p.set_transform(rot)
        self.strip.set_facecolor(
            STATE_COL.get(f.get("state") or scanner_amr.NORMAL, self.accent))
        self.tag.set_position((x, y - 0.62))
        self.tag.set_text(f"AMR-0{1 if self.arm == 'A' else 7}")

        xs = [p["x"] for p in self.traj[:min(k, len(self.traj) - 1) + 1]]
        ys = [p["y"] for p in self.traj[:min(k, len(self.traj) - 1) + 1]]
        self.trail.set_data(xs, ys)

        for i, (wx, wy, wyaw, seen) in enumerate(f["workers"]):
            if i >= len(self.w_vest):
                break
            self.w_vest[i].set_center((wx, wy))
            self.w_vest[i].angle = np.degrees(wyaw)
            self.w_head[i].set_center((wx, wy))
            self.w_vest[i].set_alpha(1.0 if seen else 0.30)
            self.w_head[i].set_alpha(1.0 if seen else 0.30)
            self.w_arrow[i].set_data([wx, wx + 0.45 * np.cos(wyaw)],
                                     [wy, wy + 0.45 * np.sin(wyaw)])
            self.w_arrow[i].set_alpha(0.9 if seen else 0.25)

        vmax = self.plat.robot.v_max
        self.speed_txt.set_text(f"{v:.2f}")
        self.bar.set_width(self.bar_w * min(1.0, max(0.0, v / vmax)))
        self.cap_txt.set_text(f"limit {f['cap']:.2f}   of {vmax:.1f} m/s rated")

        state = f.get("state")
        if state is None:
            state, col = "SUPERVISED", self.accent
        else:
            col = STATE_COL[state]
        self.lamp.set_facecolor(col)
        self.lamp_txt.set_text(state)
        self.lamp_txt.set_color(col)

        hs = "  --  " if f["h"] is None else f"{f['h']:+.2f} m"
        self.readout.set_text(
            f"{'mission clock':<17}{t_now:5.1f} s\n"
            f"{'protective field':<17}{field_radius(v, self.plat):5.2f} m\n"
            f"{'ISO margin h':<17}{hs:>7s}\n"
            f"{'standstill':<17}{f['stopped_s']:5.1f} s")

        done = 1.0 if arrived else float(np.clip(1.0 - f["dist_goal"] / GOAL[0], 0.0, 1.0))
        self.pbar.set_width(self.pb_w * done)
        if arrived:
            self.pb_txt.set_text(f"DELIVERED at {self.arrive_s:.1f} s   "
                                 f"idle {t_now - self.arrive_s:4.1f} s")
            self.badge.set_text(f"DELIVERED  {self.arrive_s:.1f} s")
        else:
            self.pb_txt.set_text(f"route {100 * done:4.1f} %   "
                                 f"{f['dist_goal']:4.1f} m to go")
            self.badge.set_text("")
        self._last = k
        return []


# ------------------------------------------------------------------------- main
def build(fps):
    v = _verifier()
    plat = load_platform("industrial")
    runs, arrive = {}, {}
    for arm in ("C", "A"):
        rec = []
        res = v.run(arm, record=rec)
        runs[arm], arrive[arm] = rec, res["t"]
        print(f"  arm {arm}: {len(rec)} steps, mission {res['t']:.1f} s, "
              f"violations {sum(e['viol'] for e in res['events'])}")
    dt = plat.robot.dt
    n_live = int(max(len(r) for r in runs.values()) * dt * fps) + 1
    out = {}
    for arm, rec in runs.items():
        idx = np.clip((np.arange(n_live) / fps / dt).astype(int), 0, len(rec) - 1)
        out[arm] = [rec[i] for i in idx]
    return out, arrive, plat, n_live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--card", type=float, default=3.0, help="seconds of summary card")
    ap.add_argument("--name", default="ac_demo")
    args = ap.parse_args()

    print("replaying both machines through the real MPC + CBF stack ...")
    traj, arrive, plat, n_live = build(args.fps)
    n_card = int(args.card * args.fps)
    n = n_live + n_card

    fig, axes = plt.subplots(2, 1, figsize=(16.0, 9.0), gridspec_kw=dict(hspace=0.05))
    fig.patch.set_facecolor("white")
    p_c = Panel(axes[0], traj["C"], plat, "C", args.fps, arrive["C"])
    p_a = Panel(axes[1], traj["A"], plat, "A", args.fps, arrive["A"])
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("position along the aisle  x  [m]", fontsize=9)
    for k, name in enumerate(["blind corner\n(nobody there)", "worker crosses\nthe aisle",
                              "occluded worker\nsteps out"]):
        axes[0].text(EVENT_X[k], AISLE_TOP + 0.05, name, ha="center", va="bottom",
                     fontsize=7.5, color="#7a5c00", linespacing=1.2)

    fig.suptitle("Same warehouse, same workers, same 27 m transport run - "
                 "identical MPC + CBF stack, two ways of choosing the speed",
                 fontsize=13, weight="bold", x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.016,
             "Shaded disc / rings = protective field: the room the machine needs to stop "
             "from the speed it is doing (ISO 13855).  A SMALLER FIELD MEANS A SLOWER "
             "ROBOT, not a safer one.\n"
             "Neither machine breaches the field on this route - the difference is "
             "throughput.  At the rated 1.5 m/s with no supervision the same route "
             "breaches ISO in 14 of 24 runs.",
             fontsize=8.5, family="monospace", va="bottom", color="#333")
    fig.subplots_adjust(left=0.035, right=0.982, top=0.905, bottom=0.115)

    fig.canvas.draw()
    for panel in (p_c, p_a):
        bb = panel.ax.get_window_extent().transformed(fig.transFigure.inverted())
        panel.build_sidebar(fig, (0.012, bb.y0, bb.x0 - 0.030, bb.height))

    card = _summary_card(fig, arrive)

    def frame(k):
        kk = min(k, n_live - 1)
        p_c.draw(kk)
        p_a.draw(kk)
        for a in card:
            a.set_visible(k >= n_live)
        return []

    if args.still is not None:
        frame(int(args.still * args.fps))
        path = OUT / f"{args.name}_t{args.still:g}.png"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")
        return

    anim = animation.FuncAnimation(fig, frame, frames=n, blit=False,
                                   interval=1000 / args.fps)
    mp4 = OUT / f"{args.name}.mp4"
    anim.save(mp4, writer=animation.FFMpegWriter(fps=args.fps, bitrate=4200), dpi=120)
    print(f"wrote {mp4}  ({n} frames, {n / args.fps:.1f} s)")
    if args.gif:
        gif = OUT / f"{args.name}.gif"
        anim.save(gif, writer=animation.PillowWriter(fps=min(args.fps, 12)), dpi=80)
        print(f"wrote {gif}")


def _summary_card(fig, arrive):
    """Overlay shown over the final freeze frame. Hidden for the whole live run."""
    ratio = arrive["A"] / arrive["C"]
    L, CA, CC, CN = 0.075, 0.500, 0.680, 0.780      # label / arm A / arm C / note columns
    art = [fig.add_artist(Rectangle((0.0, 0.0), 1.0, 1.0, transform=fig.transFigure,
                                    fc="white", alpha=1.0, ec="none", zorder=40))]
    art.append(fig.text(L, 0.855, "Same safety. Twice the throughput.",
                        fontsize=32, weight="bold", color="#22252a", zorder=41))
    art.append(fig.text(L, 0.795,
                        "27 m transport run, identical MPC + CBF stack, "
                        "identical workers and cues",
                        fontsize=12, color="#6b7079", zorder=41))
    art.append(fig.add_artist(Rectangle((L, 0.735), 0.85, 0.0035,
                                        transform=fig.transFigure, fc="#d5d8dd",
                                        ec="none", zorder=41)))
    art.append(fig.text(CA, 0.680, "COMMISSIONED AMR", ha="center", fontsize=11.5,
                        weight="bold", color=ACCENT["A"], zorder=41))
    art.append(fig.text(CC, 0.680, "OURS", ha="center", fontsize=11.5, weight="bold",
                        color=ACCENT["C"], zorder=41))
    rows = [("ISO stopping-distance violations", "0", "0", "identical"),
            ("mission time", f"{arrive['A']:.1f} s", f"{arrive['C']:.1f} s",
             f"{ratio:.2f}x faster"),
            ("barrier margin at the occluded corner", "+0.65 m", "+0.68 m", "identical"),
            ("missions completed, 96-run battery", "73 %", "100 %", "+27 points")]
    for i, (label, a, c, note) in enumerate(rows):
        yy = 0.590 - i * 0.088
        art.append(fig.text(L, yy, label, fontsize=13.5, color="#3c4048", zorder=41))
        art.append(fig.text(CA, yy, a, ha="center", fontsize=15, weight="bold",
                            color=ACCENT["A"], zorder=41))
        art.append(fig.text(CC, yy, c, ha="center", fontsize=15, weight="bold",
                            color=ACCENT["C"], zorder=41))
        art.append(fig.text(CN, yy, note, fontsize=12, color="#6b7079", zorder=41))
    art.append(fig.text(L, 0.185,
                        "Both machines are ISO 3691-4 compliant on this route. The "
                        "commissioned AMR achieves it by driving 0.50 m/s everywhere; the "
                        "learned\nsupervisor slows only where the geometry demands it. At "
                        "the rated 1.5 m/s with no supervision the same route breaches "
                        "ISO in 14 of 24 runs.",
                        fontsize=11.5, color="#555", linespacing=1.7, zorder=41))
    art.append(fig.text(L, 0.085,
                        "2D simulation - the same control stack as the Gazebo build.",
                        fontsize=10.5, color="#8a919b", zorder=41))
    for a in art:
        a.set_visible(False)
    return art


if __name__ == "__main__":
    main()
