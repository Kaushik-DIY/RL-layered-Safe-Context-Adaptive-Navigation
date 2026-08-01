"""The head-on video: an oncoming picker in a wide two-way aisle, three ways of passing.

SEPARATE FROM THE COMMISSIONING VIDEO ON PURPOSE. There are no cross-aisles on this route,
so there is nothing for an integrator to mark and the commissioning argument does not
apply. The only question here is what each machine DOES when a person walks straight at it.

    1  INDUSTRIAL AMR        the warning tier is its whole answer: anything inside a
                             ~5 m x 2.2 m forward box drops it to a flat 0.60 m/s,
                             whether or not that person is on a collision course.
    2  OURS, AS TRAINED      measured to behave almost identically -- and that is the
                             finding, not a failure to render it. The policy pins its
                             lateral request at the bottom of its action box for the
                             whole encounter and squeezes past on the centreline.
    3  OURS, USING THE AISLE  the same stack, asking for the lateral room the walls
                             already say is there. It goes ROUND instead of slowing.

The honest reading, which the closing card carries: head-on is the geometry where this
project has repeatedly measured NO supervisor advantage, and panels 1 and 2 confirm it
again. What panel 3 shows is not learned behaviour -- it is the authority the supervisor
already has and does not use.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_headon_video.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_headon_video.py --still 9
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D

from core.common.platform import load_platform
from core.demo import aisle_scene as sc
from core.demo.industrial_amr import (COMMISSIONED, NORMAL, PROT_PAD, STOPPED,
                                      WARN_HALF_W, WARNING)

OUT = Path("experiments/results")
CACHE = OUT / "headon_traj.pkl"

C_FLOOR, C_AISLE = "#e9ecef", "#f7f8fa"
C_RACK, C_RACK_EDGE, C_PALLET = "#aeb7c2", "#79828f", "#c8a06a"
C_WALL = "#5a626d"
C_BOT, C_BOT_EDGE, C_LOAD = "#333d47", "#1d242b", "#c8a06a"
C_VEST, C_HEAD = "#f5a623", "#2f2f2f"
STATE_COL = {NORMAL: "#2e8b57", WARNING: "#e8a317", STOPPED: "#cc2b1d"}

ARMS = ("industrial", "ours", "ours_lateral")
ACCENT = {"industrial": "#2e8b57", "ours": "#1b6ca8", "ours_lateral": "#7b3fa0"}
TITLE = {"industrial": "INDUSTRIAL AMR",
         "ours": "OURS, AS TRAINED",
         "ours_lateral": "OURS, USING THE AISLE"}
SUBTITLE = {"industrial": "warning tier: flat 0.60 m/s while he is in the box",
            "ours": "learned supervisor, lateral request left at its floor",
            "ours_lateral": "same stack, asking for the room the walls already show"}

RACK_DEPTH = 0.75           # shallower than the commissioning scene: the aisle is wider,
                            # and the panel aspect is what pays for it


def _gate():
    spec = importlib.util.spec_from_file_location(
        "verify_headon", Path(__file__).resolve().parent / "verify_headon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw_aisle(ax, scene, goal_x, half_w, y_lo, y_hi):
    x_lo, x_hi = sc.X_MIN - 1.1, goal_x + 2.6
    ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fc=C_FLOOR,
                           ec="none", zorder=0))
    ax.add_patch(Rectangle((x_lo, -half_w), x_hi - x_lo, 2 * half_w, fc=C_AISLE,
                           ec="none", zorder=1))
    for face, out, row in ((half_w, +1, "N"), (-half_w, -1, "S")):
        depth = RACK_DEPTH * out
        ax.add_patch(Rectangle((x_lo, min(face, face + depth)), x_hi - x_lo, RACK_DEPTH,
                               fc=C_RACK, ec=C_RACK_EDGE, lw=0.8, zorder=2))
        n_bay = max(1, int(round((x_hi - x_lo) / 2.4)))
        w = (x_hi - x_lo) / n_bay
        for b in range(n_bay):
            bx = x_lo + b * w
            ax.plot([bx, bx], [face, face + depth], color=C_RACK_EDGE, lw=1.2, zorder=3)
            ax.add_patch(Rectangle((bx + 0.28 * w, face + 0.28 * depth), 0.44 * w,
                                   0.44 * abs(depth) * np.sign(out), fc=C_PALLET,
                                   ec="#8a6c46", lw=0.5, zorder=4))
        ax.plot([x_lo, x_hi], [face, face], color=C_WALL, lw=2.0, zorder=6)

    # the two halves of a two-way aisle, which is what makes stepping aside legitimate
    ax.plot([x_lo, x_hi], [0, 0], color="#c6cbd2", lw=1.0, ls=(0, (10, 8)), zorder=2)
    ax.text(0.6, -half_w + 0.26, "AISLE 07  -  TWO-WAY, SHARED WITH PEDESTRIANS",
            fontsize=6.0, color="#98a1ad", weight="bold", zorder=5)
    sx = goal_x + 0.55
    ax.add_patch(Rectangle((sx, -1.20), 1.25, 2.4, fc="#d8dde3", ec="#79828f", lw=1.0,
                           zorder=4))
    ax.text(sx + 0.62, -1.62, "PICK P-20", fontsize=5.6, color="#5a626d", ha="center",
            zorder=5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_anchor("E")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


class Panel:
    def __init__(self, ax, traj, plat, arm, fps, arrive_s, scene, y_lo, y_hi):
        self.ax, self.traj, self.plat, self.arm = ax, traj, plat, arm
        self.accent, self.fps, self.arrive_s = ACCENT[arm], fps, arrive_s
        self.goal_x = float(scene["goal"][0])
        self.half_w = plat.robot.robot_radius + PROT_PAD
        draw_aisle(ax, scene, self.goal_x, scene["half_w"], y_lo, y_hi)

        self.prot = Rectangle((0, 0), 0.1, 2 * self.half_w, fc=self.accent, alpha=0.18,
                              ec=self.accent, lw=1.2, zorder=7)
        ax.add_patch(self.prot)
        self.warn = None
        if arm == "industrial":
            self.warn = Rectangle((0, 0), 0.1, 2 * WARN_HALF_W, fc="none", ec="#e8a317",
                                  lw=1.1, ls=(0, (5, 3)), alpha=0.9, zorder=7)
            ax.add_patch(self.warn)

        self.trail, = ax.plot([], [], color=self.accent, lw=1.4, alpha=0.55, zorder=6)
        self.body = FancyBboxPatch((-0.45, -0.31), 0.90, 0.62,
                                   boxstyle="round,pad=0.015,rounding_size=0.10",
                                   fc=C_BOT, ec=C_BOT_EDGE, lw=1.0, zorder=10)
        self.load = Rectangle((-0.30, -0.22), 0.60, 0.44, fc=C_LOAD, ec="#8a6c46",
                              lw=0.7, zorder=11)
        self.strip = Rectangle((0.36, -0.26), 0.09, 0.52, fc=STATE_COL[NORMAL],
                               ec="none", zorder=12)
        for p in (self.body, self.load, self.strip):
            ax.add_patch(p)

        self.vest = Ellipse((0, 0), 0.62, 0.46, fc=C_VEST, ec="#8a5a10", lw=0.9,
                            zorder=10)
        self.head = Circle((0, 0), 0.115, fc=C_HEAD, ec="none", zorder=11)
        ax.add_patch(self.vest)
        ax.add_patch(self.head)
        self.wtrail, = ax.plot([], [], color="#c98a1e", lw=1.1, alpha=0.45, ls=(0, (3, 3)),
                               zorder=6)
        # the measurement the whole video is about: the gap at the moment of passing
        self.gapline, = ax.plot([], [], color="#cc2b1d", lw=1.3, zorder=13)
        self.gaptxt = ax.text(0, 0, "", fontsize=6.2, color="#cc2b1d", ha="center",
                              va="bottom", weight="bold", zorder=14)
        self.badge = ax.text(0.5, 0.06, "", transform=ax.transAxes, fontsize=10.5,
                             weight="bold", color="white", ha="center", zorder=15,
                             bbox=dict(facecolor=self.accent, edgecolor="none", pad=3))

    def build_sidebar(self, fig, rect):
        sx, sy, sw, sh = rect
        T = fig.transFigure
        X = lambda u: sx + u * sw
        Y = lambda v: sy + v * sh
        fig.add_artist(Rectangle((X(0.0), Y(0.10)), 0.0030, 0.86 * sh, transform=T,
                                 fc=self.accent, ec="none"))
        fig.text(X(0.05), Y(1.0), TITLE[self.arm], transform=T, fontsize=11,
                 weight="bold", color=self.accent, va="top")
        fig.text(X(0.05), Y(0.845), SUBTITLE[self.arm], transform=T, fontsize=7.2,
                 color="#6b7079", va="top")
        self.speed_txt = fig.text(X(0.05), Y(0.68), "", transform=T, fontsize=24,
                                  weight="bold", color=self.accent, va="top")
        fig.text(X(0.30), Y(0.60), "m/s", transform=T, fontsize=9.5, color="#6b7079",
                 va="top")
        self.bar_x, self.bar_y = X(0.05), Y(0.30)
        self.bar_w, self.bar_h = 0.42 * sw, 0.055 * sh
        fig.add_artist(Rectangle((self.bar_x, self.bar_y), self.bar_w, self.bar_h,
                                 transform=T, fc="#e4e7ea", ec="#c9ccd2", lw=0.8))
        self.bar = Rectangle((self.bar_x, self.bar_y), 0.0, self.bar_h, transform=T,
                             fc=self.accent, ec="none")
        fig.add_artist(self.bar)
        self.lamp_txt = fig.text(X(0.05), Y(0.20), "", transform=T, fontsize=7.6,
                                 weight="bold", va="center")
        self.readout = fig.text(X(0.52), Y(0.72), "", transform=T, fontsize=8.6,
                                family="monospace", color="#3c4048", va="top",
                                linespacing=1.55)

    def draw(self, k):
        f = self.traj[min(k, len(self.traj) - 1)]
        t_now = k / self.fps
        x, y, yaw, v = f["x"], f["y"], f["yaw"], f["v"]
        arrived = self.arrive_s is not None and t_now >= self.arrive_s - 1e-6
        rot = Affine2D().rotate_around(0, 0, yaw).translate(x, y) + self.ax.transData

        self.prot.set_bounds(-self.half_w, -self.half_w, f["prot"] + self.half_w,
                             2 * self.half_w)
        self.prot.set_transform(rot)
        if self.warn is not None:
            self.warn.set_bounds(-self.half_w, -WARN_HALF_W, f["warn"] + self.half_w,
                                 2 * WARN_HALF_W)
            self.warn.set_transform(rot)
        for p in (self.body, self.load, self.strip):
            p.set_transform(rot)
        self.strip.set_facecolor(STATE_COL.get(f["state"], self.accent))

        upto = min(k, len(self.traj) - 1) + 1
        self.trail.set_data([p["x"] for p in self.traj[:upto]],
                            [p["y"] for p in self.traj[:upto]])
        wx, wy, wyaw, _ = f["workers"][0]
        self.vest.set_center((wx, wy))
        self.vest.angle = np.degrees(wyaw)
        self.head.set_center((wx, wy))
        self.wtrail.set_data([p["workers"][0][0] for p in self.traj[:upto]],
                             [p["workers"][0][1] for p in self.traj[:upto]])

        # draw the separation while they are actually near each other
        if f["gap"] < 4.0:
            self.gapline.set_data([x, wx], [y, wy])
            self.gaptxt.set_position(((x + wx) / 2, (y + wy) / 2 + 0.18))
            self.gaptxt.set_text(f"{f['gap']:.2f} m")
        else:
            self.gapline.set_data([], [])
            self.gaptxt.set_text("")

        self.speed_txt.set_text(f"{v:.2f}")
        self.bar.set_width(self.bar_w * min(1.0, max(0.0, v / COMMISSIONED)))
        state = f["state"] if self.arm == "industrial" else "SUPERVISED"
        col = STATE_COL.get(f["state"], self.accent) if self.arm == "industrial" \
            else self.accent
        self.lamp_txt.set_text(state)
        self.lamp_txt.set_color(col)

        mg = f.get("margin")
        mgs = f"{mg:.2f} m" if mg is not None else "  n/a "
        hs = "  --  " if f["h"] is None else f"{f['h']:+.2f} m"
        self.readout.set_text(
            f"{'clock':<16}{t_now:5.1f} s\n"
            f"{'gap to picker':<16}{f['gap']:5.2f} m\n"
            f"{'lateral offset':<16}{f['lat']:+5.2f} m\n"
            f"{'lateral asked':<16}{mgs:>7s}\n"
            f"{'ISO margin h':<16}{hs:>7s}")
        self.badge.set_text(f"DELIVERED  {self.arrive_s:.1f} s" if arrived else "")
        return []


def build(fps, gate, plat, scene, cache=True, n_battery=4):
    if cache and CACHE.exists():
        runs, arrive, summary, batt = pickle.loads(CACHE.read_bytes())
        print(f"  (replayed from {CACHE})")
    else:
        from core.rl.supervisor import SupervisorPolicy
        sup = SupervisorPolicy(gate.MODEL, platform="industrial",
                               walls=scene["walls"], posts=scene["posts"])
        runs, arrive, summary = {}, {}, {}
        for arm in ARMS:
            rec = []
            res = gate.run(arm, plat, scene,
                           sup=sup if arm.startswith("ours") else None, record=rec)
            runs[arm], arrive[arm], summary[arm] = rec, res["t"], res
        print(f"  running the {n_battery}-presentation battery ...")
        batt = gate.battery(n_battery, plat, scene, sup)
        CACHE.write_bytes(pickle.dumps((runs, arrive, summary, batt)))
    for arm in ARMS:
        r = summary[arm]
        print(f"  {arm:<14} {r['t']:5.1f} s   gap {r['min_d']:.2f} m   "
              f"v@pass {r['v_at_pass']:.2f}   lateral {r['lat_max']:.2f} m   "
              f"min_h {r['min_h']:+.2f}")
    dt = plat.robot.dt
    n_live = int(max(len(runs[a]) for a in ARMS) * dt * fps) + 1
    out = {}
    for arm, rec in runs.items():
        idx = np.clip((np.arange(n_live) / fps / dt).astype(int), 0, len(rec) - 1)
        out[arm] = [rec[i] for i in idx]
    return out, arrive, summary, batt, n_live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--card", type=float, default=5.0)
    ap.add_argument("--name", default="headon_demo")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--battery", type=int, default=4)
    args = ap.parse_args()

    gate = _gate()
    plat = load_platform("industrial")
    scene = gate.build_scene()
    hw = scene["half_w"]
    y_lo, y_hi = -(hw + RACK_DEPTH + 0.25), hw + RACK_DEPTH + 0.95

    print("replaying all three machines through the real MPC + CBF + scanner stack ...")
    traj, arrive, summary, batt, n_live = build(args.fps, gate, plat, scene,
                                                cache=not args.fresh,
                                                n_battery=args.battery)
    fig, axes = plt.subplots(3, 1, figsize=(16.0, 9.0), gridspec_kw=dict(hspace=0.10))
    fig.patch.set_facecolor("white")
    panels = [Panel(axes[i], traj[a], plat, a, args.fps, arrive[a], scene, y_lo, y_hi)
              for i, a in enumerate(ARMS)]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("position along the aisle   x   [m]", fontsize=8.5)

    fig.suptitle("A picker walks the other way down a shared 5.0 m aisle - "
                 "slow down, or step aside?",
                 fontsize=12.5, weight="bold", x=0.010, ha="left", y=0.985)
    fig.text(0.010, 0.014,
             "Solid rectangle = protective field. Dashed amber = the warning field, "
             "which only the industrial machine carries. Red line = live separation to "
             "the picker.\n"
             "'lateral asked' is the supervisor's own d_margin action, whose range is "
             "0.1-2.0 m. Head-on is a geometry where the anticipation comes from the MPC "
             "and CBF, not the supervisor.",
             fontsize=8.0, family="monospace", va="bottom", color="#333")
    fig.subplots_adjust(left=0.030, right=0.988, top=0.930, bottom=0.100)

    fig.canvas.draw()
    for p in panels:
        bb = p.ax.get_window_extent().transformed(fig.transFigure.inverted())
        p.build_sidebar(fig, (0.010, bb.y0, bb.x0 - 0.026, bb.height))

    card = _card(fig, summary, batt, scene)

    def frame(k):
        kk = min(k, n_live - 1)
        for p in panels:
            p.draw(kk)
        for a in card:
            a.set_visible(k >= n_live)
        return []

    if args.still is not None:
        frame(int(args.still * args.fps))
        path = OUT / f"{args.name}_t{args.still:g}.png"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")
        return

    n = n_live + int(args.card * args.fps)
    anim = animation.FuncAnimation(fig, frame, frames=n, blit=False,
                                   interval=1000 / args.fps)
    mp4 = OUT / f"{args.name}.mp4"
    anim.save(mp4, writer=animation.FFMpegWriter(fps=args.fps, bitrate=4600), dpi=120)
    print(f"wrote {mp4}  ({n} frames, {n / args.fps:.1f} s)")


def _card(fig, summary, batt, scene):
    L, C = 0.062, (0.470, 0.640, 0.800)
    art = [fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc="white",
                                    ec="none", zorder=40))]
    art.append(fig.text(L, 0.885, "Going round beats squeezing past.",
                        fontsize=30, weight="bold", color="#22252a", zorder=41))
    art.append(fig.text(L, 0.828,
                        f"{2 * scene['half_w']:.1f} m shared aisle, one oncoming picker, "
                        "identical protective field on all three machines",
                        fontsize=11.5, color="#6b7079", zorder=41))
    art.append(fig.add_artist(Rectangle((L, 0.782), 0.86, 0.003,
                                        transform=fig.transFigure, fc="#d5d8dd",
                                        ec="none", zorder=41)))
    for i, a in enumerate(ARMS):
        art.append(fig.text(C[i], 0.733, TITLE[a].replace(", ", ",\n"), ha="center",
                            fontsize=9.5, weight="bold", color=ACCENT[a],
                            linespacing=1.3, zorder=41))
    rows = [("passing clearance", "min_d", "{:.2f} m"),
            ("speed at the pass", "v_at_pass", "{:.2f} m/s"),
            ("lateral offset used", "lat_max", "{:.2f} m"),
            ("ISO barrier margin", "min_h", "{:+.2f} m"),
            ("mission time", "t", "{:.1f} s")]
    for r, (label, key, fmt) in enumerate(rows):
        yy = 0.645 - r * 0.077
        art.append(fig.text(L, yy, label, fontsize=12.5, color="#3c4048", zorder=41))
        for i, a in enumerate(ARMS):
            art.append(fig.text(C[i], yy, fmt.format(batt[a][key]), ha="center",
                                fontsize=14, weight="bold", color=ACCENT[a], zorder=41))
    art.append(fig.text(
        L, 0.230,
        f"Over {batt['ours']['n']} presentations. As trained, ours is the industrial "
        "machine: it slows to a crawl and passes on the centreline, using 0.07 m of a "
        "2.5 m half-aisle.\n"
        "Asked for the lateral room the walls already show, the SAME stack passes at "
        f"{batt['ours_lateral']['min_d']:.2f} m without dropping below "
        f"{batt['ours_lateral']['v_at_pass']:.2f} m/s -- more clearance, more margin, "
        "and quicker.",
        fontsize=11.5, color="#555", linespacing=1.7, va="top", zorder=41))
    art.append(fig.text(
        L, 0.125,
        "Read this as a gap in the POLICY, not a win for it. Head-on is the geometry "
        "where this project has repeatedly measured no supervisor advantage -- the "
        "slowdown is the MPC's\nhuman-cost term plus the CBF and survives deleting the "
        "supervisor, and panels 1 and 2 confirm it again. The third panel is authority "
        "the supervisor already has in its action space\n(d_margin, 0.1-2.0 m) and pins "
        "at the floor for the whole encounter. Closing that is a training question, not "
        "an architecture one. 2D simulation, same control stack as the Gazebo build.",
        fontsize=9.5, color="#8a919b", linespacing=1.7, va="top", zorder=41))
    for a in art:
        a.set_visible(False)
    return art


if __name__ == "__main__":
    main()
