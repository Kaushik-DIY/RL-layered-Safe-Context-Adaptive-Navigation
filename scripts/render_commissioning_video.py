"""The presentation video: what a site integrator has to configure, and what the
learned supervisor works out for itself.

Both machines on this route are ISO 3691-4 compliant, take zero protective stops and
never touch anybody. So there is no red flash to watch and no race to win -- the
argument is what is DRAWN ON THE MAP.

    top     COMMISSIONED INDUSTRIAL AMR   scanner + hand-marked speed zones.
            Its map is covered in engineering: a reduced-speed zone surveyed and
            entered for every cross-aisle, two field sets sized from eleven measured
            quantities. Change the layout and all of it is re-derived and re-validated.
    bottom  OURS                          the SAME map, with nothing on it. Same
            mandatory protective field, no warning tier, no zones. The corner comes
            out of the map geometry the robot already has -- `post_ahead`, the
            along-heading distance to the next mapped constriction -- and the policy
            decides what to do about it.

READ THE FIELDS CORRECTLY. The protective rectangle is the room the machine needs to
stop from the speed it is doing, so a SMALL field means a SLOW robot, not a safe one.
Both machines carry the identical one; only the top machine also drags a warning field.

All numbers on screen come from `scripts/verify_commissioning.py`, which is the gate
and must pass before this is rendered.

    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_commissioning_video.py
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_commissioning_video.py --still 18
    PYTHONPATH=$PWD .venv-navrl/bin/python scripts/render_commissioning_video.py --gif
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
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D

from core.common.platform import load_platform
from core.demo import aisle_scene as sc
from core.demo.industrial_amr import (COMMISSIONED, PROT_PAD, WARN_HALF_W, NORMAL,
                                      STOPPED, WARNING)
from core.demo.site_zones import APPROACH_MARGIN

OUT = Path("experiments/results")

# --- palette: a fleet console. One accent per machine, held for the whole video. ---
C_FLOOR, C_AISLE = "#e9ecef", "#f7f8fa"
C_RACK, C_RACK_EDGE, C_PALLET = "#aeb7c2", "#79828f", "#c8a06a"
C_WALL = "#5a626d"
C_BOT, C_BOT_EDGE, C_LOAD = "#333d47", "#1d242b", "#c8a06a"
C_VEST, C_HEAD = "#f5a623", "#2f2f2f"
C_ENG = "#b3392b"          # everything an engineer had to put on the map
C_DERIVED = "#1b6ca8"      # everything the robot worked out for itself

ACCENT = {"commissioned": "#2e8b57", "ours": "#1b6ca8"}
TITLE = {"commissioned": "COMMISSIONED INDUSTRIAL AMR",
         "ours": "MPC + CBF + LEARNED SUPERVISOR"}
SUBTITLE = {"commissioned": "scanner field sets + hand-marked speed zones",
            "ours": "same map, nothing marked on it"}
STATE_COL = {NORMAL: "#2e8b57", WARNING: "#e8a317", STOPPED: "#cc2b1d"}

RACK_DEPTH = 1.15
# The aspect-equal panel is height-limited, so the map's WIDTH is (x-range / y-range) x
# panel height and a wider map eats the gutter the instruments live in. Do not widen
# either range without re-checking the sidebar.
Y_LO, Y_HI = -5.1, 5.3       # headroom above the side aisles for the station callouts


def _gate():
    spec = importlib.util.spec_from_file_location(
        "verify_commissioning", Path(__file__).resolve().parent / "verify_commissioning.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- warehouse
def _rack_spans(walls):
    """Solid runs along each side of the main aisle, taken FROM the collision geometry
    so the racking drawn is exactly the racking the planner avoids."""
    north, south = [], []
    for x1, y1, x2, y2 in walls:
        if abs(y1 - y2) < 1e-9 and abs(abs(y1) - sc.HALF_W) < 1e-9:
            (north if y1 > 0 else south).append((min(x1, x2), max(x1, x2)))
    return north, south


def _draw_rack_run(ax, x0, x1, y_face, outward, label_row):
    depth = RACK_DEPTH * outward
    ax.add_patch(Rectangle((x0, min(y_face, y_face + depth)), x1 - x0, RACK_DEPTH,
                           fc=C_RACK, ec=C_RACK_EDGE, lw=0.8, zorder=2))
    n_bay = max(1, int(round((x1 - x0) / 2.4)))
    w = (x1 - x0) / n_bay
    for b in range(n_bay):
        bx = x0 + b * w
        ax.plot([bx, bx], [y_face, y_face + depth], color=C_RACK_EDGE, lw=1.3, zorder=3)
        for slot in (0.30, 0.62):
            ax.add_patch(Rectangle((bx + slot * w, y_face + 0.30 * depth), 0.26 * w,
                                   0.42 * abs(depth) * np.sign(outward),
                                   fc=C_PALLET, ec="#8a6c46", lw=0.5, zorder=4))
        if w > 1.8 and label_row is not None:
            ax.text(bx + w / 2, y_face + depth * 0.90, f"{label_row}-{b + 1:02d}",
                    ha="center", va="center", fontsize=4.4, color="#5c6470", zorder=5)
    ax.plot([x0, x1], [y_face, y_face], color=C_WALL, lw=2.0, solid_capstyle="butt",
            zorder=6)


def draw_warehouse(ax, scene, goal_x):
    """The static site. Nothing decorative is drawn where the robot can drive."""
    x_lo, x_hi = sc.X_MIN - 1.1, goal_x + 2.6
    ax.add_patch(Rectangle((x_lo, Y_LO), x_hi - x_lo, Y_HI - Y_LO, fc=C_FLOOR,
                           ec="none", zorder=0))
    ax.add_patch(Rectangle((x_lo, -sc.HALF_W), x_hi - x_lo, 2 * sc.HALF_W, fc=C_AISLE,
                           ec="none", zorder=1))

    north, south = _rack_spans(scene["walls"])
    for a, b in north:
        _draw_rack_run(ax, a, b, sc.HALF_W, +1, "N")
    for a, b in south:
        _draw_rack_run(ax, a, b, -sc.HALF_W, -1, "S")
    for x1, y1, x2, y2 in scene["walls"]:                # side-aisle faces
        if abs(x1 - x2) < 1e-9:
            ax.plot([x1, x2], [y1, y2], color=C_WALL, lw=2.0, zorder=6)

    ax.plot([x_lo, x_hi], [0, 0], color="#c6cbd2", lw=0.9, ls=(0, (9, 9)), zorder=2)

    # charging dock behind the start, pick station BEYOND the goal -- the AMR pulls up
    # to it, so nothing decorative sits anywhere the robot's path actually goes
    ax.add_patch(Rectangle((sc.X_MIN - 1.0, -0.75), 0.75, 1.5, fc="#5a626d",
                           ec="#3b424b", lw=0.8, zorder=4))
    ax.text(sc.X_MIN - 0.62, 1.00, "CHARGE", fontsize=5.4, color="#5a626d",
            ha="center", zorder=5)
    sx = goal_x + 0.55
    ax.add_patch(Rectangle((sx, -1.30), 1.30, 2.6, fc="#d8dde3", ec="#79828f",
                           lw=1.0, zorder=4))
    for i in range(4):
        ax.plot([sx + 0.20 + i * 0.29] * 2, [-1.15, 1.15], color="#a8b0ba", lw=1.3,
                zorder=5)
    ax.text(sx + 0.65, -1.72, "PICK P-12", fontsize=5.6, color="#5a626d", ha="center",
            zorder=5)
    ax.text(0.6, -sc.HALF_W + 0.28, "AISLE 04", fontsize=6.2, color="#98a1ad",
            weight="bold", zorder=5)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_aspect("equal")
    ax.set_anchor("E")            # collect the slack into one left gutter
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def draw_commissioning(ax, zones, plat):
    """The engineering the top machine needs and the bottom machine does not: a marked
    reduced-speed zone per cross-aisle, plus the survey each one rests on. Drawn once,
    statically, because that is exactly what it is -- work done before the robot ever
    moves."""
    # The approach: a zoned AMR has to be AT the marked limit when it crosses the line,
    # so it starts shedding speed before it. Drawn lighter because it is not part of what
    # the integrator marks -- it is what the vehicle has to do about what was marked.
    a_dec = APPROACH_MARGIN * plat.robot.a_max_mpc
    lead = (COMMISSIONED ** 2 - zones[0].v ** 2) / (2.0 * a_dec) if zones else 0.0
    for z in zones:
        ax.add_patch(Rectangle((z.x0 - lead, -sc.HALF_W), lead, 2 * sc.HALF_W,
                               fc=C_ENG, alpha=0.05, ec="none", zorder=2))
        ax.plot([z.x0 - lead, z.x0 - lead], [-sc.HALF_W, sc.HALF_W], color=C_ENG,
                lw=0.8, alpha=0.45, ls=(0, (2, 3)), zorder=3)

    for z in zones:
        ax.add_patch(Rectangle((z.x0, -sc.HALF_W), z.x1 - z.x0, 2 * sc.HALF_W,
                               fc=C_ENG, alpha=0.13, ec="none", zorder=2))
        for e in (z.x0, z.x1):
            ax.plot([e, e], [-sc.HALF_W, sc.HALF_W], color=C_ENG, lw=1.4, alpha=0.9,
                    ls=(0, (5, 3)), zorder=3)
        ax.text((z.x0 + z.x1) / 2, sc.HALF_W - 0.30,
                f"ZONE {z.label}   {z.v:.2f} m/s\n"
                f"surveyed sight line {sc.REVEAL_DISTANCE:.2f} m",
                ha="center", va="top", fontsize=5.4, color=C_ENG, weight="bold",
                linespacing=1.25, zorder=5)
        # the extent an engineer had to establish, and mark on the floor
        ax.annotate("", xy=(z.x0, -sc.HALF_W + 0.30), xytext=(z.x1, -sc.HALF_W + 0.30),
                    arrowprops=dict(arrowstyle="<->", color=C_ENG, lw=0.8), zorder=5)
        ax.text((z.x0 + z.x1) / 2, -sc.HALF_W + 0.42, f"{z.x1 - z.x0:.2f} m",
                ha="center", va="bottom", fontsize=5.0, color=C_ENG, zorder=5)


# ------------------------------------------------------------------------- panel
class Panel:
    """One machine: site layer inside the axes, instruments out in the figure gutter."""

    def __init__(self, ax, traj, plat, arm, fps, arrive_s, scene, zones):
        self.ax, self.traj, self.plat, self.arm = ax, traj, plat, arm
        self.accent, self.fps, self.arrive_s = ACCENT[arm], fps, arrive_s
        self.goal_x = float(scene["goal"][0])
        draw_warehouse(ax, scene, self.goal_x)
        if arm == "commissioned":
            draw_commissioning(ax, zones, plat)

        # Scanner fields are forward RECTANGLES in the model (`IndustrialAMR._occupied`),
        # so they are drawn as rectangles. A disc would be a prettier lie.
        self.half_w = plat.robot.robot_radius + PROT_PAD
        self.prot = Rectangle((0, 0), 0.1, 2 * self.half_w, fc=self.accent, alpha=0.18,
                              ec=self.accent, lw=1.3, zorder=7)
        ax.add_patch(self.prot)
        self.warn = None
        if arm == "commissioned":
            self.warn = Rectangle((0, 0), 0.1, 2 * WARN_HALF_W, fc="none", ec="#e8a317",
                                  lw=1.2, ls=(0, (5, 3)), alpha=0.9, zorder=7)
            ax.add_patch(self.warn)
        self.prot_txt = ax.text(0, 0, "", fontsize=5.2, color=self.accent, ha="left",
                                va="center", zorder=13)

        # ours only: the corner it inferred from the map, drawn as the policy measures it
        self.sight, = ax.plot([], [], color=C_DERIVED, lw=1.3, ls=(0, (2, 2)), zorder=8)
        self.sight_txt = ax.text(0, 0, "", fontsize=5.4, color=C_DERIVED, ha="center",
                                 va="bottom", weight="bold", zorder=13)

        self.trail, = ax.plot([], [], color=self.accent, lw=1.2, alpha=0.5, zorder=6)
        self.body = FancyBboxPatch((-0.45, -0.31), 0.90, 0.62,
                                   boxstyle="round,pad=0.015,rounding_size=0.10",
                                   fc=C_BOT, ec=C_BOT_EDGE, lw=1.0, zorder=10)
        self.load = Rectangle((-0.30, -0.22), 0.60, 0.44, fc=C_LOAD, ec="#8a6c46",
                              lw=0.7, zorder=11)
        self.strip = Rectangle((0.36, -0.26), 0.09, 0.52, fc=STATE_COL[NORMAL],
                               ec="none", zorder=12)
        for p in (self.body, self.load, self.strip):
            ax.add_patch(p)

        n_w = max((len(f["workers"]) for f in traj), default=0)
        self.w_vest, self.w_head, self.w_arrow = [], [], []
        for _ in range(n_w):
            v = Ellipse((0, 0), 0.62, 0.46, fc=C_VEST, ec="#8a5a10", lw=0.9, zorder=10)
            hd = Circle((0, 0), 0.115, fc=C_HEAD, ec="none", zorder=11)
            ar, = ax.plot([], [], color="#8a5a10", lw=1.1, zorder=10)
            ax.add_patch(v)
            ax.add_patch(hd)
            self.w_vest.append(v)
            self.w_head.append(hd)
            self.w_arrow.append(ar)

        self.badge = ax.text(0.5, 0.055, "", transform=ax.transAxes, fontsize=12,
                             weight="bold", color="white", ha="center", zorder=15,
                             bbox=dict(facecolor=self.accent, edgecolor="none", pad=4))

    # ------------------------------------------------------------------ sidebar
    def build_sidebar(self, fig, rect, n_params):
        sx, sy, sw, sh = rect
        T = fig.transFigure
        X = lambda u: sx + u * sw
        Y = lambda v: sy + v * sh
        fig.add_artist(Rectangle((X(0.0), Y(0.26)), 0.0032, 0.74 * sh, transform=T,
                                 fc=self.accent, ec="none"))
        fig.text(X(0.06), Y(1.0), TITLE[self.arm], transform=T, fontsize=10.5,
                 weight="bold", color=self.accent, va="top")
        fig.text(X(0.06), Y(0.935), SUBTITLE[self.arm], transform=T, fontsize=7.2,
                 color="#6b7079", va="top")

        # the headline of the whole video: how much of this machine is hand-set
        col = C_ENG if n_params else C_DERIVED
        fig.text(X(0.06), Y(0.875), f"{n_params}", transform=T, fontsize=17,
                 weight="bold", color=col, va="top")
        fig.text(X(0.06), Y(0.800),
                 "site parameters\nconfigured by hand" if n_params
                 else "nothing configured\nfor this site",
                 transform=T, fontsize=7.0, color=col, va="top", linespacing=1.25)

        cy0, cy1, cw = 0.455, 0.735, 0.72
        fig.add_artist(Rectangle((X(0.06), Y(cy0)), cw * sw, (cy1 - cy0) * sh,
                                 transform=T, fc="#f7f8fa", ec="#c9ccd2", lw=1.0))
        fig.text(X(0.10), Y(cy1 - 0.018), "SPEED", transform=T, fontsize=7.5,
                 color="#6b7079", weight="bold", va="top")
        self.speed_txt = fig.text(X(0.10), Y(cy1 - 0.050), "", transform=T, fontsize=26,
                                  weight="bold", color=self.accent, va="top")
        fig.text(X(0.41), Y(cy1 - 0.090), "m/s", transform=T, fontsize=10.5,
                 color="#6b7079", va="top")
        self.bar_x, self.bar_y = X(0.10), Y(cy0 + 0.056)
        self.bar_w, self.bar_h = (cw - 0.09) * sw, 0.034 * sh
        fig.add_artist(Rectangle((self.bar_x, self.bar_y), self.bar_w, self.bar_h,
                                 transform=T, fc="#e4e7ea", ec="#c9ccd2", lw=0.8))
        self.bar = Rectangle((self.bar_x, self.bar_y), 0.0, self.bar_h, transform=T,
                             fc=self.accent, ec="none")
        fig.add_artist(self.bar)
        self.cap_txt = fig.text(X(0.10), Y(cy0 + 0.014), "", transform=T, fontsize=7.4,
                                family="monospace", color="#3c4048", va="bottom")

        self.lamp = Rectangle((X(0.06), Y(0.383)), 0.028 * sw, 0.046 * sh, transform=T,
                              fc=STATE_COL[NORMAL], ec="none")
        fig.add_artist(self.lamp)
        self.lamp_txt = fig.text(X(0.16), Y(0.406), "", transform=T, fontsize=7.8,
                                 weight="bold", color="#3c4048", va="center")
        self.readout = fig.text(X(0.06), Y(0.338), "", transform=T, fontsize=8.2,
                                family="monospace", color="#3c4048", va="top",
                                linespacing=1.5)

        self.pb_x, self.pb_y = X(0.06), Y(0.088)
        self.pb_w, self.pb_h = 0.88 * sw, 0.026 * sh
        fig.add_artist(Rectangle((self.pb_x, self.pb_y), self.pb_w, self.pb_h,
                                 transform=T, fc="#e4e7ea", ec="#c9ccd2", lw=0.8))
        self.pbar = Rectangle((self.pb_x, self.pb_y), 0.0, self.pb_h, transform=T,
                              fc=self.accent, ec="none")
        fig.add_artist(self.pbar)
        self.pb_txt = fig.text(X(0.06), Y(0.048), "", transform=T, fontsize=7.4,
                               family="monospace", color="#3c4048", va="top")

    # --------------------------------------------------------------------- draw
    def draw(self, k):
        f = self.traj[min(k, len(self.traj) - 1)]
        t_now = k / self.fps
        x, y, yaw, v = f["x"], f["y"], f["yaw"], f["v"]
        # arrive_s accumulates as k*dt, so an exact >= misses its own frame
        arrived = self.arrive_s is not None and t_now >= self.arrive_s - 1e-6
        rot = Affine2D().rotate_around(0, 0, yaw).translate(x, y) + self.ax.transData

        self.prot.set_bounds(-self.half_w, -self.half_w, f["prot"] + self.half_w,
                             2 * self.half_w)
        self.prot.set_transform(rot)
        # parked just past the nose of the field, below the centre line: clear of the
        # zone labels above the band and of the zone dimension below it. Near the goal
        # there is no room left ahead, so it flips behind the robot rather than being
        # clipped off the edge of the map.
        nose = x + (f["prot"] + self.half_w + 0.20) * np.cos(yaw)
        if nose > self.ax.get_xlim()[1] - 3.2:
            self.prot_txt.set_position((x - self.half_w - 0.20, y - 0.62))
            self.prot_txt.set_ha("right")
        else:
            self.prot_txt.set_position((nose, y - 0.62))
            self.prot_txt.set_ha("left")
        self.prot_txt.set_text(f"protective {f['prot']:.2f} m")
        if self.warn is not None:
            self.warn.set_bounds(-self.half_w, -WARN_HALF_W, f["warn"] + self.half_w,
                                 2 * WARN_HALF_W)
            self.warn.set_transform(rot)

        if self.arm == "ours":
            pa = f["post_ahead"]
            if pa < 9.99:
                self.sight.set_data([x, x + pa * np.cos(yaw)],
                                    [y, y + pa * np.sin(yaw)])
                self.sight_txt.set_position((x + pa / 2 * np.cos(yaw), y - 1.20))
                fl = f.get("v_floor")
                extra = f"   ->  {fl:.2f} m/s" if fl is not None else ""
                self.sight_txt.set_text(
                    f"post_ahead {pa:.2f} m  (from the map){extra}")
            else:
                self.sight.set_data([], [])
                self.sight_txt.set_text("")

        for p in (self.body, self.load, self.strip):
            p.set_transform(rot)
        self.strip.set_facecolor(STATE_COL.get(f["state"], self.accent))

        upto = min(k, len(self.traj) - 1) + 1
        self.trail.set_data([p["x"] for p in self.traj[:upto]],
                            [p["y"] for p in self.traj[:upto]])

        for i, vest in enumerate(self.w_vest):
            if i >= len(f["workers"]):
                vest.set_alpha(0.0)
                self.w_head[i].set_alpha(0.0)
                self.w_arrow[i].set_alpha(0.0)
                continue
            wx, wy, wyaw, seen = f["workers"][i]
            vest.set_center((wx, wy))
            vest.angle = np.degrees(wyaw)
            self.w_head[i].set_center((wx, wy))
            vest.set_alpha(1.0 if seen else 0.28)
            self.w_head[i].set_alpha(1.0 if seen else 0.28)
            self.w_arrow[i].set_data([wx, wx + 0.45 * np.cos(wyaw)],
                                     [wy, wy + 0.45 * np.sin(wyaw)])
            self.w_arrow[i].set_alpha(0.9 if seen else 0.25)

        self.speed_txt.set_text(f"{v:.2f}")
        self.bar.set_width(self.bar_w * min(1.0, max(0.0, v / COMMISSIONED)))
        self.cap_txt.set_text(f"limit {f['cap']:.2f}   site max {COMMISSIONED:.2f}")

        if self.arm == "ours":
            state, col = "SUPERVISED", self.accent
        else:
            state, col = f["state"], STATE_COL[f["state"]]
            if f["state"] == NORMAL and f["zone_cap"] is not None \
                    and f["zone_cap"] < COMMISSIONED - 1e-9:
                state, col = "IN MARKED ZONE", C_ENG
        self.lamp.set_facecolor(col)
        self.lamp_txt.set_text(state)
        self.lamp_txt.set_color(col)

        hs = "  --  " if f["h"] is None else f"{f['h']:+.2f} m"
        self.readout.set_text(
            f"{'mission clock':<16}{t_now:5.1f} s\n"
            f"{'protective field':<16}{f['prot']:5.2f} m\n"
            f"{'ISO margin h':<16}{hs:>7s}\n"
            f"{'standstill':<16}{f['stopped_s']:5.1f} s")

        done = 1.0 if arrived else float(np.clip(1.0 - f["dist_goal"] / self.goal_x,
                                                 0.0, 1.0))
        self.pbar.set_width(self.pb_w * done)
        if arrived:
            self.pb_txt.set_text(f"DELIVERED at {self.arrive_s:.1f} s   "
                                 f"idle {t_now - self.arrive_s:4.1f} s")
            self.badge.set_text(f"DELIVERED  {self.arrive_s:.1f} s")
        else:
            self.pb_txt.set_text(f"route {100 * done:4.1f} %   "
                                 f"{f['dist_goal']:4.1f} m to go")
            self.badge.set_text("")
        return []


# -------------------------------------------------------------------------- main
CACHE = OUT / "commissioning_traj.pkl"


def build(fps, gate, plat, scene, cache=True, n_battery=6):
    """Replay every arm through the real MPC/CBF/scanner stack, then resample onto a
    common wall clock so the two panels stay honest about elapsed time.

    The replay is deterministic, so it is cached -- layout work must never be a reason
    to re-run the controllers, and re-running them must never be a reason to skip
    layout work. Delete the cache (or pass --fresh) after touching anything upstream.
    """
    import pickle
    if cache and CACHE.exists():
        runs, arrive, summary, batt = pickle.loads(CACHE.read_bytes())
        print(f"  (replayed from {CACHE})")
    else:
        sup_mod = __import__("core.rl.supervisor", fromlist=["SupervisorPolicy"])
        sup = sup_mod.SupervisorPolicy(gate.MODEL, platform="industrial",
                                       walls=scene["walls"], posts=scene["posts"])
        runs, arrive, summary = {}, {}, {}
        for arm in gate.ARMS:
            rec = []
            res = gate.run(arm, plat, scene, sup=sup if arm == "ours" else None,
                           record=rec)
            runs[arm], arrive[arm], summary[arm] = rec, res["t"], res
        # the nominal run is not allowed to speak for the distribution
        print(f"  running the {n_battery}-presentation battery for the card ...")
        batt = gate.battery(n_battery, plat, scene, sup)
        CACHE.write_bytes(pickle.dumps((runs, arrive, summary, batt)))
    for arm in gate.ARMS:
        res = summary[arm]
        print(f"  {arm:<13} {res['t']:5.1f} s   protective stops {res['pstops']}   "
              f"contacts {res['contacts']}   min_h {res['min_h']:+.2f}   "
              f"viol {res['viol']}")
    dt = plat.robot.dt
    n_live = int(max(len(runs[a]) for a in ("commissioned", "ours")) * dt * fps) + 1
    out = {}
    for arm, rec in runs.items():
        idx = np.clip((np.arange(n_live) / fps / dt).astype(int), 0, len(rec) - 1)
        out[arm] = [rec[i] for i in idx]
    return out, arrive, summary, batt, n_live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--card", type=float, default=5.0, help="seconds of summary card")
    ap.add_argument("--name", default="commissioning_demo")
    ap.add_argument("--fresh", action="store_true", help="ignore the replay cache")
    ap.add_argument("--battery", type=int, default=6,
                    help="presentations behind the closing card's spread line")
    args = ap.parse_args()

    gate = _gate()
    plat = load_platform("industrial")
    scene = gate.build_scene()
    zones = gate.site_zones(plat)
    n_params = len(gate.commissioning_ledger(plat))

    print("replaying all three machines through the real MPC + CBF + scanner stack ...")
    traj, arrive, summary, batt, n_live = build(args.fps, gate, plat, scene,
                                                cache=not args.fresh,
                                                n_battery=args.battery)

    n_card = int(args.card * args.fps)
    fig, axes = plt.subplots(2, 1, figsize=(16.0, 9.0), gridspec_kw=dict(hspace=0.06))
    fig.patch.set_facecolor("white")
    p_i = Panel(axes[0], traj["commissioned"], plat, "commissioned", args.fps,
                arrive["commissioned"], scene, zones)
    p_o = Panel(axes[1], traj["ours"], plat, "ours", args.fps, arrive["ours"],
                scene, zones)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("position along the aisle   x   [m]", fontsize=8.5)
    # station callouts sit ABOVE the side-aisle walls (which stop at AISLE_TOP), so
    # they never land on the geometry they are describing
    for ax in axes:
        for k, name in enumerate(gate.STATION_LABEL):
            ax.text(gate.STATION_X[k], Y_HI - 0.10, name, ha="center", va="top",
                    fontsize=6.6, color="#6b7079", linespacing=1.2)

    fig.suptitle("Same warehouse, same workers, same 31 m transport run - "
                 "the difference is how each machine knows where to slow down",
                 fontsize=12.5, weight="bold", x=0.010, ha="left", y=0.987)
    fig.text(0.010, 0.016,
             "Solid rectangle = protective field, the room the machine needs to stop "
             "from the speed it is doing (ISO 13855). BOTH machines carry the identical "
             "one. A SMALLER FIELD MEANS A SLOWER ROBOT, not a safer one.\n"
             "Dashed amber = the warning field, which only the commissioned machine "
             "carries. Red = engineering an integrator had to survey and enter for this "
             "site. Blue = read off the map the robot already had.\n"
             "Every machine here is limited to its real acceleration, so speed limits "
             "are ramped into, never stepped: the pale band before each zone is the "
             "run-up the commissioned AMR needs to be AT the marked limit on the line.",
             fontsize=8.0, family="monospace", va="bottom", color="#333")
    fig.subplots_adjust(left=0.030, right=0.988, top=0.915, bottom=0.108)

    fig.canvas.draw()
    for panel, npar in ((p_i, n_params), (p_o, 0)):
        bb = panel.ax.get_window_extent().transformed(fig.transFigure.inverted())
        panel.build_sidebar(fig, (0.010, bb.y0, bb.x0 - 0.026, bb.height), npar)

    card = _summary_card(fig, arrive, summary, batt, n_params)

    def frame(k):
        kk = min(k, n_live - 1)
        p_i.draw(kk)
        p_o.draw(kk)
        for a in card:
            a.set_visible(k >= n_live)
        return []

    if args.still is not None:
        frame(int(args.still * args.fps))
        path = OUT / f"{args.name}_t{args.still:g}.png"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")
        return

    n = n_live + n_card
    anim = animation.FuncAnimation(fig, frame, frames=n, blit=False,
                                   interval=1000 / args.fps)
    mp4 = OUT / f"{args.name}.mp4"
    anim.save(mp4, writer=animation.FFMpegWriter(fps=args.fps, bitrate=4600), dpi=120)
    print(f"wrote {mp4}  ({n} frames, {n / args.fps:.1f} s)")
    if args.gif:
        gif = OUT / f"{args.name}.gif"
        anim.save(gif, writer=animation.PillowWriter(fps=min(args.fps, 12)), dpi=80)
        print(f"wrote {gif}")


def _summary_card(fig, arrive, summary, batt, n_params):
    """Shown over the final freeze frame.

    The table rows are the run just watched. The spread line underneath is the
    `--battery` presentations, and it is what stops a good nominal run implying more
    than it should -- ours does take the occasional protective stop, and the card says
    so. The cost figures are industry context and are labelled as such.
    """
    ti, to = arrive["commissioned"], arrive["ours"]
    cost = 100.0 * (to - ti) / ti
    bi, bo = batt["commissioned"], batt["ours"]
    batt_cost = 100.0 * (bo["t"] - bi["t"]) / bi["t"]
    L, CA, CC, CN = 0.062, 0.470, 0.640, 0.735
    art = [fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc="white",
                                    ec="none", zorder=40))]
    art.append(fig.text(L, 0.880, "Same time. Same compliance. No commissioning.",
                        fontsize=29, weight="bold", color="#22252a", zorder=41))
    art.append(fig.text(L, 0.824,
                        "31 m transport run, three cross-aisles, identical workers and "
                        "identical presentation timing",
                        fontsize=11.5, color="#6b7079", zorder=41))
    art.append(fig.add_artist(Rectangle((L, 0.775), 0.86, 0.003,
                                        transform=fig.transFigure, fc="#d5d8dd",
                                        ec="none", zorder=41)))
    art.append(fig.text(CA, 0.726, "COMMISSIONED", ha="center", fontsize=11,
                        weight="bold", color=ACCENT["commissioned"], zorder=41))
    art.append(fig.text(CC, 0.726, "OURS", ha="center", fontsize=11, weight="bold",
                        color=ACCENT["ours"], zorder=41))
    rows = [
        ("site parameters configured by hand", f"{n_params}", "0",
         "no zones, no field sets, no re-validation"),
        ("ISO stopping-distance violations",
         f"{summary['commissioned']['viol']}", f"{summary['ours']['viol']}",
         "identical"),
        ("contacts / protective stops",
         f"{summary['commissioned']['contacts']} / "
         f"{summary['commissioned']['pstops']}",
         f"{summary['ours']['contacts']} / {summary['ours']['pstops']}",
         "this run - see the spread below"),
        ("barrier margin, worst over the run",
         f"{summary['commissioned']['min_h']:+.2f} m",
         f"{summary['ours']['min_h']:+.2f} m", "both clear of the limit"),
        ("mission time", f"{ti:.1f} s", f"{to:.1f} s",
         "the same run" if abs(cost) < 3.0 else f"ours is {cost:.0f} % slower"),
    ]
    for i, (label, a, c, note) in enumerate(rows):
        yy = 0.652 - i * 0.076
        art.append(fig.text(L, yy, label, fontsize=12.5, color="#3c4048", zorder=41))
        art.append(fig.text(CA, yy, a, ha="center", fontsize=14, weight="bold",
                            color=ACCENT["commissioned"], zorder=41))
        art.append(fig.text(CC, yy, c, ha="center", fontsize=14, weight="bold",
                            color=ACCENT["ours"], zorder=41))
        art.append(fig.text(CN, yy, note, fontsize=11, color="#6b7079", zorder=41))

    art.append(fig.text(
        L, 0.258,
        f"Over {bo['n']} randomised presentations, not just the run above: "
        f"{bi['t']:.1f} s against {bo['t']:.1f} s ({batt_cost:+.1f} %), zero contacts, "
        f"zero ISO violations and\n"
        f"{bo['pstops']:.2f} against {bi['pstops']:.2f} protective stops. Ours does run "
        f"closer to the limit -- worst barrier margin {bo['min_h']:+.2f} m against "
        f"{bi['min_h']:+.2f} m -- but it never crosses it.",
        fontsize=11, color="#3c4048", linespacing=1.7, va="top", zorder=41))
    art.append(fig.text(
        L, 0.188,
        "The trade: the same transport time, against every per-site speed zone, field "
        "set and re-validation the machine above needs. Industry context for what\n"
        "that engineering costs: roughly 30 % of equipment spend goes to solution design "
        "and deployment, at 8-14 weeks contract-to-production.",
        fontsize=11.5, color="#555", linespacing=1.7, va="top", zorder=41))
    scan_cost = 100.0 * (bo["t"] - batt["scanner"]["t"]) / batt["scanner"]["t"]
    art.append(fig.text(
        L, 0.118,
        "Honest limits, measured on this route only. Ours needs one MACHINE constant -- "
        "how far its sensors see past a mapped occluder -- plus the map it already has; "
        "the machine above needs\nthat same quantity surveyed per junction, then a zone "
        "speed, extent and polygon derived and validated from it. Strip its zones off "
        f"and it runs {batt['scanner']['t']:.1f} s, {scan_cost:.0f} % quicker than ours, "
        "but with\nnothing anticipating the corner its scanner cannot see round. The "
        "`crowd` station, where ours is beaten outright, is not on this route. "
        "2D simulation, same control stack as the Gazebo build.\n"
        "Cost and lead-time figures are industry sources, not measurements from this "
        "work.",
        fontsize=9.5, color="#8a919b", linespacing=1.7, va="top", zorder=41))
    for a in art:
        a.set_visible(False)
    return art


if __name__ == "__main__":
    main()
