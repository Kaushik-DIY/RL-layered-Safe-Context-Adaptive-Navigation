"""LAYER 1 -- CBF safety filter (plan D5).

Interposed between MPC (L2) and the robot. Solves, at 10 Hz on the raw u_mpc:

    min ||u - u_mpc||^2_W
    s.t.  d_stop(sigma * v_closing_i) <= d_i - d_hard   for every tracked human i
          u bounds, |du| bounds

i.e. the ISO stopping-distance principle -- the robot may only close on a human as
fast as it can still brake to a stop within the clear distance:
    d_stop(s) = s*tau + s^2 / (2 a_brake)          (tau = latency, a_brake = decel)
    v_closing = v * max(0, cos phi_i)              (only motion TOWARD the human)
This is a discrete-time control barrier condition on b(x) = d - d_hard with the exact
braking class-K; it is enforced at the latency-predicted geometry (see below) and
realized as a per-human upper bound on v, giving a tiny QP solved with proxsuite.

CRITICAL INVARIANT (plan sec. 2 & D5): NO tunable input from the RL layer. d_hard,
tau, a_brake, gamma, sigma, the protective field, and this logic are FROZEN. This
separation is what makes the safety argument airtight -- do not add RL-controlled
knobs here.

------------------------------------------------------------------------------
The per-human speed cap (exact braking class-K + actuation-latency prediction).

The command chosen now reaches the wheels only after the latency tau (D2). During
that deadtime the robot coasts at its current speed v0 and the human keeps moving,
so a one-step CBF is blind to it -- the robot drives through the boundary before the
brake engages. We therefore enforce the stopping constraint at the geometry
predicted one latency window ahead:

    p_robot' = p_robot + v0 * t * [cos theta, sin theta]     (coast at v0)
    p_human' = p_human + v_human * t                         (constant velocity)
    d' = |p_human' - p_robot'| ;  c' = [cos theta, sin theta] . (p_human'-p_robot')/d'

evaluated at SAMPLED points t in {0, tau/2, tau} with the most conservative cap
kept (S4-evaluation finding: endpoint-only enforcement at t = tau let a robot
accelerate into a pedestrian who was predicted PAST it by then -- the constraint
must hold throughout the deadtime window, not just at its end).

Only motion TOWARD the (predicted) human matters, so the robot's closing speed is
v_c = sigma * v * max(0, c'). But the HUMAN also keeps closing during the whole
stop -- the SFM adversarial battery exposed this (8/1000 violations, pedestrians
striding in at ~1.4 m/s): budgeting only the robot's stopping distance treats the
human as frozen after the one-tau prediction window and under-brakes against fast
approach. ISO 13855 (which ISO 3691-4 leans on for protective-field sizing) makes
the same split explicit: separation = human-approach contribution + machine
stopping contribution. With beta = sigma * max(0, human closing speed along the
predicted LOS) and the robot's stop lasting (tau + v_c/a_brake) beyond wheel
engagement, the constraint is

    v_c*tau + v_c^2/(2 a_brake) + beta*(tau + v_c/a_brake) <= d' - d_hard

(robot travel + human travel during the robot's stop fit the clear distance).
Solving the quadratic for the largest admissible closing speed gives the exact cap

    v_c_max = -(a*tau + beta) + sqrt((a*tau + beta)^2 + 2 a * Delta),
    Delta   = d' - d_hard - beta*tau                          (a = a_brake)

and hence  v <= v_c_max / (sigma * c')  per human (v = 0 if Delta <= 0); beta = 0
recovers the pure-braking cap. This is the discrete-time CBF for b(x)=d-d_hard with
the exact braking class-K under constant-velocity humans; because it caps the
ACHIEVABLE closing speed directly (not an incremental linearization), it is robust
at the worst case -- full speed straight at a person, person striding at the robot.
gamma is retained in config for the class-K sensitivity study (plan D5).
------------------------------------------------------------------------------

Second layer: protective-field emergency stop (ESPE logic). If any human enters
protective_radius, brake as hard as physically possible (Delta v = -a_brake*dt,
floored at 0), override the QP, and log a protective-stop event (a headline metric).
"""
from __future__ import annotations

import numpy as np
from proxsuite import proxqp

from core.common.params import CbfParams, RobotParams


def d_stop(speed: float, tau: float, a_brake: float) -> float:
    """ISO 3691-4 stopping distance for a (closing) speed: s*tau + s^2/(2a)."""
    return speed * tau + speed * speed / (2.0 * a_brake)


class CbfFilter:
    """Discrete-time CBF-QP stopping-distance filter (plan D5). Frozen constants."""

    def __init__(self, robot: RobotParams, cbf: CbfParams):
        self.robot = robot
        self.cbf = cbf
        self.dt = robot.dt
        self._v_prev = 0.0  # last commanded v (for the physical |dv| brake limit)

    def reset(self) -> None:
        self._v_prev = 0.0

    def barrier(self, x, human) -> float:
        """Current stopping-distance barrier h0 for one human (>= 0 means safe)."""
        d, c, _ = self._geometry(x, human)
        s = self.cbf.sigma * max(0.0, c)
        v0 = float(x[3])
        return d - d_stop(s * v0, self.cbf.tau, self.cbf.a_brake) - self.cbf.d_hard

    def min_barrier(self, x, humans) -> float:
        """Smallest barrier over all humans (the value G2 checks stays >= 0)."""
        if humans is None or len(humans) == 0:
            return np.inf
        return min(self.barrier(x, h) for h in humans)

    def _geometry(self, x, human):
        """Actual current distance / closing cosine (used for the barrier & ESPE)."""
        return self._geometry_at(x, human, 0.0)

    def _geometry_at(self, x, human, t_ahead: float):
        """Distance / closing cosine / human closing speed at time t_ahead, under
        coasting robot (v0 along heading) + constant-velocity human (deadtime
        compensation, D2). t_ahead = 0 is the current geometry; t_ahead = tau is
        the wheel-engagement instant of the command being chosen now.
        """
        theta, v0 = x[2], float(x[3])
        px = x[0] + v0 * np.cos(theta) * t_ahead
        py = x[1] + v0 * np.sin(theta) * t_ahead
        hx = human[0] + human[2] * t_ahead
        hy = human[1] + human[3] * t_ahead
        lx, ly = hx - px, hy - py
        d = float(np.hypot(lx, ly))
        d_safe = max(d, 1e-6)
        nx, ny = lx / d_safe, ly / d_safe                  # unit LOS robot -> human
        c = np.cos(theta) * nx + np.sin(theta) * ny
        beta = -(human[2] * nx + human[3] * ny)            # human closing speed (>0)
        return d, float(c), float(beta)

    def filter(self, x, u_mpc, humans=None):
        """Return (u_safe=[v, omega], info). x = [x, y, theta, v, omega].

        info keys: intervention (||u_safe - u_mpc||), protective_stop (bool),
        h_min (current min barrier), n_active (humans capping v), qp_solved (bool).
        """
        cbf, dt = self.cbf, self.dt
        v_mpc, w_mpc = float(u_mpc[0]), float(u_mpc[1])
        v0 = float(x[3])
        humans = [] if humans is None else list(humans)

        h_min = self.min_barrier(x, humans)

        # physical velocity box for THIS command: brake up to a_brake, ease up to a_max
        v_lo = max(self.robot.v_min, self._v_prev - cbf.a_brake * dt)
        v_hi = min(self.robot.v_max, self._v_prev + self.robot.a_max_mpc * dt)

        # --- Layer 2: protective-field emergency stop (overrides everything, D5) ---
        d_min = min((self._geometry(x, h)[0] for h in humans), default=np.inf)
        if d_min < cbf.protective_radius:
            v_safe = max(self.robot.v_min, self._v_prev - cbf.a_brake * dt)
            u_safe = np.array([v_safe, 0.0])
            self._v_prev = v_safe
            return u_safe, {"intervention": float(np.hypot(v_safe - v_mpc, w_mpc)),
                            "protective_stop": True, "h_min": h_min,
                            "n_active": len(humans), "qp_solved": True}

        # --- Layer 1: per-human stopping-distance caps -> v <= cap_i (see header) ---
        # The cap is evaluated at SAMPLED points across the latency window
        # (t+0, t+tau/2, t+tau), taking the most conservative. Enforcing only at
        # t+tau left a hole the S4 evaluation exposed (corridor, 2 collisions):
        # a pedestrian walking nearly straight through a STOPPED robot's position
        # is predicted PAST the robot at t+tau, the closing cosine flips negative,
        # and the endpoint-only check waved the robot forward into the pedestrian
        # DURING the window. The constraint must hold throughout the interval.
        # A point contributes no cap only while the robot is not closing toward
        # the human THERE (c <= 0: fleeing must never be braked).
        a, tau = cbf.a_brake, cbf.tau
        v_caps, n_active = [], 0
        for h in humans:
            cap_h, closing_somewhere = np.inf, False
            for phi in (0.0, 0.5, 1.0):
                d, c, beta_raw = self._geometry_at(x, h, phi * tau)
                if c <= 1e-6:
                    continue
                closing_somewhere = True
                beta = cbf.sigma * max(0.0, beta_raw)  # human-approach (ISO 13855)
                delta = d - cbf.d_hard - beta * tau
                if delta <= 0.0:
                    cap_h = 0.0                        # clearance already gone -> stop
                    break
                atb = a * tau + beta
                v_c_max = -atb + np.sqrt(atb * atb + 2.0 * a * delta)
                cap_h = min(cap_h, v_c_max / (cbf.sigma * c))
            if closing_somewhere:
                v_caps.append(cap_h)
                n_active += 1

        # --- solve the tiny QP: min W_v(v-v_mpc)^2 + W_w(w-w_mpc)^2 s.t. caps + box ---
        u_safe, solved = self._solve_qp(v_mpc, w_mpc, v_lo, v_hi, v_caps)
        self._v_prev = float(u_safe[0])
        return u_safe, {"intervention": float(np.hypot(u_safe[0] - v_mpc,
                                                       u_safe[1] - w_mpc)),
                        "protective_stop": False, "h_min": h_min,
                        "n_active": n_active, "qp_solved": solved}

    def _solve_qp(self, v_mpc, w_mpc, v_lo, v_hi, v_caps):
        cbf, robot = self.cbf, self.robot
        w_lo = max(robot.omega_min, w_mpc - abs(robot.omega_max))  # omega: box only
        w_hi = min(robot.omega_max, w_mpc + abs(robot.omega_max))

        # H = 2*diag(W_v, W_omega), g = -2*[W_v v_mpc, W_omega w_mpc]
        H = np.diag([2.0 * cbf.W_v, 2.0 * cbf.W_omega])
        g = np.array([-2.0 * cbf.W_v * v_mpc, -2.0 * cbf.W_omega * w_mpc])

        # inequality rows: 2 box (v, omega) + one per active human cap (v <= cap)
        rows = [[1.0, 0.0], [0.0, 1.0]]
        lo = [v_lo, w_lo]
        hi = [v_hi, w_hi]
        for cap in v_caps:
            rows.append([1.0, 0.0]); lo.append(-1e20); hi.append(cap)
        C = np.array(rows)
        l = np.array(lo)
        u = np.array(hi)

        n_in = C.shape[0]
        qp = proxqp.dense.QP(2, 0, n_in)
        qp.init(H, g, None, None, C, l, u)
        qp.solve()
        x = qp.results.x
        solved = qp.results.info.status == proxqp.QPSolverOutput.PROXQP_SOLVED
        if x is None or not np.all(np.isfinite(x)):
            # infeasible box (cap below v_lo): brake as hard as allowed
            return np.array([v_lo, np.clip(w_mpc, w_lo, w_hi)]), False
        # clamp to the box (proxqp respects it, but guard numerics) and never exceed u_mpc's v
        v = float(np.clip(x[0], v_lo, min(v_hi, v_mpc)))
        w = float(np.clip(x[1], w_lo, w_hi))
        return np.array([v, w]), solved
