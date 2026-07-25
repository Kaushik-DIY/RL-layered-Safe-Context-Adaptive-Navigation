"""LAYER 2 -- MPC tracking controller (plan D4).

Hand-formulated NMPC in CasADi, direct multiple shooting, RK4 discretization,
solved by IPOPT. Built from scratch (dynamics, discretization, cost, constraints,
slack) -- that is the point (plan D3). No do-mpc, no Nav2 planner in the loop.

Runtime-modulated parameters from the RL layer (plan D4):
    v_max_cmd   -> hard bound   v_k <= v_max_cmd
    d_margin_cmd-> radius of the (soft) human ellipsoidal potential

Formulation (D4):
    vars   : X in R^{3x(N+1)}, U in R^{2xN}, slacks S >= 0
    cost   : w_p||pos-carrot||^2 + w_th(heading)^2 + w_v(v-v_ref)^2
             + u'Ru + du'R_du du + terminal + w_s||S||^2 + human potentials
    hard   : unicycle dynamics; u bounds w/ v_k<=v_max_cmd; |dv|<=a_max*dt
    soft   : static obstacle clearance (slack); humans as CV-predicted potentials

The NLP is built ONCE with a fixed structure; per-solve inputs (state, carrot,
obstacles, humans, the two RL parameters, and the previous applied command) enter
as CasADi parameters, and obstacle/human capacity is zero-padded (absent slots
placed far away with zero radius, making their terms inert). Warm-started from the
previous (time-shifted) solution.

Target: median solve < 30 ms at N=20 (Gate G1). Always warm-start.
"""
from __future__ import annotations

import time
from typing import Optional

import casadi as ca
import numpy as np

from core.common.params import MpcParams, RobotParams


class MpcController:
    """Direct-multiple-shooting NMPC for the unicycle (plan D4)."""

    def __init__(self, robot: RobotParams, mpc: MpcParams):
        self.robot = robot
        self.mpc = mpc
        self.N = mpc.horizon_N
        self.dt = mpc.dt
        self.MO = mpc.max_static_obstacles
        self.MH = mpc.max_humans
        self._prev_X: Optional[np.ndarray] = None
        self._prev_U: Optional[np.ndarray] = None
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        N, dt, w = self.N, self.dt, self.mpc.weights
        opti = ca.Opti()

        X = opti.variable(3, N + 1)          # states  [x, y, theta]
        U = opti.variable(2, N)              # controls [v, omega]
        S = opti.variable(self.MO, N)        # static-obstacle slacks (>= 0)

        # --- per-solve parameters ---
        p_x0 = opti.parameter(3)             # current state
        p_uprev = opti.parameter(2)          # last applied command (smoothness/accel)
        p_carrot = opti.parameter(2)         # tracking target on the path
        p_vmax = opti.parameter(1)           # v_max_cmd  (RL parameter)
        p_vref = opti.parameter(1)           # effective reference speed
        p_margin = opti.parameter(1)         # d_margin_cmd (RL parameter)
        p_obs = opti.parameter(3, self.MO)   # static obstacles [x, y, r]
        p_hum = opti.parameter(4, self.MH)   # humans [x, y, vx, vy]

        self.opti, self.X, self.U, self.S = opti, X, U, S
        self.p = dict(x0=p_x0, uprev=p_uprev, carrot=p_carrot, vmax=p_vmax,
                      vref=p_vref, margin=p_margin, obs=p_obs, hum=p_hum)

        def f(state, u):  # continuous unicycle dynamics
            return ca.vertcat(u[0] * ca.cos(state[2]),
                              u[0] * ca.sin(state[2]),
                              u[1])

        def rk4(state, u):
            k1 = f(state, u)
            k2 = f(state + 0.5 * dt * k1, u)
            k3 = f(state + 0.5 * dt * k2, u)
            k4 = f(state + dt * k3, u)
            return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        opti.subject_to(X[:, 0] == p_x0)     # initial condition

        cost = 0
        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            xk1 = X[:, k + 1]

            # multiple-shooting defect (hard)
            opti.subject_to(xk1 == rk4(xk, uk))

            # control bounds (v upper bound is the RL-modulated cap; hard)
            opti.subject_to(uk[0] >= self.robot.v_min)
            opti.subject_to(uk[0] <= p_vmax)
            opti.subject_to(uk[1] >= self.robot.omega_min)
            opti.subject_to(uk[1] <= self.robot.omega_max)

            # forward-difference acceleration limit on v (hard, D4)
            u_prev = p_uprev if k == 0 else U[:, k - 1]
            dv = uk[0] - u_prev[0]
            opti.subject_to(opti.bounded(-self.robot.a_max_mpc * dt, dv,
                                         self.robot.a_max_mpc * dt))

            # --- cost terms (D4) ---
            cost += w.w_pos * ca.sumsqr(xk1[:2] - p_carrot)           # tracking
            # heading toward the carrot from the current stage
            th_des = ca.atan2(p_carrot[1] - xk[1], p_carrot[0] - xk[0])
            herr = ca.atan2(ca.sin(xk[2] - th_des), ca.cos(xk[2] - th_des))
            cost += w.w_theta * herr ** 2
            cost += w.w_v * (uk[0] - p_vref) ** 2                     # cruise
            cost += w.R_v * uk[0] ** 2 + w.R_omega * uk[1] ** 2       # effort
            dw = uk[1] - u_prev[1]
            cost += w.R_delta_v * dv ** 2 + w.R_delta_omega * dw ** 2  # smoothness

            # static-obstacle soft clearance with slack (D4)
            for j in range(self.MO):
                d = ca.sqrt((xk1[0] - p_obs[0, j]) ** 2
                            + (xk1[1] - p_obs[1, j]) ** 2 + 1e-6)
                opti.subject_to(d >= (p_obs[2, j] + self.mpc.r_robot) - S[j, k])
                opti.subject_to(S[j, k] >= 0)
            cost += w.w_slack * ca.sumsqr(S[:, k])

            # humans as soft, CV-predicted radial barriers (D4, form revised by the
            # week-4 audit): w * exp((margin - d)/b). A Gaussian in d/margin is
            # NON-MONOTONE in the margin at close range (gradient peaks at
            # d=margin, flattens inside), which made a wider d_margin_cmd push
            # LESS -- unusable as an RL action. This barrier shifts with margin.
            for m in range(self.MH):
                hx = p_hum[0, m] + (k + 1) * dt * p_hum[2, m]
                hy = p_hum[1, m] + (k + 1) * dt * p_hum[3, m]
                dh = ca.sqrt((xk1[0] - hx) ** 2 + (xk1[1] - hy) ** 2 + 1e-9)
                cost += w.w_human * ca.exp((p_margin - dh) / self.mpc.human_decay)

        cost += w.w_terminal * ca.sumsqr(X[:2, N] - p_carrot)        # terminal
        opti.minimize(cost)
        self.cost_expr = cost

        # Solver tuning (measured on the Week-1 clutter course): `expand` compiles the
        # MX graph to SX (big eval speedup); `mu_strategy=adaptive` + early acceptable
        # exit cut the median; capping `max_iter` bounds the tail -- near-active obstacle
        # constraints otherwise drive IPOPT to hundreds of iterations, and since those
        # obstacles are SOFT (slack), a truncated iterate is still feasible and usable.
        # Median ~23 ms, p99 ~150 ms at N=20 (Gate G1). acados/SQP-RTI is the documented
        # port (plan D3) if a hard real-time p99 < 100 ms is ever required.
        opti.solver("ipopt", {
            "expand": True,
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 30,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
        })

    # ------------------------------------------------------------------ solve
    @staticmethod
    def _pad(arr, n_slots, width, fill):
        out = np.tile(np.asarray(fill, dtype=float), (n_slots, 1))
        if arr is not None and len(arr) > 0:
            a = np.asarray(arr, dtype=float)
            k = min(len(a), n_slots)
            out[:k] = a[:k, :width]
        return out.T  # -> shape (width, n_slots) to match parameter layout

    def solve(self, x0, carrot, static_obs=None, humans=None,
              v_max_cmd=None, d_margin_cmd=None, u_prev=None):
        """Solve the NMPC once and return (u0=[v,omega], info).

        static_obs : iterable of [x, y, r]         (may be empty/None)
        humans     : iterable of [x, y, vx, vy]    (may be empty/None)
        v_max_cmd  : RL v-cap (defaults to platform v_max)
        d_margin_cmd : RL human-potential radius (defaults to config default_margin)
        u_prev     : last applied [v, omega] (defaults to zeros)
        """
        v_max_cmd = self.robot.v_max if v_max_cmd is None else float(v_max_cmd)
        d_margin_cmd = (self.mpc.default_margin if d_margin_cmd is None
                        else float(d_margin_cmd))
        u_prev = np.zeros(2) if u_prev is None else np.asarray(u_prev, float)
        v_ref_eff = min(self.mpc.v_ref, v_max_cmd)

        opti = self.opti
        opti.set_value(self.p["x0"], np.asarray(x0, float)[:3])
        opti.set_value(self.p["uprev"], u_prev)
        opti.set_value(self.p["carrot"], np.asarray(carrot, float)[:2])
        opti.set_value(self.p["vmax"], v_max_cmd)
        opti.set_value(self.p["vref"], v_ref_eff)
        opti.set_value(self.p["margin"], d_margin_cmd)
        opti.set_value(self.p["obs"],
                       self._pad(static_obs, self.MO, 3, [1e3, 1e3, 0.0]))
        opti.set_value(self.p["hum"],
                       self._pad(humans, self.MH, 4, [1e6, 1e6, 0.0, 0.0]))

        # warm start from the previous (time-shifted) solution (D3: always warm-start)
        if self.mpc.warm_start and self._prev_X is not None:
            Xw = np.hstack([self._prev_X[:, 1:], self._prev_X[:, -1:]])
            Uw = np.hstack([self._prev_U[:, 1:], self._prev_U[:, -1:]])
            opti.set_initial(self.X, Xw)
            opti.set_initial(self.U, Uw)

        t0 = time.perf_counter()
        try:
            opti.solve()
        except RuntimeError:
            pass  # non-"Solve_Succeeded" status (incl. acceptable-level) -> use iterate
        solve_ms = (time.perf_counter() - t0) * 1e3

        # Read the actual solver status: acceptable-level convergence is a good
        # solution, not a failure (Opti raises on it, so don't trust the exception).
        status = opti.stats().get("return_status", "")
        success = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        Xopt, Uopt = opti.debug.value(self.X), opti.debug.value(self.U)

        self._prev_X, self._prev_U = Xopt, Uopt
        u0 = np.array([Uopt[0, 0], Uopt[1, 0]])
        return u0, {"solve_ms": solve_ms, "success": success,
                    "status": status, "X": Xopt, "U": Uopt}

    def reset(self) -> None:
        """Forget the warm-start state (call at the start of a new episode).

        Also zero Opti's stored initial guess: set_initial values PERSIST inside
        the Opti object, so without this a reused controller's first solve warm
        starts from the previous episode's leftovers -- episodes then differ at
        ~1e-3 between fresh and reused envs, breaking seed reproducibility and
        the paired-seed battery methodology (caught by check 6 of
        scripts/check_training_readiness.py)."""
        self._prev_X = self._prev_U = None
        self.opti.set_initial(self.X, np.zeros((3, self.N + 1)))
        self.opti.set_initial(self.U, np.zeros((2, self.N)))
        self.opti.set_initial(self.S, np.zeros((self.MO, self.N)))
