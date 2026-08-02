"""Unicycle kinematic robot sim (plan D1/D2).

State x=[x,y,theta], control u=[v,omega], injected latency tau (D2), integrated at
dt=0.1 s with RK4 (same discretization the MPC uses, so the sim and the controller's
prediction model agree -- any mismatch is then genuinely dynamics fidelity, not a
discretization artifact). Deliberately minimal -- the fidelity gap vs Gazebo is
measured and reported, not hidden.

The injected latency is the crux of D2: commands enter a FIFO buffer of
round(tau/dt) steps, so what actually drives the wheels this tick was commanded
tau seconds ago. This is what gives the CBF stopping-distance constraint something
to bite on at TurtleBot speeds.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from core.common.params import RobotParams


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def unicycle_deriv(state: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous unicycle dynamics: d/dt [x,y,theta] for control [v,omega]."""
    theta = state[2]
    v, omega = u[0], u[1]
    return np.array([v * np.cos(theta), v * np.sin(theta), omega])


def unicycle_rk4(state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step of the unicycle model (matches the MPC prediction model)."""
    k1 = unicycle_deriv(state, u)
    k2 = unicycle_deriv(state + 0.5 * dt * k1, u)
    k3 = unicycle_deriv(state + 0.5 * dt * k2, u)
    k4 = unicycle_deriv(state + dt * k3, u)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class KinematicSim:
    """Minimal unicycle simulator with injected actuation latency (D2)."""

    def __init__(self, robot: RobotParams):
        self.robot = robot
        self.dt = robot.dt
        self.latency_steps = int(round(robot.tau_latency / robot.dt))
        self.reset(np.zeros(3))

    def reset(self, x0) -> np.ndarray:
        """Reset to pose x0=[x,y,theta]; clear the latency buffer to zero commands."""
        self._pose = np.asarray(x0, dtype=float).copy()
        self._applied_u = np.zeros(2)
        # FIFO of not-yet-applied commands; pre-filled with zeros (robot starts still).
        self._buffer: deque = deque(
            (np.zeros(2) for _ in range(self.latency_steps)),
            maxlen=self.latency_steps or 1,
        )
        return self.state()

    def step(self, u) -> np.ndarray:
        """Apply command u=[v,omega] (clamped), advance one dt. Returns full state."""
        u = np.clip(
            np.asarray(u, dtype=float),
            [self.robot.v_min, self.robot.omega_min],
            [self.robot.v_max, self.robot.omega_max],
        )
        if self.latency_steps > 0:
            self._buffer.append(u)
            applied = self._buffer.popleft()
        else:
            applied = u
        self._applied_u = applied
        self._pose = unicycle_rk4(self._pose, applied, self.dt)
        self._pose[2] = wrap_angle(self._pose[2])
        return self.state()

    def state(self) -> np.ndarray:
        """Return [x, y, theta, v, omega] (v,omega = the command currently applied)."""
        return np.array(
            [self._pose[0], self._pose[1], self._pose[2],
             self._applied_u[0], self._applied_u[1]]
        )
