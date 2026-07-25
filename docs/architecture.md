# Architecture

Three layers. The core property: **Layer 1 has no tunable input from Layer 3.** The RL
policy may request margins *above* the frozen hard floor; it can never weaken the filter.
That separation is what makes the safety argument airtight.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — RL CONTEXT POLICY (2 Hz)                          │
│  In:  goal-relative state, robot state, K=5 nearest humans   │
│       (rel. pos, rel. vel, TTC), zone flag                   │
│  Out: p = [v_max_cmd, d_margin_cmd]  (MPC parameters)        │
│  PPO, MLP 2×256, trained in fast 2D sim                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ parameters p (bounded, clipped ≥ d_hard)
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 2 — MPC TRACKING CONTROLLER (10 Hz)                   │
│  Unicycle, direct multiple shooting, CasADi/IPOPT, N=20      │
│  Cost: goal tracking + effort + smoothness (Δu)              │
│  v ≤ v_max_cmd; humans = soft ellipsoidal potentials         │
│  Out: u_mpc = (v, ω) first control of horizon                │
└──────────────────────┬──────────────────────────────────────┘
                       │ proposed control u_mpc
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 1 — CBF SAFETY FILTER (10 Hz, on raw u)  [FROZEN]     │
│  min ‖u − u_mpc‖²_W  s.t. per-human DCBF:                    │
│  h(x_{k+1}) ≥ (1−γ)·h(x_k),  h = d − d_stop(v) − d_hard      │
│  d_stop(v) = v·τ + v²/(2·a_brake)   (ISO 3691-4 principle)   │
│  + protective-field hard stop (ESPE logic, non-negotiable)   │
│  Out: u_safe → robot                                         │
└──────────────────────────────────────────────────────────────┘
```

## Software structure (plan D8)

`core/` is pure Python/CasADi with **zero ROS imports** — the identical control stack
runs in the 2D training sim and in Gazebo (wrapped by thin nodes under `ros2_ws/`).

| Node (ROS 2) | Rate | Backed by |
|---|---|---|
| `rl_supervisor` | 2 Hz | ONNX-exported policy (no torch dep at eval) |
| `mpc_controller` | 10 Hz | `core/mpc` (CasADi) |
| `cbf_filter` | 10 Hz | `core/cbf` (QP) |
| `human_tracker` | — | sim ground truth behind an interface (swap for a real detector) |
| `scenario_manager`, `metrics_logger` | — | rosbags + CSV |

## Prior-art anchors

Predictive safety filters (Wabersich & Zeilinger, Automatica 2021); CBF-QP safety-critical
control (Ames et al., TAC 2017); discrete-time CBFs for wheeled robots (Agrawal & Sreenath,
RSS 2017); RL-tunes-controller-parameters (adaptive-CBF/RHC, CDC 2024); residual/hybrid
deployment (DR-MPC, 2024).
