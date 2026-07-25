"""Config loading: YAML -> typed dataclasses.

Configs live in experiments/configs/*.yaml and are the single source of truth for
every tunable/frozen parameter (plan: "configs as YAML from day one"). Loading them
into frozen dataclasses gives autocomplete + typo protection and keeps the frozen
CBF constants (plan D5) genuinely immutable at the type level.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "experiments" / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    """Load a config file by stem (e.g. 'robot' -> experiments/configs/robot.yaml)."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class RobotParams:
    """Frozen physical platform constants (plan D2)."""

    v_max: float
    v_min: float
    omega_max: float
    omega_min: float
    a_max_physical: float
    a_max_mpc: float
    a_brake: float
    tau_latency: float
    wheelbase: float
    robot_radius: float
    dt: float

    @classmethod
    def from_yaml(cls, name: str = "robot") -> "RobotParams":
        d = load_yaml(name)
        fields = cls.__dataclass_fields__
        return cls(**{k: d[k] for k in fields})


@dataclass(frozen=True)
class MpcWeights:
    """Cost weights for the NMPC (plan D4). Report an ablation on R_delta_v."""

    w_pos: float
    w_theta: float
    w_v: float
    R_v: float
    R_omega: float
    R_delta_v: float
    R_delta_omega: float
    w_terminal: float
    w_slack: float
    w_human: float


@dataclass(frozen=True)
class MpcParams:
    """MPC tracking-controller configuration (plan D4).

    `max_static_obstacles` / `max_humans` are build-time capacities: the CasADi NLP
    has a fixed structure, so obstacle/human slots are zero-padded (absent slots are
    placed far away with zero radius, making their constraints/potentials inert).
    """

    horizon_N: int
    dt: float
    warm_start: bool
    v_ref: float
    r_robot: float
    default_margin: float
    human_decay: float
    carrot_lookahead: float
    max_static_obstacles: int
    max_humans: int
    weights: MpcWeights

    @classmethod
    def from_yaml(cls, name: str = "mpc") -> "MpcParams":
        d = load_yaml(name)
        w = d["weights"]
        weights = MpcWeights(
            w_pos=w["w_pos"],
            w_theta=w["w_theta"],
            w_v=w["w_v"],
            R_v=w["R_v"],
            R_omega=w["R_omega"],
            R_delta_v=w["R_delta_v"],
            R_delta_omega=w["R_delta_omega"],
            w_terminal=w["w_terminal"],
            w_slack=w["w_slack"],
            w_human=w["w_human"],
        )
        cap = d.get("capacities", {})
        return cls(
            horizon_N=d["horizon_N"],
            dt=d["dt"],
            warm_start=d["warm_start"],
            v_ref=d["v_ref"],
            r_robot=d["obstacle"]["r_robot"],
            default_margin=d["human_potential"]["default_margin"],
            human_decay=d["human_potential"]["decay"],
            carrot_lookahead=d["carrot"]["lookahead"],
            max_static_obstacles=cap["max_static_obstacles"],
            max_humans=cap["max_humans"],
            weights=weights,
        )


@dataclass(frozen=True)
class CbfParams:
    """FROZEN safety-filter constants (plan D5). THE certifiable layer.

    CRITICAL INVARIANT: nothing here is tunable by the RL policy. The policy may
    request a human margin ABOVE d_hard, but d_hard / tau / a_brake / gamma / sigma
    / the protective field are immutable. Constructed frozen so that immutability is
    enforced at the type level, not just by convention.
    """

    d_hard: float
    tau: float
    a_brake: float
    gamma: float
    sigma: float
    W_v: float
    W_omega: float
    protective_radius: float
    solver: str

    def __post_init__(self) -> None:
        # Cheap invariants that must hold for the safety argument (plan D5).
        assert self.d_hard > 0.0
        assert 0.0 < self.gamma <= 1.0
        assert self.sigma >= 1.0
        assert self.protective_radius > self.d_hard

    @classmethod
    def from_yaml(cls, name: str = "cbf") -> "CbfParams":
        d = load_yaml(name)
        return cls(
            d_hard=d["d_hard"],
            tau=d["tau"],
            a_brake=d["a_brake"],
            gamma=d["gamma"],
            sigma=d["sigma"],
            W_v=d["qp"]["W_v"],
            W_omega=d["qp"]["W_omega"],
            protective_radius=d["protective_field"]["radius"],
            solver=d["qp"]["solver"],
        )


@dataclass(frozen=True)
class RewardWeights:
    """Per-term reward weights (plan D6). Logged separately, always."""

    w1_progress: float
    w2_energy: float
    w3_jerk: float
    w4_cbf_intervention: float
    w5_protective_stop: float
    w6_personal_space: float
    w7_success_bonus: float
    time_penalty: float
    # w8 penalizes the ISO stopping-distance breach magnitude (max(0,-h)) while
    # MOVING -- the violation the metric counts but the reward previously ignored.
    # 0.0 on TB3 (frozen reward; the filter catches everything at d_stop 0.25 m so
    # there is nothing to penalize). >0 on the industrial platform, where reveal <
    # d_stop makes corner rushes cause filter-uncatchable violations the policy must
    # learn to pre-empt (2026-07 reward surgery). Defaulted so old configs load.
    w8_stopping_violation: float = 0.0
    # w9 rewards ANTICIPATORY corner slowing directly: the ISO "limited-visibility
    # safe speed" -- at a mapped blind constriction the robot must be able to stop
    # within the distance it can see (max(post_ahead, sight_floor)). Unlike w8 (a
    # DELAYED, PROBABILISTIC penalty that fires only when a pedestrian happens to
    # breach), w9 is DENSE and IMMEDIATE -- it fires every step the robot is too
    # fast for the corner sight distance, giving the clean gradient needed to learn
    # anticipation (added 2026-07-24 after w8-alone left corners at 4/8). 0.0 on
    # TB3; >0 industrial. See core/rl/reward.py for the exact term.
    w9_blind_corner_speed: float = 0.0
    # w10 is the VISIBLE-HUMAN analogue of w9: an anticipatory approach-margin term
    # that penalizes clip(h_buffer - h, 0, h_buffer) while moving, so the robot keeps
    # a safety BUFFER above the hard stopping-distance barrier when driving toward a
    # person -- slowing BEFORE it breaches, not after (w8 is the after). Diagnosed
    # 2026-07-24: the corner-fixed policy's residual crowd/interferer violations were
    # ALL "robot-too-fast toward a visible human", which w8 (reactive) could not
    # pre-empt. h already encodes closing geometry (0 term for humans beside/behind),
    # so w10 is inert in open lanes and does NOT over-slow for a hovering follower.
    # 0.0 on TB3; >0 industrial.
    w10_human_approach: float = 0.0


@dataclass(frozen=True)
class RlParams:
    """RL supervisor configuration (plan D6): action bounds, obs layout, reward.

    The supervisor decides at 2 Hz (every `decision_every` MPC steps) and only
    MODULATES the MPC; d_margin_low is clipped to >= cbf.d_hard by construction
    (the yaml carries the same floor -- validated in __post_init__ by the caller
    passing d_hard when available).
    """

    decision_every: int
    v_max_low: float
    v_max_high: float
    d_margin_low: float
    d_margin_high: float
    K_nearest: int
    weights: RewardWeights
    episode_timeout_s: float
    personal_space: float = 0.5   # m, the w6 intrusion radius (plan 4.3)
    # sight distance available AT a blind constriction (once at the corner the robot
    # can see this far around it) -- the floor on the w9 anticipatory-speed term.
    # ~= the tracker reveal distance; the robot must be able to stop within it.
    blind_corner_sight_floor: float = 1.2
    # safety buffer the w10 human-approach term keeps ABOVE the hard barrier h=0:
    # the robot is penalized while h < this, so it slows early when closing on a
    # person. Modest (m) so it anticipates without crawling.
    human_approach_buffer: float = 0.5

    def __post_init__(self) -> None:
        assert self.decision_every >= 1
        assert 0.0 < self.v_max_low < self.v_max_high
        assert 0.0 < self.d_margin_low < self.d_margin_high
        assert self.K_nearest >= 1

    @classmethod
    def from_yaml(cls, name: str = "rl") -> "RlParams":
        d = load_yaml(name)
        a, w = d["action"], d["reward_weights"]
        weights = RewardWeights(
            w1_progress=w["w1_progress"],
            w2_energy=w["w2_energy"],
            w3_jerk=w["w3_jerk"],
            w4_cbf_intervention=w["w4_cbf_intervention"],
            w5_protective_stop=w["w5_protective_stop"],
            w6_personal_space=w["w6_personal_space"],
            w7_success_bonus=w["w7_success_bonus"],
            time_penalty=d["time_penalty_per_step"],
            w8_stopping_violation=w.get("w8_stopping_violation", 0.0),
            w9_blind_corner_speed=w.get("w9_blind_corner_speed", 0.0),
            w10_human_approach=w.get("w10_human_approach", 0.0),
        )
        return cls(
            decision_every=d["decision_every_n_mpc_steps"],
            v_max_low=a["v_max_cmd"]["low"],
            v_max_high=a["v_max_cmd"]["high"],
            d_margin_low=a["d_margin_cmd"]["low"],
            d_margin_high=a["d_margin_cmd"]["high"],
            K_nearest=d["observation"]["K_nearest_humans"],
            weights=weights,
            episode_timeout_s=d["episode_timeout_s"],
            blind_corner_sight_floor=d.get("blind_corner_sight_floor", 1.2),
            human_approach_buffer=d.get("human_approach_buffer", 0.5),
        )
