"""Reward terms (plan D6), computed per MPC step and summed over the decision window.

Every term is returned SEPARATELY so it can be logged individually (TensorBoard /
info dict) -- reward debugging without per-term logs is the classic RL time sink
(plan D6, risk register). `total_reward` is the only place they are ever summed.

    +w1 progress along path            (dominant)
    -w2 energy proxy |a|*v*dt          (penalizes stop-start; the energy story)
    -w3 jerk^2                         (comfort)
    -w4 ||u_safe - u_mpc||             (KEY: teaches anticipation, not filter-as-crutch)
    -w5 protective-stop event          (large)
    -w6 personal-space intrusion time  (d < 0.5 m)
    +w7 terminal success bonus         (- small per-step time penalty; 60 s timeout)
    -w8 ISO stopping-distance breach   (max(0,-h) while moving; 0 on TB3, >0 industrial)
    -w9 blind-corner over-speed        (d_stop above corner sight distance; anticipatory)
    -w10 human-approach margin         (barrier h below a safety buffer; anticipatory)

Sign convention: each term in the returned dict is ALREADY signed (penalties are
negative), so total_reward is a plain sum -- no hidden sign flips downstream.
"""
from __future__ import annotations

import numpy as np

from core.common.params import RewardWeights


def reward_terms(w: RewardWeights, *, progress: float, dv: float, v: float,
                 jerk: float, intervention: float, protective_stop: bool,
                 min_human_dist: float, dt: float, success: bool,
                 personal_space: float = 0.5,
                 barrier: float = np.inf, moving: bool = True,
                 corner_speed_excess: float = 0.0,
                 approach_buffer: float = 0.0) -> dict[str, float]:
    """Signed reward terms for ONE MPC step (the env sums them over the window).

    progress      : metres of progress along the path this step (signed)
    dv            : applied velocity change this step (a*dt proxy input)
    v             : applied speed this step
    jerk          : change of dv between consecutive steps (d2v, discrete jerk*dt^2)
    intervention  : ||u_safe - u_mpc|| reported by the CBF filter
    protective_stop: the filter's ESPE override fired this step
    min_human_dist: distance to the nearest visible human (inf if none)
    success       : goal reached AT this step (terminal bonus, once)
    barrier       : the min stopping-distance barrier h vs GROUND-TRUTH humans (the
                    true safety state -- privileged in training; the policy still
                    only sees the tracked observation, so it must learn to pre-empt
                    breaches from the occlusion/post-ahead features via the critic).
    moving        : robot moving (v > threshold) -- a breach onto a STOPPED robot is
                    unpreventable and uncharged (same accountability split as w5).
    corner_speed_excess : metres by which the robot's stopping distance exceeds the
                    sight distance to the mapped blind constriction ahead, i.e.
                    max(0, d_stop(sigma*v) - max(post_ahead, sight_floor)). 0 in open
                    space (no constriction ahead) and in crowds (no posts). Computed
                    in NavEnv from the SAME post_ahead the policy observes, so the w9
                    incentive is aligned with a feature the policy can act on.
    approach_buffer : the w10 human-approach margin shortfall
                    clip(h_buffer - barrier, 0, h_buffer): >0 when the robot is
                    closing on a visible human and its safety barrier h has dropped
                    below the buffer (anticipatory -- slow BEFORE h<0). 0 when people
                    are far (h large) or beside/behind (barrier's closing term ~0),
                    so w10 is inert in open lanes and does not over-slow for a
                    hovering follower. Capped at h_buffer so w8 owns the breach zone.
    """
    breach = max(0.0, -float(barrier)) if np.isfinite(barrier) else 0.0
    return {
        "progress": w.w1_progress * progress,
        # energy proxy E = |a| * v * dt (plan 4.3): accelerating while moving costs
        "energy": -w.w2_energy * abs(dv) * v,
        "jerk": -w.w3_jerk * jerk * jerk,
        "cbf_intervention": -w.w4_cbf_intervention * intervention,
        "protective_stop": -w.w5_protective_stop * float(protective_stop),
        "personal_space": -w.w6_personal_space * dt * float(min_human_dist < personal_space),
        "stopping_violation": -w.w8_stopping_violation * breach * float(moving),
        "blind_corner_speed": -w.w9_blind_corner_speed * float(corner_speed_excess),
        "human_approach": -w.w10_human_approach * float(approach_buffer) * float(moving),
        "time": -w.time_penalty,
        "success": w.w7_success_bonus * float(success),
    }


def total_reward(terms: dict[str, float]) -> float:
    """Plain sum -- terms are already signed."""
    return float(sum(terms.values()))


def accumulate(window: list[dict[str, float]]) -> dict[str, float]:
    """Sum per-term dicts over a decision window (for logging at the RL rate)."""
    if not window:
        return {}
    keys = window[0].keys()
    return {k: float(np.sum([t[k] for t in window])) for k in keys}
