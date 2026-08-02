"""Training curriculum (plan D6): stage samplers for NavEnv's `scenario_sampler`.

TB3 track (unchanged -- the trained models on disk saw exactly this):
    A  empty world, no pedestrians        -> learn to drive fast
    B  1-2 pedestrians, simple crossings  -> learn to yield (DR switches on here)
    C  4-8 pedestrians, corridors +       -> general competence in clutter
       doorways + open crowd
    blind_corner EXCLUDED: at TB3 scale the policy cannot see the occluded
    pedestrian anyway (obs v1 carries no walls), keeping it a genuine holdout.

Industrial track (2026-07 replan; platform="industrial"):
    A  empty world                        -> drive fast (short)
    B  crossings + BLIND CORNERS + 1-2 roamers   (DR on)
    C  aisles + doorways + hall crowds + corners + INTERFERER + 4-8 roamers
    Corners are IN training here -- obs v2 carries wall/visibility features, and
    reveal < d_stop makes corner anticipation the thing to learn. The holdout is
    t_junction_interferer (occluded junction + seeker, never sampled).

Each sampler draws a fresh scenario seed from the env's np_random, so vectorized
envs decorrelate naturally while a seeded env remains reproducible.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from core.sim2d.scenarios import ScenarioSpec, free_roam, make_scenario

STAGES = ("A", "B", "C")
_IND_ARENA = (-1.0, -5.0, 15.0, 5.0)     # free-roam arena at industrial scale


# Evaluation batteries use small fixed seeds (scenarios.yaml seed_base 1000, plus
# per-battery offsets up to +5000 -> all < EVAL_SEED_CEILING). Training draws seeds
# STRICTLY ABOVE that ceiling, so a training episode can NEVER coincide with an
# evaluation episode: the generalization test (held-out scenario + fixed eval seeds)
# is disjoint from training BY CONSTRUCTION, not just with high probability.
EVAL_SEED_CEILING = 100_000


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(EVAL_SEED_CEILING, 2 ** 31))


def _stage_a(rng: np.random.Generator) -> ScenarioSpec:
    return free_roam(_seed(rng), n_pedestrians=0)


def _stage_b(rng: np.random.Generator) -> ScenarioSpec:
    if rng.random() < 0.5:
        return make_scenario("perpendicular_crossing", _seed(rng))
    return free_roam(_seed(rng), n_pedestrians=int(rng.integers(1, 3)))


def _stage_c(rng: np.random.Generator) -> ScenarioSpec:
    r = rng.random()
    if r < 0.25:
        return make_scenario("corridor_passby", _seed(rng))
    if r < 0.50:
        return make_scenario("doorway_negotiation", _seed(rng))
    if r < 0.75:
        return make_scenario("open_hall", _seed(rng))
    return free_roam(_seed(rng), n_pedestrians=int(rng.integers(4, 9)))


def _ind_stage_a(rng: np.random.Generator) -> ScenarioSpec:
    return free_roam(_seed(rng), n_pedestrians=0, arena=_IND_ARENA)


def _ind_stage_b(rng: np.random.Generator) -> ScenarioSpec:
    # corner-weighted (50%): the blind corner is the behavior the w8 reward targets
    # and the limited retrain budget must spend on it. Crossings teach basic yield.
    r = rng.random()
    if r < 0.50:
        return make_scenario("blind_corner", _seed(rng), "industrial")
    if r < 0.80:
        return make_scenario("perpendicular_crossing", _seed(rng), "industrial")
    return free_roam(_seed(rng), n_pedestrians=int(rng.integers(1, 3)),
                     arena=_IND_ARENA)


def _ind_stage_c(rng: np.random.Generator) -> ScenarioSpec:
    # FINAL-run BALANCED mix (2026-07-24): every instance exercised so the policy
    # masters all of them without forgetting the corner win. Corner keeps a strong
    # 20% (w9 + preserve the learned anticipation); crowd + interferer get 20% each
    # (w10 targets their robot-too-fast violations); doorway/crossing 15% each;
    # corridor/free-roam the remainder.
    r = rng.random()
    if r < 0.20:
        return make_scenario("blind_corner", _seed(rng), "industrial")
    if r < 0.40:
        return make_scenario("open_hall", _seed(rng), "industrial")
    if r < 0.60:
        return make_scenario("interferer", _seed(rng), "industrial")
    if r < 0.75:
        return make_scenario("doorway_negotiation", _seed(rng), "industrial")
    if r < 0.90:
        return make_scenario("perpendicular_crossing", _seed(rng), "industrial")
    if r < 0.96:
        return make_scenario("corridor_passby", _seed(rng), "industrial")
    return free_roam(_seed(rng), n_pedestrians=int(rng.integers(4, 9)),
                     arena=_IND_ARENA)


_SAMPLERS = {
    ("tb3", "A"): _stage_a, ("tb3", "B"): _stage_b, ("tb3", "C"): _stage_c,
    ("industrial", "A"): _ind_stage_a, ("industrial", "B"): _ind_stage_b,
    ("industrial", "C"): _ind_stage_c,
}


def make_sampler(stage: str,
                 platform: str = "tb3") -> Callable[[np.random.Generator], ScenarioSpec]:
    """Return a NavEnv-compatible scenario sampler for a curriculum stage."""
    try:
        return _SAMPLERS[(platform, stage.upper())]
    except KeyError:
        raise ValueError(f"unknown curriculum stage '{stage}'/platform "
                         f"'{platform}'; stages {STAGES}, platforms tb3|industrial")
