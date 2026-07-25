"""Week-3 reward-term tests (plan D6): signs, magnitudes, per-term separation."""
from __future__ import annotations

import pytest

from core.common.params import RlParams
from core.rl.reward import accumulate, reward_terms, total_reward


@pytest.fixture(scope="module")
def rl() -> RlParams:
    return RlParams.from_yaml()


def _terms(rl, **over):
    base = dict(progress=0.0, dv=0.0, v=0.0, jerk=0.0, intervention=0.0,
                protective_stop=False, min_human_dist=10.0, dt=0.1, success=False)
    base.update(over)
    return reward_terms(rl.weights, **base)


def test_weights_load(rl):
    assert rl.weights.w1_progress == 1.0
    assert rl.weights.w5_protective_stop == 5.0
    assert rl.decision_every == 5
    assert rl.d_margin_low == pytest.approx(0.30)  # action floor == cbf d_hard


def test_progress_dominates_time_penalty(rl):
    """Driving toward the goal must net positive (else standing still is optimal)."""
    t = _terms(rl, progress=0.026, v=0.26)   # one full-speed step
    assert total_reward(t) > 0


def test_all_penalties_are_negative(rl):
    t = _terms(rl, dv=0.02, v=0.2, jerk=0.05, intervention=0.1,
               protective_stop=True, min_human_dist=0.4)
    for k in ("energy", "jerk", "cbf_intervention", "protective_stop",
              "personal_space", "time"):
        assert t[k] < 0, k


def test_intervention_teaches_anticipation(rl):
    """The KEY term (D6): a filtered step must be worse than an unfiltered one."""
    filtered = _terms(rl, progress=0.02, intervention=0.13)
    clean = _terms(rl, progress=0.02, intervention=0.0)
    assert total_reward(filtered) < total_reward(clean)


def test_protective_stop_is_large(rl):
    """One protective stop must wipe out many steps of progress (it is -5)."""
    assert abs(_terms(rl, protective_stop=True)["protective_stop"]) >= 100 * 0.026


def test_personal_space_only_inside_radius(rl):
    assert _terms(rl, min_human_dist=0.49)["personal_space"] < 0
    assert _terms(rl, min_human_dist=0.51)["personal_space"] == 0.0


def test_success_bonus_once(rl):
    t = _terms(rl, success=True)
    assert t["success"] == rl.weights.w7_success_bonus


def test_accumulate_sums_per_term(rl):
    w = [_terms(rl, progress=0.01), _terms(rl, progress=0.02)]
    acc = accumulate(w)
    assert acc["progress"] == pytest.approx(0.03)
    assert total_reward(acc) == pytest.approx(sum(total_reward(t) for t in w))
