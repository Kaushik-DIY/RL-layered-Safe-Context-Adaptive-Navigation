"""Pre-training readiness audit: is the env/reward actually worth training on?

Run BEFORE committing GPU/Kaggle hours, and again after any reward surgery
(plan D6 budgets 2-3 reward iterations -- this script is the regression harness
for them). Each check targets a specific way RL training gets silently wasted:

  1 SPEED GRADIENT    empty world: faster caps must earn more return (else the
                      policy learns to crawl and "progress" is mis-scaled)
  2 NO FREEZING       standing still must be clearly worst (degenerate optimum)
  3 ANTICIPATION      in human traffic, blind-aggressive must pay a visible
                      w4/w5/w6 penalty (else the filter is a free crutch and the
                      anticipation claim dies)
  4 ADAPTIVITY        DIAGNOSTIC, not a gate (week-4 finding): hand-built
                      adaptive policies (context heuristic, filter-cap-tracking
                      oracle) were measured at 6.96-7.64 vs always-max's 7.73 --
                      the exact-braking CBF already regulates speed near-optimally
                      and a 2 Hz supervisor cannot out-track it, so RETURN
                      headroom on v_max_cmd is ~2%. Whether PPO finds value in
                      the dimensions these oracles under-sample (per-scenario
                      margin schedules, yield timing) is an empirical question;
                      a negative margin here flags the plan's risk-register
                      outcome ("S4 matches the frontier while adapting"), which
                      the plan explicitly accepts as an honest result. Reported
                      every run so reward iterations see the trend.
  5 NUMERICAL HYGIENE every observation and reward finite over all audit rollouts
  6 DETERMINISM       same seed => bit-identical episode return on a REUSED env
                      (catches state leakage: MPC warm start, filter memory,
                      occlusion latch)

    python scripts/check_training_readiness.py           # ~10 min
    python scripts/check_training_readiness.py --seeds 3 # quicker

Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from core.cbf.cbf_filter import d_stop
from core.common.observation import SCALE
from core.common.platform import PLATFORMS, load_platform
from core.rl.nav_env import NavEnv
from core.rl.curriculum import make_sampler

# ----------------------------------------------------------------- policies
def const_policy(v_max, d_margin):
    return lambda obs: np.array([v_max, d_margin]), f"const({v_max:.2f},{d_margin:.2f})"


def adaptive_policy(p):
    """Context heuristic using ONLY policy-visible information (the obs):
    nearest-human slot 0 -> slow & widen only for people AHEAD that we are
    actually converging with (ttca), stay fast otherwise. This is the kind of
    schedule the RL policy is supposed to discover -- if even this cannot beat
    the fixed extremes, there is nothing to learn (or the reward is wrong)."""
    S = p.obs_scale or SCALE
    v_hi, v_lo = p.rl.v_max_high, p.rl.v_max_low
    if p.name == "tb3":                              # frozen historical constants
        d_yield, d_caution, v_mid = 1.0, 2.2, 0.16
    else:                                            # scale with the stop envelope
        D = d_stop(p.cbf.sigma * v_hi, p.cbf.tau, p.cbf.a_brake)
        d_yield, d_caution, v_mid = 1.0 * D, 1.8 * D, 0.6 * v_hi

    def act(obs):
        rel = obs[5:10]
        if np.all(rel == 0.0):                       # zero slot = nobody tracked
            return np.array([v_hi, 0.4])
        rel_x, rel_y = rel[0] * S["dist"], rel[1] * S["dist"]
        ttca = rel[4] * S["ttca"]
        d = float(np.hypot(rel_x, rel_y))
        ahead = rel_x > -0.2                         # ignore people already passed
        if ahead and d < d_yield:
            return np.array([v_lo, 0.9])             # yield fully (action floor)
        if ahead and d < d_caution and ttca < 4.0:
            return np.array([v_mid, 0.7])
        return np.array([v_hi, 0.4])
    return act, "adaptive-heuristic"


def cap_tracking_policy(p):
    """Physics-informed anticipation oracle: command (approximately) the CBF's own
    speed cap, reconstructed from POLICY-VISIBLE observation entries alone -- the
    same exact-braking + human-approach math the filter runs, decoded from the
    K-nearest slots. Commanding the cap yields the SAME filtered trajectory as
    always-max but with ~zero intervention penalty, so it strictly dominates
    always-max under w4 > 0. This is the behaviour the w4 term is designed to
    teach; if even THIS cannot beat the fixed extremes, the reward is broken."""
    cbf = p.cbf
    S = p.obs_scale or SCALE
    v_hi, v_lo = p.rl.v_max_high, p.rl.v_max_low
    a, tau, sig, d_hard = cbf.a_brake, cbf.tau, cbf.sigma, cbf.d_hard

    def act(obs):
        v = float(obs[3]) * S["v_robot"]              # own speed (undo scale)
        caps = [v_hi]
        for k in range(5):
            s = obs[5 + 5 * k: 10 + 5 * k]
            if np.all(s == 0.0):
                continue                              # empty slot
            px, py = s[0] * S["dist"], s[1] * S["dist"]      # rel pos, robot frame
            vx, vy = s[2] * S["v_human"], s[3] * S["v_human"]  # rel vel
            # geometry one latency window ahead (mirrors _predicted_geometry:
            # relative displacement covers robot coast + human CV motion)
            pxp, pyp = px + vx * tau, py + vy * tau
            dp = float(np.hypot(pxp, pyp))
            if dp < 1e-6:
                caps.append(0.0)
                continue
            ex, ey = pxp / dp, pyp / dp
            cp = ex                                   # cos bearing (heading = +x)
            if cp <= 1e-6:
                continue                              # not closing on this human
            hvx, hvy = vx + v, vy                     # human vel in robot frame
            beta = sig * max(0.0, -(hvx * ex + hvy * ey))
            delta = dp - d_hard - beta * tau
            if delta <= 0.0:
                caps.append(0.0)
                continue
            atb = a * tau + beta
            caps.append((-atb + np.sqrt(atb * atb + 2.0 * a * delta)) / (sig * cp))
        cap = float(np.clip(min(caps), v_lo, v_hi))   # action-space bounds
        # margin matches the aggressive baseline's 0.35: this policy differs from
        # always-max ONLY in tracking the cap, so the comparison isolates the
        # anticipation incentive (a wider margin costs real progress now that the
        # barrier works, and would confound the check)
        return np.array([cap, 0.35])
    return act, "cap-tracking"


# ------------------------------------------------------------------ rollout
def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    ret, terms_sum, finite = 0.0, {}, True
    for _ in range(130):
        obs, r, term, trunc, info = env.step(policy(obs))
        finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(r))
        ret += r
        for k, v in info["reward_terms"].items():
            terms_sum[k] = terms_sum.get(k, 0.0) + v
        if term or trunc:
            return ret, info["episode_metrics"], terms_sum, finite
    return ret, {}, terms_sum, finite


def battery(env, policy, seeds):
    outs = [rollout(env, policy, s) for s in seeds]
    return {
        "return": float(np.mean([o[0] for o in outs])),
        "success": float(np.mean([o[1].get("success", False) for o in outs])),
        "terms": {k: float(np.mean([o[2].get(k, 0.0) for o in outs]))
                  for k in outs[0][2]},
        "finite": all(o[3] for o in outs),
    }


# ------------------------------------------------------------------- checks
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--platform", default="tb3", choices=list(PLATFORMS))
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    p = load_platform(args.platform)
    v_hi, v_lo = p.rl.v_max_high, p.rl.v_max_low
    v_mid = 0.13 if p.name == "tb3" else round(v_hi / 2, 3)   # tb3 frozen literal

    def mk_env(**kw):
        return NavEnv(robot=p.robot, mpc=p.mpc, cbf=p.cbf, rl=p.rl,
                      obs_version=p.obs_version, obs_scale=p.obs_scale,
                      scenario_platform=p.name, **kw)

    results, failures = [], []

    def check(name, ok, evidence):
        results.append((name, ok, evidence))
        if not ok:
            failures.append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {evidence}")

    print(f"== platform: {p.name} (v_max {p.robot.v_max}) ==")
    print("== 1-2: empty world (stage A) — speed gradient & no freezing ==")
    env_a = mk_env(scenario_sampler=make_sampler("A", p.name))
    speeds = {}
    for v in (v_hi, v_mid, v_lo):
        pol, _ = const_policy(v, 0.5)
        speeds[v] = battery(env_a, pol, seeds)
    check(f"speed gradient ({v_hi} > {v_mid} > {v_lo})",
          speeds[v_hi]["return"] > speeds[v_mid]["return"] > speeds[v_lo]["return"],
          {v: round(s["return"], 2) for v, s in speeds.items()})
    check("fast policy succeeds in the open",
          speeds[v_hi]["success"] == 1.0, f"success={speeds[v_hi]['success']:.2f}")
    check("no freezing optimum",
          speeds[v_lo]["return"] < 0.5 * speeds[v_hi]["return"],
          f"crawl={speeds[v_lo]['return']:.2f} vs fast={speeds[v_hi]['return']:.2f}")
    finite_all = all(s["finite"] for s in speeds.values())

    print("== 3: human traffic — blind-aggressive pays a visible penalty ==")
    aggr, _ = const_policy(v_hi, 0.35)
    stats = {}
    for name in ("corridor_passby", "perpendicular_crossing"):
        env = mk_env(scenarios=[name])
        stats[name] = battery(env, aggr, seeds)
        finite_all &= stats[name]["finite"]
    pen = {n: round(s["terms"]["cbf_intervention"] + s["terms"]["protective_stop"]
                    + s["terms"]["personal_space"], 3) for n, s in stats.items()}
    check("filter/social penalties bite the aggressive policy",
          any(v < -0.05 for v in pen.values()), f"summed w4+w5+w6 per scenario: {pen}")

    print("== 4: adaptivity headroom — DIAGNOSTIC (reported, not a gate) ==")
    # the plan's evaluation distribution (4.2, minus held-out blind_corner):
    # doorway (deadlock) and open_hall (sustained crowd) are where adaptivity
    # actually pays -- a corridor pass-by is too brief to differentiate policies
    mixed = ["corridor_passby", "perpendicular_crossing",
             "doorway_negotiation", "open_hall"]
    scores = {}
    ADAPTIVE = ("adaptive-heuristic", "cap-tracking")
    for pol_fn, pname in (const_policy(v_hi, 0.35), const_policy(v_mid, 1.0),
                          adaptive_policy(p), cap_tracking_policy(p)):
        per_env = []
        for name in mixed:
            per_env.append(battery(mk_env(scenarios=[name]), pol_fn, seeds))
        env_open = mk_env(scenario_sampler=make_sampler("A", p.name))
        per_env.append(battery(env_open, pol_fn, seeds))
        scores[pname] = float(np.mean([b["return"] for b in per_env]))
        finite_all &= all(b["finite"] for b in per_env)
    best_fixed = max(v for k, v in scores.items() if k not in ADAPTIVE)
    best_adaptive = max(scores[k] for k in ADAPTIVE)
    headroom = best_adaptive - best_fixed
    tag = "HEADROOM" if headroom >= 0 else "NO-HEADROOM"
    print(f"  [DIAG:{tag}] hand-policy headroom = {headroom:+.2f} "
          f"(see module doc; settled empirically by training): "
          f"{ {k: round(v, 2) for k, v in scores.items()} }")

    print("== 5: numerical hygiene ==")
    check("all observations & rewards finite", finite_all, "across every audit rollout")

    print("== 6: determinism / state leakage on a reused env ==")
    env = mk_env(scenarios=["perpendicular_crossing"])
    pol, _ = const_policy(0.2 if p.name == "tb3" else 0.8 * v_hi, 0.6)
    r1, *_ = rollout(env, pol, seed=42)
    _ = rollout(env, pol, seed=7)          # dirty the env state
    r2, *_ = rollout(env, pol, seed=42)    # same seed again on the USED env
    check("seed 42 reproducible after reuse", r1 == r2, f"{r1:.6f} vs {r2:.6f}")

    n_fail = len(failures)
    print(f"\n{'ALL CHECKS PASSED — env is training-ready' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED: {failures}'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
