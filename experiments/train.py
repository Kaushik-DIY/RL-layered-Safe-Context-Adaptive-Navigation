"""Week-4 PPO training (plan D6): the RL supervisor over the MPC+CBF stack.

    python -m experiments.train --stage A --steps 500000 --seed 0
    python -m experiments.train --stage B --resume models/ppo_A_s0_final.zip
    make train STAGE=A

Design decisions (D6, restated where they bite):
  * PPO / MlpPolicy 2x256, CPU -- tiny network, on-policy stability, few knobs.
  * The observation is normalized BY CONSTRUCTION (core.common.observation.SCALE),
    so no VecNormalize wrapper: the ONNX export (week 5) then needs no running
    statistics and the 2D->Gazebo transfer story stays clean.
  * EVERY reward term is logged separately (reward_terms/*) plus the plan-4.3
    episode metrics (metrics/*) -- reward surgery without per-term logs is the
    classic RL time sink (risk register).
  * Domain randomization switches on from stage B (curriculum table in rl.yaml).
  * Checkpoints every `checkpoint_every` steps -- training must survive
    interruption (Kaggle session limits, plan sec. 8).

Honest throughput note: one env step = `decision_every` MPC solves, so the wall
clock is MPC-bound (~20-40 ms/solve), not sim-bound; the plan's 1000x-realtime
figure applies to the raw 2D sim, not the MPC-in-the-loop env. Measure with
--smoke before committing to a step budget.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor)

from core.common.params import load_yaml
from core.common.platform import PLATFORMS, load_platform
from core.rl.curriculum import make_sampler
from core.rl.nav_env import NavEnv

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "experiments" / "models"
RUNS = REPO / "runs"

# episode metrics worth a TensorBoard curve (subset of the 4.3 families)
TB_METRICS = ("success", "collision", "time_to_goal", "min_human_dist",
              "protective_stops", "intervention_rate", "mean_intervention",
              "energy", "full_stops", "rms_jerk", "intrusion_time")


def make_env(stage: str, domain_rand: dict | None, seed: int,
             platform: str = "tb3"):
    """Top-level factory (must be picklable for SubprocVecEnv) -- the platform is
    passed by NAME and loaded inside the child process."""
    def _init() -> NavEnv:
        p = load_platform(platform)
        env = NavEnv(scenario_sampler=make_sampler(stage, platform), use_cbf=True,
                     domain_rand=domain_rand,
                     robot=p.robot, mpc=p.mpc, cbf=p.cbf, rl=p.rl,
                     obs_version=p.obs_version, obs_scale=p.obs_scale)
        env.reset(seed=seed)
        return env
    return _init


class NavLogger(BaseCallback):
    """Per-term reward + episode-metric TensorBoard logging (plan D6)."""

    def _on_rollout_start(self) -> None:
        self._terms: dict[str, float] = defaultdict(float)
        self._n_steps = 0
        self._eps: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            terms = info.get("reward_terms")
            if terms:
                for k, v in terms.items():
                    self._terms[k] += v
                self._n_steps += 1
            ep = info.get("episode_metrics")
            if ep:
                self._eps.append(ep)
        return True

    def _on_rollout_end(self) -> None:
        for k, v in self._terms.items():
            self.logger.record(f"reward_terms/{k}", v / max(self._n_steps, 1))
        for key in TB_METRICS:
            vals = [float(e[key]) for e in self._eps
                    if key in e and np.isfinite(e[key])]
            if vals:
                self.logger.record(f"metrics/{key}", float(np.mean(vals)))
        if self._eps:
            self.logger.record("metrics/episodes_in_rollout", len(self._eps))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", default="A", choices=["A", "B", "C"])
    p.add_argument("--steps", type=int, default=None,
                   help="RL decision steps (default: rl.yaml total_timesteps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--platform", default="tb3", choices=list(PLATFORMS),
                   help="parameter stack (core/common/platform.py); tb3 names "
                        "stay unchanged so existing runs/models keep their ids")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint .zip to continue from (e.g. previous stage)")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="override rl.yaml checkpoint_every (steps). Lower it for "
                        "more frequent recovery points on slow/timeout-prone runs.")
    p.add_argument("--ent-coef", type=float, default=None,
                   help="PPO entropy bonus. SB3 default 0 gives NO sustained "
                        "exploration -- fine once converged, but a resumed policy "
                        "must EXPLORE to discover a newly-rewarded behavior (e.g. "
                        "corner-slowing under w8). Set ~0.01 for the industrial "
                        "retrain; applied to fresh AND resumed models.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 2 envs, 512 steps, throughput report")
    args = p.parse_args()

    cfg = load_yaml("rl")
    tr = cfg["training"]
    n_envs = args.n_envs or tr["n_envs"]
    steps = args.steps or tr["total_timesteps"]
    if args.smoke:
        n_envs, steps = 2, 512
    domain_rand = cfg["domain_randomization"] if args.stage in ("B", "C") else None
    prefix = "ppo" if args.platform == "tb3" else f"ppo_{args.platform[:3]}"
    run_name = f"{prefix}_{args.stage}_s{args.seed}"

    fns = [make_env(args.stage, domain_rand, seed=args.seed * 1000 + i,
                    platform=args.platform)
           for i in range(n_envs)]
    venv = VecMonitor(SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns))

    MODELS.mkdir(parents=True, exist_ok=True)
    if args.resume:
        # cross-version tolerant load: a checkpoint saved under NumPy 2 (Kaggle)
        # cannot unpickle its spaces under NumPy 1 (local venv). Substitute the
        # venv's identical spaces + the config schedules instead of deserializing
        # (torch weights/optimizer load fine -- version-agnostic). Harmless when
        # the versions already match. Lets a Kaggle checkpoint resume ANYWHERE.
        custom = {"observation_space": venv.observation_space,
                  "action_space": venv.action_space,
                  "lr_schedule": lambda _: 3e-4, "clip_range": lambda _: 0.2,
                  "_last_obs": None, "_last_episode_starts": None,
                  "ep_info_buffer": None, "ep_success_buffer": None}
        model = PPO.load(args.resume, env=venv, device=tr["device"],
                         custom_objects=custom)
        model.set_random_seed(args.seed)
        model.tensorboard_log = str(RUNS)   # a Kaggle-saved model carries /kaggle/...
        if args.ent_coef is not None:       # re-open exploration for a new behavior
            model.ent_coef = args.ent_coef
        print(f"resumed from {args.resume} (at {model.num_timesteps} steps; "
              f"ent_coef={model.ent_coef})")
    else:
        # smoke: short rollouts so the run finishes in ~1 min (default n_steps=2048
        # would force a 4096-step first rollout regardless of --steps)
        ppo_kwargs = {"n_steps": 128, "batch_size": 64} if args.smoke else {}
        ent = args.ent_coef if args.ent_coef is not None else tr.get("ent_coef", 0.0)
        model = PPO(cfg["policy"], venv, seed=args.seed, device=tr["device"],
                    policy_kwargs={"net_arch": list(cfg["net_arch"])},
                    ent_coef=ent, tensorboard_log=str(RUNS), verbose=1, **ppo_kwargs)

    ckpt_every = args.checkpoint_every or tr["checkpoint_every"]
    callbacks = [NavLogger(),
                 CheckpointCallback(save_freq=max(ckpt_every // n_envs, 1),
                                    save_path=str(MODELS), name_prefix=run_name)]
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=callbacks, tb_log_name=run_name,
                reset_num_timesteps=args.resume is None)
    dt_wall = time.time() - t0

    final = MODELS / f"{run_name}_final.zip"
    model.save(final)
    sps = steps / dt_wall
    print(f"\nstage {args.stage}: {steps} steps in {dt_wall / 60:.1f} min "
          f"({sps:.1f} decision-steps/s, {sps * cfg['decision_every_n_mpc_steps']:.0f} MPC solves/s)")
    print(f"model  -> {final}")
    print(f"logs   -> {RUNS}/{run_name}_* (make tensorboard)")


if __name__ == "__main__":
    main()
