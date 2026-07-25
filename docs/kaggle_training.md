# Training on Kaggle (plan D6 / sec. 8)

The training stack (`core/` + `experiments/train.py`) is deliberately ROS-free and
imports clean in a bare environment — verified with `env -i` before writing this.

## Honest hardware note

**The GPU will sit mostly idle.** Training is MPC-bound (each decision step = 5
CasADi/IPOPT solves on CPU, ~30 ms each); the policy is a 2x256 MLP whose
forward/backward is negligible. What Kaggle actually buys us:

- dedicated CPU cores (no laptop thermal throttling, no Gazebo contention),
- 12 h sessions ≈ 1 full curriculum stage per session,
- parallel sessions = parallel seeds (the plan needs 5 seeds in week 5).

**Session settings:** accelerator = None (CPU). GPU sessions draw from the
~30 h/week GPU quota while giving the SAME ~4 vCPUs — for this workload they
spend a budget to accelerate nothing. CPU sessions have no weekly quota.

**vCPUs are not a setting.** Every session has ~4 cores; parallelism comes from
our SubprocVecEnv via `--n-envs`. Check `os.cpu_count()` in the first cell and
pass `--n-envs <cores>` — do not oversubscribe (8 envs on 4 cores just adds
context switching for pure-CPU IPOPT). Calibrate once with `--smoke --n-envs 4`.

**Parallel seeds:** run several CPU sessions concurrently, one seed each.

## Setup (first notebook cell) -- the commit-safe recipe

The dataset mount at /kaggle/input is READ-ONLY, but train.py writes checkpoints
and TensorBoard logs into the repo tree -- so copy the project to /kaggle/working
(persisted as notebook Output) and run from there.

**USE PYTHON + `subprocess`, NOT `!` SHELL CELLS.** Three separate failures came
from `!` cells, all now avoided by driving everything from plain Python:

1. `!` commands are ONE line -- backslash continuation feeds the next line to the
   Python parser (`SyntaxError: unterminated string literal`). Worse, a long
   single-line `!` command WRAPS in a terminal and copy-paste turns the wrap into a
   real newline, re-breaking it. Short Python lines paste cleanly.
2. `!` cells do NOT reliably inherit `os.environ["PYTHONPATH"]`. `subprocess.run(...,
   env=dict(os.environ, PYTHONPATH=PP))` passes it to the child EXPLICITLY.
3. A failed `!cp`/`!pip` does not stop the notebook, so the real error surfaces two
   cells later as a mystery. Python `assert` + `check=True` abort at the exact step.

**Install into the SYSTEM env (NO `--target`).** A `--target=/kaggle/working/pylibs`
install pulls a FRESH torch/numpy into pylibs (torch 2.13.0 / numpy 2.5.1 on
2026-07-17) that shadow Kaggle's tested system versions (torch 2.10.0) -- the binary
mix SEGFAULTs the moment torch is used (both model-load and training died with
SIGSEGV). A plain `pip install` keeps pip's default only-if-needed resolution, so
torch/numpy stay as Kaggle's ABI-correct image versions. This reopens the old
**proxsuite `cmeel` layout** worry (its module sits under
`.../cmeel.prefix/lib/python3.12/site-packages/proxsuite`, and a `.pth` shim is only
processed at interpreter startup) -- BUT we run every step via `subprocess.run([...])`,
and a fresh subprocess interpreter DOES process that `.pth`, so proxsuite imports
fine. We also locate the cmeel site-packages and add it to PYTHONPATH as insurance.
And do NOT hard-code the dataset slug -- locate the repo by searching for
`experiments/train.py` (the mount layout has bitten us twice).

RUN CELL 1 INTERACTIVELY (just this cell, ~3 min) and confirm `nav ready` +
`deps OK` BEFORE committing the multi-hour job. Every past failure surfaced 10 s
into a commit; a live run of the setup cell catches it while you watch.

```python
# Cell 1 -- setup (idempotent; internet ON). All lines short -> paste-safe.
import os, glob, shutil, subprocess, sys, zipfile

# 1. find the repo anywhere under /kaggle/input (any depth); if only a zip is
#    present (not auto-extracted), unpack it and search again.
hits = glob.glob("/kaggle/input/**/experiments/train.py", recursive=True)
if not hits:
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    print("no train.py yet; zips found:", zips)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall("/kaggle/working/unz")
    hits = glob.glob("/kaggle/working/unz/**/experiments/train.py",
                     recursive=True)
if not hits:
    print("=== /kaggle/input tree ===")
    for i, (d, subs, files) in enumerate(os.walk("/kaggle/input")):
        print(d, files[:6])
        if i > 60:
            break
    raise SystemExit("repo not found; paste the tree above")

root = os.path.dirname(os.path.dirname(hits[0]))
print("repo root:", root)

shutil.rmtree("/kaggle/working/nav", ignore_errors=True)
shutil.copytree(root, "/kaggle/working/nav")

# 2. resume model: Kaggle RECURSIVELY unzips nested archives (verified 2026-07-17),
#    so a bundled ppo_B_s0_final.zip arrives as a DIRECTORY 'ppo_B_s0_final' (the
#    .zip suffix stripped) holding data/policy.pth/... -> re-zip it back to a real
#    archive for PPO.load. (drop this block for a cold start with no --resume.)
base = "/kaggle/working/nav/experiments/models/ppo_B_s0_final"
mp = base + ".zip"
if not os.path.isfile(mp) and os.path.isdir(base):
    shutil.make_archive(base, "zip", base)
assert os.path.isfile(mp), "resume model missing!"
print("nav ready; model OK; cores:", os.cpu_count())

# 3. install into the SYSTEM env (NO --target) so torch/numpy come from Kaggle's
#    tested image -- a --target install pulled torch 2.13.0/numpy 2.5.1 that
#    SEGFAULTED. pip's default only-if-needed keeps the system torch/numpy.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "casadi==3.7.2", "proxsuite==0.7.3",
    "gymnasium==1.3.0", "stable-baselines3==2.9.0",
], check=True)

# 4. PYTHONPATH: repo + proxsuite's cmeel site-packages if present (a fresh
#    subprocess interpreter processes the cmeel .pth anyway; this is insurance).
cmeel = glob.glob("/usr/**/cmeel.prefix/lib/python*/site-packages/proxsuite",
                  recursive=True)
extra = os.path.dirname(cmeel[0]) if cmeel else ""
PP = ":".join(p for p in ["/kaggle/working/nav", extra] if p)
print("PYTHONPATH:", PP)

# 5. sanity: imports must resolve AND not segfault; print the versions we landed on
subprocess.run(
    [sys.executable, "-c",
     "import torch, numpy, casadi, proxsuite, gymnasium, stable_baselines3 as sb3; "
     "print('deps OK; torch', torch.__version__, 'numpy', numpy.__version__, "
     "'sb3', sb3.__version__)"],
    env=dict(os.environ, PYTHONPATH=PP), check=True,
)
```

(offline fallback: upload `navrl-wheels.zip` as a second dataset and
`pip install --no-index --find-links <wheels-dir> ...` the same four packages.)

## Train (a second cell; reuses `PP` from Cell 1)

```python
# Cell 2 -- stage C (see budgets below). Every line short -> paste-safe.
import os, sys, subprocess

subprocess.run(
    [sys.executable, "-m", "experiments.train",
     "--stage", "C", "--steps", "200000", "--seed", "0", "--n-envs", "4",
     "--resume", "experiments/models/ppo_B_s0_final.zip"],
    cwd="/kaggle/working/nav",
    env=dict(os.environ, PYTHONPATH=PP),
    check=True,
)
```

The optional pre-training audit is the same pattern with
`["scripts/check_training_readiness.py", "--seeds", "3"]` as the arg list.

For multi-hour runs use **Save Version -> Save & Run All (Commit)** -- that executes
detached for up to 12 h with outputs persisted, instead of relying on the
interactive session staying alive (an interactive kernel that drops LOSES the run).
Commit kernels start from a FRESH /kaggle/working, so Cell 1 re-runs the copy +
install every commit -- that is why it is written to be idempotent. Artifacts to
download afterwards from the notebook's Output tab: `nav/experiments/models/*.zip`
and `nav/runs/`.

Checkpoints save every 100k steps (`experiments/models/`), so a session
interruption loses at most 100k steps — resume with `--resume` on the newest
checkpoint. TensorBoard event files are in `runs/`; download and inspect
`reward_terms/*` and `metrics/*` after every stage (reward surgery decisions are
made from these, plan D6).

## Stage C (crowds + corridors + doorways) -- resume from stage B

Stage C is the G4 remedy: the stage-B policy saturates aggressive (v_max 0.26,
margin at the 0.30 floor) in corridors because corridors are OUT of distribution for
stages A/B -- it beats S1 on time-to-goal but MISSES S2 on min-distance (0.470 vs
0.572 m). Stage C's sampler is 25% corridor / 25% doorway / 25% open_hall / 25%
free-roam(4-8 peds), with domain randomization ON, so it trains the margin dimension
exactly where the miss lives.

**The resume checkpoint must be in the DATASET.** `ppo_B_s0_final.zip` is a training
OUTPUT -- re-upload the code dataset with `experiments/models/ppo_B_s0_final.zip`
bundled in (the rebuilt `context-adaptive-navigation-kaggle.zip` already contains
it), and bump the dataset to a new version so /kaggle/input has it. NOTE the stage-B
model was trained against the PRE-fix filter; stage C now trains against the FIXED
interval-sampling filter -- a small, intended distribution shift the policy adapts to.

(The stage-C training cell is the `subprocess.run(... --stage C ...)` block shown
under "Train" above -- one 12 h commit, resumes stage B, DR on, C curriculum.)

## Industrial platform training (2026-07 replan)

Fresh policy at MiR-class dynamics (`--platform industrial`: v 1.5 m/s, obs v2
with occlusion features, corners + interferer in the curriculum). No resume model
needed for commit 1 -- upload `navrl-industrial-kaggle.zip` (code only) as a new
dataset. Two commits:

1. commit 1: audit (provenance) + stage A' 50k + stage B' 150k chained (~9-10 h)
   -- B' resumes A's file locally inside the same container (the proven pattern)
2. commit 2: stage C' 200k, `--resume experiments/models/ppo_ind_B_s0_final.zip`
   (bundle the downloaded B model into the dataset -- remember Kaggle strips
   nested .zip into a DIRECTORY named without the suffix; Cell 1's re-zip block
   handles it, adjust the model name to ppo_ind_B_s0_final)

Training args (Cell-2/3 pattern, same subprocess recipe):
    ["-m", "experiments.train", "--platform", "industrial", "--stage", "A",
     "--steps", "50000", "--seed", "0", "--n-envs", "4"]
    ["-m", "experiments.train", "--platform", "industrial", "--stage", "B",
     "--steps", "150000", "--seed", "0", "--n-envs", "4",
     "--resume", "experiments/models/ppo_ind_A_s0_final.zip"]
Optional provenance cell before training:
    ["scripts/check_training_readiness.py", "--seeds", "2",
     "--platform", "industrial"]
Run names carry the platform prefix: ppo_ind_A_s0 / ppo_ind_B_s0 / ppo_ind_C_s0.

Output: `experiments/models/ppo_C_s0_final.zip` (+ `ppo_C_s0_100000_steps.zip`
checkpoint -- `checkpoint_every`=100k TOTAL steps, so a 200k run yields one interim
checkpoint + the final) and `runs/ppo_C_s0_*`. Download both. If the run is cut
short, resume from the newest `ppo_C_s0_<N>_steps.zip` in the next commit
(`reset_num_timesteps=False` continues the step count).

After stage C: download the model, `make eval MODEL=experiments/models/ppo_C_s0_final.zip`
locally to re-check Gate G4 (min-distance), then `make onnx MODEL=...` and re-run the
S4/S5 batteries with the stage-C policy.

## Step budgets (measured on Kaggle, not the plan's optimistic 2-4 h)

Observed ~6 decision-steps/s on 4 envs (pure-CPU IPOPT, MPC-bound) => 60-80k ≈
3-4 h, 200k ≈ 9-10 h (fits one 12 h commit), 500k ≈ 24 h (needs 2-3 chained
commits). Stage A needs little (empty world; 60-80k converges); spend the budget on
B/C. Gate G4 is checked after training: beat S1 on time-to-goal AND S2 on
min-distance in scenario 1 (corridor_passby).
