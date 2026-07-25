# Safe Context-Adaptive Navigation: RL-Modulated MPC with an ISO 3691-4-Informed Safety Filter

**Complete Implementation Plan — Step 0 through Final Presentation**

Target platform: TurtleBot3 Waffle · Gazebo Classic 11 · ROS 2 Humble · Timeline: 8 weeks + 1 buffer

---

## 1. Project Statement and Positioning

**One-sentence pitch:** A three-layer navigation stack in which a reinforcement learning policy adaptively modulates the operating parameters of a self-built MPC tracking controller based on predicted pedestrian context, while a control barrier function safety filter derived from the ISO 3691-4 stopping-distance principle guarantees that safety constraints are never violated, regardless of what the learned policy commands.

**The gap being addressed:** Classical MPC local planners for AMRs run with fixed, hand-tuned parameters (one speed cap, one safety margin, humans treated as static obstacles). This forces a global trade-off: conservative tuning produces stop-start motion, wasted energy, low throughput, and freezing in crowds; aggressive tuning erodes safety margins. Industry does not deploy end-to-end RL on safety-critical AMRs because learned policies are not certifiable. The industrially honest architecture is therefore: keep the certifiable classical layer, add a learned layer that only *modulates* it, and interpose a formally-guaranteed safety filter between them.

**The three claims the project must support with evidence:**

1. *Adaptivity claim:* RL-modulated MPC achieves the efficiency of aggressive tuning with the safety of conservative tuning (Pareto improvement over any fixed tuning).
2. *Safety claim:* The CBF filter guarantees zero violations of the stopping-distance constraint even under an adversarial or untrained RL policy (demonstrated, not asserted).
3. *Anticipation claim:* The learned policy decelerates in anticipation of human encounters rather than relying on the filter as a crutch (measured via filter-intervention rate and velocity-vs-distance profiles).

**What this project is NOT (state this in the README):** Not a certified ISO 3691-4 system — full compliance requires PL-rated hardware (ISO 13849-1) and certified safety controllers. The project encodes the standard's *physical principle* (speed limited by guaranteed stopping distance within the cleared protective field) as a hard constraint. This is an "ISO 3691-4-informed" design study. Stating this boundary explicitly is itself a competence signal.

---

## 2. Architecture (the thing on your first slide)

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — RL CONTEXT POLICY (2 Hz)                          │
│  In:  goal-relative state, robot state, K nearest humans     │
│       (rel. pos, rel. vel, TTC), zone flag                   │
│  Out: p = [v_max_cmd, d_margin_cmd]  (MPC parameters)        │
│  PPO, MLP 2×256, trained in fast 2D sim                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ parameters p (bounded, clipped)
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 2 — MPC TRACKING CONTROLLER (10 Hz)                   │
│  Unicycle model, direct multiple shooting, CasADi/IPOPT      │
│  Cost: goal tracking + effort + smoothness (Δu)              │
│  Constraints: v∈[0, v_max_cmd], ω, accel bounds,             │
│  soft static-obstacle constraints with slack                 │
│  Out: u_mpc = (v, ω) first control of horizon                │
└──────────────────────┬──────────────────────────────────────┘
                       │ proposed control u_mpc
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 1 — CBF SAFETY FILTER (10 Hz, runs on raw u)          │
│  Discrete-time CBF-QP: min ‖u − u_mpc‖²  s.t.                │
│  h(x_{k+1}) ≥ (1−γ)·h(x_k)  for every tracked human          │
│  h = d_human − d_stop(v) − d_hard                            │
│  d_stop(v) = v·τ + v²/(2·a_brake)   (ISO principle)          │
│  + protective-field hard stop (non-negotiable)               │
│  Out: u_safe → robot                                         │
└──────────────────────────────────────────────────────────────┘
```

Key property: Layer 1 has **no tunable inputs from Layer 3**. The RL policy can request margins *above* the hard floor (d_margin_cmd ≥ d_hard is enforced by clipping), but the floor itself, a_brake, τ, and the filter logic are frozen constants. This is the separation that makes the reliability argument airtight.

**Prior-art anchors to cite (pre-empts "is this naive?"):** predictive safety filters (Wabersich & Zeilinger, Automatica 2021), CBF-QP safety-critical control (Ames et al., TAC 2017), discrete-time CBFs for wheeled robots (Agrawal & Sreenath, RSS 2017), RL-tunes-controller-parameters pattern (adaptive-CBF/RHC works, CDC 2024), residual/hybrid deployment pattern (DR-MPC, 2024).

---

## 3. Foundational Decisions and Their Justifications

### D1. Two-simulator strategy (the single biggest de-risking decision)

**Decision:** Build a lightweight custom 2D kinematic Python simulator (unicycle robot + social-force pedestrians) for all RL training; use Gazebo Classic 11 + HuNavSim exclusively for validation, evaluation, and video generation.

**Justification:** Gazebo Classic runs at roughly real-time and cannot be trivially vectorized; PPO needs 2–5 M environment steps. Training in Gazebo would consume the entire 8 weeks on compute alone and locks training to your laptop. A 2D kinematic sim runs at 1000×+ real-time, vectorizes across 8–16 parallel envs in SB3, and runs free on Kaggle. This is the standard "train cheap, validate rich" pattern. The 2D sim and Gazebo share the *same observation construction code* and the same MPC+CBF stack (pure Python/CasADi, simulator-agnostic), so the transfer gap is only in dynamics fidelity and pedestrian behavior — which the Gazebo evaluation explicitly measures and reports (this becomes a *result*, not a weakness: "policy trained in a 60-line kinematic sim transfers to Gazebo physics with X% performance retention").

**Fallback if 2D→Gazebo transfer is poor:** fine-tune the policy for 100–200 k steps directly in Gazebo (feasible because it is only fine-tuning) with domain randomization on a_brake, sensor noise, and pedestrian speeds added in the 2D sim first.

### D2. Robot model and parameters

**Decision:** TurtleBot3 Waffle differential drive, unicycle abstraction. States x = [x, y, θ], controls u = [v, ω]. Limits: v ∈ [0, 0.26] m/s, ω ∈ [−1.82, 1.82] rad/s, |a| ≤ 0.5 m/s² (enforced tighter in MPC: a_max = 0.2 m/s² for smoothness), braking deceleration a_brake = 0.3 m/s², **injected actuation/perception latency τ = 0.4 s** (deliberate, see below). Wheelbase 0.287 m, control dt = 0.1 s.

**Justification:** TB3 Waffle is your existing, hardware-credible platform. Its low top speed makes stopping distances small, which would trivialize the safety constraint — the injected latency τ = 0.4 s (justified as modeling perception + safety-PLC reaction time, which real AMRs have; ISO 3691-4 braking analysis includes system response time) restores a meaningful d_stop of ~0.22 m at full speed, on the same order as the robot footprint, so the constraint genuinely binds. **Appendix experiment:** rerun the full evaluation in the 2D sim with an industrial-AMR parameter set (v_max = 1.5 m/s, a_brake = 0.8 m/s², τ = 0.5 s → d_stop ≈ 2.2 m) to demonstrate the framework is platform-agnostic and that effects *grow* with scale. This kills the "toy robot, toy result" objection at near-zero cost.

### D3. MPC toolchain

**Decision:** Hand-formulated NMPC in **CasADi** with direct multiple shooting, solved by **IPOPT** initially; port to **acados** (SQP-RTI) only if the 10 Hz loop budget is violated.

**Justification:** You said "build my own MPC" — CasADi symbolic formulation is the level at which you demonstrate understanding (dynamics, discretization, cost, constraints, slack handling all written by you), while not reinventing an NLP solver (which no employer wants to see). For a 3-state unicycle with N = 20, IPOPT solves in 5–20 ms on a laptop CPU — comfortably within 100 ms. do-mpc is rejected (too much abstraction — undermines the "built my own" claim); acados-first is rejected (installation and CMake friction is a notorious week-eater; it is the optimization, not the starting point). Document solve times — "median 8 ms, p99 21 ms on an i5" is itself an industry-legible result.

### D4. MPC formulation (exact)

- **Prediction model:** unicycle, RK4 discretization, dt = 0.1 s, horizon N = 20 (2.0 s).
- **Decision variables:** states X ∈ R^{3×(N+1)}, controls U ∈ R^{2×N}, slacks S ≥ 0.
- **Cost:** Σ [ w_p‖pos_k − goal‖² + w_θ(heading error)² + w_v(v_k − v_ref)² + uᵀR u + Δuᵀ R_Δ Δu ] + terminal cost + w_s‖S‖². Δu penalty is what produces smooth profiles — report an ablation on R_Δ.
- **Hard constraints:** dynamics; u bounds with **v_k ≤ v_max_cmd (RL parameter)**; |Δv| ≤ a_max·dt.
- **Soft constraints (slack):** static obstacle clearance ‖pos_k − o_j‖ ≥ r_obs + r_robot − s; humans enter the MPC cost as ellipsoidal potential terms with radius **d_margin_cmd (RL parameter)** using constant-velocity human prediction over the horizon. Humans are *soft* in MPC (feasibility preserved) because the *hard* human guarantee lives in the CBF layer — this division of labor is a deliberate, defendable design: MPC optimizes comfort/efficiency around humans; CBF guarantees safety.
- **Reference:** global path from A* on a static occupancy grid (or straight-line to goal in open scenarios); MPC tracks a moving carrot on this path. Do not use Nav2's planner inside the loop — keep the stack self-contained and inspectable.

### D5. CBF safety filter (exact conditions)

**Primary constraint (ISO stopping-distance principle), per tracked human i:**

h_i(x) = d_i − ( v·τ + v²/(2·a_brake) ) − d_hard,  with d_i the distance to human i and d_hard = 0.30 m (robot radius + person clearance).

**Directionality:** the braking constraint applies to motion *toward* the human. Use the projected closing component: replace v with v·max(0, cos φ_i)·σ, where φ_i is the bearing of human i relative to the robot heading and σ a small robustness inflation (1.1). This avoids absurd conservatism when driving away from people while remaining conservative head-on.

**Discrete-time CBF condition (relative degree 1 in v — the fiddly nonholonomic part solved):** because h depends on u only through v (not ω), the constraint h(x_{k+1}) ≥ (1−γ)h(x_k) with one-step Euler prediction of d_i (using measured human velocity, constant-velocity assumption) is affine-in-v after linearizing v² about v_k — yielding a small QP:

min ‖u − u_mpc‖²_W  s.t.  per-human linearized DCBF constraints, u bounds, |Δu| bounds.

Solved with proxsuite (or qpOASES). W weights ω-deviation cheaper than v-deviation, so the filter prefers slowing over swerving (matches AMR practice). γ = 0.3 initially; report sensitivity.

**Second layer — protective-field emergency stop (non-negotiable, mirrors ESPE logic in ISO 3691-4):** if any human enters d < d_hard + 0.1 m, command v = 0 with max braking, overriding everything, and log the event as a *protective stop*. Well-designed runs should show the RL system triggering near-zero protective stops while fixed-aggressive MPC triggers many — this count is one of your headline metrics.

**Robustness handling:** human-velocity estimate noise → inflate d_stop by σ; unmodeled dynamics → conservative a_brake (use 60% of the platform's true capability). State these as deliberate robustness margins.

### D6. RL formulation

- **Algorithm:** PPO (stable-baselines3). Justification: on-policy stability, few sensitive hyperparameters, you have prior PPO experience, and the action space is a small box — SAC's sample-efficiency edge is irrelevant when the 2D sim is nearly free. SAC is the documented fallback if PPO plateaus.
- **Action (2 Hz decision rate, i.e., every 5 MPC steps):** a = [v_max_cmd ∈ [0.05, 0.26] m/s, d_margin_cmd ∈ [d_hard, 1.2] m]. Low decision rate is deliberate: it makes the policy a *supervisor*, not a controller; smooths behavior; and shrinks the effective horizon problem. (Optional extension, only if weeks 4–5 go smoothly: add a discrete strategy head {proceed, yield-and-wait, commit-through} for the doorway scenario.)
- **Observation (dim ≈ 3 + 2 + 5×5 + 2 = 32):** goal distance & bearing, path curvature ahead; robot v, ω; K = 5 nearest humans, each as [rel. x, rel. y, rel. vx, rel. vy, time-to-closest-approach], zero-padded, sorted by distance; current (v_max, d_margin) in effect. All quantities available identically in both simulators and, later, from any real tracker — no images, no lidar rasters (keeps the policy tiny, trainable on CPU, and transferable).
- **Reward (per MPC step, summed over the 5-step decision window):**
  - +w₁ · progress along path (dominant term)
  - −w₂ · energy proxy: |a|·dt (penalizes stop-start; this is your energy story)
  - −w₃ · jerk² (comfort)
  - −w₄ · **CBF intervention magnitude ‖u_safe − u_mpc‖** ← the key design element: teaches anticipation. The policy learns to slow down *before* the filter would force it, because filter corrections are penalized. This term is what converts "safety filter as crutch" into "safety filter as verifier."
  - −w₅ · protective-stop event (large, e.g., −5)
  - −w₆ · personal-space intrusion time (d < 0.5 m to any human)
  - +w₇ · terminal success bonus; small per-step time penalty; episode timeout 60 s.
  - Initial weights w = [1.0, 0.3, 0.1, 0.5, 5.0, 0.2, 10.0]; expect 2–3 reward-tuning iterations (budgeted in Week 4–5). Log every reward component separately in TensorBoard from day one — reward debugging without per-term logs is the classic RL time sink.
- **Training regime:** 8–16 vectorized 2D envs, 3 M steps target (~2–4 h on Kaggle CPU; the MLP is tiny so GPU is optional), 5 seeds for the final policy, curriculum: Stage A empty world (learn to drive fast) → Stage B 1–2 pedestrians, simple crossings → Stage C 4–8 pedestrians, corridors and doorways. Domain randomization from Stage B: pedestrian speed [0.6, 1.5] m/s, a_brake ±20%, observation noise on human velocity (σ = 0.05 m/s), τ ±0.1 s.

### D7. Pedestrian simulation

- **2D training sim:** social-force model (Helbing–Molnár) pedestrians with randomized goals, speeds, and group behavior; scripted scenario generators for the five evaluation scenarios plus randomized free-roam. Implement SFM yourself (~80 lines) — full control, no dependency risk.
- **Gazebo evaluation:** **HuNavSim** with its Gazebo Classic wrapper (ROS 2, purpose-built for benchmarking human-aware navigation, provides behavior-realistic agents and its own metrics). **Fallback if the Classic-11 wrapper fights you (known integration risk):** Gazebo actor plugins driven by your own SFM node publishing agent states — you lose HuNavSim's behavior library but keep everything else; budget max 3 days before invoking the fallback.

### D8. Software architecture (ROS 2)

Nodes: `mpc_controller` (10 Hz, CasADi), `cbf_filter` (10 Hz, QP), `rl_supervisor` (2 Hz, ONNX-exported policy — export from SB3 to ONNX so evaluation has no torch dependency), `human_tracker` (ground truth from sim topics; abstracted behind an interface so a real detector could replace it), `scenario_manager` (spawns scenarios, logs rosbags), `metrics_logger`. The MPC+CBF core is a plain Python package with zero ROS imports, wrapped by thin ROS nodes — this is what lets the identical control code run in the 2D sim, and it reads as professional software structure in the repo.

---

## 4. Evaluation Design (the actual product)

### 4.1 Systems under test (the ablation ladder)

| ID | System | Role |
|----|--------|------|
| S1 | MPC, conservative fixed tuning (v_max = 0.13, d_margin = 1.0) | safe-but-slow baseline |
| S2 | MPC, aggressive fixed tuning (v_max = 0.26, d_margin = 0.35) | fast-but-unsafe baseline |
| S3 | MPC + RL modulation, **no** CBF filter | shows what learning adds and what it risks |
| S4 | MPC + RL + CBF (full stack) | the system |
| S5 | MPC + **adversarial/random policy** + CBF | the guarantee stress test |

S5 is the headline: feed the filter a uniformly-random policy and a deliberately adversarial one (always commands v_max, d_margin at the floor) and show **zero** stopping-distance violations and zero collisions across all runs. One plot, whole thesis proven.

Optional S6 if time allows: Nav2 MPPI with default tuning, as an external reference point (contextualizes, not core).

### 4.2 Scenarios (fixed, seeded, 100 episodes each per system)

1. **Corridor pass-by** — 2 m corridor, one oncoming pedestrian. Tests anticipatory slowdown.
2. **Perpendicular crossing** — pedestrian crosses robot path at randomized timing. Tests yield-vs-proceed judgment.
3. **Doorway negotiation** — 0.9 m gap, pedestrian arriving simultaneously. Tests deadlock/freezing.
4. **Open hall, 6–8 free-roaming pedestrians** — throughput under crowd. Tests general competence.
5. **Blind corner** — occluded pedestrian appears at 1.2 m (perception limited by wall). Tests the filter under worst-case surprise; anticipation cannot help here, so this scenario shows the *layered* value: RL helps in 1–4, CBF saves 5.

Each in the 2D sim (primary statistics) **and** Gazebo (transfer validation + videos): full 100-episode battery in 2D; 20 episodes per scenario per system in Gazebo.

### 4.3 Metrics

- **Safety:** min human distance (dist.), # stopping-distance-constraint violations, # collisions, # protective stops.
- **Efficiency:** success rate, time-to-goal, energy proxy E = Σ|a|·v·dt, # full stops (v < 0.02 m/s events), path length ratio.
- **Comfort/social:** RMS jerk, personal-space (0.5 m) intrusion time, HuNavSim's built-in social-nav metrics where applicable.
- **Filter behavior:** intervention rate (% timesteps ‖u_safe − u_mpc‖ > ε), mean intervention magnitude — S4 low vs S5 high is the anticipation proof.
- **Compute:** MPC solve time (median/p99), QP solve time, total loop latency.

**Statistics:** mean ± std over episodes, 5 policy seeds pooled; Mann-Whitney U for S4-vs-baselines on primary metrics, report p-values and effect sizes. Underpowered claims are worse than modest claims — if a difference is not significant, say so.

### 4.4 The four money plots

1. **Velocity vs. distance-to-human overlay** (S1/S2/S4 on the corridor scenario): S2 shows the cliff-edge stop, S1 crawls, S4 shows the smooth anticipatory ramp. This single figure communicates the entire project to a non-expert interviewer in five seconds.
2. **Safety–efficiency Pareto scatter** (min-distance vs time-to-goal, all systems + a sweep of fixed tunings as a frontier): S4 sits above the fixed-tuning frontier = the adaptivity claim, visually.
3. **S5 stress-test violation plot:** constraint value h(t) over 100 adversarial episodes, never crossing zero; side panel showing S3 (no filter) crossing it.
4. **Filter-intervention histogram** S4 vs S5: the anticipation claim.

---

## 5. Week-by-Week Plan (with go/no-go gates)

**Week 0 (prep, ~2 evenings):** Repo scaffold (src/core [sim-agnostic], src/ros2, experiments/, configs as YAML from day one). Install CasADi, proxsuite, SB3, verify Gazebo Classic 11 + TB3 Waffle spawns. Docker image for the training environment (ties to your Docker certification; guarantees Kaggle/laptop parity).

**Week 1 — MPC standalone.** Implement unicycle RK4 + multiple-shooting NMPC in CasADi; drive point-to-point and path-tracking in the 2D sim with static obstacles; tune weights; measure solve times. *Gate G1: MPC tracks a path through static clutter at 10 Hz with median solve < 30 ms. Miss → simplify (shorter horizon, condensing) before proceeding. Do not touch RL until G1 passes.*

**Week 2 — CBF filter + 2D pedestrians.** Implement DCBF-QP; unit-test against scripted worst cases (head-on human, crossing at speed, occluded appearance) with *scripted* aggressive commands — this is S5 brought forward, deliberately: the safety layer is verified before any learning exists. Implement SFM pedestrians + the five scenario generators. *Gate G2: zero constraint violations across 1,000 randomized scripted-adversary episodes in 2D. This gate is absolute.*

**Week 3 — Gym env + baselines + Gazebo bring-up in parallel.** Wrap 2D sim as a Gymnasium env (obs/reward per D6); run and log S1/S2 baselines over the full scenario battery (baseline numbers now exist — the project already has reportable results even if RL underdelivers). In parallel: HuNavSim integration attempt (3-day box, then fallback per D7); port MPC+CBF nodes to ROS 2 and verify against Gazebo TB3. *Gate G3: baselines logged; Gazebo runs the MPC+CBF stack closed-loop.*

**Week 4 — RL training round 1.** Curriculum stages A→B on Kaggle; TensorBoard per-term reward logging; first policy evaluated on scenarios 1–2. Expect the reward to need surgery — this week exists to find that out. *Gate G4: policy beats S1 on time-to-goal and S2 on min-distance in scenario 1, in the 2D sim.*

**Week 5 — RL training round 2 + transfer.** Stage C, domain randomization, 5 seeds, select final policy; ONNX export; run in Gazebo, measure transfer gap; fine-tune in Gazebo only if retention < 80% on success rate. *Gate G5: S4 complete in both sims.*

**Week 6 — Full evaluation battery.** All systems × all scenarios × both sims; adversarial S5 runs; statistics pipeline (pandas + a single make-all-plots script — reproducibility is a portfolio feature); draft the four money plots.

**Week 7 — Presentation assets.** Videos: per-scenario side-by-side (S1 | S2 | S4) screen captures in Gazebo with rviz overlay showing the protective field, current d_margin ring, commanded vs filtered velocity trace; the S5 adversarial video ("watch the filter refuse"); 30 s architecture animation. GIF exports matching your portfolio's white-card hover-play format. README with architecture diagram, results tables, honest-limitations section. One-page PDF summary for attaching to applications.

**Week 8 — Buffer + polish.** Absorb slippage (expect it in Weeks 4–5); appendix industrial-scale parameter run; portfolio page; two-minute spoken walkthrough you can deliver in interviews (practice it — the project is also an interview script).

**Scope-cut order if behind schedule (pre-decided so panic never decides):** cut S6 Nav2 comparison → cut scenario 4 (open hall) → cut Gazebo fine-tuning (report transfer gap as-is) → reduce to 3 seeds → cut discrete strategy head (was optional anyway). **Never cut:** S5 stress test, the ablation ladder, statistics, or the money plots — they are the project.

---

## 6. Risk Register

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| CBF conservatism freezes robot (constraint too tight) | Medium | Directional projection (D5) is the fix; tune γ up; verify in Week 2 unit tests, not Week 6 |
| HuNavSim ↔ Gazebo Classic 11 integration friction | Medium-high | 3-day timebox → actor-plugin + own SFM fallback (D7) |
| Reward hacking / policy exploits filter (drives fast, lets filter brake) | High (expected) | w₄ intervention penalty is the designed counter; per-term reward logs catch it in hours not weeks |
| RL shows no significant gain over best fixed tuning | Low-medium | The fixed-tuning Pareto sweep guarantees a visible frontier; if S4 only *matches* the frontier while adapting across scenarios, that is still a positive, honest result — "characterized when learned modulation helps" |
| IPOPT too slow at 10 Hz | Low | Shorten N to 15, warm-start (always warm-start), then acados port |
| Kaggle session limits interrupt training | Medium | Checkpoint every 100 k steps; training runs are 2–4 h, well within limits |
| Gazebo transfer gap large | Medium | Domain randomization from Week 4; Gazebo fine-tune (Week 5); worst case, report 2D as primary + Gazebo as qualitative demo with the gap analyzed — an honest gap analysis is itself industry-relevant content |
| Human velocity estimation noise breaks CBF in Gazebo | Medium | σ inflation (D5); tracker uses sim ground truth by default with injected noise as the stress condition |

---

## 7. Defense Sheet — anticipated challenges and your answers

**"Why not end-to-end RL?"** Because no one certifies it. ISO 3691-4 compliance rests on verifiable safety functions with defined performance levels; a monolithic policy cannot expose one. The architecture keeps the safety function classical and verifiable and confines learning to performance parameters — the same reason industry ships residual/hybrid learning, not end-to-end.

**"Why not just tune the MPC better / gain-schedule by hand?"** The Pareto sweep of fixed tunings is exactly that comparison, run exhaustively — the frontier of all fixed tunings is in the plots, and the adaptive policy is compared against the *frontier*, not a strawman single tuning. Hand gain-scheduling on human context is possible in principle; the RL policy is a systematic way to find that schedule from experience, and the evaluation quantifies its value.

**"Is the CBF actually a guarantee?"** It is a discrete-time forward-invariance guarantee under the stated model assumptions (unicycle dynamics, constant-velocity humans over one step, bounded velocity-estimate error absorbed by σ and conservative a_brake), verified empirically by 1,000-episode adversarial testing (G2) plus S5. State assumptions crisply; a guarantee with stated assumptions is engineering, an unqualified guarantee is marketing.

**"Why train in a 2D sim?"** Sample budget arithmetic (Section D1), same control stack in both, and the transfer is measured and reported. Training-cheap/validating-rich is standard industrial practice.

**"TurtleBot at 0.26 m/s — isn't this a toy?"** The constraint structure is speed-independent; the injected latency makes it bind at TB3 scale, and the industrial-parameter appendix shows the same system at 1.5 m/s where stopping distances are 2 m+. The framework, not the platform, is the contribution.

**"Why PPO?"** Stable, few knobs, tiny action space, nearly-free simulator makes sample efficiency moot, prior hands-on experience reduces schedule risk. SAC was the evaluated alternative; the choice is logistics, not ideology.

**"What would you do next with more time?"** Learned human-trajectory prediction replacing constant-velocity (feeds both MPC potentials and CBF), the discrete strategy head for deadlocks, hardware deployment on the Jetson/RealSense platform with a real person tracker, and a learned CBF residual with formal verification. Having a credible roadmap signals the project is a platform, not a one-off.

---

## 8. System Requirements Summary

- **Laptop (GTX 1050 3 GB, sufficient):** Gazebo Classic evaluation + video capture (GPU load is rendering only), MPC/CBF development, plotting. Gazebo Classic 11 is light; this machine is not the bottleneck anywhere in the plan.
- **Kaggle (free tier):** all PPO training on the 2D sim; CPU instances suffice (32-dim obs, 2×256 MLP); GPU optional. Checkpoint to Kaggle datasets every 100 k steps.
- **Storage/repro:** single public GitHub repo; configs in YAML; one `make reproduce` path from raw seeds to final plots; Docker image published. Rosbags for the 20-episode Gazebo batteries (~2–4 GB, manageable).

## 9. Deliverables Checklist

- [ ] Public repo: `core/` (sim-agnostic MPC+CBF+envs), `ros2_ws/`, `experiments/`, `notebooks/`, Dockerfile, `make reproduce`
- [ ] Results: metrics tables (mean ± std, significance), four money plots, industrial-scale appendix
- [ ] Videos: 5 scenario side-by-sides, S5 adversarial demo, 30 s architecture explainer, portfolio GIFs
- [ ] README with architecture diagram, honest-limitations section, ISO 3691-4 scoping statement
- [ ] One-page PDF summary for applications
- [ ] Two-minute interview walkthrough (rehearsed)
