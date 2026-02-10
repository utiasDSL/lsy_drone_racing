# AI Grand Prix 2026 (AIGP) Port & Merge Checklist

This repo is the new base (crazyflow + lsy_drone_racing architecture). We are porting the
"battle-tested training intelligence" from DronePrix:
- reward system (presets + 20+ components)
- curriculum (10-stage compressed, panic/rollback, binary gates)
- domain randomization (physics + sensors + latency/motor dynamics)
- training callbacks + evaluation harness
- competition track assets/configs

Legend:
- `[x]` done
- `[ ]` todo
- `[~]` in progress / partial

## Repo / Infra
- [x] Create AIGP branches + worktrees (`codex/aigp-*`)
- [x] Add bootstrap AIGP env id (`AIGPDroneRacing-v0`)
- [x] Add a dedicated `lsy_drone_racing/aigp/` package (instead of scattering code in `envs/`)
- [ ] Add `pixi` env / `requirements` notes for AIGP training stack (jax+mujoco+sb3)
- [ ] Add CI job subset for AIGP unit tests (reward/curriculum/DR)

## Environment & Track Support
- [x] Convert DronePrix competition track YAMLs into TOML configs
  - [x] `competition_swift`
  - [x] `competition_alphapilot`
  - [x] `competition_a2rl`
  - [x] `tii_real_track1`
  - [x] `tii_real_track2`
  - [x] `tii_real_track3`
- [x] Add stage-0 single-gate smoke config (`config/aigp_stage0_single_gate.toml`)
- [x] Pad tracks to a fixed max gate count (needed for SB3 curriculum without changing obs space)
- [x] Add per-episode active gate count (7/8/11 etc) without rebuilding the MJX model
- [x] Fix masked reset correctness (partial reset must not clobber unmasked worlds)
- [~] Add track pools (sample 1 track per episode from a weighted set)
  - [x] Env API (`set_track_pool`) + weighted sampling per reset
  - [ ] Per-world sampling for vector envs (currently global-per-reset)
- [x] Add richer `info` metrics (used by curriculum + eval):
  - [x] `success` / `completed`
  - [x] `gates_passed`, `num_gates_active`
  - [x] `lap_time_s` / `lap_time_steps`
  - [x] `crash_reason` (bounds/contact/timeout/etc)

## Reward System (Port From DronePrix)
- [x] Port RewardConfig + presets (`swift`, `grandprix`, `grandprix_lite`, `minimal`, `minimal_curiosity`)
- [~] Implement full modular reward components in JAX (vectorized over `n_envs`)
  - [x] gate passage
  - [x] progress (distance delta)
  - [x] progress-velocity (capped)
  - [x] speed bonus (with attenuation near gates)
  - [x] speed efficiency
  - [x] orientation alignment (quat-based forward vector)
  - [x] smoothness penalty (action delta)
  - [x] altitude maintenance
  - [x] boundary proximity penalty
  - [x] crash penalty
  - [x] timeout penalty
  - [x] time penalty
  - [x] completion bonus (time-based)
  - [x] approach angle
  - [x] hover penalty
  - [x] upward velocity penalty
  - [x] perception awareness (forward-vector alignment)
  - [x] lookahead alignment (next gate)
  - [~] racing line rewards (stubbed as 0.0 components)
  - [x] gate-approach curiosity (gated by success rate)
- [x] NaN/Inf guard: clamp/replace non-finite reward components
- [ ] Reward tapering across curriculum stages (preset + weight scaling)
- [~] Unit tests for reward invariants (finite, shape, component presence)
  - [ ] Add monotonicity tests for progress and caps (progress_velocity_max, speed_bonus_max)

## Curriculum (Port From DronePrix)
- [x] Create compressed 10-stage curriculum spec (`config/aigp_curriculum_10stage.toml`)
- [x] Port CurriculumManager logic:
  - [x] binary success gates (prevents fake advancement)
  - [x] stability requirements (variance gate)
  - [x] prevent force-advance "participation award" (min_episodes + SR + stability + recovery gate)
  - [x] rollback-on-stall + stage patience (hard-lock rollback)
- [x] Panic mode controller:
  - [x] reduce DR multiplier (trainer scales DR specs)
  - [x] increase assist multiplier/budget (trainer scales gate geometry)
  - [x] hard-lock detection (zero success)
  - [x] action on hard-lock: rollback stage (configurable)
- [~] Forgetting detection:
  - [x] baseline tracked on advance
  - [~] periodic stage N-1 eval + rollback in trainer (needs tuning + config knobs)
- [~] Curriculum logging:
  - [x] stage/eval log (JSONL)
  - [~] stage transition summaries (add explicit transition events + per-stage best checkpoints)
- [~] Unit tests for progression + rollback rules
  - [x] config load + basic gating tests
  - [ ] add tests for panic/recovery state transitions and forgetting baseline bookkeeping

## Domain Randomization (Merge DronePrix + crazyflow)
- [ ] Physics randomization integrated into crazyflow reset pipeline (JAX):
  - [x] mass range
  - [x] inertia perturbation
  - [x] thrust scaling (first_principles: `rpm2thrust`; TODO: sys-id models)
  - [x] motor time constant scaling (first_principles: `rotor_dyn_coef`; TODO: sys-id models)
  - [ ] motor degradation / battery discharge (thrust scaling schedule)
- [x] Wind OU process (time-correlated) as a JAX step hook (disturbance)
- [x] Sensor randomization as wrappers (VIO/IMU failures + bias + noise)
- [x] Action latency wrapper (N-step delay; vectorized)
- [ ] Optional motor dynamics wrapper (if needed beyond crazyflow rotor dynamics)
- [~] Adaptive DR tiers (stage->tier) + panic/adaptive multipliers (implemented in trainer; needs config + validation)
- [~] Unit tests for DR parameter ranges + deterministic seeding
  - [x] wrapper behavior tests
  - [ ] add tests for OU spec validation + RNG determinism (seeded rollouts)

## Training Pipeline
- [x] SB3-compatible VecEnv adapter for gymnasium.vector envs (`lsy_drone_racing/aigp/sb3_vec_env.py`)
- [x] Eval harness for curriculum gating (`lsy_drone_racing/aigp/eval.py`)
- [~] `scripts/train_aigp_curriculum.py` that:
  - [x] loads env base TOML + curriculum config
  - [~] runs PPO with evaluation-driven stage progression (needs real run verification)
  - [x] saves checkpoints per eval
  - [~] logs metrics locally (and optionally wandb; not wired yet)
- [x] Minimal smoke test: learns 1-gate policy in < ~10 minutes on CPU
- [ ] Scale test: 128-1024 envs on GPU (if available) without python bottlenecks

## Competition Readiness (Later Phases)
- [ ] SimulatorBackend ABC + DCL stub backend (SDK pending)
- [ ] Observation adapter (37D/full -> competition -> DCL)
- [ ] Vision pipeline integration hooks (gate detector + policy input)
- [ ] Policy comparison harness (benchmark across all tracks)
- [ ] Export: ONNX / packaging for submission
