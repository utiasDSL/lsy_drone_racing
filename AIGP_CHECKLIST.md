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
- [~] Add a dedicated `lsy_drone_racing/aigp/` package (instead of scattering code in `envs/`)
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
- [~] Pad tracks to a fixed max gate count (needed for SB3 curriculum without changing obs space)
- [~] Add per-episode active gate count (7/8/11 etc) without rebuilding the MJX model
- [ ] Add track pools (sample 1 track per episode from a weighted set)
- [ ] Add richer `info` metrics:
  - [ ] `success` / `completed`
  - [ ] `gates_passed`, `num_gates_active`
  - [ ] `lap_time_s` / `lap_time_steps`
  - [ ] `crash_reason` (bounds/contact/timeout/etc)

## Reward System (Port From DronePrix)
- [~] Port RewardConfig + presets (`swift`, `grandprix`, `grandprix_lite`, `minimal`, `minimal_curiosity`)
- [ ] Implement full modular reward components in JAX (vectorized over `n_envs`)
  - [ ] gate passage
  - [ ] progress (distance delta)
  - [ ] progress-velocity (capped)
  - [ ] speed bonus (with attenuation near gates)
  - [ ] speed efficiency
  - [ ] orientation alignment (quat-based forward vector)
  - [ ] smoothness penalty (action delta)
  - [ ] altitude maintenance
  - [ ] boundary proximity penalty
  - [ ] crash penalty
  - [ ] timeout penalty
  - [ ] time penalty
  - [ ] completion bonus (time-based)
  - [ ] approach angle
  - [ ] hover penalty
  - [ ] upward velocity penalty
  - [ ] perception awareness (forward-vector alignment)
  - [ ] lookahead alignment (next gate)
  - [ ] racing line rewards (optional; can be stubbed initially)
  - [ ] gate-approach curiosity (gated by success rate)
- [ ] NaN/Inf guard: clamp/replace non-finite reward components
- [ ] Reward tapering across curriculum stages (preset + weight scaling)
- [ ] Unit tests for reward invariants (finite, shape, monotonicity for progress, etc)

## Curriculum (Port From DronePrix)
- [ ] Create compressed 10-stage curriculum spec (TOML/YAML)
- [ ] Port CurriculumManager logic:
  - [ ] binary success gates (prevents fake advancement)
  - [ ] stability requirements (variance gate)
  - [ ] prevent force-advance "participation award"
  - [ ] rollback-on-stall + stage patience
- [ ] Panic mode controller:
  - [ ] reduce DR multiplier
  - [ ] increase assist multiplier/budget
  - [ ] hard-lock detection (zero success)
  - [ ] action on hard-lock: rollback stage (configurable)
- [ ] Forgetting detection:
  - [ ] after advancing, periodically evaluate stage N-1
  - [ ] rollback if catastrophic forgetting is detected
- [ ] Curriculum logging:
  - [ ] stage transition log (JSONL/CSV)
  - [ ] per-stage eval summaries
- [ ] Unit tests for progression + rollback rules

## Domain Randomization (Merge DronePrix + crazyflow)
- [ ] Physics randomization integrated into crazyflow reset pipeline (JAX):
  - [ ] mass range
  - [ ] inertia perturbation
  - [ ] thrust scaling (maps to `rpm2thrust` / `cmd_f_coef`)
  - [ ] motor time constant scaling (`rotor_dyn_coef` / `thrust_time_coef`)
  - [ ] motor degradation / battery discharge (thrust scaling schedule)
- [ ] Wind OU process (time-correlated) as a JAX step hook (disturbance)
- [ ] Sensor randomization as wrappers (VIO/IMU failures + bias + noise)
- [ ] Action latency wrapper (N-step delay; vectorized)
- [ ] Optional motor dynamics wrapper (if needed beyond crazyflow rotor dynamics)
- [ ] Adaptive DR tiers (stage->tier) + panic-mode multiplier
- [ ] Unit tests for DR parameter ranges + deterministic seeding

## Training Pipeline
- [ ] SB3-compatible VecEnv adapter for gymnasium.vector envs
- [ ] `train_curriculum.py` (new) that:
  - [ ] loads env base TOML + curriculum config
  - [ ] runs PPO with evaluation-driven stage progression
  - [ ] saves checkpoints per stage
  - [ ] logs metrics locally (and optionally wandb)
- [ ] Minimal smoke test: learns 1-gate policy in < ~10 minutes on CPU
- [ ] Scale test: 128-1024 envs on GPU (if available) without python bottlenecks

## Competition Readiness (Later Phases)
- [ ] SimulatorBackend ABC + DCL stub backend (SDK pending)
- [ ] Observation adapter (37D/full -> competition -> DCL)
- [ ] Vision pipeline integration hooks (gate detector + policy input)
- [ ] Policy comparison harness (benchmark across all tracks)
- [ ] Export: ONNX / packaging for submission

