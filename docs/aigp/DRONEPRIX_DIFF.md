# DronePrix vs AIGP Repository Differences

## 1) Scope and Baseline

This document captures the practical differences between:

- AIGP repo (this repository): `/Users/massimoraso/projects/aigp/lsy_drone_racing`
- DronePrix reference repo: `/Users/massimoraso/projects/aigp/droneprix_reference`

Snapshot used for this comparison:

- AIGP HEAD: `ceda51759a1c2844006c84b8b6140c89d092ea57`
- DronePrix HEAD: `e1fb7e5eeba1b7c4b8a6799c358fe83bcab9958b`

Git remotes at snapshot time:

- AIGP fork: `git@github.com:Mrassimo/lsy_drone_racing.git`
- AIGP upstream: `https://github.com/utiasDSL/lsy_drone_racing.git`
- DronePrix: `https://github.com/Mrassimo/DronePrix.git`

This is decision-focused and tournament-oriented, not a file-by-file parity list.

## 2) Architectural Split

AIGP keeps tournament work inside the `lsy_drone_racing` codebase and adds AIGP-specific components:

- Core package path: `lsy_drone_racing/*`
- AIGP extensions: `lsy_drone_racing/aigp/*`
- Main trainer entrypoint: `scripts/train_aigp_curriculum.py`
- Curriculum config format: TOML under `config/*.toml`

DronePrix uses a separate `src/*` stack:

- Training pipeline: `src/training/*`
- Evaluation pipeline: `src/evaluation/*`
- Environment/perception utilities: `src/envs/*`, `src/perception/*`
- Config format: YAML under `configs/*.yaml`

Implication: migration between projects is an interface translation problem (entrypoints + config format), not a simple code copy.

## 3) Training and Curriculum Differences

AIGP prioritizes explicit promotion gates and truthful logging:

- Gate diagnostics returned by curriculum decision: `lsy_drone_racing/aigp/curriculum.py`
- Explicit `block_reason` and bossfight gate fields in decision dict:
  - `block_reason`
  - `bossfight_ok`
  - `bossfight_tracks_total`
  - `bossfight_tracks_covered`
  - `bossfight_success_rate_min`
  - `bossfight_success_rate_bottom20_mean`
- Force-advance policy modes in trainer:
  - `never`
  - `if_passing`
  - `always`
  - Implemented in `scripts/train_aigp_curriculum.py`

DronePrix has a broader experimental curriculum/training surface (panic mode, speed gates, EWC/distillation options), but not the same strict AIGP JSONL schema contract used for run promotion in this repo.

## 4) Observation and Qualifier Alignment

AIGP has an explicit observation compliance mode:

- `competition_proxy` observation projection:
  - `lsy_drone_racing/aigp/observation.py`
- Trainer-level constraints:
  - `tournament_mode` requires `obs_mode=competition_proxy`
  - `scripts/train_aigp_curriculum.py`
  - `scripts/run_aigp_kaggle_session.py`

DronePrix includes privileged-teacher and distillation workflows in `configs/student_distillation.yaml` and `src/training/train_teacher.py` / `src/training/train_student.py`, but the AIGP lane here enforces qualifier-labeled runs through tournament mode at launch.

## 5) Runtime and Operations Differences

AIGP includes a Kaggle-first runtime layer aimed at reproducible bootstrap:

- Launcher: `scripts/run_aigp_kaggle_session.py`
- Kernel builder/pusher: `scripts/push_aigp_kaggle_kernel.py`
- Preflight checks with explicit failure codes: `scripts/kaggle_preflight.py`
- Runtime guide: `docs/aigp/kaggle_runtime.md`

AIGP also has integrated monitoring and W&B-aware operational tooling:

- Monitor: `scripts/monitor_aigp_run.py`
- W&B metadata logging in trainer JSONL rows:
  - `wandb.enabled`
  - `wandb.mode`
  - `wandb.run_id`
  - `wandb.run_path`
  - `wandb.run_url`

DronePrix has strong operations scripts too, but they are structured around the DronePrix `src/*` runtime and YAML configs.

## 6) Validation and Decision Pipeline

AIGP has a deterministic readiness control plane for promotion decisions:

- Readiness evaluator with named profiles:
  - `scripts/check_qualifier_readiness.py`
  - profiles include `default`, `stage_gate`, `qualifier_strict`
- Report builder:
  - `scripts/build_readiness_report.py`
  - emits `readiness/latest.json` and `readiness/history.jsonl`
- Run comparison:
  - `scripts/compare_aigp_runs.py`
  - rank/rationalize runs before policy/curriculum changes

Trainer eval rows are schema-driven (`schema_version=2`) and include:

- `run_id`
- `eval_id`
- `global_timesteps`
- `scorecard`
- `block_reason`
- `forced_advance`
- `qualifier_eval`
- `train_snapshot`

This is the main operational difference from DronePrix: readiness outputs are designed to drive promotion decisions directly.

## 7) What Remains from DronePrix (Gaps / Not Ported)

The following DronePrix capabilities are not first-class in this AIGP lane today:

- Full teacher-student distillation workflow (`src/training/train_teacher.py`, `src/training/train_student.py`)
- Broader experimental modules such as EWC/panic/recovery variants in `src/training/*`
- DronePrix submission packaging flow (`src/submission/prepare_submission.py`) as a standalone CLI pipeline
- AirSim/perception-oriented components under `src/envs/airsim_env.py` and `src/perception/*`

These are not blockers for current AIGP training operations, but they are relevant if strategy shifts toward that ecosystem.

## 8) Practical Migration Map

If you are moving from DronePrix to this AIGP repo:

1. Training entrypoint:
- Use `python scripts/train_aigp_curriculum.py train ...` instead of `python -m src.training.*`.

2. Config translation:
- Convert YAML configs (`configs/*.yaml`) into TOML configs:
  - trainer config in `config/*.toml`
  - curriculum config in `config/aigp_curriculum_*.toml`

3. Qualifier-compliant candidate runs:
- Launch with `tournament_mode=true` and `obs_mode=competition_proxy`.
- For Kaggle use `scripts/run_aigp_kaggle_session.py` and `scripts/push_aigp_kaggle_kernel.py`.

4. Promotion decisions:
- Use `scripts/build_readiness_report.py --profile qualifier_strict`.
- Compare alternatives with `scripts/compare_aigp_runs.py`.

5. Submission artifacts:
- Use the AIGP trainer export path (`export_submission_bundle`) and keep vecnorm/config metadata with model artifacts.
- If DronePrix-style packaging is needed, port that as a dedicated follow-up workstream rather than mixing interfaces ad hoc.
