# AIGP Architecture

## Control Plane

- `scripts/train_aigp_curriculum.py` orchestrates stage training, evaluation, gating, checkpoints, and exports.
- `CurriculumManager` in `lsy_drone_racing/aigp/curriculum.py` owns advancement logic and gate diagnostics.
- `scripts/monitor_aigp_run.py` provides local + optional W&B observability.

## Data Plane

- Environment construction: `lsy_drone_racing/envs/aigp_drone_race.py` + wrappers.
- Evaluation: `lsy_drone_racing/aigp/eval.py` produces per-stage aggregate and per-track metrics.
- Logging output: `runs/<run_name>/curriculum_log.jsonl` with additive schema v2 rows.

## Invariants

- `schema_version` must remain `2` until a deliberate migration is introduced.
- `eval_id` and `global_timesteps` must be strictly increasing in each run process.
- Force-advance defaults to `if_passing` and may not bypass failing scorecards.
- Every eval row should include `scorecard`, `block_reason`, `forced_advance`, and `wandb` metadata.

## Readiness Interfaces

- Run metadata contract: `runs/<run_name>/run_meta.json`.
- Submission contract: `runs/<run_name>/submission_bundle/metadata.json` + model + vecnorm + inference wrapper.
- Readiness checker: `scripts/check_qualifier_readiness.py`.
