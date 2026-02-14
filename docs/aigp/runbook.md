# AIGP Runbook

## Start New Run

1. Choose a fresh output directory and tmux session name.
2. Launch with `JAX_ENABLE_COMPILATION_CACHE=0`.
3. Keep `--force_advance_mode if_passing` unless running an explicit ablation.
4. Enable W&B when available (`--wandb_enabled true`).

## Kaggle Launch

- Kaggle session wrapper (fresh or resume-safe):

```bash
python scripts/run_aigp_kaggle_session.py \
  --repo-root /kaggle/working/lsy_drone_racing \
  --out runs/aigp_kaggle_primary \
  --wandb-enabled \
  --wandb-project drone-racing \
  --wandb-entity classimo \
  --max-walltime-s 42000
```

- Tournament candidate (competition constraints + qualification defaults):

```bash
python scripts/run_aigp_kaggle_session.py \
  --repo-root /kaggle/working/lsy_drone_racing \
  --out runs/aigp_kaggle_tournament \
  --tournament-mode true \
  --obs-mode competition_proxy \
  --qualifier-eval-profile aigp_qualifier_eval_profile_default.toml \
  --readiness-profile qualifier_strict \
  --wandb-enabled \
  --wandb-project drone-racing \
  --wandb-entity classimo \
  --max-walltime-s 42000
```

- Dry-run to print exact trainer/readiness commands:

```bash
python scripts/run_aigp_kaggle_session.py --dry-run
```

- Full Kaggle setup details:
  `/Users/massimoraso/projects/aigp/lsy_drone_racing/docs/aigp/kaggle_runtime.md`

- Build/push Kaggle kernel bundle from local:

```bash
python scripts/push_aigp_kaggle_kernel.py --help
```

## Monitor

- Live local + W&B view:

```bash
/Users/massimoraso/projects/aigp/.venv/bin/python scripts/monitor_aigp_run.py \
  --run-dir runs/<run_name>
```

- One-shot JSON status:

```bash
/Users/massimoraso/projects/aigp/.venv/bin/python scripts/monitor_aigp_run.py \
  --run-dir runs/<run_name> --once --json
```

For tournament candidates, check readiness at the same cadence:

```bash
/Users/massimoraso/projects/aigp/.venv/bin/python scripts/build_readiness_report.py \
  --run-dir runs/aigp_kaggle_tournament --profile qualifier_strict --window-evals 5 --json
```

## Readiness Gate

```bash
/Users/massimoraso/projects/aigp/.venv/bin/python scripts/check_qualifier_readiness.py \
  --run-dir runs/<run_name> --window-evals 5
```

Persist report artifacts:

```bash
/Users/massimoraso/projects/aigp/.venv/bin/python scripts/build_readiness_report.py \
  --run-dir runs/<run_name> --profile default --window-evals 5
```

Compare multiple runs:

```bash
/Users/massimoraso/projects/aigp/.venv/bin/python scripts/compare_aigp_runs.py \
  --run-dir runs/<run_a> --run-dir runs/<run_b> --profile default --window-evals 5
```

## Stop Run

1. `tmux send-keys -t <session> C-c`
2. Verify no `train_aigp_curriculum.py` process remains.
3. Close session: `tmux kill-session -t <session>`.
