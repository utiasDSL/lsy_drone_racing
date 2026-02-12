# AIGP Runbook

## Start New Run

1. Choose a fresh output directory and tmux session name.
2. Launch with `JAX_ENABLE_COMPILATION_CACHE=0`.
3. Keep `--force_advance_mode if_passing` unless running an explicit ablation.
4. Enable W&B when available (`--wandb_enabled true`).

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
