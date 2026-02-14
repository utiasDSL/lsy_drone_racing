# AIGP Kaggle Runtime (Canonical Free Lane)

## Runtime Contract

- Canonical compute backend: Kaggle notebooks.
- Interactive frontend: Colab connected to Kaggle Jupyter Server when needed.
- Canonical repo root in runtime: `/kaggle/working/lsy_drone_racing`.
- Bootstrap mode: offline/self-contained only.
- No startup dependency on GitHub clone, PyPI, or live internet package fetch.
- Default env contract:
  - `JAX_ENABLE_COMPILATION_CACHE=0`
  - `SCIPY_ARRAY_API=1`
  - `PYTHONPATH=/kaggle/working/lsy_drone_racing[:$PYTHONPATH]`

## Required Kaggle Datasets

- Source blob dataset: `massimoraso/aigp-source-blob-fix6`
- Wheelhouse dataset: `massimoraso/aigp-wheelhouse`

The kernel builder now always injects these sources in canonical mode.

## Build and Push Kernels

### Smoke kernel (required first)

```bash
cd /Users/massimoraso/projects/aigp/lsy_drone_racing
python scripts/push_aigp_kaggle_kernel.py \
  --profile smoke \
  --wandb-enabled \
  --wandb-project drone-racing \
  --wandb-entity classimo \
  --wandb-mode online \
  --push
```

Expected smoke behavior:

- preflight runs first and writes health JSON;
- short stage0 training starts;
- `curriculum_log.jsonl` appears with at least one eval row.

### Full kernel (only after smoke pass)

```bash
cd /Users/massimoraso/projects/aigp/lsy_drone_racing
python scripts/push_aigp_kaggle_kernel.py \
  --profile full \
  --tournament-mode true \
  --obs-mode competition_proxy \
  --qualifier-eval-profile aigp_qualifier_eval_profile_default.toml \
  --readiness-profile qualifier_strict \
  --wandb-enabled \
  --wandb-project drone-racing \
  --wandb-entity classimo \
  --wandb-mode online \
  --push
```

## Session Wrapper Interface

`/Users/massimoraso/projects/aigp/lsy_drone_racing/scripts/run_aigp_kaggle_session.py` supports:

- `--preflight-only`
- `--strict-preflight`
- `--pythonpath-mode {repo-root,none}`
- `--health-json <path>`

Preflight script `/Users/massimoraso/projects/aigp/lsy_drone_racing/scripts/kaggle_preflight.py`
exit codes:

- `0`: pass
- `11`: import failure
- `12`: missing asset
- `13`: path/config mismatch
- `14`: interpreter/site-packages mismatch

## Colab UI on Kaggle Backend

1. Start the Kaggle notebook/kernel.
2. Open the Kaggle Jupyter Server connection panel.
3. In Colab, connect to external/local runtime using the Kaggle server URL/token.
4. Run cells from Colab while compute stays on Kaggle.
5. If disconnected, reconnect to the same Kaggle backend session and continue.

This keeps runtime policy and artifacts identical to canonical Kaggle execution.

## Resume and Timeout Hygiene

- Use `--max-walltime-s 42000` for graceful checkpoint before 12h session cutoff.
- Re-running the same out dir auto-enables append mode when `curriculum_log.jsonl` exists.
- To seed from previous artifacts:

```bash
python scripts/run_aigp_kaggle_session.py \
  --repo-root /kaggle/working/lsy_drone_racing \
  --resume-from /kaggle/input/<previous_run_dir> \
  --out runs/aigp_kaggle_primary
```

## Pivot Policy

- If two consecutive smoke attempts fail after code fixes, temporarily pivot to pure Colab managed runtime.
- Keep the same preflight/readiness scripts and artifact schema while pivoted.
- Re-enter Kaggle canonical lane after:
  - one successful Colab smoke, and
  - one successful Kaggle revalidation smoke.

## Run Outputs

- `runs/<run_name>/health/preflight.json`
- `runs/<run_name>/curriculum_log.jsonl`
- `runs/<run_name>/readiness/latest.json`
- `runs/<run_name>/readiness/history.jsonl`

Manual readiness command:

```bash
python scripts/build_readiness_report.py \
  --run-dir runs/aigp_kaggle_primary \
  --profile qualifier_strict \
  --window-evals 5 \
  --json
```
