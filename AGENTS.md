# AIGP Agent Map

This file is the entry map for coding agents working in this repository.
Treat repository files as the source of truth; avoid relying on chat history.

## Fast Start

1. Read `/Users/massimoraso/projects/aigp/lsy_drone_racing/docs/aigp/index.md`.
2. Confirm architecture and ownership boundaries in `/Users/massimoraso/projects/aigp/lsy_drone_racing/docs/aigp/architecture.md`.
3. Check qualifier KPIs and pass criteria in `/Users/massimoraso/projects/aigp/lsy_drone_racing/docs/aigp/qualifier_kpis.md`.
4. Use the runbook for train/restart/monitor actions in `/Users/massimoraso/projects/aigp/lsy_drone_racing/docs/aigp/runbook.md`.

## Core Rules

- Keep eval logs additive and backward-compatible.
- Keep `schema_version=2` on curriculum JSONL rows.
- Keep `force_advance_mode=if_passing` as default.
- Do not auto-advance failing stages under default mode.
- Keep run output non-append by default (`--allow_append` must be explicit).
- Preserve monotonic `eval_id` and `global_timesteps` within a run.
- Keep W&B optional; training must work with W&B disabled.

## Primary Code Paths

- Trainer: `/Users/massimoraso/projects/aigp/lsy_drone_racing/scripts/train_aigp_curriculum.py`
- Curriculum gates: `/Users/massimoraso/projects/aigp/lsy_drone_racing/lsy_drone_racing/aigp/curriculum.py`
- Eval aggregation: `/Users/massimoraso/projects/aigp/lsy_drone_racing/lsy_drone_racing/aigp/eval.py`
- Env track sampling: `/Users/massimoraso/projects/aigp/lsy_drone_racing/lsy_drone_racing/envs/aigp_drone_race.py`
- Monitor: `/Users/massimoraso/projects/aigp/lsy_drone_racing/scripts/monitor_aigp_run.py`
- Repo invariant checks: `/Users/massimoraso/projects/aigp/lsy_drone_racing/scripts/check_repo_invariants.py`
- Qualifier readiness checks: `/Users/massimoraso/projects/aigp/lsy_drone_racing/scripts/check_qualifier_readiness.py`

## Required Checks Before Merge

Run from repo root:

```bash
/Users/massimoraso/projects/aigp/.venv/bin/ruff check .
/Users/massimoraso/projects/aigp/.venv/bin/pytest -q
/Users/massimoraso/projects/aigp/.venv/bin/python scripts/check_repo_invariants.py
```

For a live run gate:

```bash
/Users/massimoraso/projects/aigp/.venv/bin/python scripts/check_qualifier_readiness.py \
  --run-dir runs/<run_name> --window-evals 5
```

## Active Plan Index

- Current strategic plan pointer: `/Users/massimoraso/projects/aigp/lsy_drone_racing/docs/aigp/plans/active.md`
- Completed plans can be moved under `/Users/massimoraso/projects/aigp/lsy_drone_racing/docs/aigp/plans/completed/`.
