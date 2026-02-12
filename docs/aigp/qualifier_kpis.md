# Qualifier KPI Standard

Use these metrics to evaluate virtual-qualifier readiness from run artifacts.

## Primary KPIs

- `course_completion_rate` (from `qualifier_eval`)
- `lap_time_s_p50` and `lap_time_s_p90`
- `robustness_std`
- `track_coverage.ratio`
- `seed_coverage.ratio`
- Curriculum gate status (`scorecard.pass`, `block_reason`, `forced_advance`)

## Baseline Thresholds (Default Checker)

- mean success rate over window: `>= 0.70`
- mean course completion rate over window: `>= 0.70`
- max robustness std over window: `<= 0.20`
- min track coverage ratio over window: `>= 1.00`
- min seed coverage ratio over window: `>= 1.00`
- latest scorecard pass: `true`
- no forced advance while scorecard is failing

Tune thresholds by profile, but keep the defaults as a conservative gate.
