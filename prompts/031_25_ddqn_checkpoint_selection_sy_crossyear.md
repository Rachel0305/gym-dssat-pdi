# 031_25 DDQN checkpoint selection / early-stopping test on SY cross-year transfer

## Purpose

Test whether the Double-Dueling DQN failure mode observed in 031_24 is mainly a "bad final checkpoint" problem rather than a total algorithm failure.

The scientific question is:

> Can a literature-aligned mask-aware Double-Dueling DQN become usable if the final model is selected by a fixed validation-year checkpoint protocol instead of always taking the last training step?

## Scope

- Site: SY only.
- Training year: SYA2014.
- Validation year for model selection: SYA2012.
- Frozen test year: SYA2015.
- Seeds: 0, 1, 2.
- Maximum training budget per seed: 100000 environment steps.
- Checkpoints saved during a single continuous training run:
  - 10000
  - 20000
  - 30000
  - 50000
  - 75000
  - 100000

## Fixed settings

No change from 031_22/031_23/031_24:

- Algorithm: project-local mask-aware Double-Dueling DQN.
- Reward:

```text
r = 0.001 * (0.158 * final_yield_at_harvest - 1.1 * irrigation - 1.58 * nitrogen)
```

- Discrete action grid:
  - irrigation: [0, 6, 12, 18, 24] mm
  - nitrogen: [0, 40, 80, 120, 160] kg/ha
- Safety/action constraints:
  - seasonal irrigation cap: 160 mm
  - seasonal nitrogen cap: 250 kg/ha
  - minimum interval between irrigation events: 7 days
  - minimum interval between fertilization events: 7 days
  - irrigation allowed DAP: 1-120
  - fertilization allowed DAP: 1-90
- Network/training hyperparameters:
  - dueling MLP [64, 64]
  - learning rate 3e-4
  - gamma 1.0
  - replay capacity 20000
  - batch size 144
  - learning starts 1000
  - target update interval 1000
  - epsilon: 1.0 -> 0.05 over 70% of total steps
  - max grad norm 10

## Selection rule

For each seed, evaluate every checkpoint on SYA2012 only. Do not use SYA2015 for selection.

For SYA2012, compute gaps against the four baseline maximum values:

- `gap_yield = candidate_yield - max_baseline_yield`
- `gap_wp_et = candidate_WP_ET - max_baseline_WP_ET`
- `gap_pfp_n = candidate_PFP_N - max_baseline_PFP_N_positive_N`

Define:

```text
advisor_any_metric_strict_winner =
  gap_yield > 0 or gap_wp_et > 0 or gap_pfp_n > 0
```

Define collapse guardrails:

- total nitrogen must be positive;
- final yield must be at least 70% of the validation-year maximum baseline yield;
- episode must terminate normally.

Define normalized validation score:

```text
score = max(
  gap_yield / max_baseline_yield,
  gap_wp_et / max_baseline_WP_ET,
  gap_pfp_n / max_baseline_PFP_N_positive_N
)
```

A checkpoint is a validation pass only if it satisfies both:

- `advisor_any_metric_strict_winner == true`
- collapse guardrails pass

Per seed:

1. If one or more checkpoints pass, select the passing checkpoint with the highest validation score.
2. If no checkpoint passes, still select the checkpoint with the highest validation score among guardrail-passing checkpoints, but mark `selected_pass=false`.
3. If no checkpoint passes guardrails, select the highest-score checkpoint overall and mark both `selected_pass=false` and `selected_guardrail_pass=false`.

Tie-breaker order:

1. higher validation score;
2. higher yield gap;
3. lower total nitrogen;
4. lower total irrigation;
5. earlier checkpoint step.

## Test protocol

After selecting exactly one checkpoint per seed using SYA2012, evaluate the selected frozen checkpoint on:

- SYA2014 training year;
- SYA2012 validation year;
- SYA2015 held-out test year.

SYA2015 is never used for selection.

## Stop / interpretation rules

- If selected checkpoints improve SYA2012 but fail SYA2015, report validation overfitting; do not tune against SYA2015.
- If selected checkpoints fail even on SYA2012, report that checkpoint selection does not rescue DDQN under current setup.
- If selected checkpoints pass SYA2012 and at least 2/3 pass SYA2015, DDQN can be retained as a candidate line, but still must be compared against PPO on the same protocol.
- Do not change reward, action grid, constraints, seeds, or checkpoint list after seeing results.

## Required outputs

- Prompt MD in `prompts/`.
- Config YAML in `experiments/ppo_observed_years/`.
- Script in `src/`.
- Experiment record MD in `docs/`.
- CSVs:
  - training summary;
  - checkpoint evaluation summary;
  - checkpoint gap summary;
  - selected checkpoint summary;
  - final selected train/validation/test summary.
- Daily CSVs and DSSAT snapshots for every evaluated checkpoint.

