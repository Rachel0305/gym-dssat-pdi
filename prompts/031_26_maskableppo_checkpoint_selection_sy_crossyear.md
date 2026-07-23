# 031_26 MaskablePPO checkpoint selection / early-stopping test on SY cross-year transfer

## Purpose

Provide a fair comparison against 031_25 DDQN checkpoint selection.

031_25 showed that DDQN should not be judged only by the final 100k checkpoint; validation-year checkpoint selection can rescue useful candidates. Therefore PPO must be evaluated under the same selection protocol before deciding which algorithm is the stronger free-timing RL line.

## Scope

- Algorithm: free-timing discrete MaskablePPO.
- Site: SY only.
- Training year: SYA2014.
- Validation year for model selection: SYA2012.
- Frozen test year: SYA2015.
- Seeds: 0, 1, 2.
- Maximum training budget per seed: 100000 environment steps.
- Checkpoints:
  - 10000
  - 20000
  - 30000
  - 50000
  - 75000
  - 100000

## Fixed settings

Use the same reward/action/constraint setup as 031_17/031_18 and the same cross-year protocol as 031_25:

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
- PPO hyperparameters:
  - learning rate 3e-4
  - gamma 1.0
  - GAE lambda 1.0
  - n_steps 144
  - batch size 144
  - n_epochs 5
  - entropy coefficient 0.01
  - clip range 0.2
  - net arch [64, 64]

## Selection rule

Exactly match 031_25.

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

Collapse guardrails:

- total nitrogen must be positive;
- final yield must be at least 70% of the validation-year maximum baseline yield;
- episode must terminate normally.

Normalized validation score:

```text
score = max(
  gap_yield / max_baseline_yield,
  gap_wp_et / max_baseline_WP_ET,
  gap_pfp_n / max_baseline_PFP_N_positive_N
)
```

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

## Comparison with DDQN

After the PPO run, compare selected PPO checkpoints against 031_25 selected DDQN checkpoints using:

- SYA2012 validation pass rate;
- SYA2015 test pass rate;
- yield / WP_ET / PFP_N gaps;
- irrigation and nitrogen totals;
- action sequences and stress days.

Do not declare PPO or DDQN superior from a single metric alone; summarize tradeoffs.

