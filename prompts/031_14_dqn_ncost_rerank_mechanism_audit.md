# 031_14 DQN nitrogen-cost rerank mechanism audit

## Question

031_13 doubled the literature reward nitrogen cost from 0.79 to 1.58, but all three SYA2014 DQN seeds still evaluated at N=250 kg/ha.

This task asks whether the failure is because:

1. the nitrogen penalty is still too weak to change the true full-season ranking, or
2. the true ranking already favors lower-N staged policies, but DQN failed to learn/use that ranking.

## Scope

- Site-year: SYA2014 only.
- No new training.
- No new DSSAT calls.
- Reuse the completed 031_12 DSSAT counterfactual table:
  `benchmark_results/031_12_dqn_nitrogen_counterfactual_audit/evaluation/031_12_nitrogen_counterfactual_summary.csv`
- Compare the same already-simulated policy variants under multiple nitrogen-cost values.

## Inputs

031_12 variants:

- `original_replay`
- `stage_spread_N250`
- `stage_spread_N200`
- `stage_spread_N160`
- `early_scaled_N200`
- `early_scaled_N160`

Reward family for reranking:

```text
R_lit(c_N) = 0.158 * final_grnwt - 1.1 * total_irrigation - c_N * total_n
```

where `c_N` is the nitrogen penalty.

Primary nitrogen-cost values:

- 0.79: original literature value.
- 1.58: 031_13 doubled nitrogen-cost test.

Exploratory diagnostic values, for interpretation only:

- 2.37
- 3.16

## Pre-registered outputs

1. Per-seed variant scores and ranks under each `c_N`.
2. Winner per seed under each `c_N`.
3. Break-even nitrogen penalty where `stage_spread_N200` ties `original_replay`, per seed:

```text
c_N* = (0.158*(Y_original - Y_N200) - 1.1*(I_original - I_N200)) / (N_original - N_N200)
```

4. Interpretation branch:

- Branch A: if `stage_spread_N200` does not outrank `original_replay` at `c_N=1.58`, then 031_13 failure is consistent with nitrogen penalty still being too weak.
- Branch B: if `stage_spread_N200` does outrank `original_replay` at `c_N=1.58` for most/all seeds, then 031_13 failure is not explained by the full-season reward ranking; it points to DQN learning/exploration/Q-estimation not recovering the better ranking.

## Stop rule

This task produces a diagnostic record only. It must not launch new training, tune another nitrogen-cost value, or change caps/action space.

