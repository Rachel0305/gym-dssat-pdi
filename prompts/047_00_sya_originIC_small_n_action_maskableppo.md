# 047_00 SYA originIC small-N action MaskablePPO

## Research question

Does reducing the nitrogen dose grid improve PPO's water-nitrogen management under SYA originIC, without changing observation, reward, safety constraints, or teacher guidance?

## Controlled change

Relative to `046_10`, this experiment changes only the nitrogen action grid:

```text
046_10 irrigation: [0, 15, 30, 45] mm
046_10 nitrogen:   [0, 40, 80, 120] kg/ha

047_00 irrigation: [0, 15, 30, 45] mm
047_00 nitrogen:   [0, 10, 20, 40] kg/ha
```

The combined action count remains 16. The purpose is to test whether the previous PPO high-input tendency was partly caused by a coarse nitrogen grid whose minimum positive nitrogen action was already 40 kg/ha.

## Frozen factors

- station: `SYA`
- input profile: `originIC`
- train years: 2005-2013
- validation years: 2014-2023
- observation: raw `046_02` observation
- observation normalization: disabled
- weather forecast features: disabled
- algorithm: MaskablePPO
- reward and safety constraints: inherit the frozen `042_15 -> 046_02` chain
- recorded farmer template: frozen
- teacher guidance: disabled
- auto-0.05 teacher: not used in this experiment

## Required execution order

1. Run `--dry-run` to verify paths, frozen contracts, and action grid.
2. Run `--smoke` for 2K timesteps.
3. Inspect the smoke action audit. The smoke must show legal transmitted actions, more than one nonzero action pair, and positive actions after DAP 1.
4. Only after smoke passes, run `--formal` for 100K timesteps with checkpoints at 25K, 50K, 75K, and 100K.

## Interpretation boundary

If 047_00 improves PPO water-nitrogen efficiency, the improvement should be attributed to the smaller nitrogen action grid, not to teacher learning.

If 047_00 still behaves like a fixed high-input policy, action granularity alone is insufficient; the next experiment should introduce auto-0.05 teacher guidance while keeping this small-N action grid fixed.

