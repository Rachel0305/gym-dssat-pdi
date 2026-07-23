# 031_07 SY2014 free-timing action marginal value audit

## Purpose

031_06 showed that free-timing PPO and DQN can run, but reward v2 still failed to produce a good autonomous timing policy:

- PPO still saturated I160/N250 by DAP7.
- DQN moved away from the previous very-late failure but still saturated I160/N250 and underperformed the fixed timing references.

Before changing the RL reward again, run a zero-training DSSAT counterfactual audit to map which action timings actually have positive or negative marginal value under a fixed management background.

## Scope

- Site-year: SYA2014 only.
- No RL training.
- No checkpoint loading.
- DSSAT deterministic forward simulations only.
- Do not expand to other sites/years in this task.

## Counterfactual design

Use the 031_05 `uniform_spread` / `expert_window_budget` schedule as the common background:

```text
DAP1:   I30, N50
DAP30:  I30, N50
DAP50:  I30, N50
DAP65:  I30, N50
DAP85:  I20, N50
DAP110: I20, N0
```

For each target DAP:

```text
[1, 30, 50, 65, 85, 100]
```

replace only that day's action with one candidate from the 3x3 grid:

```text
I in [0, 20, 40]
N in [0, 40, 80]
```

Additionally include the original background action at that DAP when it is not already in the 3x3 grid, so that every target DAP can be compared against both:

- the target-DAP no-op branch;
- the unchanged background-action branch.

All non-target days use the same background schedule. Safety caps remain active:

- season irrigation <= 160 mm
- season nitrogen <= 250 kg/ha
- daily irrigation <= 40 mm
- daily nitrogen <= 80 kg/ha
- fertilization after DAP90 disabled

This means DAP100 nitrogen candidates are expected to be clipped to N0 by the safety wrapper; record the requested and actually executed action separately.

## Metrics

For every DAP-action branch, record:

- requested action;
- safe/executed action;
- final grain yield;
- final biomass;
- total irrigation;
- total nitrogen;
- simple profit = grain - irrigation - 5 * nitrogen;
- reward v2 sum;
- timing penalty sum;
- SWFAC stress days > 0.05;
- NSTRES stress days > 0.05;
- deltas relative to the target-DAP no-op branch;
- deltas relative to the background action branch.

## Output

Save:

- prompt MD;
- script;
- CSV summary;
- daily CSV files;
- experiment record MD under `docs/`;
- optional heatmap figure if plotting dependencies are available.

## Interpretation boundary

This is not an RL result. It is a DSSAT counterfactual map used to decide whether the free-timing reward provides the right learning signal.

Do not use the result to claim a trained policy is successful.
