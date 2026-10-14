# HL/FQ 8-seed historical-weather PPO: YC 055_03-style five-scenario output

Figures use completed 100K checkpoints and frozen four-baseline results. PPO WP_ET was completed by deterministic, endpoint-verified DSSAT replays of the existing checkpoints; no training, WGEN generation, or configuration edit was performed.

Each completed seed has `figures/` and `tables/` under `HL|FQ/best_seed_seedN/`, matching the YC reference folder structure.

The yearly tables contain yield / irrigation / N and WP_ET / PFP_N. PPO WP_ET is populated from endpoint-verified Summary.OUT ETCP replays. Plant N uptake is unavailable, so NUE remains blank; PFP_N is not NUE.

## Completed seed outputs

- [HL seed 0](HL/best_seed_seed0/README.md) — 23 PNG figures
- [HL seed 1](HL/best_seed_seed1/README.md) — 23 PNG figures
- [HL seed 2](HL/best_seed_seed2/README.md) — 23 PNG figures
- [HL seed 3](HL/best_seed_seed3/README.md) — 23 PNG figures
- [HL seed 4](HL/best_seed_seed4/README.md) — 23 PNG figures
- [HL seed 5](HL/best_seed_seed5/README.md) — 23 PNG figures
- [HL seed 6](HL/best_seed_seed6/README.md) — 23 PNG figures
- [HL seed 7](HL/best_seed_seed7/README.md) — 23 PNG figures
- [FQ seed 0](FQ/best_seed_seed0/README.md) — 23 PNG figures
- [FQ seed 1](FQ/best_seed_seed1/README.md) — 23 PNG figures
- [FQ seed 2](FQ/best_seed_seed2/README.md) — 23 PNG figures
- [FQ seed 3](FQ/best_seed_seed3/README.md) — 23 PNG figures
- [FQ seed 4](FQ/best_seed_seed4/README.md) — 23 PNG figures
- [FQ seed 5](FQ/best_seed_seed5/README.md) — 23 PNG figures
- [FQ seed 6](FQ/best_seed_seed6/README.md) — 23 PNG figures

## Coverage

- HL: PPO seeds 0–7 (8 seeds).
- FQ: PPO seeds 0–7 (8 completed seeds). Seed7 completed in isolated 20261012fq7 attempt; original behavioral smoke failure retained.
- The per-seed action audit reports whether its ten annual positive-action sequences are exactly identical.

[Open the complete 16-seed yearly figure gallery](GALLERY.md). Each seed has 23 figures (10 daily, 10 per-year irrigation/N bars, 3 across-year).


- [FQ seed7 figures and tables](FQ/best_seed_seed7/README.md): 23 PNG; 10 exact ETCP replays.
- [FQ eight-seed yearly metrics](FQ/best_seed_seed7/fq_8seed_100k_yearly_metrics.csv): 80 rows, fixed 100K checkpoint.
