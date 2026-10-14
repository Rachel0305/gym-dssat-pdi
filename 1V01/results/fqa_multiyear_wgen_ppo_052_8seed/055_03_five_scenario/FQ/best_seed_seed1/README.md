# FQ historical-weather PPO seed 1: five-scenario results

Compared with the frozen Null, recorded farmer template, DSSAT auto + external N, and official extension expert baselines.

- Validation years: 2014–2023 (10 years).
- PPO mean yield / irrigation / N: 7188.0 kg/ha / 45.0 mm / 120.0 kg N/ha.
- Exact same positive-action sequence across all ten years: yes (1 unique annual sequences).
- PPO WP_ET / WUE uses endpoint-matched Summary.OUT ETCP from deterministic 2014–2023 DSSAT replay; plant N uptake is unavailable, so NUE is not inferred.
- NUE: unavailable because the selected PPO validation artifacts do not contain plant N uptake. PFP_N is reported separately and must not be relabeled as NUE.
- Daily process figures show WSPD/NSTD from PlantGro.OUT for PPO and the frozen baselines; soil water comes from SoilWat.OUT.
- Validation weather was matched to the same-year frozen Null baseline by calendar date; rainfall tolerance is 1e-6 mm and temperature tolerance is 0.051°C.

## Tables

- `tables/fq_seed1_yearly_yield_irrigation_n_table.csv`: each cell is yield / irrigation / N.
- `tables/fq_seed1_yearly_wp_et_pfp_n_table.csv`: each cell is WP_ET / PFP_N; missing exact metrics are `—`.
- `tables/fq_seed1_yearly_five_scenario_metrics.csv`: tidy numeric five-scenario metrics.
- `tables/fq_seed1_five_scenario_management_events.csv`: all positive baseline and PPO water/N events.
- `tables/fq_seed1_ppo_daily_action_trace.csv`: full daily PPO action trace, including no-ops.

## Figures

- `figures/fq_seed1_2014_five_scenario_daily.png`
- `figures/fq_seed1_2014_five_scenario_management_bars.png`
- `figures/fq_seed1_2015_five_scenario_daily.png`
- `figures/fq_seed1_2015_five_scenario_management_bars.png`
- `figures/fq_seed1_2016_five_scenario_daily.png`
- `figures/fq_seed1_2016_five_scenario_management_bars.png`
- `figures/fq_seed1_2017_five_scenario_daily.png`
- `figures/fq_seed1_2017_five_scenario_management_bars.png`
- `figures/fq_seed1_2018_five_scenario_daily.png`
- `figures/fq_seed1_2018_five_scenario_management_bars.png`
- `figures/fq_seed1_2019_five_scenario_daily.png`
- `figures/fq_seed1_2019_five_scenario_management_bars.png`
- `figures/fq_seed1_2020_five_scenario_daily.png`
- `figures/fq_seed1_2020_five_scenario_management_bars.png`
- `figures/fq_seed1_2021_five_scenario_daily.png`
- `figures/fq_seed1_2021_five_scenario_management_bars.png`
- `figures/fq_seed1_2022_five_scenario_daily.png`
- `figures/fq_seed1_2022_five_scenario_management_bars.png`
- `figures/fq_seed1_2023_five_scenario_daily.png`
- `figures/fq_seed1_2023_five_scenario_management_bars.png`
- `figures/fq_seed1_five_scenario_management.png`
- `figures/fq_seed1_five_scenario_metrics.png`
- `figures/fq_seed1_five_scenario_reward.png`
