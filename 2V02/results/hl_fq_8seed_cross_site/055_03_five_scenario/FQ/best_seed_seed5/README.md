# FQ historical-weather PPO seed 5: five-scenario results

Compared with the frozen Null, recorded farmer template, DSSAT auto + external N, and official extension expert baselines.

- Validation years: 2014–2023 (10 years).
- PPO mean yield / irrigation / N: 6804.3 kg/ha / 45.0 mm / 0.0 kg N/ha.
- Exact same positive-action sequence across all ten years: yes (1 unique annual sequences).
- PPO WP_ET (ET-based WUE) is available for all ten years from verified Summary.OUT ETCP: prior seed0 exact replay or new deterministic checkpoint replay.
- NUE: unavailable because the selected PPO validation artifacts do not contain plant N uptake. PFP_N is reported separately and must not be relabeled as NUE.
- Daily process figures show baseline WSPD/NSTD alongside PPO SWFAC/NSTRES; these are the source-specific stress indicators. PPO daily soil water was not recorded.
- Weather was cross-checked against the matching site/year Null baseline by calendar date (rainfall exact; temperature tolerance 0.051°C for one-decimal baseline rounding); see the action audit JSON.

## Tables

- `tables/fq_seed5_yearly_yield_irrigation_n_table.csv`: each cell is yield / irrigation / N.
- `tables/fq_seed5_yearly_wp_et_pfp_n_table.csv`: each cell is WP_ET / PFP_N; missing exact metrics are `—`.
- `tables/fq_seed5_yearly_five_scenario_metrics.csv`: tidy numeric five-scenario metrics.
- `tables/fq_seed5_five_scenario_management_events.csv`: all positive baseline and PPO water/N events.
- `tables/fq_seed5_ppo_daily_action_trace.csv`: full daily PPO action trace, including no-ops.

## Figures

- `figures/fq_seed5_2014_five_scenario_daily.png`
- `figures/fq_seed5_2015_five_scenario_daily.png`
- `figures/fq_seed5_2016_five_scenario_daily.png`
- `figures/fq_seed5_2017_five_scenario_daily.png`
- `figures/fq_seed5_2018_five_scenario_daily.png`
- `figures/fq_seed5_2019_five_scenario_daily.png`
- `figures/fq_seed5_2020_five_scenario_daily.png`
- `figures/fq_seed5_2021_five_scenario_daily.png`
- `figures/fq_seed5_2022_five_scenario_daily.png`
- `figures/fq_seed5_2023_five_scenario_daily.png`
- `figures/fq_seed5_five_scenario_management.png`
- `figures/fq_seed5_five_scenario_metrics.png`
- `figures/fq_seed5_five_scenario_reward.png`

## 逐年管理用量柱状图

- [2014](figures/fq_seed5_2014_five_scenario_management_bars.png)
- [2015](figures/fq_seed5_2015_five_scenario_management_bars.png)
- [2016](figures/fq_seed5_2016_five_scenario_management_bars.png)
- [2017](figures/fq_seed5_2017_five_scenario_management_bars.png)
- [2018](figures/fq_seed5_2018_five_scenario_management_bars.png)
- [2019](figures/fq_seed5_2019_five_scenario_management_bars.png)
- [2020](figures/fq_seed5_2020_five_scenario_management_bars.png)
- [2021](figures/fq_seed5_2021_five_scenario_management_bars.png)
- [2022](figures/fq_seed5_2022_five_scenario_management_bars.png)
- [2023](figures/fq_seed5_2023_five_scenario_management_bars.png)
