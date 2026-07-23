# 031_05 SY2014 free-timing rule sensitivity record

## Purpose

Before changing reward or training free-timing RL, test whether SY2014 is sensitive to operation timing under the same I160/N250 caps.

## Rules

- `early_dump`: use water/N as early as possible.
- `uniform_spread`: spread I160/N250 over fixed stage-like DAPs.
- `expert_window_budget`: same DAP timing as the expert-style window reference under I160/N250; not the official expert dose.
- `stress_triggered`: only act when SWFAC or NSTRES > 0.05.
- `delayed_late`: intentionally late water/N timing.

## Summary

                rule run_status  episode_length  final_grnwt  final_topwt  total_irrigation  total_n  profit_simple  swfac_stress_days_gt_0p05  nstres_days_gt_0p05  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap                                                                                                     daily_csv_path
          early_dump         ok             144 10091.840210 18275.517578             160.0    250.0    8681.840210                          9                    9                       4              4                     1            1           benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_early_dump_daily.csv
      uniform_spread         ok             144 10908.603516 19224.992676             160.0    250.0    9498.603516                          0                    0                       6              5                     1            1       benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_uniform_spread_daily.csv
expert_window_budget         ok             144 10908.603516 19224.992676             160.0    250.0    9498.603516                          0                    0                       6              5                     1            1 benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_expert_window_budget_daily.csv
    stress_triggered         ok             144  9761.712036 17934.986572             120.0    160.0    8841.712036                          2                   27                       3              2                     1           41     benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_stress_triggered_daily.csv
        delayed_late         ok             144  6926.159668 11262.729492             160.0    250.0    5516.159668                          0                   42                       8              7                   102           83         benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_delayed_late_daily.csv

## Timing sensitivity

- Yield range across rules: 3982.44 kg/ha.
- Simple profit range across rules: 3982.44.

## Interpretation boundary

This is not RL training. It only checks whether timing choices create enough outcome contrast for a free-timing RL testbed.
