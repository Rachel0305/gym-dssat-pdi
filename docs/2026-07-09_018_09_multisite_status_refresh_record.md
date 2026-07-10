# 018_09 多站点状态刷新记录

## 目的

- 用 018_08 的 SY2014 seed1 新证据刷新 018_05 的多站点状态判断。
- 形成当前可直接汇报的总表：哪些站点年份已经稳定、哪些只是 promising、哪些还需补 seed。

## SY2014 新增证据

| seed | checkpoint | final_gwad | irrigation_total | fertilizer_total | total_reward | yield_diff_vs_null | yield_diff_vs_dssat_auto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 15000.00 | 11216.00 | 120.00 | 300.00 | 6826.54 | 8447.00 | 8492.00 |
| 1.00 | 10000.00 | 11227.00 | 90.00 | 300.00 | 6867.95 | 8458.00 | 8503.00 |

## 刷新后的站点总表

| site | station | year | dqn_yield | dqn_irrigation | dqn_nitrogen | yield_diff_vs_dssat_auto | yield_diff_vs_extension | current_evidence_level | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SY | Shenyang | 2014 | 11216.00 | 120.00 | 300.00 | 8492.00 | 139.00 | stable_success_across_seed | SY2014 已完成 seed0/seed1 复现；两粒种子都达到高产，且 seed1 在保持 300 kg/ha 施氮下比 seed0 少用 30 mm 灌溉，可作为当前最完整的稳定成功案例。 |
| FQ | Fengqiu | 2016 | 7995.00 | 60.00 | 0.00 | -17.00 | 55.00 | promising_but_needs_seed | DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。 |
| HLA | Hailun | 2010 | 7853.67 | 120.00 | 0.00 | -0.33 | -0.33 | promising_but_needs_seed | DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。 |
| YC | Yucheng | 2014 | 9418.00 | 120.00 | 250.00 | 705.00 | 1.00 | promising_but_needs_seed | DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。 |
| LC | Luancheng | 2010 | 8739.00 | 90.00 | 0.00 | 1.00 | 0.00 | yield_stable_resource_unstable | LC2010 两个 seed 都能追平 auto/extension 产量，但 seed1 会回到 N300，不满足稳定节氮，因此目前只能算产量稳定、资源效率不稳定。 |

## seed 状态总表

| site | year | seed_status | best_seed0 | best_seed1 | overall_judgement |
| --- | --- | --- | --- | --- | --- |
| FQ | 2016 | only seed0 evidence | GWAD 7995.000, I60.0, N0.0 |  | promising_but_needs_seed |
| HLA | 2010 | only seed0 evidence | GWAD 7853.665, I120.0, N0.0 |  | promising_but_needs_seed |
| LC | 2010 | seed0+seed1 verified | ckpt5000, GWAD 8739, I90, N0 | ckpt5000, GWAD 8739, I120, N300 | yield_stable_resource_unstable |
| SY | 2014 | seed0+seed1 verified | ckpt15000, GWAD 11216, I120, N300 | ckpt10000, GWAD 11227, I90, N300 | stable_success_across_seed |
| YC | 2014 | only seed0 evidence | GWAD 9418.000, I120.0, N250.0 |  | promising_but_needs_seed |

## 当前口径

- stable_success_across_seed：至少 seed0/seed1 都已补证，且结论方向一致。
- yield_stable_resource_unstable：跨 seed 产量稳定，但节水/节氮方向不稳定。
- promising_but_needs_seed：当前只有 seed0 或等价单粒种子证据，暂不能称稳定成功。

## 输出

- `DSSAT_auto_validation\extension_expert_baseline_018_03\018_09_multisite_status_refresh\018_09_site_level_audit_refreshed.csv`
- `DSSAT_auto_validation\extension_expert_baseline_018_03\018_09_multisite_status_refresh\018_09_seed_status_summary.csv`
- `DSSAT_auto_validation\extension_expert_baseline_018_03\018_09_multisite_status_refresh\018_09_multisite_compact_status_table.csv`
