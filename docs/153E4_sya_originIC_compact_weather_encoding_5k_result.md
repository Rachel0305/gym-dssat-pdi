# E4：压缩天气特征表达 5K seed0 结果

## 设计与门禁

E4 保留 E2 的奖励函数、动作评价、16 格动作网格、25+12 观测宽度、96 维双分支 extractor 和训练计划，只让 6 个天气特征保持实际值，将其余 6 个天气槽位置零。2K smoke 和 5K 正式审计均通过；10 个验证年份完整，置零列全部为 0，动作均在声明网格内。

## 5K checkpoint=5000

| 情景 | 平均产量 (kg/ha) | 平均 PFP-N (kg/kg) | 平均 ETCP (mm) | 平均 WP_ET (kg/m³) |
|---|---:|---:|---:|---:|
| E4 compact weather | 10047.51 | 41.8646 | 486.32 | 2.067 |
| E2 full weather | 10070.19 | 41.9591 | 476.07 | 2.115 |
| raw no-forecast | 10009.38 | 41.7057 | 476.89 | 2.102 |

相对 raw no-forecast，E4 产量增加 38.13 kg/ha（约 +0.38%），PFP-N 增加 0.1589；10 年中产量和 PFP-N 均为 7/10 年胜出。相对 E2，E4 产量低 22.68 kg/ha，PFP-N 低 0.0945。

但 E4 的平均 ETCP 比 raw no-forecast 高 9.43 mm，WP_ET 低 0.035 kg/m³（约 −1.7%）；逐年 WP_ET 对 E2 和 raw no-forecast 都是 0/10 胜出。因此 E4 是“产量/PFP-N 部分改善”，不是三个指标都占优的 forecast 胜出方案。

从 2K 到 5K，E4 产量由 10019.89 增至 10047.51 kg/ha，PFP-N 由 41.7496 增至 41.8646，没有出现明显坍缩。5K 行为上平均灌溉事件 7.4 次、施氮事件 6.0 次，氮肥约 200/40/0 kg/ha 分布在 DAP1–30、31–60、61–90，仍偏前置。

## 结论与下一步

精简天气特征对产量和 PFP-N 有希望，但没有改善水分生产率；当前不建议直接做 E4 独立 seed 验证。下一步应针对“产量提高但 ETCP 增加”的问题，设计一个仍不改奖励函数的低维水分需求/蒸散相关天气编码对照，再先做 2K smoke。

## 文件

- Prompt：`prompts/2026-08-16_sya_E4_compact_weather_encoding_5k.md`
- 配置：`configs/153E4_sya_originIC_compact_weather_encoding_5k_seed0.json`
- Runner：`src/run_153E4_sya_compact_weather_encoding_5k.py`
- 5K 结果：`benchmark_results/153E4_sya_originIC_compact_weather_encoding_5k_seed0/153E4_5k_result.json`
- WP_ET replay：`benchmark_results/153E4N_wp_et_5k_replay/2026-08-16_sya_E4_wp_et_5k_replay.md`
