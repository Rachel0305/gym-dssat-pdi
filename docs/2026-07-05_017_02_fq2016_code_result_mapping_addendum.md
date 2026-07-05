# 017_02 FQ2016 四情景过程图代码—结果索引

本文件是 `docs/2026-07-05_dqn_code_result_mapping.md` 的干净中文补充索引，用于避免历史编码问题影响后续追溯。

## 文件对应关系

| 内容 | 路径 |
|---|---|
| Prompt | `prompts/017_02_fq2016_four_scenario_process_plot.md` |
| 脚本 | `src/plot_fq2016_four_scenario_process_017_02.py` |
| 实验记录 | `docs/2026-07-05_017_02_fq2016_four_scenario_process_record.md` |
| 结果目录 | `DSSAT_auto_validation/fq2016_four_scenario_process_017_02` |
| 四情景日值表 | `DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_daily.csv` |
| 管理事件表 | `DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_management_events.csv` |
| 汇总表 | `DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_summary.csv` |
| 过程图 PNG | `DSSAT_auto_validation/fq2016_four_scenario_process_017_02/figures/fq2016_four_scenario_process.png` |
| 过程图 SVG/PDF | `DSSAT_auto_validation/fq2016_four_scenario_process_017_02/figures/fq2016_four_scenario_process.svg` / `.pdf` |

## 执行口径

- 本轮没有新增训练。
- 非 DQN 三个情景 `null_zero`、`recorded_shifted`、`dssat_auto` 使用 FQ2016 输入重新 forward。
- DQN 情景读取 `017_01` seed1 50K 训练中的 best-reward checkpoint 30000。
- 奖励代理值仅用于图中对齐展示，公式为：

```text
terminal max(0, GWAD - FQ2016_null_GWAD) - 1.0 × irrigation - 5.0 × fertilizer
```

## 核心结果

| 情景 | GWAD kg/ha | CWAD kg/ha | 灌溉 mm | 施氮 kg/ha | 最大水分胁迫 | 最大氮胁迫 | 累积奖励代理 |
|---|---:|---:|---:|---:|---:|---:|---:|
| null_zero | 7066 | 13148 | 0.0 | 0.0 | 0.657 | 0.012 | 0.1 |
| recorded_shifted | 7933 | 14008 | 75.0 | 144.0 | 0.373 | 0.012 | 866.6 |
| dssat_auto | 8012 | 14095 | 59.9 | 0.0 | 0.000 | 0.012 | 946.4 |
| dqn_seed1_best_reward | 7995 | 14078 | 60.0 | 0.0 | 0.050 | 0.012 | 869.2 |

## 当前解释

FQ2016 的 DQN best-reward 策略为约 `I60/N0`，产量几乎追平 DSSAT auto，并高于 recorded_shifted，同时相对 recorded_shifted 明显节氮、略节水。

该结果可以作为 FQ 后续跨年份迁移前的站点内代表过程图，但还不能替代后续跨年份/跨 seed 证据。
