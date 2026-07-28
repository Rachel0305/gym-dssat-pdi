# 038_03 历史探索线备份说明：032 系列

记录时间：2026-07-28

## 这份文件的作用

本文件说明为什么要把 032 系列提交到 GitHub，以及它的使用边界。

032 系列不是当前最终可信结果线，但它记录了自由时序强化学习路线的重要探索过程，包括：

- 自由时序 PPO/DQN smoke；
- stress-aware reward；
- LC2010 多训练步数和多 seed；
- checkpoint guardrail；
- LC 跨年迁移；
- 五站点 half-split readiness；
- 五站点 stress-aware MaskablePPO batch；
- 日过程图、累积奖励图、土壤水氮胁迫图的绘制方法。

这些内容对复盘“为什么后来需要修 IC、修 DSSAT 管理事件链、重建可信基线”很有用，因此建议备份代码、prompt 和实验记录。

## 使用边界

032 系列可以用于：

- 复盘方法演进；
- 查找旧脚本逻辑；
- 复用绘图样式；
- 对比后续 036/037 的修正；
- 解释为什么某些 reward 或管理约束被放弃。

032 系列不建议直接用于：

- 向导师汇报最终可信数值；
- 作为论文主结果；
- 证明某站点/年份最终优化成功；
- 直接与修正后的 037 基线结果混合比较。

原因是后续发现并修正了 IC 和 DSSAT 管理事件链相关问题，旧结果需要被限定为“历史探索/诊断结果”。

## 本批建议提交内容

本批只提交小体积、可复盘文件：

- `prompts/032_*`
- `docs/032_*`
- `src/*032*`
- `experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml`

本批不提交：

- `benchmark_results/032_*` 大目录；
- 模型；
- runtime；
- snapshot；
- DSSAT 中间输出。

## 推荐 commit message

```text
exp032: add historical free-timing PPO exploration records
```

