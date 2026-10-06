# SYA / LCA 天气增强实验临时冻结说明

日期：2026-10-06。目的：在现有天气资料和 WGEN 方法约束下，结束本轮 SYA/LCA 随机天气增强推进，同时保存可复核的实验链。统一状态为 `TEMPORARILY_FROZEN_FOR_CURRENT_WEATHER_AUGMENTATION_EXPERIMENT`。该状态可在数据或方法改善后重新开启。

| 站点 | 本轮天气增强 | 当前限制 | historical-weather PPO |
|---|---|---|---|
| SYA | `EXCLUDED_FROM_WGEN_FOR_CURRENT_EXPERIMENT` | `BLOCKED_BY_WEATHER_STATISTICS` | 保留当前冻结结果 |
| LCA | `EXCLUDED_FROM_WGEN_FOR_CURRENT_EXPERIMENT` | `BLOCKED_BY_JANUARY_GAMMA_SAMPLE` | 保留当前冻结结果 |

## 已排除的来源问题

RAIN 原始 XLS 空白表示 0 mm 降水。LCA cleaned CSV 与最终 WTH 的 1,113 处差异来自历史下载补充流程，差异本身不再列为来源异常。SYA 的 462 项非降水缺口确认采用旧站点×月份均值填补；LCA 残余 22 项数值与旧填补一致，但逐日外部补测链仍不完整。方法审查允许带记录的缺测填补进入候选参数拟合，不要求所有值都是原始观测。原 015 provenance gate 作为历史审查状态保留，后续方法 gate 记录具体 WGEN 限制。

## 本轮真正的 WGEN 限制

SYA train-only 候选修正了旧均值填补对 2014–2023 验证期的依赖；018 CLI 的 168 项参数和格式通过。但 2005 年 1–4 月 SRAD、TMAX、TMIN 各自整月恒定，最长连续受填补影响 139 天。均值填补不能恢复日际变率，因此 019 判 `BLOCKED_BY_WEATHER_STATISTICS`。

LCA 2005–2013 CLI 参数齐全，但 1 月只有 4 个湿日，Gamma ALPHA 达 0.998 截断上限。扩至 2000–2013 后 1 月有 26 个湿日，其中 14 个集中在 2000 年，原始 ALPHA 仍超过截断范围；1995–2013 无连续合格序列。本轮维持 `BLOCKED_BY_JANUARY_GAMMA_SAMPLE`。

## 历史天气 PPO 与论文表述

已冻结的 historical-weather PPO 结果继续按原实验范围用于当前研究。本轮仅审查随机天气增强输入，没有重新计算或重新验证 PPO 结果，也不据此判定历史天气模拟或 PPO 结果无效。论文宜写为：两站的历史天气策略结果按既有条件报告；SYA 因 2005 年连续均值填补导致月内日际变率缺失，LCA 因冬季湿日稀少且 Gamma 参数稳定性不足，未纳入本轮 WGEN 增强比较。不要把缺少随机天气对照解读为两站策略失败。

## 重新开启条件

SYA 需要可靠的 2005 年外部补测，或独立验证的天气重建方案，并重新审查月内和湿/干条件变率。LCA 需要更长、连续且质量合格的降水记录，或独立验证的冬季降水参数化方案。采用新的天气生成方法时，也需重新完成输入来源、参数和输出 QC。本轮实验到此收尾；未运行 WGEN、DSSAT 或 PPO，未修改原 WTH 或冻结 PPO 结果。
