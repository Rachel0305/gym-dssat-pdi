# 029_03 冻结 Mask-aware DQN 跨年验证与证据包

## 目标

在 029_00 已冻结的公平对比协议下，将五个锚定年份训练得到的 Mask-aware DQN 权重冻结，并在与 028 系列 MaskablePPO 完全相同的筛选年份上做零训练确定性评估。生成逐 seed、逐站点年份的算法对照，以及与 PPO 完全同构的五情景逐日证据包。

## 范围与硬约束

1. 不修改 DSSAT 输入、IC、reward、动作空间、阶段窗口、资源上限或四基线。
2. 不重新训练 DQN；只加载 029_02 中按预注册 reward 规则选中的 checkpoint。
3. 不在验证年份重新选 checkpoint 或调参。
4. 站点年份固定为 028_13 的 17 个筛选年份：
   - SY：2012、2014、2015；
   - HLA：2007、2010、2015、2016、2022；
   - YC：2008、2014；
   - FQ：2013、2014、2016、2019、2020、2023；
   - LC：2010。
5. 锚定年份直接复用 029_02，不重复 DSSAT 评估；其余 12 年才运行冻结迁移。
6. 每次只运行一个站点年份和一个 seed，避免 OOM。
7. 主判据沿用导师规则：产量、WP_ET、PFP_N 中至少一项严格超过四基线最大值；其余指标只报告差值，不追加事后阈值。
8. 主比较使用全部 seed，不以验证年 reward 事后挑选 seed 来宣称算法胜负。

## 实现与验证

1. 从 029_02 的 `result.json` 读取每个站点、每个 seed 的 selected checkpoint。
2. 复用 026/027/028 中已经验证的站点环境、scaler、四基线和逐日 DSSAT 输出解析。
3. 动作 mask 必须同时用于确定性 argmax；非法动作次数必须为 0。
4. 冻结模型文件在评估前后哈希必须一致。
5. 每季必须执行预期数量的阶段动作，保存完整 action sequence、终值指标和 DSSAT snapshot。
6. 输出 17 年 × 3 seed 汇总、逐年 winner count、PPO-DQN 成对比较。

## 逐日证据包

1. 每个站点年份选择一个仅用于展示的 DQN 代表 seed：按本地 episode reward 最大、并列取较小 seed；该选择不得用于主统计。
2. 复用 028_12 的五情景配色、8 面板布局、字段和 CSV 列名，仅把算法标签改为 `Mask-aware DQN candidate`。
3. 每个站点年份输出：
   - 五情景终值表；
   - 五情景逐日表；
   - DQN 阶段动作表；
   - 完整逐日 8 面板 PNG/SVG；
   - 终值指标对照图 PNG/SVG。

## 停止条件与结论边界

- 任一环境、哈希、mask 或输出完整性检查失败即停止对应任务并记录，不伪造结果。
- 只有在 17 年全部完成后才判断 DQN 是否值得替代 PPO。
- 若 DQN 仅在个别年份更好、但总体成功率或跨年稳定性不优于 PPO，则结论为“不建议替换，可保留为对照”。
- 本任务不开展 DQN 超参数敏感性扫描。

## 交付物

- `benchmark_results/029_03_frozen_maskaware_dqn_crossyear/`
- `benchmark_results/029_04_maskableppo_vs_maskaware_dqn_evidence/`
- `docs/2026-07-19_029_03_frozen_maskaware_dqn_crossyear_record.md`
- `docs/2026-07-19_029_04_maskableppo_vs_maskaware_dqn_comparison_record.md`
- `docs/2026-07-19_029_04_maskableppo_vs_maskaware_dqn_comparison.pptx`

