# 032_22 五站点前半训练/后半验证自由时序 stress-aware MaskablePPO 批量实验

## 背景

032_21 已确认五个站点从 2000 年以后有可用天气/场景数据，并给出固定切分：

- FQ：2005-2013 训练，2014-2023 验证
- HLA：2004-2013 训练，2014-2023 验证
- LC：2005-2013 训练，2014-2023 验证
- SY：2005-2013 训练，2014-2023 验证
- YC：2004-2013 训练，2014-2023 验证

导师要求不能将 PPO 决策限制在 expert 的固定 DAP 窗口内，因此本轮继续使用自由时序决策框架。目标是快速得到五站点独立训练结果，观察各站点是否也出现 LC 类似的“多年动作模式几乎固定”或其他不合理现象。

## 本轮目标

1. 五个站点各自独立训练一个模型，不跨站点迁移、不联合训练。
2. 每个站点使用自己的前半年份训练，后半年份验证。
3. 使用同一套自由时序 stress-aware MaskablePPO 配置。
4. 每站点训练 100,000 timesteps，保存 25k、50k、75k、100k 四个 checkpoint。
5. 每个 checkpoint 在该站点验证年份逐年冻结确定性评估。
6. 输出逐站点、逐年份、逐 checkpoint 的指标表和动作序列，便于后续绘图与筛选。

## 算法与配置

- 算法：MaskablePPO。
- 决策方式：自由时序，每日可判断是否灌溉/施氮，但受到 action mask 和总量上限约束。
- 奖励、动作空间、约束：沿用 `config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml`。
- seed：0。
- checkpoint：25,000；50,000；75,000；100,000。
- 不使用天气预报。

## 边界

- 本轮是批量探索性结果，不做参数扫描。
- 不根据单站点结果临时改奖励、改步数或改约束。
- 不跨站点迁移。
- 不重新补四情景基线；基线缺口只在后续汇总/绘图阶段处理。
- 不生成五情景过程图；本轮先生成训练和验证数值结果。

## 预期输出

- `benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/evaluation/032_22_checkpoint_validation_summary.csv`
- `benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/evaluation/032_22_training_checkpoint_inventory.csv`
- `benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/logs/032_22_training_year_reset_counts.csv`
- 中文实验记录 MD。

## 停止规则

- 若某站点训练失败，记录失败并继续下一个站点。
- 若容器或 DSSAT 环境失败，保留 partial CSV 和错误信息，不删除已有输出。
- 本轮不因为某个站点结果不好而现场调参。
