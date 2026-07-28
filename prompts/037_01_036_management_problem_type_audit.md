# 037_01：036 主线 PPO 措施问题类型审计

## 任务目标

036_05 和 037_00 已经显示：036 主线 PPO 在不少站点年份上指标表现较好，但措施过程存在可解释性问题。

本任务不训练、不重跑 DSSAT，只回答一个更具体的问题：

> 036 的措施问题主要是“频繁操作/碎片化操作”，还是“早期集中投入/early dump”？

这个判断决定下一步是否应该加“最大事件数”，还是应该优先处理“早期集中投入”。

## 输入文件

- `benchmark_results/036_05_selected_ppo_management_rationality_audit/tables/036_05_year_level_management_audit.csv`
- `benchmark_results/036_05_selected_ppo_management_rationality_audit/tables/036_05_event_level_audit.csv`
- `benchmark_results/036_04_select_checkpoint_and_plot_03601_03603_summary/tables/036_04_selected_year_level_comparison.csv`

## 预注册判据

### 频繁操作问题

定义为：

- 灌溉事件数 > 4；或
- 施氮事件数 > 4；或
- 同类操作间隔 < 7 天。

解释：

- 如果该类问题普遍存在，下一步可考虑最大事件数约束或更强间隔约束。
- 如果该类问题不普遍，则不优先加最大事件数。

### 早期集中投入问题

定义为：

- DAP1-10 灌溉量占季节总灌溉量 >= 80%；或
- DAP1-10 施氮量占季节总施氮量 >= 80%。

解释：

- 如果该类问题普遍存在，下一步不应优先加最大事件数，而应考虑：
  - 早期集中投入惩罚；
  - 生育阶段资源上限；
  - 或按生育阶段设置更合理的 action mask。

### 预防性操作

操作前 3 天胁迫很低的预防性操作不直接判为错误，只作为解释性标签保留。

## 输出

- 年份级问题类型表；
- 站点级问题类型汇总表；
- 总体问题构成图；
- 站点问题构成图；
- 中文实验记录 MD。

## 停止线

本任务只做审计，不训练模型，不重跑 DSSAT，不改 reward、动作空间或训练参数。
