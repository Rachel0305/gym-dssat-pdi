# 027_01 HLA2010 阶段型 MaskablePPO 准备、scaler 与环境 smoke 记录

## 结论

状态：`completed / A_ready_for_027_02_seed0`。

HLA2010 已完成阶段型 MaskablePPO 的训练前准备检查。四基线、24 维站点专属 scaler、六个固定决策点、动作 mask、奖励闭合和完整季节 no-op smoke 全部通过。权威 attempt2 的 no-op 总奖励严格为 0，且本任务训练步数为 0、创建 PPO 模型数为 0。

本结果只说明 HLA2010 环境已具备进入 seed0 预注册训练的工程条件，不代表 PPO 已在 HLA 成功。

## 任务边界

- 站点与年份：HLA2010。
- 决策点：DAP 1、30、50、65、85、110。
- 原始观测：24 维；不复用 SY 的 25 维 scaler。
- 只重放四条基线、拟合 scaler 和运行一个 no-op 季节。
- 不修改 IC、DSSAT 输入、reward 结构、动作空间或 PPO 超参数。
- 不进行 PPO 训练，不创建模型 checkpoint。

## 四基线复核

| 情景 | 产量（kg/ha） | 灌溉（mm） | 施氮（kg/ha） | ETCP（mm） | WP_ET（kg/m3） | PFP_N（kg/kg） |
|---|---:|---:|---:|---:|---:|---:|
| null | 6956 | 0 | 0 | 427.1 | 1.63 | 不可定义 |
| recorded/farmer | 7679 | 30 | 165 | 451.6 | 1.70 | 46.5 |
| DSSAT auto | 7854 | 190.4 | 0 | 478.7 | 1.64 | 不可定义 |
| official extension expert | 7854 | 266.1 | 300 | 482.1 | 1.63 | 26.2 |

四情景在新运行目录中重新前向执行，最大产量差为 0.453857 kg/ha，低于预注册的 2 kg/ha 容差。输入副本哈希全部匹配。

## 精度修正与 attempt 记录

第一次执行复用了 020_11 汇总表中的整数产量，因而把 `local_null_yield` 写成 6956；DSSAT 新鲜前向结果为 6956.453857，导致 no-op 季节出现约 0.000454 的微小正奖励。该次没有训练模型，其结果保留在 `benchmark_results/027_01/`，并由 `027_01_attempt_note.json` 标记为训练前被 attempt2 取代。

权威 attempt2 改为从本次确定性前向结果读取完整精度参数：

- `local_null_yield = 6956.453857421875`
- `local_feasibility_yield = max(auto, expert) = 7853.6651611328125`
- `water_cost = 1`
- `nitrogen_cost = 5`
- `feasibility_bonus = 1620`
- 季节预算：I≤120 mm、N≤300 kg/ha

recorded 不进入奖励或 checkpoint 选择，只进入最终独立比较。

## HLA 专属 scaler

scaler 只使用 HLA2010 的 null、recorded、auto、official expert 四条训练年轨迹，在六个决策点采集，共 24 个状态、每个状态 24 维。

| 检查 | 结果 |
|---|---:|
| 状态数 | 24 |
| 维度 | 24 |
| 近常量维度 | 2 |
| 非常量维标准化均值最大绝对值 | 4.35e-12 |
| 非常量维标准差与 1 的最大偏差 | 2.22e-16 |
| 逆变换最大重构误差 | 2.84e-14 |

所有 scaler 检查通过。该 scaler 只能称为 HLA2010 训练年四基线 scaler，不能称为跨站点通用 scaler。

## 完整季节 no-op smoke

- 六个决策点全部到达；六次动作均为 action0/no-op。
- 每一步 action0 均有效；无非法动作。
- 最终产量：6956.453857 kg/ha。
- 最终生物量：19344.494629 kg/ha。
- 灌溉：0 mm；施氮：0 kg/ha。
- 总奖励：0.0。
- Summary.OUT 的水氮记账与环境动作一致。

## 已解决问题

1. 证明 HLA 不能直接复用 SY 的观测维度与 scaler，并建立了独立 24 维入口。
2. 训练前发现并修正整数汇总导致的 null reward 精度偏差。
3. 确认六阶段和 mask 在 HLA2010 可完整运行。
4. 确认所有运行只使用新目录副本，源输入哈希未改变。

## 尚未回答的问题

- 尚未训练 HLA2010 PPO，不能判断其是否能学到超过 auto 和 official expert 的策略。
- 尚未做 seed1/2，不能判断跨 seed 稳定性。
- 尚未冻结 HLA checkpoint 或迁移到 HLA 其他年份。
- recorded 的 WP_ET=1.70、PFP_N=46.5 高于 auto/expert 对应可比值；是否能全面超过 recorded 必须单独报告，不能由 primary 判据替代。

## 输出文件

- `benchmark_results/027_01_attempt2/027_01_hla2010_four_baselines.csv`
- `benchmark_results/027_01_attempt2/027_01_hla2010_four_baseline_rerun_audit.csv`
- `benchmark_results/027_01_attempt2/027_01_hla2010_scaler_source_states.csv`
- `benchmark_results/027_01_attempt2/027_01_hla2010_observation_scaler.csv`
- `benchmark_results/027_01_attempt2/027_01_hla2010_reward_config.json`
- `benchmark_results/027_01_attempt2/027_01_hla2010_smoke_stage_actions.csv`
- `benchmark_results/027_01_attempt2/027_01_result.json`
- `src/run_hla2010_stage_ppo_readiness_scaler_smoke_027_01.py`

## 下一步

允许编写 027_02 预注册任务书：只训练 HLA2010 seed0、240 阶段步，并在 0/60/120/180/240 保存确定性评估。不得在看到结果后改 reward、阶段点、PPO 超参数或 checkpoint 选择规则。

