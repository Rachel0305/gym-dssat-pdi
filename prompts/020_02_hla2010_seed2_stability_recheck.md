# 020_02 HLA2010 统一 DQN 框架 seed2 稳定性复核

## 开始前的历史核对

- HLA2010 正式 baseline-relative DQN 已完成 seed0、seed1 各 50K，代码为 `src/run_hla_baseline_relative_dqn_checkpoint_015_12.py`。
- 项目中没有同一 HLA2010 正式配置的 seed2；早期 HLA2015 seed2 使用的是不同 economic-reward 线路，不能替代本实验。
- 020_01 已用 DSSAT 原生指标统一筛选 90 个旧 checkpoint。HLA 严格成功点只来自 seed0，因此本轮只补 seed2，不重做优化空间审计、不改 reward、不改 IC。

## 单变量实验设置

- 站点年份：HLA2010。
- Seed：2。
- 正式训练：50K timesteps，每 5K 保存并确定性评估一次。
- 算法及超参数：完全复用 015_12。
- Reward：

```text
reward_t = -1.0 * irrigation_t - 5.0 * nitrogen_t
reward_terminal += max(0, GWAD_final - GWAD_null_HLA2010)
```

- 动作空间：9 个离散水氮组合，I∈{0,15,30} mm，N∈{0,50,100} kg/ha。
- 预算与操作约束：I≤120 mm，N≤300 kg/ha，单次 I≤30、N≤100，最小操作间隔 7 d。
- 输入：继续使用 HLA2010 IC=1、更新后品种参数、IRRIG=L、FERTI=L 的同一输入链路。
- 不修改初始条件，不加入淋洗 reward，不调成本系数。

## 防止结果导向选择

正式训练前固定 checkpoint 选择规则：

1. 在 5K、10K、……、50K 的确定性评估结果中选择 `total_reward` 最大者；
2. 如果奖励并列，选择训练步数最早者；
3. 选择后再独立检查产量、WP_ET、灌溉、施氮及 NLCM；
4. 不允许根据最终图形或产量手动改选 checkpoint。

严格成功定义：相对 DSSAT auto 和官方推广 expert，产量与 WP_ET 不低，灌溉、施氮及 NLCM 不高。容差沿用 020_01：产量 1 kg/ha、WP_ET 0.01 kg/m³、NLCM 0.01 kg/ha。

## 执行顺序

1. 检查容器、指定虚拟环境、输入文件、IC 和管理开关。
2. 先跑 seed2 500-step smoke，仅检查流程、动作链、输出和内存，不判断策略优劣。
3. smoke 通过后运行 50K 正式训练。
4. 提取所有 checkpoint 的 DSSAT 原生指标，按预先固定规则选点，并与 seed0、seed1 对照。

## 输出

```text
DSSAT_auto_validation/HLA_2004/hla2010_seed2_stability_020_02/
docs/2026-07-10_020_02_hla2010_seed2_stability_record.md
```

保存 checkpoint summary、日值 CSV、模型 checkpoint、DSSAT 原始快照、日志、诊断图及最终判定表；不得覆盖 015_12 旧结果。
