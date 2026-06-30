# 012_11 HLA2015 DQN 训练动作覆盖诊断记录

## 目的

012_09 和 012_10 显示，HLA2015 economic DQN 的 5K–20K 训练存在明显 seed 不稳定性：

- seed0 5K：不操作，等同 null；
- seed1 5K：I60/N0，接近 fixed I60/N0；
- seed2 5K：只施氮不灌溉，低于 null 的 economic reward；
- seed0 20K：有改善，但仍偏离 I60/N0。

本轮诊断的问题是：

> 失败 seed 是不是因为训练过程中几乎没有探索到灌溉动作？

如果失败 seed 没有探索到灌溉，那么下一步应该改 exploration。

如果失败 seed 探索过灌溉，但最终确定性策略仍不选择灌溉，那么问题更可能在 Q 值学习、信用分配或策略收敛稳定性。

## 方法

在不改变原有 DQN 设置的前提下，新增训练动作日志 wrapper，记录每个训练 step：

- seed；
- episode；
- DAP；
- action index；
- raw amir/anfer；
- safe amir/anfer；
- 是否处于灌溉窗口；
- 是否处于施氮窗口；
- reward；
- grnwt/topwt；
- swfac/nstres。

## 运行设置

- 年份：HLA2015
- 算法：DQN
- seeds：0、1、2
- 每个 seed：5000 steps
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

- 训练参数保持与 012_03/012_07/012_09 相同；
- 不保存大模型文件；
- 只保存动作覆盖日志和统计表。

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_dqn_action_coverage_012_11.py --seeds 0 1 2 --timesteps 5000"
```

绘图命令：

```bash
python src/plot_hla2015_dqn_action_coverage_012_11.py
```

## 输出文件

- 诊断脚本：

```text
src/run_hla2015_dqn_action_coverage_012_11.py
```

- 绘图脚本：

```text
src/plot_hla2015_dqn_action_coverage_012_11.py
```

- 三 seed 覆盖汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dqn_action_coverage_012_11/hla2015_dqn_action_coverage_5000steps_summary.csv
```

- 合并评估结果后的汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dqn_action_coverage_012_11/figures/hla2015_dqn_action_coverage_with_eval_summary.csv
```

- 诊断图：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dqn_action_coverage_012_11/figures/hla2015_dqn_action_coverage_diagnostic.png
```

- 每个 seed 的训练动作日志：

```text
DSSAT_auto_validation/HLA_2004/hla2015_dqn_action_coverage_012_11/seed0_5000steps/training_action_log.csv
DSSAT_auto_validation/HLA_2004/hla2015_dqn_action_coverage_012_11/seed1_5000steps/training_action_log.csv
DSSAT_auto_validation/HLA_2004/hla2015_dqn_action_coverage_012_11/seed2_5000steps/training_action_log.csv
```

## 关键结果

| seed | 灌溉窗口内选择灌溉动作比例 | 至少一次有效灌溉的episode比例 | 平均每episode有效灌溉量 mm | 最终评估灌溉 mm | 最终评估施氮 kg/ha | 最终评估reward |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.269 | 0.938 | 90.00 | 0 | 0 | 6485.77 |
| 1 | 0.514 | 1.000 | 120.00 | 60 | 0 | 7572.28 |
| 2 | 0.324 | 0.969 | 101.25 | 0 | 50 | 6272.87 |

## 解释

### 1. 失败 seed 并不是完全没有探索到灌溉

训练阶段：

- seed0 有 93.8% 的 episode 至少执行过一次有效灌溉；
- seed2 有 96.9% 的 episode 至少执行过一次有效灌溉；
- seed0 平均每个 episode 有效灌溉 90 mm；
- seed2 平均每个 episode 有效灌溉 101.25 mm。

因此，seed0 和 seed2 的失败不能简单归因于“训练过程中从来没试过灌溉”。

### 2. 成功 seed 的灌溉探索覆盖更强

seed1 在灌溉窗口内选择灌溉动作的比例最高：

- seed0：0.269；
- seed1：0.514；
- seed2：0.324。

这说明探索覆盖强度仍然有影响。seed1 训练中更频繁地在正确窗口尝试灌溉，这可能帮助它最终学到 I60/N0。

但 seed0/seed2 也不是完全没有覆盖，因此“只增加随机探索次数”未必能完全解决问题。

### 3. 关键问题更可能发生在最终确定性策略阶段

训练过程中 seed0 和 seed2 都经历过大量有效灌溉 episode，但最终 deterministic evaluation 中：

- seed0 选择 0 mm 灌溉；
- seed2 选择 0 mm 灌溉；
- 只有 seed1 保留了 60 mm 灌溉策略。

这说明问题更像是：

> DQN 在训练过程中见过有收益的灌溉动作，但 Q 值学习/策略收敛不稳定，最终确定性策略没有稳定选择这些动作。

## 当前结论

012_11 排除了一个简单解释：

> 失败不是因为 DQN 完全没有探索过灌溉。

更准确的判断是：

> HLA2015 DQN 失败 seed 的训练过程中存在有效灌溉覆盖，但最终 Q 策略没有稳定保留灌溉动作；因此问题更接近 DQN 的值函数学习稳定性、信用分配或最终策略选择不稳定，而不是单纯探索缺失。

## 对下一步的影响

现在不建议只做“提高 epsilon 随机探索”这一件事，因为训练中已经有大量灌溉覆盖。

更值得考虑的是：

1. 保存 replay buffer / Q 值诊断；
   - 检查关键 DAP 状态下 action 0/1/2/3 的 Q 值排序；
   - 看失败 seed 是否低估了灌溉动作。

2. 改 DQN 稳定性机制；
   - Double DQN；
   - Dueling DQN；
   - Prioritized replay；
   - n-step return；
   - 更慢或更稳定的 target network update。

3. 简化信用分配；
   - 阶段化动作，而不是日动作；
   - 或者给灌溉后的短期水分改善增加辅助奖励，但这会重新引入 reward 设计问题。

## 给导师汇报时的表述

可以这样说：

> 我们进一步检查了 DQN 的训练动作覆盖。失败 seed 在训练中并不是没有尝试灌溉，事实上大多数 episode 都执行过有效灌溉；但最终 deterministic policy 没有稳定保留灌溉动作。这说明当前问题不只是探索不足，而是 DQN 在该作物管理任务中的值函数学习和策略稳定性问题。下一步若继续算法线，应优先考虑 Double/Dueling/Prioritized replay 等 DQN 稳定性改进，而不是继续简单增加训练步数或随机调 reward。

