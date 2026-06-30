# 012_15 HLA2015 文献对齐版 DQN 奖励/动作空间探针记录

## 实验目的

参考文献 **A comparative study of deep reinforcement learning for crop production management** 中 mixed fertilization and irrigation task 的 DQN 设定，测试“终端产量奖励 + 水氮投入成本 + 5×5 离散动作空间”是否能改善 HLA2015 中 DQN seed 不稳定的问题。

本实验不是完全复现论文环境，而是在当前 HLA2015、IC=1、已更新品种参数、窗口/预算约束不变的前提下，做一个低成本兼容探针。

## 输入和环境

- 站点年份：HLA 2015
- 模型接口：PDI/gym-DSSAT 4.8.0
- Docker 容器：`b2fd6726c8c1`
- Python：`/opt/gym_dssat_pdi/bin/python`
- 脚本：`src/run_hla2015_literature_aligned_dqn_012_15.py`
- prompt：`prompts/012_15_hla2015_literature_aligned_dqn_probe.md`

## 固定约束

- 灌溉窗口：DAP 20–35, 45–65, 70–95
- 施氮窗口：DAP 25–40, 55–70
- 总灌溉预算：I ≤ 120 mm
- 总施氮预算：N ≤ 150 kg/ha
- 最小操作间隔：7 天

## 文献对齐改动

### 奖励函数

```text
非终止步：R_t = - w2 * N_t - w3 * W_t
终止步：R_T = w1 * GRNWT_final - w2 * N_T - w3 * W_T
```

本轮使用：

```text
w1 = 0.158
w2 = 0.79
w3 = 1.10
w4 = 0
```

暂不加入硝态氮淋失项，因为当前 wrapper 没有稳定整理该变量，而且文献示例中 `w4=0`。

### 动作空间

动作空间从原来的 4 个动作改为 25 个水氮组合：

```text
灌溉：0, 6, 12, 18, 24 mm
施氮：0, 40, 80, 120, 160 kg/ha
```

注意：在当前 HLA 约束下，160 kg/ha 施氮动作会被 N150 总预算裁剪。

## smoke test

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_literature_aligned_dqn_012_15.py --year 2015 --timesteps 200 --seed 0 --label smoke_literature_aligned"
```

结果目录：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_aligned_dqn_012_15/2015/smoke_literature_aligned_seed0_200steps
```

smoke test 正常完成，CSV、event summary、PDI snapshot 均生成。200 步结果不用于判断策略优劣。

## 5K seed0 正式探针

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_literature_aligned_dqn_012_15.py --year 2015 --timesteps 5000 --seed 0 --label literature_aligned"
```

结果目录：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_aligned_dqn_012_15/2015/literature_aligned_seed0_5000steps
```

## 主要结果

| 指标 | 数值 |
|---|---:|
| Null 产量 | 6486 kg/ha |
| DQN 产量 | 7652 kg/ha |
| 最终 GRNWT | 7651.86 kg/ha |
| 最终 TOPWT | 19050.05 kg/ha |
| 总灌溉 | 102 mm |
| 总施氮 | 150 kg/ha |
| 最大水分胁迫 | 0.000 |
| 最大氮胁迫 | 0.0145 |
| 文献式累计 reward | 978.29 |
| 投入成本项合计 | 230.70 |
| 灌溉事件数 | 7 |
| 施肥事件数 | 2 |

## 管理动作

| DAP | 灌溉 mm | 施氮 kg/ha | 说明 |
|---:|---:|---:|---|
| 46 | 6 | 0 | 灌溉 |
| 53 | 12 | 0 | 灌溉 |
| 60 | 12 | 120 | 灌溉 + 大量施氮 |
| 67 | 0 | 30 | 剩余施氮预算补满 |
| 74 | 24 | 0 | 灌溉 |
| 82 | 24 | 0 | 灌溉 |
| 89 | 12 | 0 | 灌溉 |
| 96 | 12 | 0 | 灌溉 |

## 初步结论

1. **流程已跑通**：文献式终端产量奖励和 25 动作空间可以在当前 HLA2015 wrapper 中正常训练、评估和保存结果。
2. **seed0 不再退化为 null**：相比 012_05 economic reward seed0 的 I0/N0，本轮 seed0 学会了灌溉，并把产量提升到 7652 kg/ha。
3. **但氮策略仍不理想**：HLA2015 反事实扫描 012_08 已显示 N0 条件下 I60 附近已经接近经济最优；本轮 DQN 仍使用 N150，说明终端产量奖励没有解决“氮投入信用分配/过量施氮”的问题。
4. **当前结果不能说优于前一条 economic reward 线**：它改善了 seed0 的灌溉学习，但把氮又推回了上限，整体仍存在策略稳定性和投入合理性问题。

## 追加低成本复评分

为了判断“DQN 用 N150”到底是因为奖励函数本身偏好 N150，还是 DQN 没学到正确排序，使用同一套文献式 reward 对 012_08 的固定水量扫描结果进行了离线复评分：

```text
score = 0.158 × yield - 1.10 × I - 0.79 × N
```

输出文件：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_aligned_dqn_012_15/literature_reward_rescore_vs_012_08.csv
```

| 情景 | 产量 kg/ha | I mm | N kg/ha | 文献式复评分 |
|---|---:|---:|---:|---:|
| I60_N0 | 7624.80 | 60 | 0 | 1138.72 |
| I90_N0 | 7645.12 | 90 | 0 | 1108.93 |
| I120_N0 | 7645.12 | 120 | 0 | 1075.93 |
| I30_N0 | 7017.07 | 30 | 0 | 1075.70 |
| I0_N0 | 6485.77 | 0 | 0 | 1024.75 |
| 012_15_DQN_seed0 | 7651.86 | 102 | 150 | 978.29 |

这个复评分结果说明：**012_15 的文献式 reward 本身并不偏好 N150。按该 reward，I60/N0 仍然高于 DQN 学到的 I102/N150。**

因此，本轮失败更像是 DQN 训练/信用分配/Q 排序问题，而不是奖励公式直接把策略推向 N150。

## 下一步建议

不建议立刻继续跑 seed1/seed2，也不建议继续只换 Double/Dueling 这类单个 DQN 变体。更合理的下一步是围绕“为什么 Q 排序学不到 I60/N0”做更有针对性的算法诊断：

1. 检查 012_15 训练过程中的 replay buffer / action coverage，确认 I60/N0 类轨迹是否被充分采样；
2. 做 012_15 Q-value probe，看最终策略为什么把 N150 动作排在更高位置；
3. 如果继续改算法，优先考虑能改善稀疏终端奖励信用分配的方案，例如 n-step return 或 prioritized replay，而不是继续单独试 Double/Dueling。
