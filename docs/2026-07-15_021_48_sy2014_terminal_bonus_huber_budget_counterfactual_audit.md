# 021_48 SY2014 terminal bonus 的 Huber 梯度预算反事实审计

## 1. 背景

021_47 发现，含 terminal feasibility bonus 的 agent transition 并没有因 PER 而少见：

- direct bonus transition 共被抽中 1,703 次；
- 5-step return 含 bonus 的 transition 共被抽中 1,890 次；
- 但对应 n-step TD 残差仍约为 7,360。

因此提出候选解释：`+1620` bonus 是否把 TD target 抬得过高，使 Huber loss 长期处于线性区，而 1K 的更新预算不足以拟合该目标。

本任务不启动训练，只用 021_47 已保存的样本级日志做固定样本路径下的反事实计算。

## 2. 方法

### 2.1 数据来源

输入：

`benchmark_results/021_47/021_47_agent_sample_draws.csv`

该文件来自 021_45 的逐位精确复现，包含每次 agent replay draw 的 chosen Q、1-step/n-step target、discounted bonus、importance weight 和 transition 来源。

### 2.2 无 bonus 反事实

对所有 `nstep_contains_bonus=True` 样本：

```text
target_n_no_bonus = target_n - discounted_bonus_in_nstep
```

对 direct terminal bonus 样本：

```text
target_1_no_bonus = target_1 - 1620
```

Huber delta 固定为 1，对 chosen Q 的导数为：

```text
dL/dQ = clip(chosen_q - target, -1, 1)
```

本反事实固定 021_47 实际抽中的样本，不把原 importance weight 和 PER probability 当作无 bonus 训练中的反事实权重。真实去掉 bonus 后，priority 和采样分布本来也会变化。

### 2.3 同一 transition 的经验拟合速度

同一个 optimizer update 内若重复抽中同一 transition，先合并为一个 chosen-Q 均值。对至少覆盖 20 个不同 update 的 direct terminal transition，比较其首末 10% update 的 chosen Q，计算实际残差闭合比例和 OLS `Q/update` 斜率。

该斜率只能描述此次 1K 固定-target 诊断中已经观察到的变化速度，不能当作理论所需训练步数。

## 3. 数据与恒等式校验

- 反事实恒等式最大误差：0；
- direct terminal draws：1,703；
- direct terminal transitions：4；
- 5-step return 含 bonus draws：1,890；
- 5-step return 含 bonus transitions：20。

数据字段和反事实计算通过，允许进入科学解释。

## 4. 结果

### 4.1 去掉 bonus 后，终端 target 仍然很大

| 类别 | 实际 target 均值 | 去 bonus target 均值 | 实际残差均值 | 去 bonus 残差均值 |
|---|---:|---:|---:|---:|
| direct terminal | 7,390.03 | 5,770.03 | 7,377.38 | 5,757.38 |
| 5-step contains bonus | 7,372.33 | 5,756.18 | 7,359.83 | 5,743.68 |

对 direct terminal 样本，`+1620` 平均只占实际残差的 **21.96%**。即使完全移除该 bonus，残差仍约为 5,757，而不是降到 immediate resource cost 类别约 309 的量级。

因此，不能把“约 7,360 vs 309”的全部差异归因于 bonus。终端产量增益本身已经形成了数千量级的 value target。

### 4.2 Huber 局部梯度完全没有改变

对 1,703 个 direct terminal draws：

- 去掉 bonus 后仍处于 Huber 线性区：**100%**；
- 去掉 bonus 前后，`dL/dQ` 完全相同：**100%**。

对 1,890 个 5-step bonus draws，结论同样是：

- 去掉 bonus 后仍处于 Huber 线性区：**100%**；
- 去掉 bonus 前后，局部 Huber 导数完全相同：**100%**。

这不表示 bonus 对完整训练过程毫无影响。bonus 会改变 priority、采样分布和长期网络状态；但在本次固定样本路径的直接局部比较里，它没有改变 Huber 梯度的方向或绝对值。

### 4.3 实际 1K 中，同一终端 transition 的残差几乎没有闭合

| global index | 不同 update 数 | draws | 首段 Q | 末段 Q | 残差闭合比例 |
|---:|---:|---:|---:|---:|---:|
| 319 | 456 | 681 | 7.459 | 13.784 | 0.0856% |
| 479 | 332 | 467 | 9.873 | 12.935 | 0.0416% |
| 639 | 256 | 344 | 13.308 | 13.937 | 0.0085% |
| 799 | 158 | 211 | 13.194 | 13.747 | 0.0075% |

四条 direct terminal transition 的平均残差闭合比例仅为 **0.0358%**，中位数为 **0.0250%**，远低于预注册的 1% 界线。

这只能说明：在 021_45 的 1K、fixed-target、当前 optimizer/batch/PER 条件下，chosen Q 对这些终端 target 的实际拟合非常慢。不能把观测斜率线性外推为“理论上还需多少训练步”。

## 5. 预注册分支判定

判定为 **B**：

- 100% direct 样本去 bonus 后仍在 Huber 线性区；
- 100% direct 样本的 Huber 局部导数完全相同；
- direct transition 的平均经验残差闭合比例仅 0.0358%，低于 1%。

## 6. 结论

Claude 提出的“bonus 量级使 Huber 梯度预算不足”只得到部分支持，必须收紧为：

1. `+1620` 确实进一步抬高了 target，但只解释约 22% 的 direct terminal 残差；
2. 移除 bonus 后，终端产量增益 target 仍约为 5,770，value 拟合困难在 bonus 加入之前已经存在；
3. 在 Huber 已饱和的固定样本局部比较中，去掉 bonus 不会改变单样本梯度方向和大小；
4. 因此，“降低 bonus 系数”不能被现有证据支持为下一步的直接修复，更不能把 7,360 的残差简单除以学习率来估计所需训练步数；
5. 当前更准确的问题是：**终端产量价值目标本身很大，而在 1K fixed-target 的 DQfD/DQN 更新条件下，网络几乎没有拟合该终端价值。**

## 7. 与传播问题的关系

021_47 已证明：

- bonus 样本不缺采样；
- target 在 1K 内冻结；
- 标准 5-step bootstrap 通道不能逐段把终端价值传播到距离终止 83–110 步的 DAP29/42/56；
- 神经网络参数共享仍可能产生不可预测的间接泛化影响。

021_48 又证明：即使只看终端 transition 本身，value target 也几乎没有被拟合。因此，“局部 value 拟合缓慢”和“标准 bootstrap 传播受限”可以同时存在，目前不能把任何一个单独宣布为唯一根因。

## 8. 下一步边界

本任务不直接启动完整 return-to-go。若下一步测试完整 episode return，应先写新的预注册任务，并满足：

- 不把完整 return 与可能方向冲突的 1-step TD 等权直接叠加；
- 明确采用 replacement 或 mask 的一致性目标；
- 先做离线符号/梯度单元测试，防止重复 021_27 已发现的精确抵消；
- 仍然只做一个 seed 的短期对照，不同时调整 bonus、训练时长和 target 同步。

## 9. 输出文件

- `prompts/021_48_sy2014_terminal_bonus_huber_budget_counterfactual_audit.md`
- `src/audit_sy2014_terminal_bonus_huber_budget_counterfactual_021_48.py`
- `benchmark_results/021_48/021_48_bonus_sample_counterfactual.csv`
- `benchmark_results/021_48/021_48_direct_bonus_counterfactual.csv`
- `benchmark_results/021_48/021_48_direct_transition_empirical_fit.csv`
- `benchmark_results/021_48/021_48_category_summary.csv`
- `benchmark_results/021_48/021_48_terminal_bonus_huber_counterfactual.png`
- `benchmark_results/021_48/021_48_terminal_bonus_huber_counterfactual.svg`
- `benchmark_results/021_48/021_48_summary.json`

## 10. 状态

- 状态：completed
- 新训练：无
- DSSAT 调用：无
- Git commit/push：本任务未执行

