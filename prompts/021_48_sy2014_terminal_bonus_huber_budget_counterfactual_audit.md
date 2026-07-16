# 021_48 SY2014 terminal bonus 的 Huber 梯度预算反事实审计

## 研究问题

021_47 已确认：terminal bonus 样本并未因 PER 而少见，含 bonus 的 transition 在 1K 训练中被抽中 1,890 次，但其 n-step TD 残差仍约为 7,360。Claude 提出“bonus 把目标抬高后，Huber 线性区的有限梯度预算可能不足”的候选解释。

本任务先做零训练成本的反事实审计，回答：

1. 去掉 `+1620` bonus 后，同一批已抽中样本的 TD 残差是否会离开 Huber 线性区；
2. 去掉 bonus 后，Huber 对 chosen Q 的导数方向或大小是否改变；
3. 当前大残差究竟主要由 bonus 新增，还是在 bonus 之前已由终端产量增益目标形成；
4. 对同一条 terminal transition，实际 1K 更新期间 chosen Q 以多快速度接近固定 target。

## 范围与禁止事项

- 只读取 `benchmark_results/021_47/021_47_agent_sample_draws.csv`。
- 不运行 DSSAT，不训练模型，不修改 reward、IC、动作空间、网络或 replay。
- 不把现有 importance weight / PER probability 冒充为“无 bonus 训练”中的反事实权重。去掉 bonus 后，priority 和采样分布本来也会改变；本任务只隔离**固定样本路径下，target 数值变化对 Huber 局部导数的直接影响**。
- 不用 `residual / learning_rate` 声称理论所需更新步数。神经网络 Jacobian、Adam、batch 聚合、importance weight 和参数共享使这种算法不成立。
- 允许报告基于已观测 chosen-Q 轨迹的描述性线性外推，但必须标记“不是理论训练预算，也不是因果预测”。
- 不测试 full return-to-go；若未来测试，必须预注册 coherent replacement/mask，不能把可能方向冲突的完整回报与 1-step TD 等权直接相加。

## 反事实定义

对每条 `nstep_contains_bonus=True` 的样本：

```text
target_n_no_bonus = target_n - discounted_bonus_in_nstep
residual_n_actual = target_n - chosen_q
residual_n_no_bonus = target_n_no_bonus - chosen_q
```

对 `direct_bonus=True` 的终止样本：

```text
target_1_no_bonus = target_1 - 1620
```

非 direct bonus 样本的 one-step target 不含 bonus，不作 one-step 反事实扣减。

Huber delta 固定为 1。对 chosen Q 的导数定义为：

```text
dL/dQ = clip(chosen_q - target, -1, 1)
```

## 同 transition 的经验拟合速度

- 先按 `(global_index, update)` 聚合，同一次 optimizer update 内重复抽到同一 transition 只保留一个 chosen-Q 均值，避免把重复 draw 当成多次参数更新。
- 对 direct bonus transition，要求至少覆盖 20 个不同 update。
- 记录首末 decile chosen-Q、target、初末残差、残差闭合比例、OLS `Q/update` 斜率。
- 若斜率为正，可给出 `末端残差/斜率` 的描述性外推，但必须注明它假设未来保持线性，不能当作所需训练步数。

## 预注册判据

- **A：bonus 是局部 Huber 梯度变化的主要直接来源候选。** 去掉 bonus 后，超过 50% direct bonus 样本离开 Huber 线性区，或超过 50% 样本的 Huber 导数发生变化/反号。
- **B：bonus 不是大残差和局部 Huber 饱和的主要直接来源。** 至少 95% direct bonus 样本去掉 bonus 后仍在 Huber 线性区，且至少 95% 样本的 Huber 导数完全相同；同时 direct transition 的经验残差闭合比例低于 1%。这支持“终端价值拟合困难在 bonus 之前已存在，bonus 只进一步抬高目标”的解释。
- **C：混合结果。** 不满足 A 或 B。
- **D：数据字段、反事实恒等式或聚合校验失败。** 停止科学解释。

## 必须输出

- 样本级反事实 CSV；
- direct transition 经验拟合 CSV；
- 类别汇总 CSV；
- PNG/SVG 图；
- JSON 摘要；
- 中文实验记录，明确适用边界和下一步，不自动启动新训练。

