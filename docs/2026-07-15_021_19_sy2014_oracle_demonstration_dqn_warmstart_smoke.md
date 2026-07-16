# 021_19 SY2014 oracle 示范引导 DQN warm-start smoke 记录

## 背景

021_18 在 SY2014 IC=2、冻结9动作、共享7 d间隔、I≤120/N≤300约束内找到确定性候选：HWAM=11205 kg ha-1、I75/N200。021_19 检查该候选能否被转换成冻结 DQN 环境中的真实 observation→action 示范数据，以及一次极小行为克隆 warm-start 能否自由复现它。

本实验不进行 RL 训练，不能作为 DQN 正式结果。

## 冻结条件

- 输入：SY2014 IC=2，PDI/DSSAT 4.8.0，复用021_14输入。
- 9动作、I120/N300预算、单次I30/N100上限、DAP1–120窗口、共享7 d间隔不变。
- baseline-relative reward、observation、SB3 DQN MLP不变。
- 未加载历史checkpoint；未修改IC、reward、DSSAT输入。

## Phase A：示范数据真实性

目标动作：DAP22/79 action1，DAP29/56 action4，DAP42 action7，其余action0。

| 指标 | 结果 |
|---|---:|
| observation/action条数 | 160 / 160 |
| 非零操作数 | 5 |
| 全部观测有限 | True |
| 非零操作无裁剪 | True |
| HWAM | 11205 kg ha-1 |
| CWAM | 20132 kg ha-1 |
| 灌溉 | 75 mm |
| 施氮 | 200 kg ha-1 |
| 相对021_18产量误差 | 0 kg ha-1 |

结论：示范数据链路严格通过，可作为后续算法输入。

## Phase B：行为克隆 warm-start

使用100 epoch、项目本地Q网络、逆频率类别加权交叉熵；同步online/target网络后独立自由回放。该设计预先写入prompt，没有在看到结果后调epoch或权重。

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| 最终loss | 0.5460 | 仅记录 | - |
| 监督整体准确率 | 62.5% | ≥95% | 失败 |
| 5个非零操作准确率 | 100% | 100% | 通过 |

整体门槛未通过。逆频率加权成功识别稀少的非零操作，但牺牲了大量no-op状态的识别。

这里还需要和最简单的基线比较：160个示范状态中有155个目标动作是no-op，因此“永远预测no-op”的朴素分类器准确率已经是155/160=96.875%。本轮逆频率加权模型的62.5%不仅没有达到预注册的95%，而且明显低于该朴素基线。这个结果说明简单逆频率加权把模型过度推向了少数非零动作，不能把62.5%解释成“接近通过”。

## 自由回放

| 指标 | warm-start自由回放 | oracle示范 |
|---|---:|---:|
| HWAM | 11154 | 11205 |
| CWAM | 20041 | 20132 |
| 灌溉 | 120 | 75 |
| 施氮 | 300 | 200 |
| baseline-relative reward total | 4126.13 | 未在本表重算 |

自由回放实际执行8次操作：DAP12/19/26灌溉15；DAP33/40灌溉15+施氮50；DAP47灌溉15+施氮100；DAP54/61灌溉15+施氮50。模型获得高产，但提前、频繁操作并打满I120/N300，没有复现oracle的节水节氮时序。

该自由回放是纯冻结、确定性评估：使用`model.predict(..., deterministic=True)`，没有调用`learn()`、没有梯度更新，也没有探索动作。因此其偏离oracle的行为可归因于行为克隆所得网络本身，而不是回放期间继续学习或探索造成的漂移。

## 结论

1. 021_18 oracle可以无误转换为DQN observation→action示范数据。
2. 简单的逆频率加权行为克隆不是充分方案：它把“识别稀少操作”变成了“过度操作”。
3. 该结果不能写成DQN成功，也不能因自由回放产量高而忽略资源饱和。
4. 按预注册停止规则，不临时增加epoch、不扫类别权重、不启动RL微调。

## 下一步建议

若继续示范引导路线，应采用能够同时保留TD学习与示范动作约束的方法，而不是继续调简单分类权重。优先考虑项目本地、可审计的 DQfD-style 最小实现：

- 用021_19的真实 transition/reward/done构建示范回放；
- TD loss保持DQN目标；
- 只在示范样本上加大间隔动作排序损失；
- 示范样本在replay中保留，但不把示范策略硬编码为最终动作；
- 先做极小离线单元测试和5K单seed A/B，再决定是否多seed或长训练。

这属于算法结构改动，必须另立任务并与无示范DQN做单变量对照。

## 输出

- `benchmark_results/021_19/021_19_demonstration_dataset.npz`
- `benchmark_results/021_19/021_19_demonstration_steps.csv`
- `benchmark_results/021_19/021_19_demonstration_audit.json`
- `benchmark_results/021_19/021_19_warmstart_losses.csv`
- `benchmark_results/021_19/dqn_oracle_warmstart_before_rl.zip`
- `benchmark_results/021_19/021_19_warmstart_free_eval_daily.csv`
- `benchmark_results/021_19/021_19_summary.json`

## 状态

`partial`：示范数据链路成功；简单行为克隆warm-start未达到预注册门槛，未进入RL微调。
