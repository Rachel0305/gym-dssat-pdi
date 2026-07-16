# 021_21 SY2014 文献对齐 DQfD 离线组件与单元审计

## 1. 目的

021_18 已证明 SY2014 存在 HWAM=11205 kg/ha、I75/N200 的高产节水节氮 oracle；021_19 的简单行为克隆和 021_20 的最小 DQfD-style 均未能把该稀疏时序稳定传给 DQN。021_21 不再训练模型，只建立并审计一个更接近 Hester et al. (2018) 的 DQfD replay/loss 基础组件，为下一轮是否允许5K A/B提供工程证据。

本任务不调用 DSSAT、不启动 DQN 训练、不修改任何站点输入、IC、reward、动作、预算或旧结果。

## 2. 方法来源与准确命名

方法来源：Hester, T. et al. *Deep Q-learning from Demonstrations*, AAAI 2018, DOI 10.1609/aaai.v32i1.11757。

预注册文献参数：

- prioritized replay exponent `alpha=0.4`；
- importance-sampling initial exponent `beta=0.6`；
- demonstration priority bonus `epsilon_demo=1.0`；
- agent priority bonus `epsilon_agent=0.001`；
- large margin `0.8`；
- `lambda_n_step=1.0`；
- `lambda_margin=1.0`；
- `lambda_l2=1e-5`。

项目适配：沿用冻结主线的 `gamma=0.99` 和 `n_step=5`，而不是机械复制论文的训练规模和任务设定。后续若进入训练，仍应标注为“literature-aligned DQfD adaptation”，不能写成原论文的逐项完整复现。

## 3. 优先回放定义

建立项目本地 `PrioritizedDemonstrationReplay`：

- 示范 transition 使用独立永久区，永不被 agent ring buffer 淘汰；
- agent transition 使用固定容量环形区；
- 两类样本进入统一优先采样分布；
- 原始优先级：`p_i = abs(td_error_i) + epsilon_type`；
- 采样概率：`P(i) = p_i ** alpha / sum(p ** alpha)`；
- importance weight：`w_i = (N * P(i)) ** (-beta)`，并除以 batch 最大值归一化到 `(0, 1]`；
- 采样后允许根据新 TD error 更新优先级，但示范身份和永久性不能改变。

本轮不实现 beta 随训练进度退火，因为没有训练；仅验证 beta=0.6 的公式与数值。

## 4. 损失定义与日志要求

对合成 batch 实现并拆分记录：

- 1-step TD Huber loss；
- n-step TD Huber loss；
- demonstration-only large-margin loss；
- Q-network L2 loss；
- 原始分量；
- 乘 lambda 后的加权分量；
- 总损失；
- 各分量相对于1-step TD的比值。

margin loss 必须只作用于 demonstration mask；agent 样本即使动作相同也不能获得监督 margin 项。

## 5. 必须通过的离线单元测试

1. 复用 `benchmark_results/021_20/021_20_demonstration_transitions.npz`，确认160条示范及所有字段有限；
2. agent ring buffer 插入超过容量的样本后，160条示范仍完整且内容哈希不变；
3. 相同 TD error 下，示范样本优先级严格高于 agent 样本；
4. alpha=0.4 的统一采样概率和手算结果一致，概率和为1；
5. beta=0.6 的IS权重与概率负相关、最大值为1且全部有限；
6. priority update 只改变指定样本的数值，不改变身份、内容和示范永久性；
7. large-margin满足“expert领先至少0.8时为0，否则为正”；
8. agent-only batch的margin loss严格为0；
9. 混合batch的四项原始/加权损失及总损失全部有限，分项求和与总损失误差≤1e-6；
10. 固定随机种子重复采样产生完全相同的索引和权重；
11. 不进行任何网络训练或DSSAT调用。

任一核心测试失败即停止，不现场修改参数后重试；若是纯代码缺陷，只能备份失败现场、记录修复、重新运行同一预注册测试，不得修改科学参数。

## 6. 损失量级处理纪律

本轮只报告各项损失的数值及比值，不据此现场调整 lambda。即使发现某项显著大于其他项，也只能形成下一任务的预注册依据，不能在021_21中改权重或启动5K。

## 7. 输出

- `src/literature_aligned_dqfd.py`
- `src/test_literature_aligned_dqfd_offline_021_21.py`
- `benchmark_results/021_21/021_21_unit_test_results.json`
- `benchmark_results/021_21/021_21_priority_table.csv`
- `benchmark_results/021_21/021_21_loss_component_audit.csv`
- `docs/2026-07-15_021_21_sy2014_literature_aligned_dqfd_offline_unit_audit.md`

## 8. 资源与版本纪律

- 使用指定容器 `b2fd6726c8c1` 和 `/opt/gym_dssat_pdi/bin/python`；
- 纯CPU、纯离线、小数组测试，禁止训练和DSSAT；
- 不覆盖021_18–021_20或其他历史结果；
- 不安装新包；
- 本轮结束后只汇报是否具备进入下一轮5K设计的工程条件，不自动启动5K。

