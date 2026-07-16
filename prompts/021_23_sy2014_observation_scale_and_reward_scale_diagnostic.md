# 021_23 SY2014 观测尺度审计与 reward×0.1 短诊断

## 目标

在不进入 5K 训练、不修改观测信息和奖励结构的前提下，回答两个彼此分离的问题：

1. 当前 25 维观测是否存在明显的数值尺度不均衡；
2. 在严格复刻 021_22 的网络、示范数据、随机种子、更新次数和采样机制时，仅将用于训练的即时奖励与 n-step return 整体乘以 0.1，是否会降低 TD 损失、参数梯度和梯度裁剪频率。

## 背景与技术边界

- 021_22 发现示范预训练和在线短诊断中 100% 更新触发 `max_grad_norm=10` 裁剪。
- 大 TD loss 不等于参数梯度必然按误差幅度同比增大；Smooth L1/Huber loss 在大误差区间对 Q 输出的梯度饱和。因此，本任务不能预设 reward 尺度就是大梯度的唯一原因。
- 观测归一化只改变同一信息的数值表示，不等同于加入新特征；但本任务只审计，不实施归一化。
- 本任务不得进入 5K，不得修改 reward 相对权重、IC、动作空间、示范数据、PER 参数、网络结构或梯度裁剪阈值。

## 预注册设计

### A. 零训练成本观测尺度审计

读取 021_20 已保存的 160 条示范 transition 的 25 维观测，逐维输出：

- min、max、mean、median、std、range、max_abs；
- 非零比例、唯一值数量；
- 相对全体非零维中位尺度的倍数；
- 仅作描述性的尺度警报，不据此直接宣称因果。

如真实环境可以低成本读取观测变量顺序，则登记变量名称；否则使用稳定的 `obs_00...obs_24` 标签，并保留后续映射说明。

### B. reward×0.1 单变量短诊断

严格复刻 021_22：

- SY2014、IC=2、seed=0；
- 同一 DQN 配置与网络初始化；
- 同一 160 条示范数据；
- 同一 literature-aligned DQfD 组件参数；
- 100 次示范预训练；
- 100 个真实环境交互步，step 50 起在线更新，共 51 次；
- 同一 replay RNG、batch size、n-step、探索率日程与 `max_grad_norm=10`。

唯一科学变量：

- 示范和 agent transition 中用于训练的 `reward` 乘以 0.1；
- 与该 reward 对应的 `n_step_return` 同步乘以 0.1；
- 环境原始 reward 必须另列保存，不得覆盖或丢失。

### C. 实现生效校验

正式比较前必须验证：

- 示范 `scaled_reward/raw_reward = 0.1`；
- 示范 `scaled_n_step_return/raw_n_step_return = 0.1`；
- agent `training_reward/raw_reward = 0.1`（非零项）；
- 分项缩放后求和与总 reward 外层缩放一致；
- 不修改已有 021_22 文件。

## 对比指标

分别对示范预训练和在线更新比较 021_22 未缩放基准与 021_23 缩放版本：

- TD 1-step、n-step、margin、L2 loss；
- 各损失分量梯度范数与总梯度范数；
- 梯度裁剪比例；
- Q 绝对值均值/最大值；
- 单次及累计参数相对变化；
- 示范/agent 采样比例。

## 预注册解释分支

1. **TD 与梯度、裁剪均明显下降**：reward 数值尺度是参与因素，但仍不能单独称为根因。
2. **TD loss 明显下降，但梯度与裁剪未同步下降**：更支持观测/激活或采样结构是大梯度的直接来源。
3. **TD、梯度均未明显下降**：reward 缩放不是主要解释，应优先检查输入/激活、重复高优先级采样及 1-step+n-step 联合作用。

无论出现哪一分支，都不得在本任务自动进入 5K 或现场修改参数。

## 输出

- `benchmark_results/021_23/021_23_observation_scale_audit.csv`
- `benchmark_results/021_23/021_23_pretrain_update_log.csv`
- `benchmark_results/021_23/021_23_online_update_log.csv`
- `benchmark_results/021_23/021_23_agent_interactions.csv`
- `benchmark_results/021_23/021_23_stage_comparison.csv`
- `benchmark_results/021_23/021_23_scale_validation.json`
- `benchmark_results/021_23/021_23_summary.json`
- PNG/SVG 诊断图
- `docs/2026-07-15_021_23_sy2014_observation_scale_and_reward_scale_diagnostic.md`

## 资源纪律

- 只允许一次 100+100 步短诊断；
- 单进程、单环境，不并行启动 DSSAT；
- 不保存大型模型；
- 发生失败必须保留失败原因，不得覆盖旧结果；
- 不自动 Git push。
