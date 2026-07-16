# 021_19 SY2014 oracle 示范引导 DQN warm-start smoke

## 目的

021_18 已证明当前 SY2014 IC=2、9动作、共享7 d间隔、I≤120/N≤300约束内存在产量11205 kg ha-1、I75/N200的高质量确定性时序。021_19 只验证这条时序能否被转换成**冻结 DQN 环境中的真实 observation→action 示范数据**，以及一个极小的行为克隆 warm-start 是否能复现它。

这不是正式训练，不得写成 DQN 成功。

## 冻结项

- 同一 SY2014 IC=2 输入；PDI/DSSAT 4.8.0。
- 同一9动作表、I120/N300预算、单次上限、DAP1–120窗口、共享7 d间隔。
- 同一 observation、baseline-relative reward 和 DQN MLP结构。
- 不修改 IC、reward、动作空间、DSSAT输入。
- 不启动25K/50K RL训练。

## 示范时序

- 灌溉：DAP22/29/42/56/79各15 mm；
- 施氮：DAP29=50、DAP42=100、DAP56=50 kg ha-1；
- 对应动作索引：DAP22/79为 action1，DAP29/56为 action4，DAP42为 action7，其余为 action0。

## Phase A：示范数据真实性 smoke

用冻结 budget/reward wrapper 逐日执行上述动作，保存每个**动作前** observation、请求动作、wrapper实际动作、reward、终止状态。

通过条件：

- DSSAT产量与021_18的11205相差≤2 kg ha-1；
- 实际I/N为75/200；
- 5个非零操作全部按预期执行，无裁剪、无漏动作；
- observation/action条数一致且全部有限。

## Phase B：行为克隆 warm-start smoke

- 新建未训练的 SB3 DQN，不加载历史问题checkpoint。
- 对同一条示范轨迹做加权交叉熵warm-start；只影响Q网络初始化，不改RL reward。
- 同步online/target网络后，保存warm-start模型。
- 报告整体动作准确率、非零操作准确率、逐类准确率、loss轨迹。
- 独立自由回放一次，检查网络能否在不强制示范动作的情况下复现合理管理。

最低通过门槛：监督数据上整体准确率≥95%，5个非零操作准确率=100%；自由回放的结果单独报告，不因监督准确率高而宣称策略已稳定。

## 停止规则

- Phase A失败：停止，不做warm-start。
- Phase B自由回放失败：记录失败，不临时调epoch、权重或网络结构追结果。
- 不启动RL微调；微调必须另立021_20并做示范/无示范单变量多seed对照。

## 输出

- `benchmark_results/021_19/`：示范CSV/NPZ、warm-start loss/准确率、模型、自由回放CSV和DSSAT快照、审计JSON。
- `docs/2026-07-15_021_19_sy2014_oracle_demonstration_dqn_warmstart_smoke.md`。
- 保存所有失败与修正，不覆盖021_18或旧模型。
