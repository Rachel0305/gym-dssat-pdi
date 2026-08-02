# 043_06 SYA lowIC binary-forecast BC-only imitation audit

## 背景

043_04 证明：lowIC teacher 的优质轨迹具有年份差异，说明 DSSAT/lowIC 的优质水氮措施确实会响应年份天气。

043_05 的 2K smoke 暴露了一个中间层问题：即使使用 teacher warm-start，policy 仍可能输出十年同一模板。这说明问题可能不在 PPO 强化学习步数，而在 teacher imitation 是否真正把年份差异写进 policy。

## 本任务目标

只训练 imitation / behavior cloning，不做 PPO `model.learn()`，回答一个基础问题：

> PPO 的 policy 网络能不能仅凭 teacher 标签学出随年份变化的动作序列？

如果 BC-only 都无法产生非模板化动作，那么继续 PPO 长训没有意义；需要先改 teacher 数据构造、样本平衡、动作表达或 observation。

## 固定条件

- 输入数据：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 年份：2014–2023
- observation：沿用 043_02 的 30 维天气/预报/归一化 observation
- 动作空间：沿用 042_10 binary timing
  - 灌溉 `[0,45]` mm
  - 施氮 `[0,80]` kg/ha
- safety mask：沿用 040_36/042_10
- teacher：沿用 041_02 lowIC teacher
- 不运行 PPO 强化学习，不改 reward，不改 DSSAT 输入

## Teacher 动作映射

- teacher 当天灌溉量 > 0，则映射为 `I45`
- teacher 当天施氮量 > 0，则映射为 `N80`
- 水氮都为 0，则 no-op
- 若映射后的正动作被当前 safety mask 禁止，则记录到 skipped 表，并用 no-op 继续 rollout

## 训练快照

BC epoch 快照固定为：

- 0
- 5
- 20
- 50

这些快照只用于诊断 imitation 是否能产生年份差异，不用于事后挑选最终模型。

## 判据

主要看：

- `unique_action_signatures` 是否大于 1，最好达到 3 以上；
- 各年份灌溉/施氮总量和时机是否有差异；
- BC 非零动作准确率，而不是只看总 accuracy；
- 如果指标尚可但十年动作仍同一模板，则判定 imitation 没有学进 teacher 的年份差异。

## 停止规则

- 若 BC 数据收集失败，停止；
- 若可表达 teacher 正动作过少，停止并说明动作空间或约束过窄；
- 若 50 epoch 后仍是单一模板，则不进入 PPO 100K，先改 imitation 机制。
