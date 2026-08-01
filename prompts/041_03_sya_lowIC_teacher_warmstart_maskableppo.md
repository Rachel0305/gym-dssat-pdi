# 041_03 SYA lowIC teacher warm-start MaskablePPO

## 背景

041_02 已经把 lowIC 条件下的 teacher 轨迹整理成分层 imitation 数据集：

- strong teacher：5 年；
- near-miss teacher：5 年；
- 每日 imitation 样本：1398；
- 非零动作样本：73。

040_53 显示当前 040_40 PPO 对已解析观测变量扰动不敏感，灌溉时机高度固定。041_03 的目标是检验：先用 teacher 轨迹做 policy warm-start，能否改善 PPO 在自由时序环境里的学习起点。

## 任务性质

这是 teacher-assisted PPO 的方法可行性训练，不是严格跨年泛化验证。

原因：teacher 来自 2014–2023，因此本任务在 2014–2023 上训练和评估只能回答“PPO 能否吸收 teacher 并在自由时序环境中保持/改善策略”，不能作为未见年份泛化结论。

若 041_03 通过，后续另开 041_04 做训练年份 teacher → 验证年份迁移。

## 固定配置

- 算法：MaskablePPO
- 输入根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 训练年份：2014–2023
- 评估年份：2014–2023
- 初始阶段：behavior cloning / imitation warm-start
- 后续阶段：自由时序 PPO fine-tune
- 动作边界：沿用 040_40
  - 灌溉档位：0、30、45 mm
  - 施氮档位：0、80、120 kg/ha
  - 季节灌溉上限：240 mm
  - 季节施氮上限：250 kg/ha
  - 最小灌溉间隔：7 天
  - 最小施氮间隔：7 天
  - DAP90 后禁氮
  - DAP≤90 累计灌溉最多 195 mm
- PPO 超参数：沿用 040_40
- reward：沿用 040_40

## imitation 权重

沿用 041_02 预注册权重：

- strong teacher 非零动作日：1.0
- strong teacher no-op 日：0.2
- near-miss teacher 非零动作日：0.5
- near-miss teacher no-op 日：0.1

## 训练步骤

1. 重放 041_02 选中的 teacher schedule，采集 PPO 实际 observation 向量、action mask、teacher action index；
2. 初始化 MaskablePPO；
3. 用加权 cross-entropy 做 policy warm-start；
4. 保存 BC 初始化模型；
5. 在 2014–2023 自由时序环境中继续 PPO fine-tune；
6. 保存 checkpoint；
7. 对每个 checkpoint 做 2014–2023 五情景指标评估。

## 预注册默认参数

- BC epoch：20
- BC batch size：256
- BC learning rate：1e-4
- PPO timesteps：100000
- checkpoint：25000、50000、75000、100000

smoke 可用：

```bash
python run_sya_lowIC_teacher_warmstart_maskableppo_041_03.py --timesteps 2000 --checkpoint-steps 1000,2000 --suffix smoke2k
```

## 停止线

- 如果 041_02 数据集缺失，停止；
- 如果采集到的 teacher action 不在当前动作 mask 内，停止；
- 如果 BC loss 为 NaN，停止；
- 如果 smoke 失败，不能进入 100k；
- 不根据结果现场改 BC 权重、reward、PPO 超参数。

