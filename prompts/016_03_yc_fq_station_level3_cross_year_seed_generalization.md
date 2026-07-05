# 016_03 YC/FQ 第3层：同站点跨年份、跨 seed 泛化诊断

## 目标

把禹城站（YC）和封丘站（FQ）推进到“第3层”诊断：判断它们是否能像海伦站（HLA）一样，在同一站点内部实现：

1. 一个训练年份得到的 DQN 策略能迁移到其他年份；
2. 不同 seed 得到的策略是否表现一致；
3. 最多可以跨多少年仍保持可解释的水氮管理效果。

## 执行原则

- 节省算力，不做新的长训练。
- 先复用已有模型、已有日值 CSV 和已有基线结果。
- 不覆盖旧结果，所有输出写入新目录。
- 明确区分“真模型迁移”和“动作策略回放”，不能把回放冒充为模型泛化。
- 所有记录用中文。

## 已知输入

### FQ

FQ2016 的 DQN checkpoint 模型已经保存：

```text
DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed0_50000steps/models/
```

其中可加载 checkpoint，例如：

```text
dqn_baseline_relative_checkpoint_5000.zip
dqn_baseline_relative_checkpoint_25000.zip
```

FQ 多年 null / recorded_shifted / DSSAT auto 基线已经在：

```text
DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/
```

因此 FQ 可以做真正的“加载 FQ2016 模型，迁移到 FQ 其他年份”的评估。

### YC

YC2014 的 seed0 / seed1 checkpoint 评估结果已经存在，但当时没有保存 DQN `.zip` 模型：

```text
DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/
DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_seed1_015_05/
```

所以 YC 当前只能先做“动作策略回放诊断”：提取 YC2014 最佳 checkpoint 的管理事件，在 YC2008 / YC2014 上回放，判断策略形态是否有跨年可移植迹象。若要做真正模型迁移，需要后续重跑 YC2014 并保存模型。

## 本次任务

1. 生成脚本：

```text
src/summarize_yc_fq_station_level3_cross_year_seed_016_03.py
```

2. FQ 部分：

- 加载 FQ2016 已保存 DQN checkpoint；
- 在 FQ 2000–2023 年逐年评估；
- 与已有 null / recorded_shifted / DSSAT auto 基线合并；
- 输出每年产量、水氮用量、胁迫最大值、相对 auto 的产量差和节水节氮量；
- 总结哪些年份达到“接近/超过 auto，且更节水节氮”的标准。

3. YC 部分：

- 从 YC2014 seed0 / seed1 最佳 checkpoint 日值 CSV 中提取实际灌溉和施肥事件；
- 在 YC2008 / YC2014 上做固定事件回放；
- 仅作为“动作策略回放诊断”，不得写成“模型迁移成功”。

4. 输出：

```text
DSSAT_auto_validation/yc_fq_station_level3_cross_year_seed_016_03/
  fq_model_transfer_summary.csv
  fq_model_transfer_daily.csv
  yc_action_replay_summary.csv
  yc_action_replay_daily.csv
  figures/
docs/2026-07-02_016_03_yc_fq_station_level3_cross_year_seed_record.md
```

5. 运行要求：

- 使用指定 Docker 容器：

```text
b2fd6726c8c1
```

- 使用指定虚拟环境：

```text
/opt/gym_dssat_pdi/bin/python
```

- 先 smoke：只跑少量年份确认能通；
- smoke 成功后再跑 FQ 2000–2023 全年评估。

## 判定口径

### 真模型迁移成功

只适用于 FQ：

- 直接加载训练年份 checkpoint；
- 目标年份不重新训练；
- DQN 产量达到或接近 DSSAT auto / recorded_shifted；
- 水氮用量低于或不显著高于 auto / recorded_shifted；
- 管理事件不是明显无意义的早期打满。

### 动作回放可移植迹象

只适用于 YC：

- 回放 YC2014 checkpoint 学到的事件时间和用量；
- 若在 YC2008 仍能明显优于 null、接近 recorded/auto，说明策略形态有跨年可移植迹象；
- 但不能称为“DQN 模型跨年泛化”，因为没有加载模型。

