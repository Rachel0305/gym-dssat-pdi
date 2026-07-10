# 018_02 官方农技推广 extension expert baseline 小试：HLA2010 与 YC2014

## 任务背景

导师提出：论文中的 expert strategy 应优先参考 `references/【农技推广】玉米大豆水肥一体化单产提升技术方案.pdf` 里的区域水肥一体化推荐制度。

此前使用的 recorded/expert management 更准确应称为：

```text
farmer practice / recorded management
```

本轮目标不是重新训练 DQN，也不是修改奖励函数，而是低成本检查：新增官方农技推广推荐方案作为 `extension expert` baseline 后，现有 HLA 和 YC 结果叙事是否发生明显变化。

## 核心原则

1. 不修改 DQN 奖励函数。
2. 不重新训练 DQN。
3. 不修改原始输入文件。
4. 不覆盖旧结果。
5. 只新增一个管理情景：`extension_expert_fixed_dap`。
6. 先只测试两个代表站点年份：
   - HLA2010：代表东北及长城沿线春玉米区。
   - YC2014：代表华北黄淮和汾渭平原夏玉米区。

## 为什么不改奖励函数

当前要检验的是“expert baseline 定义变化”对结果的冲击。如果同时修改 DQN reward，会混入第二个变量，导致无法判断叙事变化来自专家基线，还是来自 DQN 目标函数。

因此本轮只新增 extension expert baseline，保留现有 DQN 结果和现有 reward 结论。

## 推文区域匹配

### HLA2010

使用推文表1：东北及长城沿线春玉米区水肥一体化灌溉施肥制度。

推荐总量：

- 氮肥 N：18–22 kg/亩，即 270–330 kg/ha。
- 灌水：155–200 方/亩，即 232.5–300 mm。

本轮使用中位数：

- N = 300 kg/ha。
- irrigation = 266.25 mm。

### YC2014

使用推文表3：华北黄淮和汾渭平原夏玉米区水肥一体化灌溉施肥制度。

推荐总量：

- 氮肥 N：15–18 kg/亩，即 225–270 kg/ha。
- 灌水：130–175 方/亩，即 195–262.5 mm。

本轮使用中位数：

- N = 247.5 kg/ha。
- irrigation = 228.75 mm。

## 固定 DAP 映射

本轮不使用实测生育期，也不使用 DSSAT 事后模拟生育期匹配。原因：

- 实测生育期不完整；
- 推文阶段比实测记录更细；
- DSSAT 事后生育期匹配可能引入 hindsight 信息；
- 固定 DAP 方案更适合作为五站点全年份统一 baseline。

### HLA/SY：东北春玉米固定 DAP

| 推文阶段 | DAP | 说明 |
|---|---:|---|
| 播种期/基肥 | 0 | 出苗水/基肥 |
| 小喇叭口期 | 30 | 第一次追肥灌水 |
| 大喇叭口期 | 50 | 关键营养生长期 |
| 抽雄散粉期 | 65 | 生殖转换关键期 |
| 灌浆初期 | 85 | 灌浆前期 |
| 乳熟末期 | 110 | 后期补水 |

### LC/YC/FQ：华北黄淮夏玉米固定 DAP

| 推文阶段 | DAP | 说明 |
|---|---:|---|
| 出苗水 | 7 | 出苗水 |
| 小喇叭口期 | 30 | 第一次关键追肥灌水 |
| 大喇叭口期 | 45 | 快速生长期 |
| 抽雄散粉期 | 60 | 生殖关键期 |
| 灌浆期 | 80 | 灌浆期 |
| 乳熟期 | 100 | 后期补水 |

## 输出

输出目录：

```text
DSSAT_auto_validation/extension_expert_baseline_018_02/
```

需要保存：

1. `extension_expert_schedule.csv`
2. HLA2010 extension expert 日值 CSV
3. YC2014 extension expert 日值 CSV
4. HLA2010 新增 extension expert 后的对比 summary
5. YC2014 新增 extension expert 后的对比 summary
6. 可读性图：至少包含产量、水氮总量对比；如时间允许，生成过程图
7. 中文实验记录 MD：

```text
docs/2026-07-09_018_02_extension_expert_baseline_hla2010_yc2014_record.md
```

## 判定重点

| 问题 | 解释 |
|---|---|
| extension expert 是否显著高于 recorded/farmer？ | 如果是，说明过去 farmer baseline 偏弱 |
| extension expert 是否显著高于 DQN？ | 如果是，DQN 需改为“逼近高投入推广方案”叙事 |
| DQN 是否接近 extension expert 但用水氮更少？ | 最理想结果 |
| extension expert 是否投入远高于 DQN budget？ | 说明它是高投入专家方案，不能与预算受限 DQN 做完全公平比较 |

