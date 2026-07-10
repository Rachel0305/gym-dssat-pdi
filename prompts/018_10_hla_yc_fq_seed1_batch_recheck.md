# 018_10 HLA2010 + YC2014 + FQ2016 批量 seed1 复核

## 目标

对当前 3 个最关键但仍缺少 `seed1` 证据的站点年份做统一复核：

- `HLA2010`
- `YC2014`
- `FQ2016`

目的不是重新设计框架，也不是调新参数，而是严格沿用各自当前最优 `seed0` 所对应的训练框架、奖励函数、动作空间、预算和输入数据，只补做 `seed1` 最小复现，判断这些案例能否像 `SY2014` 一样升级为跨 seed 稳定成功。

## 统一原则

- 必须使用指定 Docker：`b2fd6726c8c1`
- 必须使用指定 Python：`/opt/gym_dssat_pdi/bin/python`
- 先做小成本 smoke，再决定是否继续 formal
- 不覆盖任何旧结果目录
- 每个站点单独建新目录、新 summary、新 doc
- 只允许“复制框架 + 改 seed + 改输出目录”，不允许顺手改 reward、budget、action space

## 需要复核的三个案例

### 1) HLA2010

- 现有代表性结果来源：
  - `DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/`
- 当前口径：
  - `seed0` promising
  - 产量追平 auto / extension
  - 灌溉更省
  - 缺 `seed1`
- 任务：
  - 做 `HLA2010 seed1`
  - 先 5K smoke，再按原训练长度补 formal
  - 最终输出 `seed0 vs seed1` 对比表

### 2) YC2014

- 现有代表性结果来源：
  - `DSSAT_auto_validation/yc2014_formal_four_scenario_015_06/`
- 当前口径：
  - `seed0` promising
  - 超过 auto，接近/追平 extension
  - 缺 `seed1`
- 任务：
  - 做 `YC2014 seed1`
  - 先 5K smoke，再按原训练长度补 formal
  - 最终输出 `seed0 vs seed1` 对比表

### 3) FQ2016

- 现有代表性结果来源：
  - `DSSAT_auto_validation/fq2016_four_scenario_process_017_02/`
  - 训练脚本基础：`src/run_fq2016_baseline_relative_dqn_checkpoint_015_14.py`
- 当前口径：
  - `seed0` promising
  - 基本追平 auto，并明显优于 extension expert 的资源使用
  - 缺 `seed1`
- 任务：
  - 做 `FQ2016 seed1`
  - 先 5K smoke，再按原训练长度补 formal
  - 最终输出 `seed0 vs seed1` 对比表

## 每个站点必须产出

1. `baseline daily/events/summary`
2. `smoke checkpoint daily/events/summary`
3. `formal checkpoint daily/events/summary`（仅 smoke 成功后）
4. `seed0 vs seed1 comparison.csv`
5. 中文实验记录 `.md`

## 执行顺序建议

按风险从低到高：

1. `YC2014`
2. `HLA2010`
3. `FQ2016`

原因：

- `YC2014` 现有框架与文档最完整，最适合先补
- `HLA2010` 目前结果也较稳，但历史分支较多，需谨慎核对脚本
- `FQ2016` 结果贴近 auto，最需要最后单独看波动

## 判定标准

若满足以下条件，可升级为 `stable_success_across_seed`：

- `seed0` 与 `seed1` 都得到同方向的高产结果
- 相对 `null` 明显增产
- 相对 `DSSAT auto` 和 `extension expert` 不出现方向性反转
- 资源使用模式没有从“节水/节氮”突然塌到明显更差

若两粒种子都能追平产量，但资源用量方向反复，则归为：

- `yield_stable_resource_unstable`

若 `seed1` 明显失败或塌陷，则维持：

- `promising_but_needs_more_diagnosis`
