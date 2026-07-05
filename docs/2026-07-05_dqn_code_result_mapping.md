# DQN 代码-结果对应表

日期：2026-07-05

目的：把当前 HLA、YC、FQ 的 DQN 训练代码、奖励函数、结果目录和实验记录一一对应起来，避免把早期探索线、正式统一线和后处理绘图线混在一起。

## 0. 当前最重要结论

当前 HLA、YC、FQ 已经有一条共同的 baseline-relative DQN 框架线。该线的核心奖励函数一致：

```text
非终止步：
reward_t = - 1.0 * I_t - 5.0 * N_t

终止步：
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

即：

```text
reward = terminal max(0, final_grain - local_null_grain) - water_cost * irrigation - nitrogen_cost * nitrogen
```

共同设置：

| 项目 | 当前统一设置 |
|---|---|
| 算法 | DQN |
| 动作空间 | 9-action |
| 灌溉动作 | 0 / 15 / 30 mm |
| 施氮动作 | 0 / 50 / 100 kg/ha |
| 灌溉总预算 | 120 mm |
| 施氮总预算 | 300 kg/ha |
| 单次灌溉上限 | 30 mm |
| 单次施氮上限 | 100 kg/ha |
| 最小操作间隔 | 7 days |
| 水成本 | 1.0 |
| 氮成本 | 5.0 |
| baseline | 各站点年份自己的 null 产量 |

注意：早期 `unified/economic` DQN 线使用过 `max(0, delta_grnwt) - cost`，但那不是当前 HLA/YC/FQ baseline-relative 主线。

## 1. HLA baseline-relative DQN

### 1.1 训练脚本

| 内容 | 路径 |
|---|---|
| 主训练脚本 | `src/run_hla_baseline_relative_dqn_checkpoint_015_12.py` |
| 主要记录 MD | `docs/2026-07-01_015_12_hla2010_baseline_relative_dqn_checkpoint_record.md` |
| 主要记录 MD | `docs/2026-07-01_015_12_hla2015_baseline_relative_dqn_checkpoint_record.md` |
| seed 稳定性记录 | `docs/2026-07-01_015_13_hla2010_2015_baseline_relative_seed1_validation_record.md` |

脚本中实际奖励函数位置：

```text
src/run_hla_baseline_relative_dqn_checkpoint_015_12.py
line 69: water_cost_term = yc.WATER_COST * irrigation
line 70: nitrogen_cost_term = yc.NITROGEN_COST * nitrogen
line 71: yield_gain = max(0.0, grnwt - self.null_baseline_yield) if bool(terminated or truncated) else 0.0
line 72: reward = float(yield_gain - water_cost_term - nitrogen_cost_term)
```

### 1.2 结果目录

| 年份 | seed | 结果目录 |
|---:|---:|---|
| 2010 | 0 | `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed0_50000steps` |
| 2010 | 1 | `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps` |
| 2015 | 0 | `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed0_50000steps` |
| 2015 | 1 | `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed1_50000steps` |

每个目录内核心文件：

| 文件 | 含义 |
|---|---|
| `checkpoint_summary.csv` | 每个 checkpoint 的水氮总量、产量、reward 汇总 |
| `dqn_eval_daily.csv` | 对应 checkpoint 评估日值 |
| `figures/*.png` | checkpoint 诊断图 |
| `models/dqn_baseline_relative_checkpoint_*.zip` | 保存的 DQN checkpoint 模型 |

### 1.3 已核对关键结果

HLA2010：

| seed | 代表 checkpoint | I mm | N kg/ha | GWAD kg/ha | total_reward | 当前解释 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 15000/25000/35000 | 120 | 0 | 7854 | 777.7 | 可追平 DSSAT auto，但 seed1 未完全复现 |
| 1 | 25000 | 60 | 0 | 7573 | 557.0 | 优于 null，但低于 recorded/auto |

HLA2015：

| seed | 代表 checkpoint | I mm | N kg/ha | GWAD kg/ha | total_reward | 当前解释 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 45000 | 75 | 0 | 7653 | 1092.2 | 略高于 DSSAT auto，明显节水 |
| 1 | 40000 | 75 | 0 | 7653 | 1092.1 | 与 seed0 高度一致 |
| 1 | 50000 | 45 | 50 | 7653 | 872.2 | 同产量，更少水但用 50 kg/ha N |

### 1.4 HLA 跨年份迁移

| 内容 | 路径 |
|---|---|
| 跨年总结脚本 | `src/summarize_hla2010_dqn_cross_year_transfer_015_20.py` |
| 跨年记录 MD | `docs/2026-07-02_015_20_hla2010_dqn_cross_year_transfer_summary.md` |
| 汇总目录 | `DSSAT_auto_validation/HLA_2004/hla2010_dqn_cross_year_transfer_summary_015_20` |

记录中明确写明：

| 项目 | 内容 |
|---|---|
| 训练年 | HLA2010 |
| 迁移验证年 | HLA2007、HLA2015、HLA2016、HLA2022 |
| 是否重新训练 | 否，只加载 HLA2010 checkpoint |
| 主要结论 | 同站点跨年份迁移有迹象；不等于跨站点泛化 |

## 2. YC baseline-relative DQN

### 2.1 训练脚本

| 内容 | 路径 |
|---|---|
| 主训练脚本 | `src/run_yc2014_baseline_relative_dqn_015_10.py` |
| 主记录 MD | `docs/2026-07-01_015_10_yc2014_baseline_relative_dqn_record.md` |
| checkpoint refresh 记录 MD | `docs/2026-07-05_016_08_yc2014_baseline_relative_checkpoint_refresh_record.md` |

脚本中实际奖励函数位置：

```text
src/run_yc2014_baseline_relative_dqn_015_10.py
line 68: water_cost_term = yc.WATER_COST * irrigation
line 69: nitrogen_cost_term = yc.NITROGEN_COST * nitrogen
line 70: yield_gain = max(0.0, grnwt - self.null_baseline_yield) if bool(terminated or truncated) else 0.0
line 71: reward = float(yield_gain - water_cost_term - nitrogen_cost_term)
```

### 2.2 结果目录

| 内容 | 路径 |
|---|---|
| 原始 baseline-relative 结果 | `DSSAT_auto_validation/yc2014_baseline_relative_dqn_015_10` |
| checkpoint refresh 结果 | `DSSAT_auto_validation/yc2014_baseline_relative_checkpoint_refresh_016_08` |

核心文件：

| 文件 | 含义 |
|---|---|
| `seed*/null/015_10_yc2014_null_summary.csv` | YC2014 本地 null baseline |
| `seed*/dqn_baseline_relative_checkpoint/015_10_yc2014_baseline_relative_checkpoint_summary.csv` | checkpoint 汇总 |
| `seed*/dqn_baseline_relative_checkpoint/models/dqn_baseline_relative_checkpoint_*.zip` | 保存的 DQN checkpoint 模型 |

### 2.3 已核对关键结果

`016_08` checkpoint refresh：

| seed | 代表 checkpoint | I mm | N kg/ha | GWAD kg/ha | total_reward | 当前解释 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 5000 | 120 | 300 | 9418 | -27.1 | 高产 checkpoint |
| 0 | 25000 | 120 | 0 | 8657 | 711.9 | 高 reward checkpoint |
| 1 | 30000 | 30 | 300 | 9418 | 62.7 | 高产且少灌溉 checkpoint |
| 1 | 50000 | 75 | 0 | 8657 | 756.6 | 高 reward checkpoint |

### 2.4 YC 跨年份迁移

| 内容 | 路径 |
|---|---|
| 真模型迁移脚本 | `src/run_yc2014_station_level3_true_model_transfer_016_04.py` |
| 真模型迁移记录 MD | `docs/2026-07-02_016_04_yc2014_station_level3_true_model_transfer_record.md` |
| 真模型迁移结果目录 | `DSSAT_auto_validation/yc2014_station_level3_true_model_transfer_016_04` |
| 修正后的四情景图 | `DSSAT_auto_validation/yc2014_cross_year_transfer_success_plots_016_11_fixed` |
| 修正后的汇总图表 | `DSSAT_auto_validation/yc2014_cross_year_transfer_summary_report_016_12_fixed` |

`016_04` 脚本明确引用 `015_10` 的模型与动作表：

```text
src/run_yc2014_station_level3_true_model_transfer_016_04.py
line 33: from run_yc2014_baseline_relative_dqn_015_10 import BaselineRelativeRewardWrapper
line 34: from run_yc2014_baseline_relative_dqn_015_10 import ACTION_TABLE_9
line 42: YC_MODEL_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_dqn_015_10"
```

代表性迁移年：

| 年份 | DQN I mm | DQN N kg/ha | DQN GWAD | 相对 expert | 相对 auto |
|---:|---:|---:|---:|---|---|
| 2006 | 0 | 300 | 9618 | 高 103 kg/ha，水氮更少 | 高 922 kg/ha，多施 N |
| 2009 | 0 | 300 | 9322 | 低 80 kg/ha，水氮更少 | 高 524 kg/ha，多施 N |
| 2015 | 0 | 300 | 9361 | 低 196 kg/ha，水氮更少 | 高 184 kg/ha，多施 N |
| 2018 | 0 | 300 | 7495 | 低 79 kg/ha，水氮更少 | 高 176 kg/ha，多施 N |

重要修正：

早期 `016_11` 图漏画了真实管理事件，因为只从 daily action 字段画管理措施。已在 `016_11_fixed` 中改为读取 DSSAT `MgmtEvent.OUT`，因此 expert/auto 的灌溉施肥事件已恢复。

## 3. FQ baseline-relative DQN

### 3.1 训练脚本

| 内容 | 路径 |
|---|---|
| 主训练脚本 | `src/run_fq2016_baseline_relative_dqn_checkpoint_015_14.py` |
| 主记录 MD | `docs/2026-07-01_015_14_fq2016_baseline_relative_dqn_checkpoint_record.md` |

脚本中实际奖励函数位置：

```text
src/run_fq2016_baseline_relative_dqn_checkpoint_015_14.py
line 71: water_cost_term = yc_dqn.WATER_COST * irrigation
line 72: nitrogen_cost_term = yc_dqn.NITROGEN_COST * nitrogen
line 73: yield_gain = max(0.0, grnwt - self.null_baseline_yield) if bool(terminated or truncated) else 0.0
line 74: reward = float(yield_gain - water_cost_term - nitrogen_cost_term)
```

### 3.2 结果目录

| 内容 | 路径 |
|---|---|
| FQ2016 checkpoint 结果 | `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14` |
| FQ/YC level-3 跨年诊断 | `DSSAT_auto_validation/yc_fq_station_level3_cross_year_seed_016_03` |
| FQ2016 seed1 稳定性复核记录 | `docs/2026-07-05_017_01_fq2016_baseline_relative_seed1_validation_record.md` |

核心文件：

| 文件 | 含义 |
|---|---|
| `seed0_5000steps/checkpoint_summary.csv` | 5K 低成本 checkpoint 结果 |
| `seed0_50000steps/checkpoint_summary.csv` | 50K checkpoint 结果 |
| `yc_fq_station_level3_cross_year_seed_016_03/fq_model_transfer_summary.csv` | FQ2016 模型跨年迁移汇总 |
| `yc_fq_station_level3_cross_year_seed_016_03/fq_model_transfer_daily.csv` | FQ2016 模型跨年迁移日值 |

### 3.3 已核对关键结果

`015_14` 记录中的 FQ2016 5K 结果：

| checkpoint | I mm | N kg/ha | GWAD kg/ha | total_reward | 当前解释 |
|---:|---:|---:|---:|---:|---|
| 1000 | 120 | 300 | 7970 | -716.5 | 接近 auto，但水氮用满 |
| 2000 | 120 | 300 | 7988 | -698.0 | 接近 auto，但水氮用满 |
| 3000 | 90 | 300 | 7980 | -675.9 | 接近 auto，少一点水 |
| 4000 | 120 | 300 | 8012 | -673.6 | 与 auto 持平，但水氮用满 |
| 5000 | 120 | 300 | 8012 | -673.6 | 与 auto 持平，但水氮用满 |

`017_01` seed1 稳定性复核：

| seed | best reward checkpoint | I mm | N kg/ha | GWAD kg/ha | total_reward | 当前解释 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 25000 | 60 | 0 | 7779 | 652.6 | 低于 auto/recorded，但优于 null，节水省氮 |
| 1 | 30000 | 60 | 0 | 7995 | 869.2 | 几乎追平 auto，高于 recorded，节水省氮 |

`016_03` 记录中的 FQ 模型迁移：

| 迁移年 | checkpoint | GWAD kg/ha | I mm | N kg/ha | 相对 auto |
|---:|---:|---:|---:|---:|---|
| 2010 | 25000 | 6636 | 30 | 0 | 达到 100% auto，少水 |
| 2017 | 25000 | 7706 | 30 | 0 | 达到 100% auto，少水 |

注意：`016_03` 同时记录了很多 FQ 年份，但部分年份高投入 checkpoint 用满 I120/N300，部分年份无操作 checkpoint 产量很低。结合 `017_01`，FQ2016 已经显示出跨 seed 的 `I60/N0` 节水省氮策略迹象，但还没有像 HLA/YC 那样整理成最终汇报图组。

## 4. 历史线与不要混用的结果

| 历史线 | 脚本/记录 | 奖励 | 当前用途 |
|---|---|---|---|
| YC 015_01-015_05 unified/economic DQN | `src/run_yc2014_unified_dqn_*.py`，`docs/2026-06-30_015_01...015_05...md` | `max(0, delta_grnwt) - cost` | 早期探索，不作为当前统一主线 |
| HLA 015_09 unified DQN | `src/run_hla_unified_dqn_long_train_015_09.py` | `max(0, delta_grnwt) - cost` | 早期/并行探索，不作为当前 baseline-relative 主线 |
| YC/FQ 016_03 中的 YC action replay | `src/summarize_yc_fq_station_level3_cross_year_seed_016_03.py` | 动作回放，不是真模型迁移 | 后来已被 `016_04` YC 真模型迁移替代 |
| YC 016_11 初版图 | `DSSAT_auto_validation/yc2014_cross_year_transfer_success_plots_016_11` | 绘图后处理问题 | 管理事件漏画，使用 `016_11_fixed` 替代 |

## 5. 当前可对导师说明的口径

1. 当前 HLA、YC、FQ 的 baseline-relative DQN 主线奖励函数已经统一，不是三个站点各自乱改 reward。
2. 统一公式是：收获时奖励“相对本地 null 的产量增益”，全过程扣水氮成本。
3. HLA2015 是目前最稳定的 HLA 成功案例；HLA2010 有优化潜力但 seed 稳定性较弱。
4. YC2014 已经完成真模型跨年迁移，代表年显示“相对 expert 更省水氮、产量接近；相对 auto 多施氮、产量更高”。
5. FQ2016 已有 baseline-relative DQN、seed1 复核和部分跨年候选结果；下一步需要整理 FQ 四情景过程图和候选年份迁移图组，暂不应和 HLA/YC 的完整证据等级完全等同。

## 6. 下一步建议

建议先继续 FQ，而不是退回重改 HLA reward。理由：

- HLA 和 YC 当前主线奖励已经一致；
- FQ 脚本也采用同一类 baseline-relative 奖励；
- 下一步的重点是把 FQ 做到与 HLA/YC 同等证据等级：同站点候选年份筛选、seed 复核、跨年迁移、四情景过程图与日值表。
