# 040_00 SYA lowIC 自由时序 MaskablePPO 训练任务书

## 任务目的

在已经通过 039_00/039_02 审计的 lowIC 输入数据上，重新运行 SYA 站点的自由时序 stress-aware MaskablePPO。

本任务只回答一个问题：

> 在其他训练框架不变的情况下，把 DSSAT 输入从 originalIC 切换为 lowIC 后，SYA 的 PPO 是否能在更明显的初始水氮限制下学出合理策略？

## 科学边界

本任务只改变一件事：

- 输入数据根目录从  
  `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`  
  改为  
  `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`

以下内容保持不变：

- 算法：MaskablePPO
- 训练脚本主逻辑：沿用 `032_22` half-split 多年训练框架
- 站点内训练/验证年份划分：沿用 `032_21_half_split_years.csv`
- 训练步数：100,000 timesteps
- checkpoint：25k / 50k / 75k / 100k
- seed：0
- 动作空间：灌溉 `{0, 15, 30, 45}` mm；施氮 `{0, 40, 80, 120}` kg/ha
- 管理约束：单季水氮软上限、7 天最小操作间隔、DAP90 后禁氮等均沿用原配置
- 奖励函数：沿用 `032_00` stress-aware reward，不重新调参
- 不修改 DSSAT 环境原生逻辑

## 年份范围

本轮只跑 SYA。

039_02 的 lowIC 可用性分类显示 SYA 全部 19 个年份均为：

```text
A_RL_training_candidate
```

因此 040_00 不再额外筛选 SY 年份，直接沿用 032_21 的 SYA half-split 年份划分。

## 执行规则

1. 先运行 dry-run，确认：
   - 输入根目录是 lowIC；
   - 只选择 SYA；
   - 年份划分正确；
   - 参数没有意外改变。
2. dry-run 通过后，用户手动运行正式训练。
3. 本任务不在运行中临时改 reward、步数、约束或年份。
4. 如果训练失败，只记录失败原因，不现场改参数补救。

## 推荐运行命令

在指定 Docker 容器的指定虚拟环境中运行：

```bash
cd /workspace/src
python run_sya_lowIC_free_timing_maskableppo_040_00.py --dry-run
python run_sya_lowIC_free_timing_maskableppo_040_00.py
```

## 预期输出

- `benchmark_results/040_00_sya_lowIC_free_timing_maskableppo/`
- `docs/040_00_sya_lowIC_free_timing_maskableppo_record.md`

其中主要表格包括：

- `040_00_half_split_selection.csv`
- `040_00_training_checkpoint_inventory.csv`
- `040_00_training_year_reset_counts.csv`
- `040_00_checkpoint_validation_summary.csv`
- `040_00_validation_summary_by_station_checkpoint.csv`
- `040_00_result.json`

