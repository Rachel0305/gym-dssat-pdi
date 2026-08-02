# 043_00 SYA lowIC 奇数年训练 / 偶数年验证 binary-timing MaskablePPO

## 任务目的

在不改变 042_10 主线 PPO 算法、reward、lowIC 输入、动作安全约束和二元剂量动作空间的前提下，只改变年份划分方式：

- 奇数年：训练集
- 偶数年：验证集

目的是检查此前“前半段年份训练、后半段年份验证”的时间外推划分，是否让气候分布偏移过大，从而导致 PPO 学到较模板化的策略。043_00 不是新调参，而是一个 split 设计对照。

## 固定继承项

继承 042_10/040_36：

- 站点：SYA
- 输入目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 环境：每日运行、每日观测、自由时序决策
- 算法：MaskablePPO
- 动作空间：
  - irrigation levels: `[0, 45]` mm
  - nitrogen levels: `[0, 80]` kg/ha
  - 组合动作数：4
- 约束：
  - 单次灌溉最大 45 mm
  - 单次施氮最大 80 kg/ha
  - 灌溉最小间隔 7 天
  - 施氮最小间隔 7 天
  - 季节灌溉上限/安全层继承 040_36
  - 季节施氮上限/安全层继承 040_36
  - DAP90 后禁氮
  - 后期灌溉 reserve mask 继承 040_36
- reward：继承 040_36/040_28，不新增 reward 项，不修改权重。

## 唯一计划改动

年份划分从 042_10 的前半段训练/后半段验证，改为：

- train: 所有奇数年份
- validation: 所有偶数年份

这个改动用于判断“交错年份训练”能否让模型接触到更均衡的天气分布，从而提高对验证年份天气/土壤差异的响应。

## 预注册判读

043_00 不以单一 checkpoint 的最终指标直接宣布方法成功，而看三类证据：

1. 验证集指标：
   - 产量、WP_ET、PFP_N 相对四情景的表现；
   - 是否至少保持 042_10/042_12 的基本水平。
2. 措施响应性：
   - 验证年份之间是否出现不同管理序列；
   - 灌溉是否仍固定在少数模板日；
   - 是否能减少 2017 类水分胁迫失败。
3. 措施合理性：
   - 不出现频繁小灌；
   - 不出现一天集中打满；
   - 不违反 7 天间隔、DAP90 后禁氮和季节上限。

若交错划分只改善指标但措施仍完全模板化，则只能说明 split 带来性能改善，不能说明 PPO 已学到足够天气响应性。

## 执行命令

先做 dry-run：

```bash
cd /workspace/src
python run_sya_lowIC_odd_even_binary_timing_maskableppo_043_00.py --dry-run
```

可选 2K smoke：

```bash
cd /workspace/src
python run_sya_lowIC_odd_even_binary_timing_maskableppo_043_00.py --timesteps 2000 --checkpoint-steps 1000,2000 --suffix smoke2k
```

正式训练：

```bash
cd /workspace/src
python run_sya_lowIC_odd_even_binary_timing_maskableppo_043_00.py
```

## 输出

- 记录文档：`docs/043_00_sya_lowIC_odd_even_binary_timing_maskableppo_record.md`
- 结果目录：`benchmark_results/043_00_sya_lowIC_odd_even_binary_timing_maskableppo/`
- 主要表格：
  - `evaluation/043_00_training_checkpoint_inventory.csv`
  - `evaluation/043_00_checkpoint_validation_summary.csv`
  - `evaluation/043_00_validation_summary_by_station_checkpoint.csv`
  - `configs/043_00_odd_even_selection.csv`

