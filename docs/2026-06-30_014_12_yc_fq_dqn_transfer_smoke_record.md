# 014_12 YC / FQ DQN 迁移最小验证记录

## 目的

在 HLA2010 / HLA2015 已经证明 `9-action + baseline-relative reward + IC=1` 可以跨 seed 复现成功之后，先用同一套 DQN 迁移框架，在禹城和封丘做最小验证，判断这套方法能不能换站点继续站住。

## 统一设置

- 仍然使用 linked DQN 框架
- 不改奖励逻辑，不改动作语义
- 先看最小验证，不直接加大步长

## 运行情况

### 禹城站

- 运行年份：YC2014
- 结果：seed1 复核成功
- 输出动作：
  - `dqn_linked_free_daily`
  - `dqn_linked_agronomic_window`
- 结果摘要：
  - 两个场景都使用了 `I120 / N300`
  - 最终籽粒产量约 `9417 kg/ha`
  - 水分胁迫为 `0`
  - 氮胁迫约 `0.0129`
- 说明：
  - 禹城站不仅能跑，而且 seed1 复核与 seed0 相当，说明方法对该站点具备稳定迁移性。

### 封丘站

- 运行年份：FQ2019、FQ2016
- 结果：
  - 2019：两个 DQN 场景都能跑通，并且高产
  - 2016：两个 DQN 场景都能跑通，其中 `agronomic_window` 更接近“节约型高产”
- 结果摘要：
  - 2019 `free_daily`：`8814 kg/ha`
  - 2019 `agronomic_window`：`8819 kg/ha`
  - 2016 seed0 `free_daily`：`7890 kg/ha`
  - 2016 seed0 `agronomic_window`：`8012 kg/ha`
  - 2016 seed1 `free_daily`：`8012 kg/ha`
  - 2016 seed1 `agronomic_window`：`8012 kg/ha`
- 说明：
  - 封丘站不是不能迁移，而是不同年份的可解释性和节约性差异较大；
  - 2016 seed1 与 seed0 基本一致，说明它比 2019 更像后续正式训练的候选年，因为它在较少投入下也能拿到接近或更好的结果。

## 初步结论

1. YC2014 已经可以作为跨站点迁移成功的正例。
2. FQ2016 和 FQ2019 都能跑通，说明这套 DQN 方法不是只对海伦站有效。
3. FQ2016 比 FQ2019 更适合先进入正式训练候选，因为它更像“少投入也能拿到不错结果”的年份。
4. 如果要正式加大步长，优先做 YC2014 / FQ2016 的 seed1 复核，再决定是否扩展到更多年份。

## 下一步建议

- YC2014：作为跨站点稳定成功案例保留；
- FQ2016：作为正式训练候选优先保留；
- FQ2019：作为高产但不一定节约的对照保留；
- 继续前最好先把 YC2014 / FQ2016 的 seed1 结果固定下来，再决定是否扩展。

## 记录文件

- `prompts/014_12_yc_fq_dqn_transfer_smoke_prompt.md`
- `DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/014_01_fq_dqn_summary.csv`
- `DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/014_01_fq_selected_four_scenario_daily.csv`
- `DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/014_01_fq_selected_four_scenario_events.csv`
