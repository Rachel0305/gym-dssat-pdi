# 014_02 HLA 全年份筛选与 DQN 迁移准备

## 目的

在已经完成禹城站和封丘站 DQN 探索后，重新评估海伦站是否值得继续做 DQN 水氮优化。

本轮不直接训练 DQN，先用已有的 HLA 2004–2023 IC=1 null 与 DSSAT 自动灌溉诊断结果，筛选有优化空间的年份，并整理已有 HLA DQN 结果，决定下一步是否重启 HLA DQN。

## 输入依据

- 工作目录：`C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi`
- 只读取项目内已有结果，不访问项目外文件。
- 主要输入：
  - `DSSAT_auto_validation/HLA_2004/hla_ic1_yearly_diagnostics_2004_2023/hla_original_ic1_null_vs_auto_candidate_years.csv`
  - `DSSAT_auto_validation/HLA_2004/hla_ic1_yearly_diagnostics_2004_2023/hla_ic1_yearly_summary.csv`
  - `DSSAT_auto_validation/HLA_2004/hla_2010_2015_four_scenario_with_ppo/hla_2010_2015_four_scenario_summary.csv`
  - `DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/hla2010_dqn_economic_compare_summary.csv`
  - `DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_05_hla2015/hla2015_economic_dqn_four_scenario_summary.csv`

## 筛选标准

候选年份优先满足：

1. null 和 DSSAT auto 都能正常完成并产生非零产量；
2. DSSAT auto 相对 null 有明显增产，说明该年份存在水分管理优化空间；
3. null 情景出现水分胁迫，说明不是完全无管理需求年份；
4. 不优先选择 2004、2012 这种 null 或 auto 失败/极端异常年份；
5. 结合已有 2010、2015 DQN 结果，判断海伦站继续训练是否值得。

## 输出

- `DSSAT_auto_validation/HLA_2004/hla_all_year_screen_and_dqn_transfer_014_02/014_02_hla_candidate_year_screen_summary.csv`
- `DSSAT_auto_validation/HLA_2004/hla_all_year_screen_and_dqn_transfer_014_02/014_02_hla_existing_dqn_summary.csv`
- `DSSAT_auto_validation/HLA_2004/hla_all_year_screen_and_dqn_transfer_014_02/figures/014_02_hla_null_auto_yield_gain.png`
- `docs/2026-06-30_014_02_hla_all_year_screen_and_dqn_transfer_record.md`

## 运行原则

- 本轮只做低成本整理和绘图，不训练、不运行 DSSAT。
- 如果筛选结果显示 HLA 仍有可探索年份，再单独写下一轮 prompt 做 smoke test。
- 保持中文实验记录。
- 不覆盖旧结果。
