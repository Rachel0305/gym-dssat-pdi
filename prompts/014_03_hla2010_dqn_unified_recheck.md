# 014_03 HLA2010 DQN 统一流程复核

## 目的

在 014_02 中，HLA2010 被筛选为海伦站优先复核年份：

- null 产量非零；
- DSSAT auto 相对 null 有明显增产；
- null 有明显水分胁迫；
- 旧的 HLA2010 economic DQN 曾经达到 DSSAT auto 同量级，但需要用现在 YC/FQ 统一流程复核。

本轮目标是检查：用当前 YC/FQ 的 linked DQN 方法迁移到 HLA2010，是否仍能产生合理水氮决策。

## 输入与环境

- 工作目录：`C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi`
- Docker 容器：`b2fd6726c8c1`
- Python：`/opt/gym_dssat_pdi/bin/python`
- HLA 输入来源：已有 HLA IC=1、新品种参数输入
  - `DSSAT_auto_validation/HLA_2004/hla_new_cultivar_candidate_year_screening/null/2010/input`
- 品种参数：已更新的 HLA `MZCER048.CUL`
- 管理方式：由脚本插入 gym-DSSAT Jinja 占位符，并切换为 linked 管理，确保 DQN 动作可以传入 DSSAT。

## DQN 方法

沿用当前 YC/FQ linked DQN 设置：

- 离散动作：
  - 0：不操作
  - 1：灌溉 30 mm
  - 2：施氮 100 kg/ha
  - 3：灌溉 30 mm + 施氮 100 kg/ha
- 总预算：
  - 灌溉上限 120 mm
  - 施氮上限 300 kg/ha
- 单次上限：
  - 灌溉 30 mm
  - 施氮 100 kg/ha
- 最小操作间隔：7 天
- 两种窗口：
  - free daily：DAP 1–120 都允许水氮操作；
  - agronomic window：灌溉 DAP 35–65，施氮 DAP 1–10 和 35–55。
- 奖励函数：
  - `reward = delta_grnwt - 1.0 * irrigation - 5.0 * nitrogen`

## 执行顺序

1. 先运行 200 step smoke test，确认：
   - 环境能正常 reset/step；
   - DQN 动作能写入 DSSAT；
   - 评估 CSV、MgmtEvent.OUT、event summary 正常；
   - 不 OOM、不卡死。
2. smoke 正常后，再决定是否运行 5K seed0。
3. 如果 5K seed0 合理，再讨论 seed1，不直接多 seed 大跑。

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_dqn_unified_recheck_014_03/`
- smoke/5K 每个 run 保存：
  - `dqn_eval_daily.csv`
  - `event_summary.json`
  - `pdi_tmp_snapshot_eval/`
  - debug log
- 汇总：
  - `014_03_hla2010_dqn_unified_recheck_summary.csv`
  - `docs/2026-06-30_014_03_hla2010_dqn_unified_recheck_record.md`

## 注意事项

- 本轮先小测试，避免浪费算力。
- 不覆盖 012、013、014_01、014_02 的旧结果。
- 如果发现 linked 管理未生效，立即停止，不继续训练。
