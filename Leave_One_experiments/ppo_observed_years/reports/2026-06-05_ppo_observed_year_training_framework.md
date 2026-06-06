# PPO observed-year leave-one training framework

Generated at: 2026-06-06 11:16:06

## 1. 本阶段目标
基于已经跑通的 observed-year smoke test，生成可逐站点调试的 PPO 训练、验证、绘图和策略选择框架。本阶段不批量训练全部 PPO，只运行 HLA 2007 seed0 小步数 debug PPO。

## 2. 为什么现在可以进入 PPO 框架生成
上一阶段 smoke test timeout 已排查完成，29 个原失败案例全部重跑成功。LCA/SYA/YCA 的问题来自临时 rendered DSSAT management section 和跨年份静态管理日期，已在安全渲染逻辑中修复。

## 3. 为什么不直接批量训练全部站点
完整计划共有 14 个 observed-year PPO 模型。当前阶段只验证框架和 HLA 2007 单模型流程，避免长时间训练时难以定位环境、渲染、动作记录或 OOM 问题。

## 4. 实测年份和训练-验证组合
组合表：`Leave_One_experiments/ppo_observed_years/configs/ppo_observed_year_experiment_plan.csv`

| station   | experiment_group    |   train_year | train_year_label                  | validation_years   | validation_year_labels                                                           |   num_observed_years | cross_validation_type     | notes                             |
|:----------|:--------------------|-------------:|:----------------------------------|:-------------------|:---------------------------------------------------------------------------------|---------------------:|:--------------------------|:----------------------------------|
| HLA       | HLA_train2007_seed0 |         2007 | observed_low_rain_year            | 2011,2009          | observed_mid_rain_year,observed_high_rain_year                                   |                    3 | three_year_leave_one      | observed_year_leave_one           |
| HLA       | HLA_train2011_seed0 |         2011 | observed_mid_rain_year            | 2007,2009          | observed_low_rain_year,observed_high_rain_year                                   |                    3 | three_year_leave_one      | observed_year_leave_one           |
| HLA       | HLA_train2009_seed0 |         2009 | observed_high_rain_year           | 2007,2011          | observed_low_rain_year,observed_mid_rain_year                                    |                    3 | three_year_leave_one      | observed_year_leave_one           |
| SYA       | SYA_train2014_seed0 |         2014 | observed_low_rain_year            | 2015,2012          | observed_mid_rain_year,observed_high_rain_year                                   |                    3 | three_year_leave_one      | observed_year_leave_one           |
| SYA       | SYA_train2015_seed0 |         2015 | observed_mid_rain_year            | 2014,2012          | observed_low_rain_year,observed_high_rain_year                                   |                    3 | three_year_leave_one      | observed_year_leave_one           |
| SYA       | SYA_train2012_seed0 |         2012 | observed_high_rain_year           | 2014,2015          | observed_low_rain_year,observed_mid_rain_year                                    |                    3 | three_year_leave_one      | observed_year_leave_one           |
| LCA       | LCA_train2010_seed0 |         2010 | observed_low_rain_year            | 2011,2008,2009     | observed_mid_rain_year,observed_intermediate_rain_year_3,observed_high_rain_year |                    4 | four_year_leave_one       | observed_year_leave_one           |
| LCA       | LCA_train2011_seed0 |         2011 | observed_mid_rain_year            | 2010,2008,2009     | observed_low_rain_year,observed_intermediate_rain_year_3,observed_high_rain_year |                    4 | four_year_leave_one       | observed_year_leave_one           |
| LCA       | LCA_train2008_seed0 |         2008 | observed_intermediate_rain_year_3 | 2010,2011,2009     | observed_low_rain_year,observed_mid_rain_year,observed_high_rain_year            |                    4 | four_year_leave_one       | observed_year_leave_one           |
| LCA       | LCA_train2009_seed0 |         2009 | observed_high_rain_year           | 2010,2011,2008     | observed_low_rain_year,observed_mid_rain_year,observed_intermediate_rain_year_3  |                    4 | four_year_leave_one       | observed_year_leave_one           |
| FQA       | FQA_train2008_seed0 |         2008 | observed_lower_rain_year          | 2010               | observed_higher_rain_year                                                        |                    2 | two_year_cross_validation | limited_two_year_cross_validation |
| FQA       | FQA_train2010_seed0 |         2010 | observed_higher_rain_year         | 2008               | observed_lower_rain_year                                                         |                    2 | two_year_cross_validation | limited_two_year_cross_validation |
| YCA       | YCA_train2014_seed0 |         2014 | observed_lower_rain_year          | 2008               | observed_higher_rain_year                                                        |                    2 | two_year_cross_validation | limited_two_year_cross_validation |
| YCA       | YCA_train2008_seed0 |         2008 | observed_higher_rain_year         | 2014               | observed_lower_rain_year                                                         |                    2 | two_year_cross_validation | limited_two_year_cross_validation |

## 5. 安全 rendered input 逻辑
PPO 训练和评估使用 `src/ppo_safe_rendering.py`，继承 smoke test debug 的修复：不覆盖 my_data 原始模板；每个站点-年份生成临时 rendered input；启用 MI/MF；保证 irrigation/fertilizer sections 存在；替换跨年份残留的静态灌溉/施肥事件；复制对应 QC WTH 到临时目录。

rendered input 检查表：`Leave_One_experiments/ppo_observed_years/evaluation/rendered_input_check.csv`

| check_item                  | status   |   count |
|:----------------------------|:---------|--------:|
| fertilizer_section_present  | pass     |       4 |
| irrigation_section_present  | pass     |       4 |
| mi_mf_enabled               | pass     |       4 |
| planting_section_present    | pass     |       4 |
| simulation_controls_present | pass     |       4 |
| template_exists             | pass     |       4 |

## 6. 生成的代码和配置
- `src/ppo_experiment_plan.py`
- `src/ppo_safe_rendering.py`
- `src/ppo_train.py`
- `src/ppo_evaluate.py`
- `src/ppo_plot_results.py`
- `src/ppo_strategy_selection.py`
- `experiments/ppo_observed_years/config_ppo_observed_years.yaml`
- `experiments/ppo_observed_years/generate_experiment_plan.py`
- `experiments/ppo_observed_years/train_one_policy.py`
- `experiments/ppo_observed_years/evaluate_one_policy.py`
- `experiments/ppo_observed_years/run_debug_hla_2007.py`
- `experiments/ppo_observed_years/run_all_trainings_DISABLED_BY_DEFAULT.py`
- `experiments/ppo_observed_years/run_all_evaluations_DISABLED_BY_DEFAULT.py`

## 7. HLA 2007 pretrain smoke check
pretrain smoke check 表：`Leave_One_experiments/ppo_observed_years/smoke_checks/pretrain_smoke_check_summary.csv`

| station   |   train_year | policy_name     | run_status   | episode_completed   |   error_message | daily_csv_path                                                                                             | notes                |
|:----------|-------------:|:----------------|:-------------|:--------------------|----------------:|:-----------------------------------------------------------------------------------------------------------|:---------------------|
| HLA       |         2007 | null_zero       | ok           | True                |             nan | Leave_One_experiments/ppo_observed_years/smoke_checks/daily_outputs/HLA/HLA_2007_null_zero_daily.csv       | pretrain_smoke_check |
| HLA       |         2007 | fixed_low_input | ok           | True                |             nan | Leave_One_experiments/ppo_observed_years/smoke_checks/daily_outputs/HLA/HLA_2007_fixed_low_input_daily.csv | pretrain_smoke_check |

## 8. HLA 2007 debug PPO
- station: `HLA`
- train_year: `2007`
- seed: `0`
- configured debug total_timesteps: `1000`
- note: SB3 PPO 按 n_steps rollout 成批更新，因此实际日志显示 total_timesteps 可略高于配置值。
- status: `success`

模型文件：
- `Leave_One_experiments/ppo_observed_years/models/HLA/HLA_train2007_seed0_debug.zip`

## 9. train/eval daily output 和图
- daily CSV 数：3
- 响应图 PNG 数：15

| station   | policy_name               |   train_year | train_year_label       |   eval_year | eval_year_label         |   seed | model_path                                                                        | run_status   |   error_message | episode_completed   |   episode_length |   final_grnwt |   final_topwt |   final_xlai |   total_irrigation |   total_n_fertilizer |   mean_swfac |   mean_nstres |   mean_reward |   sum_reward | daily_csv_path                                                                                    | figure_dir                                                                               | notes      |
|:----------|:--------------------------|-------------:|:-----------------------|------------:|:------------------------|-------:|:----------------------------------------------------------------------------------|:-------------|----------------:|:--------------------|-----------------:|--------------:|--------------:|-------------:|-------------------:|---------------------:|-------------:|--------------:|--------------:|-------------:|:--------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|:-----------|
| HLA       | HLA_train2007_seed0_debug |         2007 | observed_low_rain_year |        2007 | observed_low_rain_year  |      0 | Leave_One_experiments/ppo_observed_years/models/HLA/HLA_train2007_seed0_debug.zip | ok           |             nan | True                |              149 |       4463.16 |       7902.86 |  0.288843    |            3067.78 |              14503.8 |  7.7416e-05  |   0.000471106 |      -301.047 |     -44855.9 | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2007_seed0_daily.csv | Leave_One_experiments/ppo_observed_years/figures/HLA/HLA_train2007_seed0_debug/eval_2007 | debug_eval |
| HLA       | HLA_train2007_seed0_debug |         2007 | observed_low_rain_year |        2011 | observed_mid_rain_year  |      0 | Leave_One_experiments/ppo_observed_years/models/HLA/HLA_train2007_seed0_debug.zip | ok           |             nan | True                |              169 |       4443.52 |       6699.7  |  8.02002e-05 |            3486.41 |              16457.5 |  0           |   0.00105403  |      -315.571 |     -53331.5 | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2011_seed0_daily.csv | Leave_One_experiments/ppo_observed_years/figures/HLA/HLA_train2007_seed0_debug/eval_2011 | debug_eval |
| HLA       | HLA_train2007_seed0_debug |         2007 | observed_low_rain_year |        2009 | observed_high_rain_year |      0 | Leave_One_experiments/ppo_observed_years/models/HLA/HLA_train2007_seed0_debug.zip | ok           |             nan | True                |              159 |       4614.29 |       7281.79 |  0.0634939   |            3278.6  |              15485.3 |  0.000700003 |   0.000577318 |      -309.078 |     -49143.4 | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2009_seed0_daily.csv | Leave_One_experiments/ppo_observed_years/figures/HLA/HLA_train2007_seed0_debug/eval_2009 | debug_eval |

## 10. 策略选择脚本输出
策略选择输出：`Leave_One_experiments/ppo_observed_years/strategy_selection/best_policy_by_site.csv`

| station   | best_policy_name          |   best_train_year | best_train_year_type   | validation_years   |   mean_yield |   std_yield |   mean_reward |   std_reward |   mean_irrigation |   mean_n_fertilizer |   stability_score | reason                       | model_path                                                                        |
|:----------|:--------------------------|------------------:|:-----------------------|:-------------------|-------------:|------------:|--------------:|-------------:|------------------:|--------------------:|------------------:|:-----------------------------|:----------------------------------------------------------------------------------|
| HLA       | HLA_train2007_seed0_debug |              2007 | observed_low_rain_year | 2009,2011          |      4528.91 |     120.753 |      -312.325 |        4.591 |           3382.51 |             15971.4 |              -0.5 | single_seed_cross_year_score | Leave_One_experiments/ppo_observed_years/models/HLA/HLA_train2007_seed0_debug.zip |

## 11. 仍需人工确认的 PPO 超参数
最终批量训练超参数仍保留为 null，避免把 debug 参数误当最终参数。需要确认：learning_rate, gamma, n_steps, batch_size, ent_coef, clip_range

## 12. 下一步建议
可以进入下一步 HLA/SYA/LCA 的逐站点批量训练准备，但建议顺序仍然是：每次只启动一个模型，先跑对应 pretrain smoke check，再训练，再评估，确认 daily CSV 和图完整后再进入下一个模型。
