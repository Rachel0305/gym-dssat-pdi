# 221YCA YCA/YC lowIC 十年六天气情景 PPO 实验记录

- 阶段：`100K paired exploration`；状态：`completed`。
- 新工作线；复用 054/042/032 的训练、奖励、安全约束和原始验证代码。
- 唯一实验性变化：训练集由 10 个原始天气 episode 扩展为 10×6=60 个天气 episode。
- 为防 OOM，RandomYearEnv 的采样分布保持 uniform random，但环境对象缓存上限为 2；这是资源管理措施，不改变奖励或动作评价。
- WP_ET 只接受 Summary.OUT/ETCP 精确回放；本阶段若尚未回放则明确标为 pending。

## 数据与代码 provenance

```json
{
  "task": "221YCA_yc_ppo_domain_augmentation",
  "reference_run": "055_00_yca_lowIC_expanded_action_maskableppo",
  "source_input_root": "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual",
  "augmented_input_root": "DSSAT_auto_validation/yca_lowIC_weather_scenario_bank_217_v1_fixed_width",
  "source_manifest": "benchmark_results/217YCA_yca_lowIC_weather_scenario_bank_v1_fixed_width/217YCA_weather_scenario_manifest.csv",
  "train_source_years": [
    2004,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2013
  ],
  "train_scenario_rows": 60,
  "train_variant_counts": {
    "early_dry": 10,
    "early_wet": 10,
    "late_dry": 10,
    "mid_dry": 10,
    "mid_wet": 10,
    "original": 10
  },
  "validation_original_years": [
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023
  ],
  "validation_hash_mismatch_rows": 0,
  "unique_training_target_wth": 60,
  "actions": {
    "irrigation_levels_mm": [
      0.0,
      15.0,
      30.0,
      45.0
    ],
    "nitrogen_levels_kg_ha": [
      0.0,
      40.0,
      80.0,
      120.0
    ]
  },
  "observation_contract": {
    "base": "046_02_raw_observation",
    "normalization_enabled": false,
    "weather_forecast_enabled": false
  },
  "reward_and_safety": "055_00/040_36/042_10 unchanged",
  "runtime_safety": {
    "environment_cache_limit": 2,
    "memory_warning_mb": 1536,
    "memory_stop_mb": 2048,
    "single_process_training": true,
    "device": "cpu",
    "scenario_sampler": "balanced_shuffled_cycle_without_replacement",
    "cycle_definition": "each 60-episode cycle visits every source_year_x_variant pair exactly once"
  },
  "code_sha256": {
    "src/run_sya_lowIC_binary_timing_maskableppo_042_10.py": "8ec856b4cadbb01a26de8fb87e681393f43916244c124f86a21368910679a16c",
    "src/run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py": "b79e024168fdcae75edc193e747ca4de72a79d32b2965baf71c018b6456ef0b8",
    "src/run_all_year_direct_action_safe_ppo.py": "8fbd6f5a5d9772faa791b6d3ebb692249b0f7cf4f8513fab5e08190a74903074",
    "src/ppo_safe_rendering.py": "0714deea114c3fdb029baf1210a1a09e650f4418ef6aa52dfc9e90de2a4e15cd",
    "src/054_hla_lowIC_site_transfer/run_055_00_hla_lowIC_expanded_action_maskableppo.py": "707e7c2b4ea7f59fe10b1eee8406e090d803b5dfba958c937673ca37c3b549fa",
    "src/weather_scenario_bank_160.py": "d4acd414d4f887256c70a63c457f2e13cdf876304ad84b0a341a3dbc4e736477",
    "src/build_202HLA_hla_lowIC_weather_scenario_bank.py": "23cc71c3a45034aa94901ec0ae3d3bfee08566f626d26c25880aaa36abcc8797",
    "src/run_203HLA_hla_lowIC_10y_six_weather_aug_maskableppo.py": "49ab8926b7e8f2788d2f7c6c273bc28bfd5aa03ccfeeec061142c34826c55cd7",
    "experiments/221_yc_ppo_domain_augmentation/scripts/run_221YCA_yc_ppo_domain_augmentation.py": "62dbd686cd34b3ba8dadf835598f605a16b6bed66a394a2cc134dad16b1c04cc",
    "experiments/221_yc_ppo_domain_augmentation/configs/221YCA_yc_ppo_domain_augmentation.json": "50a968e0c43ebc6f69f0790633b3fca3c495c2e264675ca4b888346e8b6983b9",
    "prompt_01/002_codex_prompt_yc_ppo_domain_augmentation.md": "14f9a85232870e072766d52f817013375a11e49c4aafd344e2f2ec1fe339c682",
    "benchmark_results/217YCA_yca_lowIC_weather_scenario_bank_v1_fixed_width/217YCA_weather_scenario_manifest.csv": "2c9e658983ec2fc828a3f4c462fcac2dcc10cfdc49bb39a9413e261972989b1e",
    "benchmark_results/217YCA_yca_lowIC_weather_physical_gate/217YCA_result.json": "58b5d79f5c65ed404fd1b447b9321d2f3720b090a9d18c420af4d91f25d19c0f"
  },
  "issues": [],
  "next_step_allowed": true,
  "station_mapping": {
    "station_code": "YCA",
    "site": "YC",
    "short_input_dir": "YC"
  },
  "paired_baseline_alias": "benchmark_results/221YCA_paired_baseline_055_00_alias/evaluation/055_00_checkpoint_validation_summary.csv",
  "paired_baseline_source": "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/evaluation/055_00_checkpoint_validation_summary.csv",
  "augmentation_factor": "training-domain diversity only",
  "required_physical_gate": "benchmark_results/217YCA_yca_lowIC_weather_physical_gate/217YCA_result.json",
  "physical_gate": {
    "all_six_scenarios_completed": true,
    "all_weather_log_rain_matches_wth_within_2mm": true,
    "all_scenarios_matured": true,
    "all_scenarios_positive_yield": true,
    "physical_yield_response_detected": true,
    "old_055_policy_action_response_detected": false,
    "allow_2k_smoke": true
  },
  "ic_profile": "lowIC only; originIC not mixed",
  "cultivar": "ZD0985 only; genotype randomization skipped"
}
```

## 实际生效 PPO/奖励/安全配置

```json
{
  "seed": 0,
  "total_timesteps": 100000,
  "checkpoint_steps": [
    25000,
    50000,
    75000,
    100000
  ],
  "ppo": {
    "learning_rate": 0.0003,
    "gamma": 1.0,
    "gae_lambda": 1.0,
    "n_steps": 144,
    "batch_size": 144,
    "n_epochs": 5,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "net_arch": [
      64,
      64
    ]
  },
  "reward": {
    "reward_type": "harvest_yield_minus_water_nitrogen_cost_plus_stress_relief_scaled_0p001",
    "yield_coef": 0.158,
    "nitrogen_cost": 1.58,
    "water_cost": 1.1,
    "water_stress_relief_coef": 10.0,
    "nitrogen_stress_relief_coef": 5.0,
    "reward_scale": 0.001,
    "swfac_guardrail_process_penalty": {
      "enabled": true,
      "threshold": 0.05,
      "coef": 50.0,
      "formula": "coef * max(0, swfac_after_step - threshold) * reward_scale"
    }
  },
  "action_safety": {
    "enabled": true,
    "daily_irrigation_max": 45.0,
    "daily_n_max": 120.0,
    "season_irrigation_soft_limit": 240.0,
    "season_n_soft_limit": 250.0,
    "min_days_between_irrigation": 7,
    "min_days_between_fertilization": 7,
    "irrigation_allowed_dap_range": [
      1,
      120
    ],
    "fertilization_allowed_dap_range": [
      1,
      90
    ],
    "late_irrigation_reserve_mask": {
      "enabled": true,
      "dap_end": 90,
      "pre_late_cum_irrigation_cap_mm": 195.0,
      "reserved_for_after_dap90_mm": 45.0,
      "reason": "reserve at least one 45 mm irrigation opportunity after DAP90; do not force an exact irrigation day"
    }
  },
  "discrete_actions_after_203_patch": {
    "irrigation_levels_mm": [
      0.0,
      15.0,
      30.0,
      45.0
    ],
    "nitrogen_levels_kg_ha": [
      0.0,
      40.0,
      80.0,
      120.0
    ]
  },
  "reward_changed_by_203": false,
  "action_safety_changed_by_203": false,
  "observation_changed_by_203": false
}
```

## 训练情景覆盖与内存

```json
{
  "scenario_rows": 60,
  "covered_scenarios": 60,
  "coverage_ratio": 1.0,
  "covered_source_years": 10,
  "covered_variants": 6,
  "min_episode_count": 15,
  "max_episode_count": 16,
  "total_episode_starts": 941
}
```

## 原始验证集 checkpoint 指标

|   checkpoint_step |   validation_years |   mean_final_grnwt |   std_final_grnwt |   mean_PFP_N |   mean_total_irrigation |   mean_total_n |   mean_reward |   mean_gap_yield_vs_four_max |   mean_gap_pfp_n_vs_four_max |   unique_full_action_signatures |   unique_total_irrigation |   unique_total_n |   mean_requested_noop_rate |   mean_positive_action_rows |   max_swfac |   max_nstres |   off_grid_rows | all_years_same_action_signature   | all_years_zero_management   |
|------------------:|-------------------:|-------------------:|------------------:|-------------:|------------------------:|---------------:|--------------:|-----------------------------:|-----------------------------:|--------------------------------:|--------------------------:|-----------------:|---------------------------:|----------------------------:|------------:|-------------:|----------------:|:----------------------------------|:----------------------------|
|             25000 |                 10 |            8202.93 |           1033.56 |      34.1789 |                   214.5 |            240 |      0.728001 |                    -0.595898 |                      1.08635 |                               7 |                         3 |                1 |                   0.878966 |                        12.3 |   0         |    0.0146757 |               0 | False                             | False                       |
|             50000 |                 10 |            8144.45 |           1009.64 |      67.8704 |                   180   |            120 |      0.952849 |                   -59.0844   |                     34.7778  |                               3 |                         3 |                1 |                   0.901509 |                        10   |   0.665563  |    0.370107  |               0 | False                             | False                       |
|             75000 |                 10 |            8202.06 |           1034.88 |      34.1753 |                   193.5 |            240 |      0.771859 |                    -1.47047  |                      1.0827  |                               3 |                         2 |                1 |                   0.861998 |                        14   |   0.0924641 |    0.0146757 |               0 | False                             | False                       |
|            100000 |                 10 |            8199.74 |           1034.64 |      34.1656 |                   220.5 |            240 |      0.742004 |                    -3.79493  |                      1.07302 |                               2 |                         2 |                1 |                   0.883825 |                        11.8 |   0         |    0.0146757 |               0 | False                             | False                       |

## 与未增强 055_00 的相同步数配对比较

```json
{
  "new_task_id": "221YCA",
  "matched_checkpoint": 100000,
  "old_054_validation_rows": 10,
  "new_augmented_validation_rows": 10,
  "old_mean_final_grnwt": 6072.488906860352,
  "new_mean_final_grnwt": 8199.735961914062,
  "delta_mean_final_grnwt": 2127.247055053711,
  "old_mean_PFP_N": 37.953055667877194,
  "new_mean_PFP_N": 34.165566507975264,
  "delta_mean_PFP_N": -3.7874891599019307,
  "old_mean_total_irrigation": 45.0,
  "new_mean_total_irrigation": 220.5,
  "old_mean_total_n": 160.0,
  "new_mean_total_n": 240.0,
  "common_checkpoint_comparison": [
    {
      "checkpoint_step": 25000,
      "old_mean_final_grnwt": 8201.599243164062,
      "new_mean_final_grnwt": 8202.934997558594,
      "delta_mean_final_grnwt": 1.33575439453125,
      "old_mean_PFP_N": 34.173330179850254,
      "new_mean_PFP_N": 34.178895823160815,
      "delta_mean_PFP_N": 0.005565643310561086,
      "old_mean_total_irrigation": 228.0,
      "new_mean_total_irrigation": 214.5,
      "old_mean_total_n": 240.0,
      "new_mean_total_n": 240.0,
      "old_unique_action_signatures": 5,
      "new_unique_action_signatures": 7
    },
    {
      "checkpoint_step": 50000,
      "old_mean_final_grnwt": 6825.94596862793,
      "new_mean_final_grnwt": 8144.446472167969,
      "delta_mean_final_grnwt": 1318.500503540039,
      "old_mean_PFP_N": 28.441441535949707,
      "new_mean_PFP_N": 67.8703872680664,
      "delta_mean_PFP_N": 39.42894573211669,
      "old_mean_total_irrigation": 75.0,
      "new_mean_total_irrigation": 180.0,
      "old_mean_total_n": 240.0,
      "new_mean_total_n": 120.0,
      "old_unique_action_signatures": 1,
      "new_unique_action_signatures": 3
    },
    {
      "checkpoint_step": 75000,
      "old_mean_final_grnwt": 6140.797073364258,
      "new_mean_final_grnwt": 8202.060424804688,
      "delta_mean_final_grnwt": 2061.2633514404297,
      "old_mean_PFP_N": 40.73720916112264,
      "new_mean_PFP_N": 34.17525177001953,
      "delta_mean_PFP_N": -6.561957391103114,
      "old_mean_total_irrigation": 45.0,
      "new_mean_total_irrigation": 193.5,
      "old_mean_total_n": 152.0,
      "new_mean_total_n": 240.0,
      "old_unique_action_signatures": 2,
      "new_unique_action_signatures": 3
    },
    {
      "checkpoint_step": 100000,
      "old_mean_final_grnwt": 6072.488906860352,
      "new_mean_final_grnwt": 8199.735961914062,
      "delta_mean_final_grnwt": 2127.247055053711,
      "old_mean_PFP_N": 37.953055667877194,
      "new_mean_PFP_N": 34.165566507975264,
      "delta_mean_PFP_N": -3.7874891599019307,
      "old_mean_total_irrigation": 45.0,
      "new_mean_total_irrigation": 220.5,
      "old_mean_total_n": 160.0,
      "new_mean_total_n": 240.0,
      "old_unique_action_signatures": 1,
      "new_unique_action_signatures": 2
    }
  ],
  "WP_ET_status": "pending exact Summary.OUT replay; never inferred from daily CSV"
}
```

## 技术门槛

```json
{
  "status_completed": true,
  "final_checkpoint": 100000,
  "observed_validation_rows": 10,
  "expected_validation_rows": 10,
  "training_metric_update_rows": 695,
  "training_scenarios_covered": 60,
  "training_scenario_coverage_ratio": 1.0,
  "all_six_variant_types_seen": true,
  "all_ten_source_years_seen": true,
  "peak_rss_mb": 1124.65625,
  "memory_warning_exceeded": false,
  "memory_stop_exceeded": false,
  "final_actions_on_grid": true,
  "final_not_all_zero_management": true,
  "final_unique_action_signatures": 2,
  "next_step_allowed": true
}
```

## 221YCA 单因素控制

- 唯一主动变化：训练域情景多样性；使用 10 个源年 × 6 个完整天气情景，采用分层无放回循环。
- reward 保持 055_00 的 040_36/042_10 stress-aware + swfac guardrail；没有 simple-profit 替换。
- IC 固定 lowIC，不混合 originIC；cultivar 固定 ZD0985；二者均不作为本轮增强因素。
- WP_ET 不从 daily CSV 反推，若无 Summary.OUT/ETCP 精确回放则保持 pending。
