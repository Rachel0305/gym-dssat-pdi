# 219YCA YC norm-observation PPO pilot record

## Boundary

- Site: YCA/YC lowIC.
- Data: same 10-year x 6 RAIN-augmented weather bank as 218YCA.
- Action grid unchanged: irrigation [0, 15, 30, 45] mm x N [0, 40, 80, 120] kg/ha.
- Reward unchanged: grain yield - 1.1 x irrigation - 1.58 x N, scaled by 0.001.
- Single learning change: VecNormalize observation normalization; no forecast features.
- Resource fix: release each closed DSSAT environment's own atexit callback.
- 2014-2023 is treated as development validation here, not a pristine final test.

## Why This Change

The diagnosis found very high first-layer saturation and near-zero action response to water/N-only state swaps under fixed masks.
This is evidence for testing observation normalization, not causal proof that normalization alone will solve the policy problem.

```json
{
  "diagnosis_result": "benchmark_results/219YCA_training_regression_diagnosis",
  "218YCA_reward_identity_passed": true,
  "218YCA_matched_saved_actions": true,
  "218YCA_sensitivity": [
    {
      "checkpoint": 5000,
      "intervention": "all_observed_state_same_dap",
      "eligible_rows": 886,
      "changed_action_rows": 88,
      "action_change_rate": 0.09932279909706546,
      "mean_probability_l1": 0.01779639720916748,
      "max_probability_l1": 0.13015440106391907
    },
    {
      "checkpoint": 5000,
      "intervention": "water_n_features_only",
      "eligible_rows": 876,
      "changed_action_rows": 0,
      "action_change_rate": 0.0,
      "mean_probability_l1": 7.453441503457725e-05,
      "max_probability_l1": 0.0017790049314498901
    },
    {
      "checkpoint": 100000,
      "intervention": "all_observed_state_same_dap",
      "eligible_rows": 886,
      "changed_action_rows": 59,
      "action_change_rate": 0.06659142212189616,
      "mean_probability_l1": 0.05023610219359398,
      "max_probability_l1": 0.6364133358001709
    },
    {
      "checkpoint": 100000,
      "intervention": "water_n_features_only",
      "eligible_rows": 876,
      "changed_action_rows": 0,
      "action_change_rate": 0.0,
      "mean_probability_l1": 0.00015937785792630166,
      "max_probability_l1": 0.005353309214115143
    },
    {
      "checkpoint": 200000,
      "intervention": "all_observed_state_same_dap",
      "eligible_rows": 886,
      "changed_action_rows": 0,
      "action_change_rate": 0.0,
      "mean_probability_l1": 0.015301133506000042,
      "max_probability_l1": 0.2804580330848694
    },
    {
      "checkpoint": 200000,
      "intervention": "water_n_features_only",
      "eligible_rows": 876,
      "changed_action_rows": 0,
      "action_change_rate": 0.0,
      "mean_probability_l1": 3.910735176759772e-05,
      "max_probability_l1": 0.0016945339739322662
    }
  ],
  "218YCA_memory": {
    "created": 4,
    "retained_after_normal_close_gc": 4,
    "retained_after_own_atexit_unregister_gc": 0,
    "callbacks": [
      "DssatPdi._cleanup_process",
      "DssatPdi._cleanup_process",
      "DssatPdi._cleanup_process",
      "DssatPdi._cleanup_process"
    ],
    "interpretation": "Controlled reachability test; does not by itself assign all RSS growth to these objects."
  }
}
```

## Checkpoint Summary

| seed | checkpoint_step | validation_years | mean_final_grnwt | mean_simple_profit | mean_WP_ET_kg_m3 | mean_PFP_N | mean_total_irrigation | mean_total_n | mean_irrigation_events | mean_n_events | unique_action_signatures | unique_total_irrigation | unique_total_n | mean_entropy | mean_top_probability | first_layer_saturation | critic_G_RMSE | water_n_action_change_rate | water_n_probability_l1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2016 | 2 | 7200.1544 | 6655.9544 | 2.105 | 30.0 | 150.0 | 240.0 | 8.0 | 2.0 | 2 | 1 | 1 | 0.9995 | 0.2621 | 0.016 | 7.4657 | 0.1111 | 0.0127 |
| 0 | 5040 | 10 | 7981.9776 | 7437.7776 | 2.194 | 33.27 | 150.0 | 240.0 | 5.9 | 3.8 | 10 | 1 | 1 | 0.9976 | 0.3071 | 0.0273 | 8.1069 | 0.3305 | 0.0285 |
| 0 | 10080 | 10 | 7880.674 | 7336.474 | 2.148 | 32.83 | 150.0 | 240.0 | 4.7 | 4.0 | 8 | 1 | 1 | 0.9953 | 0.2869 | 0.029 | 5.4455 | 0.2738 | 0.0319 |
| 0 | 20160 | 10 | 7755.7138 | 7211.5138 | 2.126 | 32.32 | 150.0 | 240.0 | 7.8 | 2.0 | 8 | 1 | 1 | 0.9237 | 0.4743 | 0.0285 | 1.6262 | 0.2313 | 0.1323 |
| 1 | 2016 | 2 | 8133.1888 | 7588.9888 | 2.25 | 33.85 | 150.0 | 240.0 | 6.0 | 2.0 | 1 | 1 | 1 | 0.9968 | 0.2515 | 0.0612 | 8.6804 | 0.0 | 0.0072 |
| 1 | 5040 | 10 | 7803.005 | 7258.805 | 2.124 | 32.53 | 150.0 | 240.0 | 4.0 | 2.7 | 3 | 1 | 1 | 0.9947 | 0.2936 | 0.0547 | 8.4397 | 0.08 | 0.0221 |
| 1 | 10080 | 10 | 7834.9877 | 7290.7877 | 2.134 | 32.65 | 150.0 | 240.0 | 4.0 | 2.5 | 2 | 1 | 1 | 0.9615 | 0.2894 | 0.0641 | 5.8253 | 0.125 | 0.0469 |
| 1 | 20160 | 10 | 8117.1107 | 7572.9107 | 2.294 | 33.83 | 150.0 | 240.0 | 5.3 | 2.0 | 9 | 1 | 1 | 0.8867 | 0.4342 | 0.0154 | 1.6342 | 0.0422 | 0.0752 |

## Result Manifest

```json
{
  "task": "219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot",
  "status": "completed",
  "output_root": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot",
  "record_md": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/219YCA_record.md",
  "elapsed_seconds": 1046.6649911580025,
  "seeds": [
    0,
    1
  ],
  "actual_checkpoint_steps": [
    2016,
    5040,
    10080,
    20160
  ],
  "primary_final_checkpoint": 20160,
  "preflight": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/configs/219YCA_preflight.json",
  "effective_training_contract": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/configs/219YCA_effective_training_contract.json",
  "checkpoint_inventory": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/tables/219YCA_checkpoint_inventory.csv",
  "checkpoint_summary": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/tables/219YCA_checkpoint_summary.csv",
  "episode_metrics": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/tables/219YCA_development_validation_episode_metrics.csv",
  "decision_states": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/tables/219YCA_development_validation_decision_states.csv",
  "state_sensitivity": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/tables/219YCA_development_validation_state_sensitivity.csv",
  "seed_results": [
    {
      "seed": 0,
      "status": "completed",
      "error": "",
      "checkpoints": [
        {
          "seed": 0,
          "checkpoint_step": 2016,
          "target_step": 2016,
          "model_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed0_ckpt2016.zip",
          "model_sha256": "de30fb2ea752c5bd3285be9ed5c18d3d938e14e9b30bda13f1e4698705fc6212",
          "vecnormalize_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed0_ckpt2016_vecnormalize.pkl",
          "vecnormalize_sha256": "e0d50c3f40aa50598c10905f08893e8fe6edcff9bdee9e799464e1f5044312a1",
          "eval_years": "2014,2015",
          "rss_mb": 386.6796875,
          "status": "completed"
        },
        {
          "seed": 0,
          "checkpoint_step": 5040,
          "target_step": 5040,
          "model_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed0_ckpt5040.zip",
          "model_sha256": "8234466f3387495e4abfe8308eecd4e47e3c0a9eeb39a97d74bbd5107b82cb1a",
          "vecnormalize_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed0_ckpt5040_vecnormalize.pkl",
          "vecnormalize_sha256": "849b983832427443a4b0fd10d2d2743b3c5b161c0dcedce5bd5211ffcf78694e",
          "eval_years": "2014,2015,2016,2017,2018,2019,2020,2021,2022,2023",
          "rss_mb": 392.3046875,
          "status": "completed"
        },
        {
          "seed": 0,
          "checkpoint_step": 10080,
          "target_step": 10080,
          "model_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed0_ckpt10080.zip",
          "model_sha256": "b76ee9f77fe9df016d1429fb47c0c2c0a113a32f245cf87d6f8f28b198cf8ceb",
          "vecnormalize_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed0_ckpt10080_vecnormalize.pkl",
          "vecnormalize_sha256": "e9f09bfd5f2935a399668d046a3feea807def873be7290dfe01d1cb62b27dfa5",
          "eval_years": "2014,2015,2016,2017,2018,2019,2020,2021,2022,2023",
          "rss_mb": 393.8046875,
          "status": "completed"
        },
        {
          "seed": 0,
          "checkpoint_step": 20160,
          "target_step": 20160,
          "model_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed0_ckpt20160.zip",
          "model_sha256": "493255dcd3f50ac92e47d8a98f2c0e4a252bd2a05ff162b2e328a8f89912128b",
          "vecnormalize_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed0_ckpt20160_vecnormalize.pkl",
          "vecnormalize_sha256": "482d931cede41e4de7b798eedb4fade4bae2a1ced5a20ed766e34ed3b7a61e00",
          "eval_years": "2014,2015,2016,2017,2018,2019,2020,2021,2022,2023",
          "rss_mb": 394.9296875,
          "status": "completed"
        }
      ]
    },
    {
      "seed": 1,
      "status": "completed",
      "error": "",
      "checkpoints": [
        {
          "seed": 1,
          "checkpoint_step": 2016,
          "target_step": 2016,
          "model_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed1_ckpt2016.zip",
          "model_sha256": "da2ab57e830a04a1717ab3798c535efe3379f29edbc20c9e98b6dd672b3a05c4",
          "vecnormalize_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed1_ckpt2016_vecnormalize.pkl",
          "vecnormalize_sha256": "bb8add8bf1170407f018685ac77877838b2892c54583eaffcc8cf95e04e78e0e",
          "eval_years": "2014,2015",
          "rss_mb": 395.9296875,
          "status": "completed"
        },
        {
          "seed": 1,
          "checkpoint_step": 5040,
          "target_step": 5040,
          "model_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed1_ckpt5040.zip",
          "model_sha256": "f03b86ee557ee792ee06ac2254eb5f689b6c9286977bd023bd11f35d1b87040c",
          "vecnormalize_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed1_ckpt5040_vecnormalize.pkl",
          "vecnormalize_sha256": "852ae413652839147ec351da5f5b7e41c01406fa348779f02292f5b311ff2780",
          "eval_years": "2014,2015,2016,2017,2018,2019,2020,2021,2022,2023",
          "rss_mb": 396.0546875,
          "status": "completed"
        },
        {
          "seed": 1,
          "checkpoint_step": 10080,
          "target_step": 10080,
          "model_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed1_ckpt10080.zip",
          "model_sha256": "4dcc9a22e9b6075b031261bab406026ded4aa7bb80b29e3bdaf552a4e751903b",
          "vecnormalize_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed1_ckpt10080_vecnormalize.pkl",
          "vecnormalize_sha256": "47d51026e6a91c848deb5a324541b2b710c61e5596e62d0e4d5075ae93fb7606",
          "eval_years": "2014,2015,2016,2017,2018,2019,2020,2021,2022,2023",
          "rss_mb": 396.8046875,
          "status": "completed"
        },
        {
          "seed": 1,
          "checkpoint_step": 20160,
          "target_step": 20160,
          "model_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed1_ckpt20160.zip",
          "model_sha256": "d9b6198c74cefbf2090ff10f069398558d731470e2cd09406bbafe3d08d51d4c",
          "vecnormalize_path": "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/models/YCA/YCA_normobs_simple_profit_seed1_ckpt20160_vecnormalize.pkl",
          "vecnormalize_sha256": "5dd9737e6bd4a20f2c8d12c62edacc9041e7743a71e1f712ac02c6aef185cd01",
          "eval_years": "2014,2015,2016,2017,2018,2019,2020,2021,2022,2023",
          "rss_mb": 397.203125,
          "status": "completed"
        }
      ]
    }
  ]
}
```
