# 043_06 SYA lowIC binary-forecast BC-only imitation audit 记录

## 一句话结论

分支：`C_bc_still_template_like`。本任务只做 teacher imitation / BC，不做 PPO fine-tune。

## 关键边界

- 未调用 PPO `model.learn()`；
- 未修改 reward；
- 未修改 DSSAT 输入；
- 使用 043_02 的 30维天气/预报/归一化 observation；
- 使用 042_10 binary action 与既有 safety mask。

## BC 数据

- 样本数：1398
- 年份数：10
- 被 mask 跳过的 teacher 正动作数：6

### BC 训练日志

| bc_epoch | bc_loss | bc_full_action_accuracy | bc_full_nonzero_action_accuracy | bc_full_nonzero_pred_rate |
| --- | --- | --- | --- | --- |
| 0 |  | 0.4134 | 0.5373 | 0.6109 |
| 5 | 1.078 | 0.402 | 0.6269 | 0.628 |
| 20 | 0.9877 | 0.4041 | 0.6716 | 0.628 |
| 50 | 0.8456 | 0.3991 | 0.5672 | 0.628 |

### BC-only rollout 汇总

| stage | checkpoint_step | validation_years | mean_yield | mean_wp_et | mean_pfp_n | mean_irrigation | mean_nitrogen | any_metric_win_years | all3_win_years | unique_action_signatures | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_only | 0 | 10 | 9633.1 | 2.013 | 40.13 | 225.0 | 240.0 | 9 | 2 | 2 | 1.0 | 0.1409 |
| bc_only | 5 | 10 | 9634.2 | 2.013 | 40.14 | 225.0 | 240.0 | 9 | 2 | 1 | 1.0 | 0.1409 |
| bc_only | 20 | 10 | 9634.2 | 2.013 | 40.14 | 225.0 | 240.0 | 9 | 2 | 1 | 1.0 | 0.1409 |
| bc_only | 50 | 10 | 9634.2 | 2.013 | 40.14 | 225.0 | 240.0 | 9 | 2 | 1 | 1.0 | 0.1409 |

### 模型清单

| station_code | site | seed | checkpoint_step | stage | model_path | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 0 | 0 | bc_only | benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/models/SYA/SYA_binary_forecast_bc_only_seed0_epoch0.zip | 985cb2382f27eb40d6740ac72b261f8d761c032af6f4bfdbdcee57629b50ddd6 |
| SYA | SY | 0 | 5 | bc_only | benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/models/SYA/SYA_binary_forecast_bc_only_seed0_epoch5.zip | 20c02de18bcaa070870de430202936c130fa65005590685c3948e0d82fa6c646 |
| SYA | SY | 0 | 20 | bc_only | benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/models/SYA/SYA_binary_forecast_bc_only_seed0_epoch20.zip | e60f958adfd2bee2e08d512dafe3fc2b07a510be51ae9e8eb461672b71fc3c65 |
| SYA | SY | 0 | 50 | bc_only | benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/models/SYA/SYA_binary_forecast_bc_only_seed0_epoch50.zip | fb0f43ec7bc3230583184031af9cf2fc9ca8ed85648503efa324fed3336521ac |

## 输出文件

```json
{
  "record_md": "docs/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit_record.md",
  "bc_log": "benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/logs/043_06_bc_snapshot_log.csv",
  "inventory": "benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/evaluation/043_06_bc_only_model_inventory.csv",
  "validation_summary": "benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/evaluation/043_06_bc_only_validation_summary.csv",
  "by_epoch": "benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/evaluation/043_06_bc_only_summary_by_epoch.csv",
  "bc_dataset_meta": "benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/bc_dataset/043_05_bc_dataset_meta.csv",
  "skipped_masked_teacher_actions": "benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/bc_dataset/043_05_bc_skipped_masked_teacher_actions.csv",
  "result_json": "benchmark_results/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit/043_06_result.json"
}
```
