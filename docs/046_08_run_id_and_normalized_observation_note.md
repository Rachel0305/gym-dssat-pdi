# 046 系列结果目录防覆盖与 046_07 归一化阴性记录

## 1. 为什么增加 `--run-id`

046_03、046_04、046_05、046_06 原来使用固定输出目录。  
这会导致一个问题：如果只跑某一年 smoke（例如 `--years 2014`），会覆盖之前全年结果，后续五情景汇总可能出现行数不完整。

因此这四个脚本新增可选参数：

```bash
--run-id <同一个批次标识>
```

建议每次完整基线/auto/汇总/日值图流程使用同一个时间戳作为 `run-id`。  
如果不传 `--run-id`，脚本仍然使用旧的固定目录，兼容过去结果。

## 2. 推荐运行方式

在容器 `/workspace/src` 下：

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
echo $RUN_ID

python run_sya_baselines_configured_046_03.py \
  --config ../configs/046_08_sya_originIC_binary_timing_ppo_auto_nstd005.json \
  --run-id $RUN_ID

python run_sya_external_auto_n_rule_046_04.py \
  --config ../configs/046_08_sya_originIC_binary_timing_ppo_auto_nstd005.json \
  --run-id $RUN_ID

python build_sya_configured_reporting_046_05.py \
  --config ../configs/046_08_sya_originIC_binary_timing_ppo_auto_nstd005.json \
  --checkpoint 100000 \
  --run-id $RUN_ID

python build_sya_configured_five_scenario_daily_plots_046_06.py \
  --config ../configs/046_08_sya_originIC_binary_timing_ppo_auto_nstd005.json \
  --checkpoint 100000 \
  --run-id $RUN_ID
```

这样会生成带 `_run_<时间戳>` 的独立目录，不会覆盖旧结果。

## 3. 046_07 的当前判定

046_07 是“只把已有 observation 用训练年统计量归一化”的 PPO 变体。  
当前观察到的 PPO 行为是近似不管理模板：

- 灌溉主要只在 DAP1 执行 45 mm；
- 施氮主要只在 DAP1 执行 80 kg/ha；
- 措施合理性和指标表现均不如先前候选。

因此目前不把 046_07 作为候选策略继续推进。  
后续若要再做归一化，应作为新分支处理，例如“归一化 + 天气预报特征 + 重新设计动作/guardrail”，不能把 046_07 当作已经成功的改进。
