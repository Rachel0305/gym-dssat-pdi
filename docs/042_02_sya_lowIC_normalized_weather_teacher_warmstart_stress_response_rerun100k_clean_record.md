# 042_02 SYA lowIC 归一化天气/预报 observation + teacher warm-start + 氮胁迫响应 PPO 记录

## 任务性质

本任务用于检验：在 `041_04` balanced teacher warm-start 基础上，加入全输入归一化、当日天气、过去/未来 7 天天气窗口，以及与既有 SWFAC guardrail 对称的 NSTRES 过程惩罚，是否能让 PPO 从“阶段模板策略”进一步转向“天气/胁迫响应策略”。

## 运行环境和输入

- 容器：`b2fd6726c8c1`
- Python：`/opt/gym_dssat_pdi/bin/python`
- 项目目录：`/workspace/src`
- 数据源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- lowIC authoritative audit：通过，`issue_count=0`
- 站点：`SYA`
- 年份：2014--2023 selected teacher trajectories

## 代码和 prompt

- prompt：`prompts/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response.md`
- 脚本：`src/run_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_042_02.py`
- 完整运行结果目录：`benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k`

注意：无 suffix 的 `benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response` 是一次被中断的半截训练，只生成了 `bc_init/25K/50K` 模型，没有完整评估汇总；正式可用结果使用 `rerun100k` 目录。

## 预检

dry-run 在容器内通过：

- base observation：25 维；
- enhanced observation：30 维；
- 新增变量：`rain_today_mm`, `tmin_today_c`, `rain_past7_mm`, `rain_future7_mm`, `tmean_future7_c`；
- normalization scale 数量：30；
- teacher 年份：2014--2023；
- teacher tier：`strong_all3=5`, `near_miss=5`；
- action safety 沿用当前主线：7 天最小间隔、粗动作、I240/N250、DAP90 后禁氮、DAP90 前留 45 mm 灌溉机会。

## 训练配置

- 算法：MaskablePPO；
- seed：0；
- BC warm-start：沿用 `041_04` balanced nonzero sampling；
- BC epoch：20；
- BC batch size：256；
- BC learning rate：1e-4；
- PPO fine-tune：100000 steps；
- checkpoint：25000, 50000, 75000, 100000。

## 新增 reward 项

沿用 `040_40` 的 reward 和约束，只新增 NSTRES 过程惩罚：

```text
nstres_penalty_unscaled = 50 * max(0, NSTRES_after_step - 0.05)
reward = reward_04040 - nstres_penalty_unscaled * reward_scale
```

参数没有扫描：

- threshold = 0.05，与现有 SWFAC guardrail 一致；
- coef = 50，与现有 SWFAC guardrail 一致。

## 结果汇总

| stage | checkpoint | mean_yield | mean_WP_ET | mean_PFP_N | mean_I | mean_N | any_metric_win | all3_win | max_SWFAC | max_NSTRES |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BC init | 0 | 9955.0 | 2.104 | 41.49 | 240 | 240 | 10/10 | 2/10 | 0.806 | 0.259 |
| PPO | 25K | 3847.4 | 0.893 | NA | 225 | 0 | 0/10 | 0/10 | 0.000 | 0.556 |
| PPO | 50K | 3847.4 | 0.893 | NA | 225 | 0 | 0/10 | 0/10 | 0.000 | 0.556 |
| PPO | 75K | 9931.7 | 2.145 | 41.36 | 225 | 240 | 10/10 | 6/10 | 0.859 | 0.238 |
| PPO | 100K | 3847.4 | 0.893 | NA | 225 | 0 | 0/10 | 0/10 | 0.000 | 0.556 |

## 75K 逐年表现

75K 是本次完整训练中表现最好的 checkpoint：

- 10/10 年份至少一个指标超过四基线最高；
- 10/10 年份 PFP_N 超过四基线最高；
- 6/10 年份三指标全部超过四基线最高：2015, 2016, 2018, 2019, 2020, 2021；
- 2014, 2017, 2022, 2023 未三指标全胜，主要输在产量和 WP_ET。

## 判读

本任务得到一个有价值但不完全稳定的正向信号：

1. 加入 normalization + weather/forecast + NSTRES 响应后，75K checkpoint 明显优于 `041_04` 的 100K/75K 三指标全胜数量，达到 6/10。
2. 但是继续训练到 100K 后策略退化为不施氮，说明 PPO fine-tune 仍然存在训练后期漂移。
3. 因此本任务不应直接采用 100K；下一步应围绕 checkpoint 选择/早停或训练稳定性做约束，而不是继续盲目加步数。

## 下一步建议

优先做纯审计，不立刻新训练：

1. 比较 `042_02` 75K 与 100K 的 action sequence，定位为什么后期变成 N=0；
2. 检查训练 reset 年份分布，确认是否有年份采样偏差；
3. 检查 25K/50K/100K 是否为同一动作模板、是否由 policy drift 导致；
4. 若证明确实是后期 PPO fine-tune 漂移，再预注册一个早停/validation checkpoint selection 规则，而不是事后挑 75K。

