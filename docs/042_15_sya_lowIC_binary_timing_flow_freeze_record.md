# 042_15 SYA lowIC binary-timing PPO 阶段性冻结说明

## 冻结对象

本次冻结的是 SYA/SY 站点在 lowIC 输入条件下的 binary-timing MaskablePPO 流程。

核心思想：

- 每日环境运行，agent 每日观察；
- 但单次剂量固定为二元动作：
  - 灌溉：`0` 或 `45 mm`
  - 施氮：`0` 或 `80 kg/ha`
- 继续保留既有安全边界：
  - 灌溉/施氮最小间隔 7 天；
  - 季节灌溉上限；
  - 季节施氮上限；
  - DAP90 后禁氮；
  - 后期灌溉 reserve mask。

这条线的目的不是证明 PPO 已经“完美自主响应天气”，而是获得一个比前序连续剂量 PPO 更可解释、更少频繁小灌/小肥的自由时序候选策略。

## 已冻结任务链

| 任务 | 作用 | 关键文件 |
|---|---|---|
| 042_10 | binary-timing PPO smoke，确认动作空间和训练链路可跑 | `src/run_sya_lowIC_binary_timing_maskableppo_042_10.py` |
| 042_11 | 训练长度曲线，比较 1K/2K/5K/10K/25K checkpoint | `src/run_sya_lowIC_binary_timing_training_length_curve_042_11.py` |
| 042_12 | 对 25K checkpoint 生成五情景指标柱状图和关键年份日过程图 | `src/build_sya_lowIC_binary_timing_25k_candidate_audit_042_12.py` |
| 042_13 | 审计 25K 的天气/胁迫响应性 | `src/audit_sya_lowIC_binary_timing_25k_weather_response_042_13.py` |
| 042_14 | 建立 checkpoint 综合 guardrail 草案 | `src/audit_sya_lowIC_binary_timing_checkpoint_guardrail_042_14.py` |

## 当前结论

042_11 的 checkpoint 曲线中，25K 是目前唯一通过 042_14 诊断性 guardrail 草案的 checkpoint。

042_12 对 25K 的五情景指标审计显示：

- 10/10 验证年份至少一项指标超过四基线最高值；
- 7/10 验证年份产量超过四基线最高值；
- 9/10 验证年份 WP_ET 超过四基线最高值；
- 10/10 验证年份 PFP_N 超过四基线最高值。

042_13 的动作响应性审计显示：

- 25K 有 7 种动作签名，不是完全固定模板；
- 但仍有固定模板成分：
  - DAP1 灌溉；
  - DAP2 施氮；
  - DAP91 灌溉；
- 中期灌溉/施氮 DAP 会随年份移动。

因此当前应表述为：

> SYA lowIC binary-timing PPO 已获得一个指标表现良好、措施较前序方案更可解释、但仍带模板成分的自由时序候选策略。可以作为 SY 阶段性成果冻结，并进入下一个站点复用同一流程。

## 复现命令

以下命令均在 Docker 容器内运行：

```bash
cd /workspace/src
/opt/gym_dssat_pdi/bin/python run_sya_lowIC_binary_timing_maskableppo_042_10.py --dry-run --timesteps 2000 --checkpoint-steps 1000,2000 --suffix smoke2k
/opt/gym_dssat_pdi/bin/python run_sya_lowIC_binary_timing_maskableppo_042_10.py --timesteps 2000 --checkpoint-steps 1000,2000 --suffix smoke2k
/opt/gym_dssat_pdi/bin/python run_sya_lowIC_binary_timing_training_length_curve_042_11.py
/opt/gym_dssat_pdi/bin/python build_sya_lowIC_binary_timing_25k_candidate_audit_042_12.py
/opt/gym_dssat_pdi/bin/python audit_sya_lowIC_binary_timing_25k_weather_response_042_13.py
/opt/gym_dssat_pdi/bin/python audit_sya_lowIC_binary_timing_checkpoint_guardrail_042_14.py
```

## GitHub 上传范围

本次提交应包含：

- `prompts/042_10`–`042_14`
- `docs/042_10`–`042_15`
- `src/042_10`–`042_14` 相关脚本
- 运行这些脚本所需的轻量依赖脚本
- `benchmark_results/042_10`–`042_14` 的轻量结果、表格、图件

本次提交不应包含：

- `models/`
- `.zip` checkpoint
- `tensorboard/`
- 大型临时缓存

## 后续建议

下一站点不要重新发明奖励或约束，优先复用 042_10–042_14 这套模板：

1. lowIC/input preflight；
2. binary-timing PPO smoke；
3. 训练长度曲线；
4. 25K 或其他候选 checkpoint 的五情景指标图；
5. 天气/胁迫响应性审计；
6. checkpoint guardrail 审计。

