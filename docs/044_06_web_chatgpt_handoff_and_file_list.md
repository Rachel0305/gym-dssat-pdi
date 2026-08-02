# 044_06 给网页版 ChatGPT / 其他 AI 的交接说明

## 一句话背景

本项目是基于 DSSAT / gym-DSSAT 的玉米水氮强化学习优化。当前主线已经从旧 DQN 转到 SYA lowIC 自由时序 PPO；DQN / DQfD 已整理为对照证据，下一步要继续改进 PPO 的“天气响应性”和“跨年份策略合理性”。

## 当前最重要结论

1. **DSSAT 输入链路已经经历过多次修正**  
   早期结果曾受 `IC=0`、DSSAT 管理模式、originIC/lowIC 路径混用等问题影响。当前主线使用：

   ```text
   DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual
   ```

   后续任何实验开始前，必须先做输入 preflight，确认确实使用 lowIC，且 DSSAT 管理模式允许外部灌溉/施肥动作进入。

2. **DQN / DQfD 当前不是正向主线**
   - 普通 DQN 040_01：动作模板化，平均产量偏低。
   - 严格 MaskableDQN 040_02：训练加深后退化。
   - Demo-DQN / DQfD 044_00–044_02：teacher 非零动作没有成为高 Q 值动作，策略仍偏 no-op。
   - 已整理成导师汇报包，见 `044_03` 和 `044_05`。

3. **PPO 当前主问题**
   PPO 已能在部分配置下得到指标较好的策略，但策略仍容易模板化；即不同验证年份天气不同，PPO 输出的灌溉/施肥时机变化不够明显。下一步不是换算法，而是审计和增强 PPO 对天气、土壤水分/氮状态的响应。

## 建议下一个任务编号

```text
045_00_sya_lowIC_ppo_weather_responsiveness_audit
```

性质：只审计，不训练。

目标：比较当前最好 PPO 与 lowIC teacher 在 2014–2023 验证年份上的水氮分配是否随天气变化。

建议检查：

- 年总灌溉、总施氮是否随降雨、初始水分、胁迫强度变化；
- DAP1–30、31–60、61–90、91+ 四个时段的水氮分配；
- 干旱年份是否保留中后期灌溉；
- 降雨多或未来 7 天降雨高时是否避免无意义灌溉；
- PPO 是否存在全验证年同一动作序列。

通过该审计后，再决定是否进入：

```text
045_01_checkpoint_weather_response_guardrail
```

或：

```text
045_02_weather_response_reward_v3
```

## 给网页版 ChatGPT 时建议附上的文件

如果只想让它快速接手，不要发整个仓库，优先发以下文件。

### A. 当前 DQN 对照结果

```text
docs/044_03_dqn_results_for_advisor_summary_record.md
docs/044_05_dqn_bad_case_examples_for_advisor_record.md
benchmark_results/044_03_dqn_results_for_advisor_summary/tables/044_03_dqn_checkpoint_mean_metrics.csv
benchmark_results/044_03_dqn_results_for_advisor_summary/tables/044_03_demo_dqn_q_ranking_summary.csv
benchmark_results/044_05_dqn_bad_case_examples_for_advisor/tables/044_05_ordinary_dqn_action_sequence_counts.csv
benchmark_results/044_05_dqn_bad_case_examples_for_advisor/tables/044_05_q_ranking_bad_case_table.csv
```

### B. 当前 PPO 天气响应性相关结果

```text
docs/043_01_sya_lowIC_04215_policy_input_sensitivity_audit_record.md
docs/043_02_sya_lowIC_binary_timing_forecast_normalized_maskableppo_record.md
docs/043_03_sya_lowIC_04302_checkpoint_outcome_audit_record.md
docs/043_04_sya_lowIC_teacher_weather_responsiveness_audit_record.md
docs/043_06_sya_lowIC_binary_forecast_bc_only_imitation_audit_record.md
docs/043_07_sya_lowIC_bc_failure_mechanism_audit_record.md
```

### C. 关键代码

```text
src/audit_sya_lowIC_04215_policy_input_sensitivity_043_01.py
src/run_sya_lowIC_binary_timing_forecast_normalized_maskableppo_043_02.py
src/audit_sya_lowIC_teacher_weather_responsiveness_043_04.py
src/run_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_043_05.py
src/run_sya_lowIC_binary_forecast_bc_only_imitation_audit_043_06.py
src/audit_sya_lowIC_bc_failure_mechanism_043_07.py
src/build_dqn_results_for_advisor_summary_044_03.py
src/build_dqn_bad_case_examples_for_advisor_044_05.py
```

### D. 图表

```text
benchmark_results/044_03_dqn_results_for_advisor_summary/figures/
benchmark_results/044_05_dqn_bad_case_examples_for_advisor/figures/
benchmark_results/044_03_dqn_results_for_advisor_summary/ppt/044_03_dqn_results_for_advisor_summary_minimal.pptx
```

## 可以直接复制给网页版 ChatGPT 的问题描述

```text
我在做 DSSAT/gym-DSSAT 玉米水氮强化学习优化。当前站点是 SYA，输入数据使用 lowIC：
DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual。

我们已经尝试过 DQN、Strict MaskableDQN、Demo-DQN/DQfD，结论是当前 DQN 路线不稳定或偏 no-op，已经作为对照保留。现在主线回到 MaskablePPO。

当前 PPO 的问题不是完全不能优化，而是策略对不同年份天气/土壤状态的响应不够强，容易输出模板化水氮管理。已有 043_04 审计显示 teacher/优质轨迹本身是随天气变化的，但 043_02 PPO 更模板化。

请你基于我提供的 docs、CSV 和代码，帮我设计下一步 045_00：
1. 不重新训练；
2. 审计当前最好 PPO 与 teacher 的天气响应性差异；
3. 量化水氮在 DAP1–30、31–60、61–90、91+ 的分配；
4. 检查灌溉是否随季节降雨、未来 7 天降雨、土壤水分胁迫变化；
5. 输出中文实验记录、表格和图；
6. 不要改 DSSAT 输入，不要改 reward，不要启动长训练。
```

## 注意事项

- 不要把旧 originIC 结果和当前 lowIC 主线混在一起比较。
- 不要把 DQN 阴性结果说成“DQN 理论上不可能成功”，只能说“当前设置下没有形成稳定正向策略”。
- 不要事后硬挑 checkpoint；如果要 checkpoint guardrail，必须先预注册规则。
- 不要提交大型模型文件、DSSAT runtime 文件、rendered_inputs 或缓存文件到 GitHub。

