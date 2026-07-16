# 021_16 SY2014 施氮时点确定性反事实对照

## 目的

在不训练 DQN、不修改 reward、IC、观测空间、动作空间或 DSSAT 输入的前提下，检验 021_15 观察到的产量差异是否主要来自施氮时点。

## 固定条件

- 站点年份：SY2014，使用 021_14 已固化的 IC=2 输入。
- 总灌溉量：120 mm。
- 总施氮量：300 kg/ha。
- 使用 PDI/DSSAT 4.8.0 原始环境做确定性前向模拟。
- 不经过预算 wrapper，直接回放已经实际执行过的管理事件，避免请求动作裁剪或动作混叠。
- 所有来源事件必须从 `benchmark_results/021_15/021_15_management_events_with_phenology.csv` 自动读取，不手工改写数值。

## 2×2 对照

从 10K checkpoint 提取：

1. `unscaled_021_10` 的灌溉时序（I-early）与施氮时序（N-early）。
2. `scaled_0p1_021_14` 的灌溉时序（I-late）与施氮时序（N-late）。

运行四个组合：

- I-early + N-early：原高产时序复现。
- I-early + N-late：固定高产灌溉，只交换为晚施氮（核心反事实）。
- I-late + N-early：固定低产灌溉，只交换为早施氮（核心反事实）。
- I-late + N-late：原低产时序复现。

## 执行纪律

1. 先只运行 I-early + N-early 作为 smoke/fidelity test。
2. 必须核对实际执行总量为 I120/N300，并核对产量是否接近原 checkpoint 的 11046 kg/ha。
3. fidelity 通过后才运行另外三组。
4. 保存每组日值 CSV、动作 CSV、DSSAT 原始输出快照、汇总 CSV 和 JSON。
5. 记录任何失败、偏差和解释边界；不把两次观察性 checkpoint 对比冒充因果证据。

## 判定

- 若在同一灌溉时序下，仅将 N-early 换为 N-late 就造成显著减产，且反向交换能恢复产量，则支持“施氮时点是主要原因”。
- 若交叉组合变化很小，则不能把原产量差异主要归因于氮时点，需要继续检查其他状态或执行差异。
- 本实验只验证 DSSAT 对时点的确定性响应，不证明 DQN 训练不稳定的唯一根因，也不直接批准观测空间修改。

## 输出

- `benchmark_results/021_16/`
- `docs/2026-07-15_021_16_sy2014_nitrogen_timing_deterministic_counterfactual.md`

