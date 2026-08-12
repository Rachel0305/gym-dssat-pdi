# 2026-08-11 LC forecast 最小反事实审计

## 结论

对已完成的 `057_00` 2K smoke checkpoint 2K 进行固定状态、固定 mask 的最小反事实推理审计后，未来天气特征的跨年交换与随机打乱均没有引起任何确定性动作改变。因此，这个 2K policy 没有提供“使用未来天气预报做决策”的证据。

**不允许进入 100K。** 本次没有重新训练，也没有把反事实动作写入 DSSAT；冻结的 `053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000` 未被修改。

## 受审计对象与来源

- 容器：`nifty_taussig`；解释器：`/opt/gym_dssat_pdi/bin/python`；`sb3_contrib` 已导入。
- 模型：`benchmark_results/057_00_lca_lowIC_forecast_engineered_maskableppo_smoke2k/models/LCA/LCA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip`。
- 年份：2014--2023；seed=0；单进程；只读模型推理。
- 输入 root：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`；MZX `LC/CNLC0801.MZX` 存在。
- 057 合约：原始 26 维 observation 后附加 11 个 engineered perfect-hindcast 特征；动作网格为灌溉 `[0,15,30,45]` mm × 施氮 `[0,40,80,120]` kg/ha。
- 运行输出：`benchmark_results/057_00_lca_lowIC_forecast_engineered_maskableppo_smoke2k_counterfactual_audit/`。

## 方法

新增的隔离 harness 为 `src/audit_057_lca_engineered_forecast_counterfactual.py`。它不改动训练 wrapper，也不调用 `learn()`：

1. 用真实 forecast 执行一次确定性 rollout，仅收集每个 pre-action state、对应 action mask、26 维 base observation 与 11 维 forecast 尾部；真实动作才会正常通过安全 wrapper 和 DSSAT。
2. 对每个收集的同一 state 与同一 mask，分别输入：真实 forecast、同 DAP 的下一验证年 forecast（可复现 swap）、及固定 RNG seed 的同 DAP 跨年 shuffled forecast。
3. 只调用 `model.predict(..., deterministic=True, action_masks=mask)`；swap/shuffle 预测不调用 `env.step()`，不写入 DSSAT，因此不把状态变化混入 forecast 因果检验。

同 DAP 配对避免了把不同物候日的特征误当作 forecast 效应。代码、状态 trace、决策 CSV 和 summary JSON 都在新的审计目录中。

## 结果

| 指标 | 结果 |
| --- | ---: |
| 固定 decision states | 938 |
| swap forecast 动作变化率 | 0.0% |
| shuffled forecast 动作变化率 | 0.0% |
| swap 的 DAP1 后动作变化数 | 0 |
| shuffled 的 DAP1 后动作变化数 | 0 |
| real/swap/shuffled 动作均在当日 mask 内 | 是 |
| swap/shuffle 动作是否执行至 DSSAT | 否（故意不执行） |

这不是“forecast 无效”的一般性结论；它是更窄但足够明确的结论：当前 LC 2K checkpoint 在这 938 个已访问决策状态上，对所附的 11 维未来天气向量没有可检测的 deterministic action sensitivity。

## 泄漏、动作链路与可比性

- `057_00_forecast_calendar_audit.csv` 已验证 future window 从 `current_date+1` 开始；本 harness 沿用相同 wrapper，未修改特征计算。
- 真实 rollout 中只有真实预测动作进入 `env.step()`；其 action mask 固定后复用于所有反事实预测。反事实动作不进入 safety/DSSAT，所以不能报告反事实 yield、`WP_ET` 或 `PFP_N`。
- 原 2K 输出已确认 raw action 到 safe action 全行一致、安全触发为 0；该事实只能支持真实 rollout 的请求到安全链路，不能替代反事实 DSSAT 写入证据。
- 反事实审计重新执行真实 forecast rollout 时的 endpoint 均值为 grain yield `9202.13`、`PFP_N=38.34`，但与原 smoke summary 的 `8335.04`、`34.73` 不一致。该重放 endpoint 不作为新结果或非劣证据；它表明 renderer/input 或评价调用的逐步 provenance 尚未完全闭环。`WP_ET` 也不可从此审计获得。

## 闸门决定

| 条件 | 状态 |
| --- | --- |
| 2K 工程运行、网格、日期窗口、特征列 | 通过（见 2026-08-10 记录） |
| future forecast 的反事实动作敏感性 | **失败：0%** |
| shuffled forecast 负对照不复制效应 | 不适用：真实 swap 效应本身为 0 |
| 与冻结 LC no-forecast 的 yield / `WP_ET` / `PFP_N` 非劣 | 未通过或未评估；原 2K yield/PFP_N 低，WP_ET 缺失 |
| replay renderer/input provenance | 未闭环（endpoint 不一致） |
| 允许 100K | **否** |

下一步不应加长训练。只有先定位重放 endpoint 差异、在同一冻结可比协议下复核输入/renderer 及 `WP_ET`，并在新的小规模机制试验中得到非零且可解释的 forecast-swap 响应后，才可重新评估是否授权正式训练。
