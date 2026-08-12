# 2026-08-10 LC engineered forecast 2K 执行与反事实闸门

## 执行结论

`057_00_lca_lowIC_forecast_engineered_maskableppo` 已在项目容器 `nifty_taussig` 内完成单进程、seed0 的 2K smoke（约 117 秒）。结果写入独立目录 `benchmark_results/057_00_lca_lowIC_forecast_engineered_maskableppo_smoke2k`，没有改写或替换冻结的 `053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000`。

工程自动 smoke gate 通过，但**不批准进入 100K**。原因有二：

1. 2K checkpoint 的验证均值产量与 `PFP_N` 均低于冻结 LC 参考，且该 smoke 没有输出可比较的 `WP_ET`；
2. 代码没有 forecast-swap 或 shuffled-forecast 反事实接口，尚未证明动作变化是由未来天气信息造成，而非状态、DAP 或训练随机性造成。

## 实际运行环境与命令

容器中确认：`/opt/gym_dssat_pdi/bin/python` 存在；`sb3_contrib` 可导入；项目目录为 `/workspaces/gym-dssat-pdi`；lowIC MZX `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC/CNLC0801.MZX` 存在；DSSAT 根为 `/opt/dssat_pdi`。

renderer 模块的默认根为 originIC，但 `057` 运行时会将 `ppo_safe_rendering.MULTISITE_INPUT_ROOT` 临时改为配置的 lowIC 根，并在结束时恢复；dry-run 的 input provenance 与该配置一致。

实际命令：

```text
docker exec -w /workspaces/gym-dssat-pdi nifty_taussig \
  /opt/gym_dssat_pdi/bin/python \
  src/run_057_lca_lowIC_forecast_engineered_maskableppo.py --dry-run

docker exec -e PYTHONPATH=/workspaces/gym-dssat-pdi/src \
  -w /workspaces/gym-dssat-pdi nifty_taussig \
  /opt/gym_dssat_pdi/bin/python \
  src/run_057_lca_lowIC_forecast_engineered_maskableppo.py --smoke
```

dry-run 通过：站点 LCA/LC、lowIC 输入根、2005--2013 训练年、2014--2023 验证年、seed0 和 16 个水氮组合均与 `057_00` 配置一致。实际 smoke 固定为 2,000 steps，checkpoint 为 1K/2K；没有运行 100K。

## 逐项闸门

| 检查项 | 状态 | 证据 |
| --- | --- | --- |
| 隔离输出、未触及冻结 LC 结果 | 通过 | 新目录为 `benchmark_results/057_00_lca_lowIC_forecast_engineered_maskableppo_smoke2k`。 |
| manifest/config provenance | 通过 | `057_00_run_manifest.json` 记录 lowIC root、LCA、年份、seed0、16 动作网格和 engineered perfect-hindcast 模式。 |
| future window 排除当前日 | 通过 | `057_00_forecast_calendar_audit.csv` 中 2014--2015 的 DAP 1/15/30/60/90 均为 `future_window_starts = current_date + 1`，`future_window_excludes_today=True`。 |
| forecast 特征可见且季内变化 | 通过 | 每个验证年有 22/22 raw+normalized forecast 列，且 22 列均非恒定。 |
| 观测维度 | 通过 | base 26 维，增强后 37 维，增加 11 个已登记 feature。 |
| 动作网格 | 通过 | 2K 的 10 个验证年中 off-grid irrigation/N 行均为 0。 |
| request 到 safe | 通过 | 2K daily 输出中 `raw_action_amir/anfer` 与 `safe_action_amir/anfer` 全行一致，safety trigger 为 0。 |
| safe 到 DSSAT 传输 | 部分通过 | daily 输出记录了安全动作与 DSSAT 日状态，训练完成且 10/10 daily 文件存在；但没有独立的 `dssat_action_*` 字段，故不能从 CSV 单独逐行证明 renderer 写入值。 |
| DAP1 后动作 | 通过 | 2K 每个验证年均有 4--6 个 DAP1 后正动作（2014--2023）。 |
| 跨年动作多样性 | 通过（仅机制） | 2K 下 10 年出现 7 个正动作签名；这不等于 forecast 因果响应。 |
| 反事实 forecast-swap | 未评估 | 当前实现没有可调用接口。 |
| shuffled-forecast 负对照 | 未评估 | 当前实现没有可调用接口。 |
| 允许 100K | **否** | 指标非劣与反事实两项均未通过。 |

`forecast_observation_date` 与 daily CSV 的 DSSAT `date` 不是严格逐行一一对应：wrapper 记录的是生成下一 observation 的日期，而 daily 行反映 step 的状态/输出。日历审计和源代码确认 future window 从 observation date+1 开始；不过若要做正式论文证据，反事实脚本应显式记录“策略输入日期、动作日期和 DSSAT 写入日期”的对应关系。

## 2K 指标（checkpoint 2K；2014--2023 平均）

| 指标 | 057 forecast smoke | 冻结 LC 参考 | 判断 |
| --- | ---: | ---: | --- |
| grain yield (kg/ha) | 8,335.04 | 9,140.6--9,140.7 | 低约 806 kg/ha，不通过非劣。 |
| total irrigation (mm) | 175.5 | 约 200 | 不单独构成成功。 |
| total N (kg/ha) | 240.0 | 240 | 持平但未提高氮效率。 |
| `PFP_N` (kg/kg) | 34.73 | 38.09 | 低约 3.36，不通过非劣。 |
| `WP_ET` | 本 smoke summary 未输出 | 2.723 | 未评估，不能声称非劣。 |

2K 是训练机制烟雾测试，不可用来选择论文 checkpoint；这里只将其用于排除明显的输入、动作网格或写入链路问题。

## 反事实缺口与最小实施方案

`src/forecast_engineered_observation_056_057.py` 目前包含 observation smoke、calendar audit 与 action/forecast-column audit，但不包含对已训练模型注入替代 forecast 向量的评价接口。因此本次没有伪造反事实结果，也没有改动 wrapper。

恢复执行应仅新增一个小型推理审计脚本，不修改训练逻辑：固定 checkpoint、year、seed、DSSAT 状态及 `action_masks()`，取得同一 decision state 的 base observation 后，分别输入原 forecast、同 DAP 跨年份交换 forecast、按 year/DAP shuffled forecast；输出动作变化率、动作值、动作 DAP、mask 合法性及 forecast 雨量/干旱风险差异。只有原 forecast 的合理响应可复现且 shuffled 对照不能复制该响应，才可把“forecast 特征存在”提升为“策略使用 forecast”的证据。

## 下一阶段结论

**禁止 100K。** 正确下一步不是扩展训练，而是先以最小、只读 checkpoint 推理审计补齐 forecast-swap 和 shuffled-forecast 负对照。此后仍需在相同站点、年份、seed、动作网格、奖励与安全规则下同时满足 yield、`WP_ET`、`PFP_N` 对冻结 no-forecast LC PPO 非劣，才可重新申请正式训练。
