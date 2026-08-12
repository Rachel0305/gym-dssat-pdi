# 2026-08-10 天气预报分支执行审计与 LC 2K 闸门

## 结论

- 当前仓库中存在可读的 `056_00`（SYA）与 `057_00`（LCA/LC）工程化天气预报实现；它们不是仅有设计文档。
- 本次**未运行 LC 2K smoke**，因此 `057_00` 不允许进入 100K。原因不是算法结果，而是当前可用运行环境不完整：项目记录指定的 WSL Python `/opt/gym_dssat_pdi/bin/python` 不存在；当前默认 WSL 仅显示 `docker-desktop`，且没有可执行的 `bash`。为避免在 Windows Python 或错误容器中造成 DSSAT 输入/依赖/输出来源混乱，本次停止在静态审计。
- 即使恢复正确运行环境，现有 `057` smoke 的自动 gate 也尚不足以证明“策略使用了未来天气”：它检查日期窗口、观测列、动作网格和 DAP1 以后的动作，但没有实现 forecast-swap/扰动反事实测试或 shuffled-forecast 负对照。因此在补齐这两项检验前，也不得把通过现有 2K gate 解释为天气预报有效。

## 三类天气相关实验必须分开

| 分支 | 实际信息结构 | 是否是天气预报 | 当前可作的结论 |
| --- | --- | --- | --- |
| `046_09` SYA raw forecast observation | observation 附加当天降雨、过去 7 天降雨及未来 7 天降雨/温度（历史 WTH 的 perfect hindcast）；4 动作网格 | 是，但为原始特征版本 | 100K 实现了预报列，但 10 个验证年均为 DAP1 `I45/N80`，唯一动作签名，不能证明天气响应。 |
| `056_00` / `057_00` engineered forecast | 在原始观测后附加 11 个归一化摘要：未来 3/7/14 天降雨、未来温度/热日/辐射、干旱/强降雨/淋失标志等；未来窗口从当日+1开始；16 动作网格 | 是，且是尚未运行的工程化 perfect-hindcast 实现 | 实现存在但没有结果目录，也没有 `forecast_calendar_audit.csv` 产物。 |
| `061_00` / `061_01` LC weather augmentation | 用早/中期降雨乘数构建伪年份以扩增训练天气世界；验证仍用原始年份；配置明确 `weather_forecast_enabled: false` | 否，是训练期 weather/domain randomization | 增加了表面动作签名多样性，但不能作为“使用预报”的证据；`061_01` 总施氮仍固定 240 kg/ha，不能替代冻结的 `053_03` LC 结果。 |

## 已核对的 057 实现与输入来源

- 配置：`configs/057_00_lca_lowIC_forecast_engineered_maskableppo.json`。
- 入口：`src/run_057_lca_lowIC_forecast_engineered_maskableppo.py`；公共实现：`src/forecast_engineered_observation_056_057.py`。
- 已声明 LC lowIC 输入根：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`；静态检查确认 `LC/CNLC0801.MZX` 存在。
- 训练/验证年份保持为 2005--2013 / 2014--2023，seed=0，动作网格为灌溉 `[0,15,30,45]` mm × 施氮 `[0,40,80,120]` kg/ha。
- 实现的 `raw_forecast_features_for_date()` 以 `date + 1` 为所有 future 窗口起点；`forecast_calendar_audit()` 会输出该起点，故静态代码层面未发现当前日混入 future window。
- 计划的 smoke 是 2,000 steps，checkpoint 为 1K/2K；正式 100K 入口还会要求 smoke 结果存在且其已有 gate 通过。

## 当前 smoke 结果与阻断点

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 现有 057 smoke/output | 不存在 | `benchmark_results/057_00_lca_lowIC_forecast_engineered_maskableppo_smoke2k` 不存在。 |
| 正确 Python 运行时 | 不可用 | 尝试执行 `/opt/gym_dssat_pdi/bin/python ... --dry-run` 返回 `No such file or directory`；默认 WSL 没有项目 Linux 发行版。 |
| 输入 MZX 静态存在 | 通过 | `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC/CNLC0801.MZX` 存在。 |
| future window 排除当日 | 静态通过 | 实现为 `[date+1, date+3/7/14]`。 |
| 动作合法网格、动作传输、daily forecast 列 | 已有 smoke gate 会检查 | 尚未实际运行，不能报告为通过。 |
| 交换/扰动 forecast 后动作改变 | 未实现/未检验 | 现有代码没有 forecast-swap 或 shuffled-forecast 评价程序。 |

## 恢复执行后的最小顺序（只限 LC 2K）

1. 在已确认的项目 DSSAT 容器/WSL 环境中运行 `src/run_057_lca_lowIC_forecast_engineered_maskableppo.py --dry-run`，确认 Python、`sb3_contrib`、DSSAT、lowIC input root 和 renderer root 一致。
2. 仅运行 `--smoke`（2K，seed0），不得改为 100K；检查生成的 manifest、`057_00_forecast_calendar_audit.csv`、observation smoke、daily CSV 和 action/forecast audit。
3. 对相同已训练 checkpoint 做两组反事实评估：保持当前 DSSAT 状态与合法动作 mask 不变，只交换两个年份同 DAP 的未来预报向量；再对未来预报向量做跨日/跨年 shuffled 负对照。记录动作改变率、改变的动作是否仍在声明网格，及是否发生在 DAP1 以后。
4. 只有下列全部成立，才允许申请 100K：无当前日/未来泄漏；特征列和输入来源可审计；动作传输且在网格内；动作不坍缩为 DAP1 模板；forecast-swap 出现可复现的合理动作差异；shuffled forecast 不产生同等或更强的“效果”。

## 论文成功标准

工程化 forecast PPO 只能在与冻结 no-forecast LC PPO 完全一致的站点、年份、seed、动作网格、奖励和安全规则下比较。候选 checkpoint 必须同时满足：

- yield、`WP_ET`、`PFP_N` 相对冻结 no-forecast PPO 非劣（按配置中的容差或预先登记的等价规则）；
- 反事实 forecast-swap 能改变至少一部分非 DAP1 决策，且变化方向与降雨/干旱风险在农学上相容；
- shuffled forecast 负对照不能复制同样的性能或动作敏感性；
- 报告全部 checkpoint 与选择规则，不能仅挑选事后表现好的 checkpoint。

在这些条件前，论文中只能称 `046_09` 为“未成功的原始预报特征消融”，`061_00/061_01` 为“天气扩增机制实验”，不能声称已解决天气预报决策问题。
