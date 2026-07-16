# 021_12 SY2014 target 更新幅度与五站点 reward 量级离线审计

## 目标

在不训练、不调用 DSSAT、不修改 reward 或观测空间的前提下，回答两个问题：

1. SY2014 延长探索配置下，seed0/seed1 的 target 网络更新幅度，是否与固定状态上的 Q 值变化和动作排序重组同步出现；
2. 现有统一口径产物能否支持“SY reward 数值尺度明显高于其他站点”，以及可支持的倍数范围是多少。

## 输入

- `021_09` SY2014 seed0 5K–25K checkpoint；
- `021_10` SY2014 seed1 5K–25K checkpoint；
- `021_06` 保存的 18 个固定真实观测及 9 动作映射；
- 五站点已有 `selected_checkpoint.json` / `season_summary.csv` 等标准化结果。

## 边界

- 只做离线分析，不训练、不调用 DSSAT；
- 不修改 reward、IC、动作空间、预算、探索率、target interval 或观测空间；
- 不把少量 target 更新事件做成统计因果结论；
- 不混用不同 reward 公式、不同配置或 smoke test 的数字；若无法确认统一口径，明确标记为不可直接比较；
- 不把“reward 较大”直接写成训练振荡的根因；
- 不启动 reward 除以 10 的正式训练。

## 方法

1. 对 seed0/seed1 的 10K、15K、20K、25K checkpoint 读取 online/target 参数；
2. 对相邻 checkpoint 计算参数绝对 L2 变化、相对 L2 变化；
3. 对同一组 18 个固定观测计算 online/target Q 的平均绝对变化、最大绝对变化、全局 argmax 改变数和按灌溉档位的 N0/N50/N100 完整排序改变数；
4. 将 target 更新窗口与 target 冻结窗口并排展示。由于事件数量很少，只描述，不计算或解释显著相关性；
5. 自动发现五站点标准化正式结果，排除名称含 `smoke` 的实验，记录 reward 公式/配置来源是否可确认；
6. 仅对可确认同一冻结 reward 口径的结果计算中位数、范围和 SY/其他站点倍数；无法确认的站点保留为缺失或 provisional；
7. 输出 CSV、JSON、PNG 和中文实验记录。

## 判定纪律

- target 参数变化与 Q 排序重组同时增大，只支持“同步变化”，不证明 target 更新造成策略变化；
- 如果非更新窗口也存在大量 online 排序漂移，必须保留“online 学习本身不稳定”的解释；
- reward 量级比较必须写明样本、实验配置和 provisional 状态；
- 只有离线证据支持尺度假设后，下一轮才允许设计显式 reward scale smoke test；smoke 必须验证 replay reward 比例确实为 0.1。

## 输出

- `benchmark_results/021_12/021_12_network_change_events.csv`
- `benchmark_results/021_12/021_12_fixed_state_q_change_summary.csv`
- `benchmark_results/021_12/021_12_reward_scale_evidence.csv`
- `benchmark_results/021_12/021_12_summary.json`
- `benchmark_results/021_12/021_12_target_q_reward_scale_audit.png`
- `docs/2026-07-15_021_12_sy2014_target_update_magnitude_and_reward_scale_audit.md`

