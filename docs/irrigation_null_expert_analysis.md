# 五站点灌溉 Null/Expert 结果异常的检查结论

本次检查按另一台设备的口径处理数据文件：如果 `my_data` 中有同名 `(1)` 文件，则优先使用 `(1)` 版本；没有 `(1)` 的站点土壤文件保持原名。现有结果显示，Null 灌溉在五个站点上 reward 和产量经常高于固定 expert 灌溉，这更像是“该年雨水和初始土壤水足够，固定灌溉没有边际收益”的结果，而不是 PPO 一定学错。

## 主要代码原因

1. `evaluate_hl(1).py` 在 irrigation 模式下用 `obs_dict = dict(zip(obs_keys, observation))` 记录 `swfac/nstres`，但该模式的 observation vector 不一定包含这两个变量，所以海伦站 CSV 中的 `swfac/nstres` 实际是 `NaN`。后续如果用 pandas 求和，`NaN` 会显示成 0，容易误读成“swfac 全部为 0”。

2. `evaluate_sy(1).py`、`evaluate_lc(1).py`、`evaluate_fq(1).py` 当前只设置了 `n_episodes=1`，而且 `while not done:` 的缩进位置异常，容易导致只保存一个 episode 或保存逻辑跟 episode 循环脱节。因此这三个站点现在只能作为快速排查，不适合作为正式统计。

3. `totir` 是 DSSAT 输出中的累计灌溉量，不能把每天的 `totir` 再累加。正式比较灌水量时应优先使用每日真实动作 `real_action_amir` 的和，或使用 episode 结束时最后一天的 `totir`。

4. 部分旧脚本把 `action_amir` 当成灌水量保存，但它可能仍是归一化动作。正式分析应保存两列：`raw_action_amir` 和 `real_action_amir`，其中 `real_action_amir` 来自 wrapper 反归一化或 `env.history['action'][-1]['amir']`。

5. `ExpertAgent` 的 irrigation policy 是固定 `{49: 10, 70: 10, 95: 10}`，没有区分站点、年份和降雨过程。若该年降雨充足，这个 expert 会变成“额外加水”的策略，自然可能降低 reward。

## 主要数据原因

`(1)` WTH 文件的年降雨和生长季降雨都偏充足：

| 站点 | WTH 文件 | 年降雨 mm | DOY 120-300 降雨 mm |
| --- | --- | ---: | ---: |
| HL | `CNHL0701(1).WTH` | 428.0 | 373.6 |
| SY | `CNSY1201(1).WTH` | 909.7 | 753.3 |
| YC | `CNYC0801(1).WTH` | 580.3 | 521.9 |
| LC | `CNLC0801(1).WTH` | 558.8 | 502.4 |
| FQ | `CNFQ0701(1).WTH` | 593.2 | 484.6 |

在这种天气下，Null 不灌水仍然可能接近水分最优；如果 expert 继续固定灌溉，增产不足以抵消水量惩罚，甚至可能因过湿、淋洗或生长过程扰动导致产量下降。

## 对 all 模式的改良方向

1. all 模式正式评估时必须从原始 history 或 `last_obs_dict` 记录 `swfac/nstres/trnu/totir/rain/runoff/ep`，不要只依赖 observation vector。

2. reward 中保留基础灌水成本，并新增“无水分胁迫时灌水额外惩罚”。当 `swfac <= threshold` 时，如果 agent 仍然灌水，应额外扣分，鼓励策略只在水分胁迫出现或即将出现时灌水。

3. 多站点训练/评估不要硬编码海伦路径，应通过 `--site HL/SY/YC/LC/FQ` 和 `--prefer-suffix "(1)"` 自动选择输入文件。新文件 `train_hl_all_multisite.py` 与 `evaluate_hl_all_multisite.py` 已按这个原则准备，不覆盖原脚本。

4. 后续如果要写进论文的“历史天气作为完美预报”，建议把未来 3/7/14 天降雨、温度、辐射统计作为 observation 扩展，而不是直接让 agent 读取整年天气。第一阶段可先用历史 WTH 生成完美预报特征，第二阶段再替换为真实天气预报。
