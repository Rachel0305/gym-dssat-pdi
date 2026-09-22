# YC weather audit plan

日期：2026-09-22

## 研究路线调整

当前阶段只处理 YC/YCA。目标不是马上大规模训练，而是先确认天气管线、随机性和输入文件边界，为后续“历史天气 vs 增强天气”的公平对照做准备。

## 为什么先做 YC

- YC/YCA 已有 `055_00` 低初始条件 PPO 迁移链路。
- 当前任务只改变天气训练域，不改变 PPO 算法、reward、网络结构、动作空间或安全约束。
- LC、SY 已冻结，本任务不触碰其数据、配置、脚本或结果。

## 审计问题

1. 当前 YC PPO 训练到底读取哪些天气文件。
2. `random_weather` / WGEN 是否启用。
3. episode 间天气是否变化。
4. PPO seed 和天气抽样 seed 是否独立。
5. YC experiment、weather、soil、cultivar、climate 参数的路径。
6. 当前 gym-DSSAT 是否支持 `random_weather` 与 `.CLI` 传入。
7. 下一阶段如何做最小改动的公平对照。

## 当前审计结论

- `055_00` 当前 `random_weather=false`。
- episode 变化来自 2004-2013 训练年份历史 `.WTH` 抽样。
- 当前 YC lowIC 输入目录无 `.CLI`。
- 当前没有证据显示默认 Florida / UF 气候参数污染了 `055_00`。
- `ppo_seed` 与训练年抽样 seed 当前未分离，均为 0。

## 下一阶段实验结构

`baseline_weather`：

- 沿用当前历史天气机制。
- 保持 `random_weather=false`。
- 训练 episode 抽取 2004-2013 真实历史年份。

`augmented_weather`：

- 仅改变训练天气来源。
- 天气增强参数只由 YC 训练期历史天气估计。
- 记录每个 episode 的完整天气情景来源。

两组共同冻结：

- PPO algorithm
- reward function
- action space
- observation space
- training steps
- hyperparameters
- evaluation years
- evaluation seeds

## 独立测试原则

- train、validation、test 必须按完整生长季划分。
- 增强天气参数只能由 train 估计。
- validation / test 真实天气不得参与天气生成器参数拟合。

## 成功标准记录结构

后续至少报告每颗 seed 的：

- yield
- irrigation
- fertilizer
- WUE 或 `WP_ET`
- NUE 或 `PFP_N`
- constraint violations
- reward
- 管理时序摘要
- 成功 seed 数 / 总 seed 数

`WP_ET` 和 NUE 类指标只在 replay 证据齐全时报告，不能从不完整日值表推断。
