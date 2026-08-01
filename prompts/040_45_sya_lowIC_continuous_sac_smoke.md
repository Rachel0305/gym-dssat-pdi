# 040_45 SYA lowIC 连续动作 SAC smoke

## 背景

040_40 MaskablePPO 的综合指标较好，但 040_43/040_44 指向一个潜在问题：灌溉策略可能接近固定日程，对不同年份天气响应不足。

SAC（Soft Actor-Critic）是连续动作空间常用的 off-policy actor-critic 算法。已有农业 RL 文献将 SAC 用于氮肥管理或灌溉调度。它不天然支持 action mask，但适合直接输出连续水氮剂量。因此本任务只做小规模候选算法 smoke：在相同 lowIC 输入、相同训练/验证年份、相同主要安全约束下，将算法从 MaskablePPO 换成连续动作 SAC，观察它是否比 PPO 产生更有年际差异的措施，以及指标是否接近或优于 PPO。

## 关键边界

- 本任务不是正式替代 PPO。
- 本任务不是 MaskableSAC；SAC 原生输出连续动作，随后由同一安全层裁剪/执行。
- 本任务只在 SYA lowIC 上做。
- 不做参数扫描。
- 默认训练 200,000 timesteps，checkpoint 为 50k/100k/150k/200k。

## 年份划分

- 训练年份：SYA 2005–2013
- 验证年份：SYA 2014–2023

## 输入条件

- `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 运行前必须输出 input_root 和年份划分，确认没有回到 originIC。

## 动作空间

SAC 输出连续动作：

- 灌溉：0–45 mm/day
- 施氮：0–120 kg/ha/day

执行前通过安全层约束：

- 单季灌溉上限：240 mm
- 单季施氮上限：250 kg/ha
- 灌溉最小间隔：7 天
- 施氮最小间隔：7 天
- 灌溉允许 DAP：1–120
- 施氮允许 DAP：1–90
- 分阶段灌溉上限：DAP1–30 不超过75 mm，DAP1–60 不超过150 mm
- DAP90 前累计灌溉不超过195 mm，保证 DAP91+ 至少保留45 mm

## reward

保持 040_40 的主要 reward 逻辑：

- 终季产量收益：`yield_coef * final_grnwt`
- 灌溉成本：`water_cost * irrigation`
- 施氮成本：`nitrogen_cost * nitrogen`
- 胁迫缓解奖励：沿用 032/040 系列 stress relief 项
- 水分胁迫过程惩罚：沿用 040_28 SWFAC guardrail
- 终季保产 guardrail：沿用 040_40 的 `0.98 * same-year official expert yield`
- reward_scale：0.001

## 评估

对每个 checkpoint，在 2014–2023 上确定性评估，输出：

- 每年产量、灌溉总量、施氮总量、PFP_N、水分胁迫、氮胁迫；
- 每年动作序列；
- 与 PPO 040_40 checkpoint100k 的逐年差值；
- by-checkpoint 平均指标；
- 灌溉序列唯一模式数和施氮序列唯一模式数。

## 通过/停止解释

- 若 SAC 指标接近 PPO 且动作序列明显更具年际差异，则值得后续正式立项。
- 若 SAC 指标显著差于 PPO，或仍学成固定模板，则说明问题不只是 PPO 算法本身，后续应优先考虑天气预报输入、reward 结构或状态表示，而不是盲目换算法。

