# YC DQN readiness audit v2

审计日期：2026-09-11  
审计范围：只读审计；未训练、未修改现有正式 PPO/DQN 代码、未覆盖既有结果。独立产物目录为 `experiments/225_yc_dqn_readiness_audit/`。

## 结论先行

最终分类：`READY_WITH_MINIMAL_ADAPTER`

最终建议：`Proceed to matched PPO–DQN comparison: YES`，但这个 YES 仅表示进入“最小适配器实现 → smoke gate → 再决定正式训练”的流程；不表示可以直接运行当前 032/031 标准 SB3 DQN，也不表示本轮已训练 DQN。

核心原因：

1. `032_00` 和 `031_19` 使用标准 SB3 `DQN`。它们的训练探索、训练时 greedy、Bellman target 都没有使用动态 action mask；032 只有 evaluation 的手写 `masked_greedy_action`。
2. `src/mask_aware_dqn_029.py` 已经实现了所需的三处 mask 和 replay mask，并有 `029_01` 的工程 smoke 通过证据；但它默认是旧的六阶段、9-action、stage observation 契约，不能直接当作当前 YC Original daily 16-action DQN。
3. 因此最小适配是“复用 mask-aware 核心 + 接入当前 055 YC lowIC daily env/reward/safety/mapping”，不是重做算法，也不是把 032 DQN 的结果拿来比较。

## 1. 仓库中发现的 DQN 路径

### 标准 SB3 路径：不满足当前 mask 安全要求

- `src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:244-258`：实例化 `stable_baselines3.DQN`；`action_masks()` 只在 wrapper 和 evaluation 手写 greedy 中出现。
- `src/run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19.py:210-230`：同类标准 SB3 DQN；evaluation 也单独调用手写 masked greedy。
- 本地 SB3 2.7.1 的关键行为：`common/off_policy_algorithm.py:367-399` 在 warmup 使用 `action_space.sample()`；`dqn/dqn.py:233-261` 的 predict 不接动态 mask；`dqn/dqn.py:187-208` 对 target Q 在所有动作上直接 `max`。

### 自定义 mask-aware 路径：安全核心可复用，但契约不匹配

- `src/mask_aware_dqn_029.py:150-157`：epsilon 随机和 greedy 只从/选择合法动作。
- `src/mask_aware_dqn_029.py:162-183`：Bellman target 在 `next_masks` 上先屏蔽再 max，terminal 不 bootstrap。
- `src/mask_aware_dqn_029.py:47-83`：replay 同时保存当前/下一状态 mask，并拒绝非法 collection action 与非 terminal 空 next mask。
- `src/run_five_site_stage_mask_aware_dqn_029_02.py:105-143`：旧版 runner 将当前 mask、action、step 后的 next mask 放入 transition。
- `benchmark_results/029_01_sy2014_mask_aware_dqn_smoke/029_01_result.json`：60 steps、60 个合法 action、zero invalid attempts、一次 optimizer update、一次 target sync、checkpoint round-trip 等工程检查均通过。

这组通过证据证明的是旧 stage/9-action 路径的工程闭环，不是当前 YC 055 daily/16-action readiness。

## 2. 当前 YC Original PPO reference contract

当前对照基准是 `configs/055_00_yca_lowIC_expanded_action_maskableppo.json` 及其 manifest，而不是名称相似的历史目录：

| 契约项 | 当前 reference |
|---|---|
| 站点/输入 | YCA/YC，`lowIC`，YC `CNYC0801.MZX`，`MZCER048.CUL` |
| 年份 | train 2004–2013；validation 2014–2023 |
| 观测 | `046_02_raw_observation`，raw/no normalization，no weather forecast，daily state |
| 动作 | 16 个 joint actions：I `[0,15,30,45]` × N `[0,40,80,120]`，row-major |
| 日约束 | I≤45 mm/day；N≤120 kg/ha/day；I/N 最小间隔均 7 天 |
| DAP 窗口 | irrigation `[1,120]`；fertilization `[1,90]` |
| 季节/阶段安全 | I=240 season cap；DAP≤30/60/90 阶段上限 75/150/195；late reserve 45 mm |
| reward/safety | 当前 040_28 SWFAC threshold 0.05、penalty coefficient 50，并沿 040_26/040_36 reserve chain |
| 输出 | 每个站点/年份/seed/checkpoint 的 daily/output/provenance；yield、I、N、PFP_N；WP_ET/NUE 需一致的 Summary.OUT/ETCP replay |
| checkpoint | 25K/50K/75K/100K；正式 report checkpoint 100K |

032 的静态 action grid 与日窗口/最小间隔相似，但 I seasonal budget、staged reserve、SWFAC reward、YC 年份/输入和评估范围均不相同。

## 3. 四项 mask readiness gate

| gate | 032/031 标准 SB3 DQN | 029 自定义核心（旧契约） | 当前 YC 结论 |
|---|---|---|---|
| exploration 只探索合法 action | `NO` | `YES_IN_LEGACY_SCOPE` | 未通过直接复用；适配器必须接当前 YC mask |
| greedy argmax 只选合法 action | `NO`（训练时）；eval 手写路径才 mask | `YES_IN_LEGACY_SCOPE` | 未通过直接复用 |
| Bellman target 只在 next-state legal action 上 max | `NO` | `YES_IN_LEGACY_SCOPE` | 未通过直接复用 |
| replay action == 实际执行 action/reward/next_state | `NO_OR_UNSAFE`；SB3 存 buffer action，wrapper 可把非法请求 fallback 为 no-op | `YES_IN_LEGACY_SCOPE` | 必须在 YC daily adapter 中重新证明 |

032 的一份已有 evaluation daily 文件有 95 行，观测到 `mask_forced_noop=0`、mapped action difference=0、safety trigger=0；这只能说明该次 deterministic eval 没触发 fallback，不能证明标准 SB3 训练 collection 与 Bellman update 是 mask-safe。

## 4. action grid 与 mapping

当前 YC 16-action mapping 已固化到 `results/yc_dqn_action_grid_mapping.csv`：

`index = irrigation_index * 4 + nitrogen_index`

因此 index 0–3 是 I=0 的四个 N 档，index 4–7 是 I=15，index 8–11 是 I=30，index 12–15 是 I=45。mask 是 state-dependent；index 存在不等于当时合法。

029 的默认 `action_dim=9`，其 stage runner 也明确构造 `next_mask=np.zeros(9, dtype=bool)`，所以不能加载/解释为当前 16-action YC DQN。

## 5. DQN hyperparameter inventory

完整清单在 `results/yc_dqn_hyperparameter_inventory.csv`。

- 032 historical SB3 settings：learning rate `1e-5`、buffer `100000`、learning starts `1024`、batch `1024`、gamma `0.99`、train frequency `4`、gradient steps `1`、target update `120`、epsilon `1.0→0.0` over fraction `1.0`、MLP `[256,256,256]`、weight decay `0.001`。
- 029 custom core settings：Adam learning rate `3e-4`、gamma `1.0`、replay `10000`、batch `30`、MLP `32→32`、epsilon `1.0→0.05` over 70% of 240 steps、gradient clip `10`、target sync every 60 env steps、checkpoint `0/60/120/180/240`。
- 本审计不把任何一套旧超参直接宣布为 YC formal DQN 超参。适配器 smoke 前只冻结“契约和 mask 行为”，正式 DQN 的 seed/checkpoint/budget 需要在 smoke PASS 后另行登记。
- 029 `save()` 不保存 replay buffer；checkpoint evaluation round-trip 已通过，但若要求 resume-from-replay，需要单独适配。

## 6. 历史证据与 checkpoint re-audit 判断

历史 DQN 产物确实存在，完整目录索引在 `results/yc_historical_dqn_inventory.csv`。其中 `029_01/029_02/029_03` 对自定义旧版 DQN 的 mask/transition/evaluation 有可复核证据；`032/031` 对标准 SB3 DQN 有模型、daily 和 eval 输出。

但没有一份历史产物同时满足“当前 YC 055 的 lowIC、2004–2013 train、2014–2023 validation、daily raw observation、16-action、I240 staged reserve、040_28/040_36 reward/safety、当前 checkpoint/output contract”。所以历史产物可用于实现审计和回归参考，不能替代当前 YC checkpoint re-audit，也不能支持直接进入正式 matched DQN training。

## 7. 最小适配器与下一步（本轮不执行训练）

建议只做以下最小改动后重新 smoke：

1. 复用 `MaskAwareDQN` 核心，显式设置当前 YC observation dimension 与 `action_dim=16`。
2. 接入 055_00 的 YC lowIC daily env、当前 reward/safety wrapper chain、10+10 年份选择和相同 cultivar/input provenance。
3. 使用同一个 row-major action mapping 生成 `action_masks()`、`select_action()`、replay action、evaluation action 和 daily log。
4. transition 写入前断言：selected action 合法；env 实际执行 index 等于 replay action；reward/next observation/next mask 来自同一次 `env.step`；terminal next mask 不用于 bootstrap。
5. 先做小 smoke gate：至少覆盖 invalid-mask rejection、epsilon legality、greedy legality、masked Bellman target、transition closure、16-action mapping、YC wrapper 约束和 checkpoint reload；任一失败即停止，不启动长训练。
6. smoke PASS 后再制定与 PPO 相同站点/年份/seed/checkpoint/output 口径的 matched comparison；保留所有失败记录。

## 8. 对 prompt 要求的逐项回答

1. **存在哪些 DQN 实现？** 标准 SB3 DQN（032/031 等）和项目自定义 `MaskAwareDQN`（029 系列）；另有历史 replay/evaluation runner。
2. **哪些路径可复用？** 029 的 mask-aware 核心可复用；032/031 的标准 DQN 训练路径不可复用为当前 matched DQN。
3. **当前 YC PPO contract 是什么？** 见本报告第 2 节及 contract CSV：YCA/YC lowIC，2004–2013/2014–2023，raw daily observation，16 actions，I240 staged reserve，N250，7-day intervals，040_28/040_36 reward/safety chain。
4. **探索是否只选合法动作？** 032/031：否；029 旧契约：是。
5. **greedy 是否只选合法动作？** 032/031 训练时否，evaluation 手写路径是；029 旧契约是。
6. **Bellman target 是否 mask？** 032/031 否；029 使用 `next_masks`，是（旧契约）。
7. **replay action 是否等于实际执行 action？** 032/031 不能保证，标准 SB3 存 buffer action 而 wrapper 可 fallback；029 旧 runner 的 60-step smoke 通过。
8. **action grid/mapping 是否一致？** 032 静态 16-grid 的排列与当前 YC 一致；029 默认 9-action 不一致；当前 DQN 尚未绑定 YC wrapper。
9. **主要 contract mismatch 是什么？** YC 年份/输入、daily vs stage、16 vs 9、I240/staged reserve、040_28/040_36 reward/safety、checkpoint/output protocol。
10. **超参是什么？** 见 hyperparameter CSV；032 与 029 分列，未将旧超参直接宣布为 YC formal 配置。
11. **历史 evidence/checkpoint re-audit 是否足够？** 历史证据足够审计旧实现，但不足以完成当前 YC checkpoint re-audit。
12. **readiness classification？** `READY_WITH_MINIMAL_ADAPTER`。
13. **Proceed to matched PPO–DQN comparison？** `YES`，条件是最小适配器 smoke PASS；当前 032/031 标准 SB3 DQN 不能直接训练/比较，本轮未训练。

## Evidence index

- `results/yc_dqn_readiness_contract_audit.csv`
- `results/yc_dqn_action_mask_audit.csv`
- `results/yc_dqn_hyperparameter_inventory.csv`
- `results/yc_historical_dqn_inventory.csv`
- `results/yc_dqn_action_grid_mapping.csv`
- `results/yc_dqn_code_dependency_map.csv`
- `scripts/build_audit.py`
