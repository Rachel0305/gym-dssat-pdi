# 021_04 SY2014 IC=2 统一 DQN seed1/2 稳定性复核

## 目的

021_03 在冻结统一配置下得到 SY2014 seed0 的 10K 高产候选：HWAM 11176 kg/ha、I120/N300，高于官方推广 expert 11077 kg/ha，同时少灌 146.1 mm、施氮相同。021_04 只改变随机种子，检验这一结果能否在 seed1、seed2 复现。

## 严格冻结项

- SY2014 treatment 2 / IC=2，源 MZX SHA-256：`20b071bc49549cbf561be3aae81caa355aa0582564db524e17d68ab6a274418a`；
- DQN、`n_steps=5`；
- 9 个离散动作：I `[0,15,30]` × N `[0,50,100]`；
- I≤120 mm、N≤300 kg N/ha；
- 共享最小操作间隔 7 DAP，决策窗口 DAP 1–120；
- reward：`max(0, HWAM_DQN-HWAM_null)-1×I-5×N`；
- 训练 50K，每 5K 保存和确定性评估 checkpoint；
- 最大 reward checkpoint 为该 seed 候选，并列时选择较早 checkpoint。

禁止修改 IC、reward、动作、预算、DSSAT 输入、checkpoint 选择规则或训练步数。

## 分级执行

1. 对 smoke config dry-run，必须只展开 SY2014 seed1、seed2 两个 case。
2. seed1、seed2 各训练 5K；Benchmark Runner 顺序执行，禁止并行。
3. 两个 smoke 均须通过 runtime audit、预算、7 DAP、MgmtEvent 和有限值检查。
4. smoke 全部通过后，使用独立 formal config 顺序训练 seed1、seed2 各 50K。
5. 保存全部 checkpoint 结果，但不把模型、replay buffer、PDI 临时文件提交 Git。

## 稳定性判据

正式基准：

- null：5408 kg/ha，I0/N0；
- recorded：9613 kg/ha，I0/N293；
- DSSAT auto：5498 kg/ha，I66/N0；
- 官方推广 expert：11077 kg/ha，I266.1/N300。

### 强复现

seed1、seed2 的最佳 checkpoint 均满足：

- HWAM ≥ 11077 kg/ha；
- I ≤ 266.1 mm；
- N ≤ 300 kg/ha；
- runtime audit 通过。

### 部分复现

只有一个新增 seed 达到上述要求，或者两个 seed 均明显高于 null/recorded，但未同时超过官方 expert。

### 未复现

两个新增 seed 最佳 checkpoint 均未形成明显高于 null 的可解释策略，或存在运行/输入异常。

即使强复现，也应报告 checkpoint 随训练步数的非单调性，不得隐藏后期塌缩。

## 输出

- `benchmark_results/021_04/` 下 smoke 与 formal 运行目录；
- 三 seed 最佳 checkpoint 汇总；
- 三 seed 10 个 checkpoint 轨迹 CSV；
- 产量、reward、水、氮的跨 seed 图；
- 与四个非 DQN 基准的比较表；
- 中文实验记录和 Git 本地提交；未经用户确认不 push。

