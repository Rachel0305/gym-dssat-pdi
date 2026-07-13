# 021_04 SY2014 IC=2 统一 DQN 多 seed 稳定性复核

## 1. 背景与问题

021_02 已确认 SY2014 treatment 2 使用完整 IC=2 初始条件；021_03 在冻结统一 DQN 配置下得到 seed0 的高产候选。021_04 只改变随机种子，检验该候选能否在 seed1、seed2 复现。没有修改 IC、reward、动作空间、预算、DSSAT 输入或 checkpoint 选择规则。

## 2. 冻结配置

- 输入：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX`
- 输入 SHA-256：`20b071bc49549cbf561be3aae81caa355aa0582564db524e17d68ab6a274418a`
- treatment 2，IC=2，6 层初始水氮剖面
- DQN，`n_steps=5`
- 离散动作：I `[0, 15, 30]` × N `[0, 50, 100]`，共 9 个动作
- 季节预算：I≤120 mm，N≤300 kg/ha
- 操作间隔≥7 DAP，决策窗口 DAP 1–120
- reward：`max(0, HWAM_DQN - HWAM_null) - I - 5N`
- 训练：50K，每 5K 保存并确定性评估 checkpoint
- 选择：每个 seed 取 reward 最大 checkpoint；并列时取较早者

## 3. 分级执行与运行审计

1. dry-run 只展开 SY2014 seed1、seed2 两个 case，未复用旧结果。
2. seed1、seed2 的 5K smoke 顺序执行，均正常完成。
3. smoke 的动作空间、预算、单次上限、7 DAP 间隔、MgmtEvent 传递、产量和生物量有限值检查全部通过。
4. 随后顺序完成 seed1、seed2 的 50K，不并行，未发生 OOM。
5. 新增 20 个正式 checkpoint 的 runtime audit 全部通过；加上 021_03 seed0，共汇总 30 个 checkpoint。

## 4. 每个 seed 的最佳 checkpoint

| seed | 最佳步数 | HWAM (kg/ha) | CWAM (kg/ha) | I (mm) | N (kg/ha) | reward | 相对官方 expert 产量 | 严格达标 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 10K | 11176 | 20029 | 120 | 300 | 4147.95 | +99 | 是 |
| 1 | 5K | 11097 | 20039 | 120 | 300 | 4069.16 | +20 | 是 |
| 2 | 5K | 11068 | 19549 | 120 | 300 | 4040.36 | -9 | 否 |

严格达标定义：HWAM≥11077 kg/ha、I≤266.1 mm、N≤300 kg/ha，且 runtime audit 通过。

三个 seed 最佳产量均值为 11113.7 kg/ha，标准差 55.9 kg/ha，CV=0.503%；资源动作完全一致，均为 I120/N300。按任务书的严格阈值，2/3 seed 超过官方 expert，seed2 仅低 9 kg/ha。因此结论是“跨 seed 高度一致的高产候选、2/3 严格达标”，不能写成“3/3 全面超过”。

## 5. 与基准的统一比较

| 情景 | HWAM (kg/ha) | CWAM (kg/ha) | I (mm) | N (kg/ha) | 同公式 reward | IWP (kg/m³) | PFP-N (kg/kg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| null | 5408 | 10603 | 0 | 0 | 0.0 | — | — |
| recorded | 9613 | 18194 | 0 | 293 | 2740.0 | — | 32.809 |
| DSSAT auto | 5498 | 10688 | 66 | 0 | 24.0 | 8.330 | — |
| 官方推广 expert | 11077 | 19522 | 266.1 | 300 | 3902.9 | 4.163 | 36.923 |
| DQN seed0 best | 11176 | 20029 | 120 | 300 | 4148.0 | 9.313 | 37.253 |
| DQN seed1 best | 11097 | 20039 | 120 | 300 | 4069.0 | 9.248 | 36.990 |
| DQN seed2 best | 11068 | 19549 | 120 | 300 | 4040.0 | 9.223 | 36.893 |

相对官方 expert，三个 DQN 均少灌 146.1 mm、施氮相同，IWP 明显更高，PFP-N 接近；产量差异为 +99、+20、-9 kg/ha。该证据支持“稳定节水且产量基本持平”的表述，不支持“所有 seed 均严格增产”。

## 6. 训练轨迹的关键限制

三个 seed 的最佳 checkpoint 都出现在 5K–10K。之后策略均明显退化：

- seed0 从 15K 起转向低产策略，20K–45K 多次接近 null；
- seed1 在 10K 已降至 9041 kg/ha，15K 后多次接近 null；
- seed2 在 10K 已降至 5522 kg/ha，15K 后基本维持 null-like 行为。

因此，长训练并非单调改善。当前协议必须保留独立确定性评估和 best-checkpoint 选择；不能把 50K 最终模型直接当作最终策略，也不能隐去后期坍缩。

## 7. 结论与下一步

### 已解决

- SY2014 权威工作输入已固化为 IC=2，输入哈希可追溯。
- 冻结统一 DQN 配置可在三个 seed 产生高度接近的最佳产量和完全一致的季节资源总量。
- 2/3 seed 严格超过官方 expert；第三个 seed 只低 9 kg/ha。

### 尚未解决

- 尚未实现 3/3 seed 严格超过官方 expert。
- 三个 seed 都存在训练后期策略坍缩；稳定的是“最佳 checkpoint 候选”，不是 50K 终点策略。
- 当前仅验证 SY2014，不能外推到 SY 其他年份或其他站点。

### 建议

将 SY2014 标为“跨 seed 高度一致的高产节水候选”，保留现有 reward/IC/动作框架，不继续做无目的系数扫描。下一步应先诊断早期高产策略为何在 10K–15K 后坍缩，再决定是否改变训练稳定性设置；任何改变都必须作为新实验线，不能覆盖本次冻结结果。

## 8. 输出文件

- prompt：`prompts/021_04_sy2014_ic2_dqn_multiseed_validation.md`
- smoke config：`configs/experiments/021_04_sy2014_ic2_dqn_seed12_smoke.yaml`
- formal config：`configs/experiments/021_04_sy2014_ic2_dqn_seed12_50k.yaml`
- 汇总代码：`src/finalize_sy2014_ic2_dqn_multiseed_021_04.py`
- 30-checkpoint 表：`benchmark_results/021_04/021_04_sy2014_multiseed_checkpoint_summary.csv`
- 最佳 checkpoint 表：`benchmark_results/021_04/021_04_sy2014_multiseed_best_checkpoint_summary.csv`
- 基准比较表：`benchmark_results/021_04/021_04_sy2014_multiseed_baseline_comparison.csv`
- 轨迹图：`benchmark_results/021_04/021_04_sy2014_multiseed_checkpoint_trajectory.png`

模型、replay buffer 和 PDI 临时快照保留在本地运行目录，不提交 Git。
