# 021_03 SY2014 IC=2 统一 DQN seed0 验证记录

## 1. 目的

在 021_02 已确认 SY2014 IC=2 输入和 null/recorded/DSSAT-auto 前向链路后，使用五站点冻结统一 DQN 配置，先进行 5K smoke，再进行正式 50K seed0 训练。该实验只回答“SY2014 IC=2 是否能产生值得继续多 seed 验证的 DQN 候选”，不回答跨 seed 稳定性。

## 2. 冻结设置

- 算法：Stable-Baselines3 DQN，`n_steps=5`；
- 动作：I `[0,15,30]` × N `[0,50,100]`，共 9 动作；
- 预算：I≤120 mm，N≤300 kg N/ha；
- 共享最小操作间隔：7 DAP；
- 决策窗口：DAP 1–120；
- reward：`max(0, HWAM_DQN - HWAM_null) - 1×I - 5×N`；
- null baseline：5408 kg/ha；
- 未修改 IC、reward、动作、预算、天气、土壤、品种和 DSSAT 输入；未加入氮淋洗惩罚。

## 3. 分级执行

### 3.1 dry-run

- 只展开 SY2014 seed0 一个 case；
- 输入 SHA-256 与 021_02 一致：`20b071bc49549cbf561be3aae81caa355aa0582564db524e17d68ab6a274418a`；
- 输出目录不存在，未覆盖旧结果。

### 3.2 5K smoke

| 项目 | 结果 |
|---|---:|
| HWAM | 11143 kg/ha |
| CWAM | 20032 kg/ha |
| 灌溉 | 120 mm |
| 施氮 | 300 kg N/ha |
| reward | 4115.43 |

runtime audit 全部通过：9 动作、I/N 预算、单次上限、7 DAP 间隔、动作到 MgmtEvent 传输、产量/生物量有限值均正常。PDI checkpoint 快照中的 treatment 2 仍为完整 IC=2。

### 3.3 正式 50K seed0

| checkpoint | HWAM | CWAM | I | N | reward |
|---:|---:|---:|---:|---:|---:|
| 5K | 11143 | 20032 | 120 | 300 | 4115.43 |
| **10K** | **11176** | **20029** | **120** | **300** | **4147.95** |
| 15K | 5536 | 12072 | 0 | 200 | -871.94 |
| 20K | 5408 | 10603 | 0 | 0 | 0.10 |
| 25K | 5408 | 10603 | 0 | 0 | 0.10 |
| 30K | 5408 | 10603 | 0 | 0 | 0.10 |
| 35K | 5408 | 10603 | 0 | 0 | 0.10 |
| 40K | 5411 | 10607 | 15 | 0 | -11.67 |
| 45K | 5408 | 10603 | 0 | 0 | 0.10 |
| 50K | 5514 | 10709 | 45 | 0 | 61.01 |

冻结的 checkpoint 选择规则为“最大 reward；并列时选择更早 checkpoint”，因此选择 10K，而不是最终 50K。

## 4. 最佳策略动作

10K checkpoint 的有效操作集中在 DAP 52–101：

| DAP | 灌溉 (mm) | 施氮 (kg N/ha) | 累积灌溉 | 累积施氮 |
|---:|---:|---:|---:|---:|
| 52 | 30 | 50 | 30 | 50 |
| 59 | 15 | 100 | 45 | 150 |
| 66 | 0 | 50 | 45 | 200 |
| 73 | 15 | 100 | 60 | 300 |
| 80 | 15 | 0 | 75 | 300 |
| 87 | 15 | 0 | 90 | 300 |
| 94 | 15 | 0 | 105 | 300 |
| 101 | 15 | 0 | 120 | 300 |

它不是播种后立即打满，而是在 DAP 52 后分阶段用完预算；但季节总量仍达到 I120/N300 上限。

## 5. 与非 DQN 基准比较

| 情景 | HWAM | CWAM | I | N | 同公式 reward | IWP | PFP-N |
|---|---:|---:|---:|---:|---:|---:|---:|
| null | 5408 | 10603 | 0 | 0 | 0 | NA | NA |
| recorded | 9613 | 18194 | 0 | 293 | 2740 | NA | 32.81 |
| DSSAT auto | 5498 | 10688 | 66 | 0 | 24 | 8.33 | NA |
| 官方推广 expert | 11077 | 19522 | 266.1 | 300 | 3902.9 | 4.16 | 36.92 |
| DQN 10K | 11176 | 20029 | 120 | 300 | 4148 | 9.31 | 37.25 |

IWP 按 `HWAM/(10×灌溉mm)` 计算，因为 1 mm·ha = 10 m³。相对 recorded，DQN 增产 1563 kg/ha（16.3%），多用 120 mm 水和 7 kg N/ha，PFP-N 提高约 13.5%。相对 auto，DQN 增产 5678 kg/ha，IWP 提高约 11.8%，但多用 54 mm 水和 300 kg N/ha。相对官方推广 expert，DQN 增产 99 kg/ha、少灌 146.1 mm、施氮相同，IWP 从 4.16 提高到 9.31 kg/m³，PFP-N 从 36.92 提高到 37.25 kg/kg。

因此，seed0 相对官方推广 expert 已满足“产量不低、灌溉更少、施氮不高、效率更高”；但它没有比 recorded 少用水，也没有比 auto 少用氮，不能写成对所有基准全面胜出。SY 当前仍只能进入多 seed 验证。

## 6. 训练后期塌缩

同一 seed 在 15K 后从高产满预算策略退化为接近 null/no-op 的策略，50K 终点也没有恢复到 10K 水平。由于所有 checkpoint runtime audit 均通过，这不是输入、预算或通信失败，而是当前 DQN 学习轨迹的非单调性。该现象说明：

1. 不能以最后一步模型代替最佳 checkpoint；
2. 10K 候选必须由 seed1/2 复核；
3. 若其他 seed 同样“早期高产、后期塌缩”，后续应研究训练稳定性或早停协议，而不是继续无限增加步数。

## 7. 安全结论

- **已证实**：冻结统一框架可在 SY2014 IC=2 上训练出明显高于 null、recorded 和 auto 的 seed0 高产候选；该候选相对官方推广 expert 同时达到产量略高、显著少灌水、施氮相同和水氮效率更高。
- **未证实**：跨 seed 稳定、绝对节水节氮全面胜出、跨年份或跨站点泛化。
- **下一步**：保持完全相同的配置，分别运行 seed1、seed2；不得根据 seed0 结果修改 reward 或 IC。

## 8. 输出

- Prompt：`prompts/021_03_sy2014_ic2_dqn_seed0_validation.md`
- Smoke config：`configs/experiments/021_03_sy2014_ic2_dqn_smoke.yaml`
- Formal config：`configs/experiments/021_03_sy2014_ic2_dqn_seed0_50k.yaml`
- 汇总脚本：`src/finalize_sy2014_ic2_dqn_021_03.py`
- Checkpoint 汇总：`benchmark_results/021_03/021_03_sy2014_checkpoint_summary.csv`
- 基准比较：`benchmark_results/021_03/021_03_sy2014_baseline_comparison.csv`
- 轨迹图：`benchmark_results/021_03/021_03_sy2014_checkpoint_trajectory.png`
- 正式训练目录：`benchmark_results/021_03/021_03_sy2014_ic2_dqn_seed0_50k__sy_2014_seed0/`
