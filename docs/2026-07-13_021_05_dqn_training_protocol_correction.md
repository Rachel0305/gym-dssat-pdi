# 021_05 DQN 探索率时间轴修复与 SY2014 seed0 最小验证

## 状态

- 训练协议修复：**completed**
- checkpoint/resume 保留：**completed（纯 SB3 保存/加载单元测试通过）**
- SY2014 seed0 修复版 5K smoke：**completed**
- SY2014 seed0 修复版 50K 单变量对照：**completed**
- 修复后训练稳定性：**failed（20K 起仍发生策略坍缩）**
- 五站点正式结论：**provisional，等待修复协议复核**

## 背景与问题

此前正式 DQN 训练按每 5K 步分段调用：

```python
model.learn(total_timesteps=5000, reset_num_timesteps=False)
```

计划总训练量虽然是 50K，`exploration_fraction=0.35`，但每次 `learn()` 只看到本段 5K 的时间轴，探索率约在 1.75K 步内便降到最低值 0.05，而不是按全局 17.5K 步衰减。直接读取保存模型确认 HLA、YC、FQ、LC、SY 的 5K checkpoint 均为 `exploration_rate=0.05`，因此该问题影响五站点旧正式 DQN 结果。

该缺陷只影响 DQN 训练协议，不影响 null、recorded、official expert、DSSAT auto、确定性 DSSAT 前向扫描及输入溯源结论。

## SB3 机制核查

SB3 在 `reset_num_timesteps=False` 时，会把当前 `model.num_timesteps` 加到本次 `learn(total_timesteps=...)` 的参数上，形成内部 `_total_timesteps`。因此：

- 旧方案：每段传 5K，探索率在每段的短时间轴上过早衰减；
- 错误替代：每段都传 50K，第二段内部目标会漂移到 55K；
- 正确方案：每段传 `planned_total - model.num_timesteps`，并由 callback 在下一个绝对 checkpoint 停止；内部目标始终为计划总量 50K。

## 实现

修改 `benchmark/train_runner.py`：

1. 新增绝对步数停止 callback；
2. 每段将剩余全局训练步数传给 SB3；
3. 保持 `reset_num_timesteps=False`；
4. 新增 `run_until_timesteps`，允许 5K smoke 使用 50K 探索时间轴；
5. 每个 checkpoint 记录探索率和 SB3 内部总时间轴；
6. 加入模型步数与 checkpoint 一致性检查。

未修改 IC、DSSAT 输入、reward、动作空间、预算、网络、学习率、target update 或其他超参数。

## 单元测试

纯 SB3 CartPole 单元测试未调用 DSSAT，结果如下：

| 方法 | 100 步 epsilon | 200 步 epsilon | 200 步内部目标 |
|---|---:|---:|---:|
| 旧增量分段 | 0.0500 | 0.0500 | 200 |
| 每段错误传完整总量 | 0.7313 | 0.5090 | 1100 |
| 剩余全局步数 | 0.7313 | 0.4599 | 1000 |
| 一次连续训练 | — | 0.4599 | 1000 |
| 保存加载后续跑 | 0.7313 | 0.4599 | 1000 |

“剩余全局步数”与“一次连续训练”在 200 步的探索率一致，且保存、加载、续跑后仍一致，说明修复不需要放弃断点续跑。

## SY2014 IC=2 seed0 smoke

- 全局计划：50K；执行到：5K；
- 5K epsilon：0.728626（旧方案为 0.05）；
- SB3 内部目标：50000；
- runtime audit：全部通过；
- 输入仍为已确认的 SY2014 treatment 2、IC=2；
- 无 OOM、无旧结果覆盖。

smoke 的产量不用于科学优越性结论。

## 修复前后 50K 单变量对照

两次实验均为 SY2014、IC=2、seed0，仅训练协议不同。

| checkpoint | 旧协议产量 | 修复协议产量 | 修复 epsilon | 修复灌溉 | 修复施氮 |
|---:|---:|---:|---:|---:|---:|
| 5K | 11143 | 11093 | 0.729 | 120 | 300 |
| 10K | 11176 | 11131 | 0.457 | 120 | 300 |
| 15K | 5536 | 10746 | 0.186 | 120 | 300 |
| 20K | 5408 | 5711 | 0.050 | 120 | 0 |
| 50K | 5514 | 5732 | 0.050 | 120 | 0 |

修复使高产策略从旧协议的约 10K 延续到 15K，并改变了坍缩形态：旧协议快速趋向 I0/N0，修复版保留 I120，但施氮从 N300 降为 N0。说明探索率缺陷确实影响训练轨迹，但不是后期坍缩的唯一原因。

## 独立确定性复评

修复版 10K checkpoint 在两个独立输出目录中重复确定性评估，汇总值和逐日行完全一致：

- 产量：11131 kg/ha；
- 生物量：20064 kg/ha；
- 灌溉：120 mm；
- 施氮：300 kg/ha；
- reward：4103.077881。

因此 10K 高产点不是评估脚本的随机误差，但它仍只是训练过程中出现的候选，不等同于稳定收敛策略。

## Q 网络与 replay buffer 诊断

| checkpoint | epsilon | action0 抽样占比 | online Q 最大值 | target Q 最大值 |
|---:|---:|---:|---:|---:|
| 5K | 0.729 | 13.5% | 973 | 899 |
| 10K | 0.457 | 12.1% | 1027 | 898 |
| 15K | 0.186 | 8.7% | 1241 | 1025 |
| 20K | 0.050 | 7.9% | 1839 | 1022 |
| 25K | 0.050 | 18.1% | 3710 | 1963 |

修复后，早期 replay buffer 不再快速被 no-op 主导；但 15K–25K online Q 与 target Q 的差距扩大，同时策略在 20K 丢失全部施氮动作。该证据支持继续检查 Q/TD 稳定性、target network 更新和 reward 数值尺度，但尚不能把其中任何一项写成已确认根因。

## 五站点影响范围

| 范围 | 判定 | 处理 |
|---|---|---|
| HLA | 旧 5K 模型 epsilon=0.05 | 旧 DQN 正式结论暂定，需最小复核 |
| YC | 同上 | 产量/迁移及系数不敏感结论暂定 |
| FQ | 同上 | 产量/迁移及系数不敏感结论暂定 |
| LC | 三个 seed 均命中 | 90 mm 资源差异需修复协议复核 |
| SY | 三个 seed 均命中 | 021_03/021_04 仅作修复前对照 |
| 非 DQN 基线与前向模拟 | 不调用 `learn()` | 不受该缺陷影响 |

## 当前结论

1. Claude 对“分段训练可能破坏探索率时间轴”的质疑成立，并已被保存模型和 SB3 单元测试确认。
2. 断点续跑不需要放弃；采用“剩余全局步数 + 绝对 checkpoint callback”即可保持正确时间轴。
3. 修复确实延迟并改变了 SY2014 的策略坍缩，但没有消除坍缩。
4. 修复版 10K 高产候选可重复评估，但不能作为稳定收敛或正式优越性证据。
5. 当前所有五站点旧 DQN 科学结论必须标记为 provisional；不删除历史结果，它们保留为修复前对照。

## 下一步边界

不立即重跑五站点，也不同时修改多个超参数。建议新建 021_06，先对 SY2014 seed0 做低成本、单变量的 Q/TD 稳定性诊断：

1. 保存训练 loss、TD target、online/target Q 和动作分布；
2. 优先判断统一 reward 数值缩放是否只改善数值稳定性而不改变目标相对权重；
3. 若需要训练对照，只运行一个 seed 的 smoke/短训练；
4. 找到能消除坍缩的证据后，依次复核 SY 多 seed、LC，再考虑 HLA/YC/FQ。

## 输出文件

- `prompts/021_05_dqn_training_protocol_correction.md`
- `benchmark/train_runner.py`
- `src/verify_segmented_exploration_schedule_021_05.py`
- `src/reevaluate_sy2014_protocol_fix_021_05.py`
- `src/finalize_dqn_training_protocol_correction_021_05.py`
- `configs/experiments/021_05_sy2014_ic2_dqn_protocol_fix_smoke.yaml`
- `configs/experiments/021_05_sy2014_ic2_dqn_protocol_fix_seed0_50k.yaml`
- `benchmark_results/021_05/021_05_segmented_exploration_unit_test.csv`
- `benchmark_results/021_05/021_05_old_vs_corrected_checkpoint_summary.csv`
- `benchmark_results/021_05/021_05_old_vs_corrected_protocol.png`
- `benchmark_results/021_05/021_05_corrected_q_replay_diagnostics.csv`
- `benchmark_results/021_05/021_05_training_protocol_impact_scope.csv`
- `configs/final_dqn_candidate.yaml`（状态改为 provisional；原文件已备份）

## Git

本节在提交后补充 commit hash。按项目规则，未经用户明确要求不执行 push。
