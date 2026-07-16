# 022_03 SY2014 固定网格经验预填充 DQN seed1 记录

## 结论

预注册判定为 **C_failed**。48 个固定阶段候选生成的 288 条真实 DSSAT transition 全部通过回放审计，但预填充 uniform replay 后，四个 checkpoint 仍为高产高投入策略，0/4 通过联合门槛。

## 数据集验证

- 48 个唯一固定网格场景；28 个主判据成功、20 个失败，18 个严格资源成功。
- 共 288 条有限 25 维阶段 transition，每场景 6 条。
- 最大 HWAM 回放误差 0.485 kg/ha。
- 灌溉和施氮总量最大误差均为 0。
- 使用全部好坏方案，没有只选成功候选，也没有行为克隆标签。

## checkpoint 结果

| season | HWAM | I | N | WP_ET | PFP_N | 主判据 | 动作序列 |
|---:|---:|---:|---:|---:|---:|---|---|
| 15 | 11201.76 | 75 | 300 | 2.25 | 37.3 | False | [7,7,7,7,7,0] |
| 30 | 11201.76 | 90 | 300 | 2.24 | 37.3 | False | [7,7,7,7,7,1] |
| 45 | 11201.76 | 90 | 300 | 2.24 | 37.3 | False | [7,7,7,7,7,1] |
| 60 | 11201.76 | 90 | 300 | 2.24 | 37.3 | False | [7,7,7,7,7,1] |

与 022_02 相比，固定网格预填充只把后期灌溉从 I105 降到 I90，没有把 N300 降到 N200，不能声称成功。

## 新诊断线索：未覆盖动作外推

按阶段统计 288 条经验后发现：

- DAP1 数据只覆盖动作 0/1/3/4，但四个 checkpoint 均选择未覆盖动作7（I15/N100）。
- DAP1 已覆盖动作中，动作3（I0/N50）的平均完整回报最高（6.056），且 16/16 对应场景通过主判据。
- 网络选择动作7不能解释为“从固定网格经验学到动作7最好”，而是对未覆盖动作头的无约束 Q 外推。
- 固定网格各阶段支持集分别为：DAP1 `{0,1,3,4}`；DAP30 `{4,5,7,8}`；DAP50 `{4,5,7,8}`；DAP65 `{1,2,4,5,7,8}`；DAP85 `{1,2,4,5}`；DAP110 `{0,1}`。

因此下一步应先对现有 checkpoint 做零训练的行为支持约束评估，而不是增加步数或修改 reward。该约束来自全部 48 个好坏候选的实际覆盖，不根据成功标签筛选。

## 输出

- `prompts/022_03_sy2014_fixed_grid_replay_prefill_dqn_seed1.md`
- `src/run_sy2014_fixed_grid_replay_prefill_dqn_seed1_022_03.py`
- `benchmark_results/022_03/022_03_dataset_validation.json`
- `benchmark_results/022_03/022_03_scenario_replay_validation.csv`
- `benchmark_results/022_03/022_03_fixed_grid_transition_manifest.csv`
- `benchmark_results/022_03/022_03_fixed_grid_replay_dataset.npz`
- `benchmark_results/022_03/022_03_checkpoint_evaluation_summary.csv`
- `benchmark_results/022_03/022_03_checkpoint_stage_actions.csv`
- `benchmark_results/022_03/022_03_checkpoint_daily_values.csv`
- `benchmark_results/022_03/022_03_training_seasons.csv`
- `benchmark_results/022_03/022_03_training_stage_actions.csv`
- `benchmark_results/022_03/022_03_update_log.csv`
- `benchmark_results/022_03/022_03_training_diagnostics.png/.svg`
- `benchmark_results/022_03/checkpoints/*.pt`
- `benchmark_results/022_03/022_03_result.json`

未执行 Git commit 或 push。
