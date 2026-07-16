# 023_00 SY2014 阶段独立 Q-network ensemble 结构审计记录

## 1. 背景

022_20 的统一共享 Q 网络在三 seed 中均能学习 DAP65 的 3 个因果排序，但 DAP110 只能保持 5–6/8 个因果排序，最终 0/3 seed 达标。该结果证明当前共享网络训练方案未成功，但没有证明 DQN 在其他结构下不可能成功。

023_00 只检查一种针对性结构修正是否在工程上成立：六个固定生育阶段是否能够使用参数完全独立的小型 Q 网络，并在更新一个阶段时保证其他阶段完全不变。

## 2. 预注册结构

- 固定阶段：DAP 1、30、50、65、85、110；
- 每阶段网络：`25 -> 64 -> 64 -> 9`；
- 每阶段参数量：6,409；
- ensemble 总参数量：38,454；
- 六阶段不共享 trunk、head、参数或 optimizer state；
- 初始权重分别复制自 022_08 seed0/1/2 的原单网络 checkpoint；
- 动作合法性继续使用 022 阶段框架的固定 mask；
- 正式 DQN 训练 0 步、DSSAT 调用 0 次、在线交互 0 次。

## 3. 执行过程

### 3.1 第一次启动失败

第一次使用本机 Python 启动时，在 PyTorch/Matplotlib 导入阶段遇到重复 Intel OpenMP runtime：

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

未采用 `KMP_DUPLICATE_LIB_OK=TRUE` 不安全绕过。失败尝试原样保留在：

```text
benchmark_results/023_00_failed_attempt_1_local_openmp_conflict/
```

随后使用项目既有 Docker 容器 `b2fd6726c8c1` 和 `/opt/gym_dssat_pdi/bin/python` 原样执行，未修改科学设计。

### 3.2 正式审计内容

对 seed0/1/2 分别检查：

1. 六阶段参数存储地址是否独立；
2. 拆分前后初始 Q 值是否逐位相同；
3. dispatcher 与直接调用阶段网络是否逐位相同；
4. DAP65 单次单元更新是否只改变 DAP65；
5. DAP110 单次单元更新是否只改变 DAP110；
6. 序列化/反序列化后参数哈希和 Q 值是否逐位相同；
7. 训练集与测试集是否覆盖全部六阶段。

六次 Adam 更新仅用于结构单元测试，更新后的模型没有保存，也不构成 DQN 训练证据。

## 4. 结果

### 4.1 总判定

**A_structure_isolated：3/3 seed 全部通过。**

| Seed | 参数存储独立 | 初始Q差异 | dispatcher差异 | DAP65隔离 | DAP110隔离 | round-trip | 通过 |
|---:|---|---:|---:|---|---|---|---|
| 0 | 是 | 0 | 0 | 是 | 是 | 逐位一致 | 是 |
| 1 | 是 | 0 | 0 | 是 | 是 | 逐位一致 | 是 |
| 2 | 是 | 0 | 0 | 是 | 是 | 逐位一致 | 是 |

### 4.2 单阶段更新隔离

| Seed | 更新阶段 | 目标阶段最大Q变化 | 非目标阶段最大Q变化 |
|---:|---:|---:|---:|
| 0 | 65 | 0.030575 | 0 |
| 0 | 110 | 0.029252 | 0 |
| 1 | 65 | 0.029718 | 0 |
| 1 | 110 | 0.021235 | 0 |
| 2 | 65 | 0.031462 | 0 |
| 2 | 110 | 0.020018 | 0 |

这证明跨阶段参数耦合在该结构中被工程性消除：更新目标阶段时，其他五阶段的参数哈希和固定观测 Q 值均逐位不变。

### 4.3 数据覆盖

训练集每阶段 36 条 transition，测试集每阶段 12 条 transition；全部六阶段均有覆盖。DAP110 继续受固定动作 mask 约束，其合法动作为 0、1、2。

## 5. 可以与不能得出的结论

可以确认：

- 阶段独立 Q-network ensemble 实现正确；
- 从原单网络复制后的初始预测没有发生漂移；
- DAP65 与 DAP110 的参数更新可以严格隔离；
- 工程上允许另立 023_01 固定离线训练实验。

不能确认：

- 不能声称 DAP65/DAP110 在正式训练后一定同时正确；
- 不能声称已获得优于 expert 与 auto 的管理策略；
- 不能声称 022_20 的失败根因已经被完全证明；
- 不能把六次单元更新当成 DQN 训练或农学结果。

## 6. 下一步边界

允许编写 023_01 预注册任务，但不在本任务自动启动。023_01 应固定网络结构、训练数据、MC target、两组因果标签、学习率、训练步数和三 seed 判据，不扫描网络大小或 loss 权重。至少 2/3 seed 离线通过后，才允许进入唯一一次 DSSAT 确定性在线验证。

## 7. 输出文件

- `prompts/023_00_sy2014_stage_separated_q_ensemble_structure_audit.md`
- `src/stage_separated_q_ensemble_023.py`
- `src/audit_sy2014_stage_separated_q_ensemble_023_00.py`
- `benchmark_results/023_00/023_00_architecture_manifest.csv`
- `benchmark_results/023_00/023_00_stage_data_coverage.csv`
- `benchmark_results/023_00/023_00_seed_structure_checks.csv`
- `benchmark_results/023_00/023_00_stage_isolation_checks.csv`
- `benchmark_results/023_00/023_00_result.json`
- `benchmark_results/023_00/023_00_stage_isolation_audit.png`
- `benchmark_results/023_00/023_00_stage_isolation_audit.svg`

## 8. Methods Source

- 网络基础结构：`src/run_sy2014_stage_mc_dqn_seed1_short_022_02.py`；
- 阶段动作与 mask：`src/stage_based_dqn_core_022.py`；
- 起点 checkpoint 与固定拆分：`benchmark_results/022_08/`；
- 双因果标签来源：`benchmark_results/022_19/`；
- 022 共享网络最终阴性对照：`benchmark_results/022_20/`。

当前未执行 Git commit 或 push。
