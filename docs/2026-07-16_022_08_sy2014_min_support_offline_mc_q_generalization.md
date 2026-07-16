# 022_08 SY2014 最小样本支持离线 MC-Q 泛化测试

## 1. 背景

022_07 因 DAP85/action5 只有1条样本，无法满足训练集和测试集双侧覆盖要求，在零梯度更新时停止。022_08 预注册与 target 数值无关的最小支持规则：总样本数≥2才进入主要留出评价；singleton 明确标记且保留在训练侧。

## 2. 数据与划分

- 数据：022_03 的48个完整场景、288条阶段 transition；
- 完整场景隔离：训练36个场景（216条）、测试12个场景（72条）；
- split seed：20260716；布尔覆盖搜索第74次找到可行解；
- 训练和测试均含成功、失败场景；
- 所有可估计阶段-动作组合在两侧均至少出现1次；
- 唯一 singleton：DAP85/action5（I30/N50），来源场景 `W120_critical__N200_spread`，只进入训练侧并从主要排序指标排除；
- 划分算法不读取 target 或产量好坏。

## 3. 冻结设置

- 25→64→64→9、ReLU；
- recorded action 的 scaled terminal-complete Monte Carlo target；
- SmoothL1、Adam 1e-4、full batch；
- seed0/1/2，每个固定2000次更新；
- 同初始化未训练网络作对照；
- DSSAT调用0、在线交互0；未加入PER、demo、TD、target network或reward修改。

## 4. 测试集结果

| seed | recorded-action Spearman | MAE | 常数基线MAE | MAE改善 | DAP1首选一致 | 阶段聚合Spearman中位数 | 全部判据 |
|---:|---:|---:|---:|---:|---|---:|---|
| 0 | 0.800 | 0.363 | 0.668 | 0.305 | 是 | 0.529 | 通过 |
| 1 | 0.800 | 0.377 | 0.668 | 0.291 | 是 | 0.557 | 通过 |
| 2 | 0.802 | 0.372 | 0.668 | 0.296 | 是 | 0.471 | 未全部通过 |

seed2 只因阶段聚合 Spearman 中位数0.471低于预注册0.50而未全部通过；其测试 recorded-action Spearman、MAE改善和 DAP1 排序均通过。

## 5. 分支判定

结果为 **A：离线排序学习可行**，因为2/3 seed满足全部预注册条件。

可以支持的结论：

1. 当前网络并非完全无法学习固定网格 MC 价值关系；
2. 在严格场景留出条件下，三 seed 的 recorded-action 排序相关均约0.80，并显著优于常数MAE基线；
3. DAP1 同状态动作首选在三 seed 中均恢复为 action4；
4. 因而 022_05 的在线不稳定更可能发生在“离线学到的排序如何在在线交互中保持”这一环，而不是基本函数拟合能力完全不足。

不能支持的结论：

- 不能说 DQN 已经得到全面优于 expert/auto 的策略；
- DAP30以后阶段聚合排序受历史状态混杂，不能写成动作因果效应；
- DAP85/action5 无独立测试证据，必须保留 `not_estimable_singleton` 标记；
- 不能据此直接扩展站点、年份或长训练。

## 6. 下一步

按预注册规则，允许另立一次短在线 warm-start smoke：

- 优先采用 seed0 或 seed1 的离线 checkpoint；
- 在线实验只改变初始化权重，其他设置与022_05冻结一致；
- 必须保留最小支持 mask和singleton限制；
- 训练前先做冻结 checkpoint 的确定性DSSAT评估，确认其初始策略不是异常动作；
- 短 smoke 需同时比较 warm-start 与同seed随机初始化 control；
- 若离线优势进入在线后迅速消失，应停止并定位在线分布漂移，而不是现场加步数。

## 7. 输出

- `prompts/022_08_sy2014_min_support_offline_mc_q_generalization.md`
- `src/run_sy2014_min_support_offline_mc_q_generalization_022_08.py`
- `benchmark_results/022_08/022_08_stage_action_identifiability.csv`
- `benchmark_results/022_08/022_08_scenario_split.csv`
- `benchmark_results/022_08/022_08_identifiable_stage_action_split_coverage.csv`
- `benchmark_results/022_08/022_08_seed_test_metrics.csv`
- `benchmark_results/022_08/022_08_seed_stage_action_q_target.csv`
- `benchmark_results/022_08/022_08_training_loss.csv`
- `benchmark_results/022_08/checkpoints/offline_mc_q_seed0.pt`、`seed1.pt`、`seed2.pt`
- `benchmark_results/022_08/022_08_offline_mc_q_generalization.png`
- `benchmark_results/022_08/022_08_offline_mc_q_generalization.svg`
- `benchmark_results/022_08/022_08_result.json`
