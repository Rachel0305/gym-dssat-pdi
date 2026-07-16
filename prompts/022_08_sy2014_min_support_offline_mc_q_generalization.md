# 022_08 SY2014 最小样本支持离线 MC-Q 泛化测试

## 1. 依据

022_07 在任何训练前判为 D：DAP85/action5 在全部 48 个场景中只有1条，无法同时覆盖互斥的训练集和测试集。因此 022_07 没有产生 DQN 学习结论。

本任务在不查看 target 好坏的前提下，采用可识别性规则修正划分。

## 2. 最小样本支持规则

- 阶段-动作总样本数≥2：进入主要场景留出泛化评价；
- 总样本数=1：标记为 `not_estimable_singleton`，不进入主要排序指标；
- singleton 所属完整场景强制放入训练侧，使该动作不会在测试侧形成完全未见外推；
- singleton 仍保留在训练数据和输出清单中，不静默删除；
- 规则只依赖样本频数，不读取产量、target 或成功标签。

## 3. 场景划分

- 36 个完整场景训练、12 个完整场景测试；
- 同一场景不得跨集合；
- 对所有可估计阶段-动作组合，训练集和测试集均至少出现1次；
- 两集合均包含成功和失败场景；
- 使用固定 split seed=20260716，通过布尔覆盖矩阵搜索，不使用 target 数值；
- 先完成划分可行性检查，失败则零训练停止。

## 4. 冻结离线学习设置

- 数据：022_03 的 288 条固定网格 transition；
- 网络：25→64→64→9、ReLU；
- recorded action 的 scaled terminal-complete MC target；
- SmoothL1、Adam 1e-4、full batch；
- 每个 seed 固定2000次更新；seed 0/1/2；
- 同初始化未训练网络作为对照；
- 不调用 DSSAT、不做在线交互、不加入 PER/demo/TD/target network/reward修改。

## 5. 评价与判据

测试集记录：recorded-action MAE/RMSE/Spearman、相对训练均值常数基线的MAE改善、DAP1首选动作、六阶段可估计动作聚合Spearman。DAP30以后仍标记为历史状态混杂的描述性指标。

- **A 可行**：至少2/3 seed 同时满足 Spearman≥0.60、MAE优于常数、DAP1首选一致、阶段聚合Spearman中位数≥0.50；允许另立短在线 warm-start smoke。
- **B 部分可行/不稳定**：至少一个seed全部通过，或多数seed误差改善但排序条件未稳定通过；只报告，不自动在线训练。
- **C 不泛化**：所有seed均未全部通过，且多数seed误差或排序不足；停止分支。
- **D 实现失败**：划分、覆盖、数值或更新检查失败。

不得现场改变最小支持阈值、split、更新次数、seed或判据。

## 6. 输出

- singleton清单、场景split、覆盖表；
- 三seed对照/训练指标、loss、阶段动作Q-target表；
- checkpoint、PNG/SVG、结果JSON；
- 中文记录。
