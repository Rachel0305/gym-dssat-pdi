# 022_03 SY2014 固定网格经验预填充 DQN seed1 单变量验证

## 1. 背景

022_01 在独立来源的官方固定 DAP 阶段空间中完成 48 组确定性方案，并找到 28 个主科学成功、18 个严格资源成功方案。022_02 使用相同阶段结构进行纯在线 60 季训练，四个 checkpoint 均学到约 11202 kg/ha 的高产策略，但使用 N300、I75–105，WP_ET=2.24–2.25，0/4 通过联合门槛，判定 C 分支。

022_02 同时证明：问题不是该阶段空间不存在好方案，而是纯在线 60 季经验没有让 Q 网络稳定识别 I60/N200 等更高资源效率方案。

## 2. 唯一科学变量

相对 022_02，唯一改动是：

> 在在线训练开始前，用 022_01 已预注册并执行的全部 48 个固定阶段候选重新回放，采集 48×6=288 条真实 DSSAT 阶段 transition，并预填充 uniform replay。

其余全部冻结：

- SY2014、IC=2、PDI/DSSAT 4.8.0；
- DAP 1/30/50/65/85/110；
- 9 动作、I120/N300、DAP110 禁氮；
- 021_24 固定 25 维标准化器；
- terminal-complete Monte Carlo return，整体除以 1000；
- 25→64→64→9、Adam 1e-4、SmoothL1；
- uniform replay capacity=2000、batch=32；
- 第 5 个在线完整季节后学习，每在线季 6 次更新；
- seed1、60 个在线季、epsilon 1.0→0.2；
- checkpoint 15/30/45/60；
- 判据完全复用 022_02。

不增加 offline pretraining update，不改变 learning-start，不做 PER、demo loss、行为克隆、target network、1-step/5-step TD，也不改变动作 mask。这样可以把结果差异只归因于 replay 初始经验内容。

## 3. 为什么不是“把答案硬塞进去”

- 使用 022_01 的全部 48 个候选，不只使用成功方案；
- 数据同时包含高产低投入、高产高投入和未达标方案；
- 标签来自每个方案真实 DSSAT 完整季节回报，不是人工动作分类标签；
- 网络仍需从状态、动作和回报中学习价值排序；
- 48 组网格来自官方固定阶段与预注册水氮组合，不按 022_02 结果事后挑选。

方法应称为：`stage-based DQN-style Q network with fixed-grid model-generated replay prefill and terminal-complete Monte Carlo targets`。

## 4. 数据集构建门槛

训练前必须完成并通过：

1. 022_01 candidate summary 恰有 48 个 `fixed_official_stage_grid` 场景；
2. 每个场景 schedule 可无歧义映射为 6 个动作索引；
3. 每次回放阶段 DAP 严格为 1/30/50/65/85/110；
4. 每个场景 HWAM 与 022_01 保存值误差≤2 kg/ha；
5. I/N 总量与 022_01 保存值完全一致；
6. 每个状态是有限的 25 维标准化向量；
7. 共得到 288 条 transition；
8. 数据至少同时包含主判据成功和失败场景；
9. replay 预填充后大小严格为 288。

任一失败即停止，不训练。

## 5. 训练与评估

严格复刻 022_02 的 seed1 训练循环。在线 season1 开始前 replay 已有 288 条固定网格经验；前 4 个在线季仍只采集、不更新；从在线 season5 开始每季 6 次 uniform replay 更新。

保存：

- 固定网格数据集 NPZ/CSV/JSON；
- 每场景回放一致性表；
- 在线训练季轨迹；
- update loss/gradient/Q 日志；
- 15/30/45/60 checkpoint 与确定性评估；
- 阶段动作和逐日表；
- PNG/SVG；
- 中文实验记录。

## 6. 预注册判定

沿用 022_02：

- **A 初步成功**：≥3/4 checkpoint 通过主判据，season60 通过，且≥1 checkpoint 通过严格判据；允许补一个独立 seed。
- **B 有信号但不稳定**：≥1 checkpoint 通过主判据但不满足 A；停止自动扩展，报告轨迹。
- **C 失败**：0/4 通过；停止本分支，不现场增加预训练更新或挑选成功样本。
- **D 实现失败**：数据集门槛、回放、总量、Summary 或数值检查失败；停止科学解释。

不得因接近阈值而事后放宽 WP_ET/PFP_N/产量门槛。
