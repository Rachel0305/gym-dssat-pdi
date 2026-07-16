# 022_05 SY2014 行为支持约束固定网格 replay DQN seed1

## 1. 依据

022_03 将 48 组固定网格的 288 条好坏经验预填充 replay 后，DQN 仍选择未被数据覆盖的 DAP1 动作7并得到 0/4 联合成功。022_04 对同一 checkpoint 做零训练支持约束评估，发现 24 个阶段状态中有 8 次原始 argmax 越出行为支持；加入支持 mask 后 season15 得到 11199 kg/ha、I90/N300、WP_ET=2.29、PFP_N=37.3，并通过主判据，但仅 1/4 checkpoint 通过。

本任务检验：若从训练第一步就阻止未覆盖动作外推，能否把单点 post-hoc 成功转化为稳定训练轨迹。

## 2. 唯一科学变量

相对 022_03，唯一变化是：在在线训练的随机探索、贪心动作选择以及 checkpoint 确定性评估中，统一使用由全部 48 个固定网格候选自动推导的阶段行为支持 mask。

支持集冻结为：

- DAP1：0/1/3/4
- DAP30：4/5/7/8
- DAP50：4/5/7/8
- DAP65：1/2/4/5/7/8
- DAP85：1/2/4/5
- DAP110：0/1

支持集必须从 `benchmark_results/022_03/022_03_fixed_grid_transition_manifest.csv` 自动重算并与上述值一致；不得手写覆盖运行值，不得按成功标签筛选。

## 3. 其余冻结设置

- SY2014、IC=2、PDI/DSSAT 4.8.0；
- 021_24 固定 25 维标准化器；
- terminal-complete Monte Carlo return，统一除以1000；
- 48个场景、288条 transition 全部预填 uniform replay；
- 不增加 offline pretraining update；
- 25→64→64→9、ReLU、Adam 1e-4、SmoothL1；
- replay capacity=2000、batch=32、第5在线季开始学习、每季6次更新；
- seed1、60在线季、epsilon 1.0→0.2；
- checkpoint 15/30/45/60；
- 原 I120/N300 预算与裁剪逻辑保持不变；本轮不同时引入 budget-exact mask；
- 不用 demo、DQfD、PER、target network、1-step/5-step TD。

## 4. 训练前门槛

1. 022_03 数据集验证仍为 passed；
2. replay 预填严格为288；
3. 六个阶段支持集与预注册一致；
4. 每个支持集都来自成功和失败场景；
5. 所有支持动作均满足原始阶段合法性；
6. 随机探索和 greedy 单元测试均不能返回支持集外动作；
7. DAP110 仍不能施氮。

任一失败停止训练。

## 5. 判定

- **A 初步成功**：≥3/4 checkpoint 通过主判据、season60通过且≥1个严格成功；允许补独立seed。
- **B 有信号但不稳定**：≥1个主判据成功但不满足A；停止自动扩展。
- **C 失败**：0/4主判据成功；停止支持约束训练分支。
- **D 实现失败**：支持集、动作、数据或指标检查失败。

不因接近门槛而修改支持集、WP_ET、PFP_N或训练季数。

## 6. 输出

- 支持集与训练前检查 JSON/CSV；
- 训练季、阶段动作、update日志；
- 四个checkpoint、评估汇总、逐日值；
- 与022_03对比图PNG/SVG；
- 中文记录。
