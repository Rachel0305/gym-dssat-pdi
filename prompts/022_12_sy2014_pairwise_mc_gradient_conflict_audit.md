# 022_12 SY2014 受控pairwise目标与MC loss梯度冲突审计

## 1. 依据

022_11在三个不同水分前缀下3/3确认：DAP65 action1相对action7的因果回报优势约0.5（scaled），而022_08三个offline checkpoint仍把action7排在action1前。本任务在任何新训练前，检查加入受控pairwise目标是否与原MC回归目标发生严重梯度冲突，以及是否扰动其他阶段。

## 2. 固定pairwise形式

对三个受控前缀的同一DAP65状态：

`d_i = (G_action1,i − G_action7,i) / 1000`

`L_pair = mean SmoothL1[(Q(s_i,a1) − Q(s_i,a7)), d_i]`

不使用人工hinge margin；目标差直接来自022_10/022_11确定性DSSAT回报。

原目标：022_08训练集216条记录动作的

`L_MC = mean SmoothL1[Q(s,a_recorded), G_MC/1000]`

## 3. 固定审计权重公式

仅用于虚拟单步审计：

`lambda_norm = ||grad_shared(L_MC)|| / ||grad_shared(L_pair)||`

使pairwise与MC在共享层的梯度范数相等。lambda不扫描、不裁剪；若分母为0或lambda非有限则判实现失败。该值只作为下一任务候选依据，本任务不保存更新后的checkpoint。

## 4. 审计范围

对022_08 seed0/1/2 checkpoint分别：

- 计算全参数和共享层的梯度范数、余弦相似度；
- 计算符号相反的非零梯度比例；
- 构造一次虚拟combined step：`theta' = theta − 1e−4[g_MC + lambda_norm*g_pair]`；
- 比较step前后L_MC、L_pair；
- 在022_08留出测试状态上检查DAP1、30、50、65、85、110支持动作argmax变化数及Q绝对变化；
- 特别要求DAP1和DAP110原本较稳定阶段不能发生任何支持集argmax改变；
- 虚拟step后不保存模型，不进入在线DSSAT。

## 5. 预注册兼容判据

单seed兼容必须同时满足：

1. shared-gradient cosine≥−0.20；
2. 虚拟combined step后L_pair下降；
3. L_MC相对增加≤1%；
4. DAP1和DAP110支持集argmax改变数均为0；
5. 数值全部有限。

- **A 兼容**：至少2/3 seed兼容；允许另立一次固定lambda推导的一次性离线训练。
- **B 局部冲突**：仅1/3兼容；停止，不现场改loss或阈值。
- **C 严重冲突**：0/3兼容；停止pairwise路线。
- **D 实现失败**：状态、回报、梯度或虚拟step检查失败。

不得因为结果接近门槛而调整−0.20、1%、学习率、loss形式或lambda公式。

## 6. 输出

- 三受控状态及回报差CSV；
- seed梯度指标与阶段扰动表；
- JSON、PNG/SVG和中文记录；
- DQN训练0、DSSAT调用0。
