# 022_00 SY2014 生育阶段型 DQN：预注册设计与离线单元测试记录

## 1. 为什么结束021旧结构线

021_43–021_48已经形成以下证据：

- 原始资源成本reward会在部分seed中偏好低于expert产量的低氮策略；
- terminal feasibility bonus离线重排序正确，但在旧1K在线结构中没有改变终态；
- bonus样本并不缺少PER采样；
- 旧1K诊断中target全程冻结，5-step标准bootstrap不能逐段覆盖距终止83–110步的关键操作；
- 即使终端transition被大量抽中，value residual仍几乎没有闭合；
- 去掉1620 bonus后，终端target仍约5770，Huber局部梯度100%不变。

因此，继续在“每日动作+5-step+稀疏终端价值”上调整bonus或训练步数，边际信息价值已经很低。022线改为阶段决策，并在训练前设置明确停止条件。

## 2. 本轮做了什么

本轮只完成结构定义和离线验证：

- 新增中文预注册prompt；
- 冻结SY2014阶段DQN配置；
- 新增阶段动作、预算裁剪、action mask和terminal-complete return的纯函数；
- 运行24项离线单元测试；
- 未调用DSSAT，未训练模型。

## 3. 阶段点的独立来源

阶段点没有照抄021_18 oracle的DAP22/29/42/56/79，而是采用018_02/018_03已经登记的官方推文表1东北春玉米固定DAP：

| 推文阶段 | 原固定DAP | gym执行DAP | SY2014 DSSAT物候解释 |
|---|---:|---:|---|
| 播种/基肥 | 0 | 1 | 出苗前 |
| 小喇叭口 | 30 | 30 | 幼龄期 |
| 大喇叭口 | 50 | 50 | 花芽分化至吐丝 |
| 抽雄散粉 | 65 | 65 | 花芽分化至吐丝 |
| 灌浆初期 | 85 | 85 | 花芽分化至吐丝 |
| 乳熟末期 | 110 | 110 | 灌浆期 |

SY2014物候来自021_15：出苗9、幼龄结束34、花芽分化44、75%吐丝89、灌浆开始99、灌浆结束136、成熟139。物候数据只作解释，不用于事后移动阶段点。

oracle时点到最近官方阶段点的距离为：

| oracle DAP | 最近官方阶段距离（天） |
|---:|---:|
| 22 | 8 |
| 29 | 1 |
| 42 | 8 |
| 56 | 6 |
| 79 | 6 |

这说明两者时间尺度相近，但不证明固定官方DAP下能够复现oracle产量。

## 4. 动作和观测设计

- 动作保持9档：I∈{0,15,30}、N∈{0,50,100}；
- 季节预算保持I120/N300；
- 只有DAP1/30/50/65/85/110是决策日，其余日强制no-op；
- DAP110只允许action 0/1/2，即N=0；使用显式action mask，不把含N请求静默裁剪成同一个动作；
- 预算不足时记录请求动作并按剩余额度裁剪；
- 阶段间隔均超过7天，因此不再叠加操作间隔惩罚。

021_17已经确认实际25维观测包含精确DAP、累计I/N、`istage`和`vstage`。本轮没有重复添加这些信息，也没有把事后才能知道的未来物候日期输入agent。

## 5. 一致的Q目标

阶段成本与终止价值为：

```text
r_stage = -I_stage - 5*N_stage
r_terminal = max(0, Y_final-5408) + 1620*1[Y_final>=11077]
```

一季最多6个stage transition，使用gamma=1的完整季节return作为唯一Q目标：

```text
G_t = sum(k=t..T) r_k
loss = Huber(Q_online(s_t,a_t), G_t)
```

不同时保留1-step TD，不叠加另一个5-step loss，也不使用target-network bootstrap。因此022结构不会重复021_27发现的“完整return和1-step TD在Huber饱和区方向相反、等权抵消”。

准确方法名称应为：

> stage-based DQN-style Q network with terminal-complete Monte Carlo targets

不能把它写成未经修改的SB3标准DQN。

## 6. 离线测试结果

- 测试总数：24；
- 通过：24；
- 失败：0；
- DSSAT调用：0；
- 训练调用：0。

覆盖内容包括：

- prompt/YAML/代码中的阶段点一致；
- 阶段点严格递增且在成熟前；
- 9动作笛卡尔积正确；
- 非阶段日只能no-op；
- DAP110禁止N动作且不会产生动作混叠；
- I/N预算裁剪正确；
- oracle时点未被写入阶段配置；
- terminal-complete return手算通过：以Y11205、I75/N200为例，`G0=5797+1620-75-5×200=6342`；
- YAML明确关闭1-step、5-step并行目标和target bootstrap；
- 022_00明确关闭训练和DSSAT。

## 7. 现在还不能说什么

022_00只证明设计在代码和算术上自洽，**没有证明官方固定DAP动作空间内存在达标策略**，也没有证明DQN能够学习。

特别不能把“oracle时点离官方阶段点只有1–8天”解释成“移动后产量一定不变”。021_16已证明施氮时机非常敏感，这一可行性必须用DSSAT前向模拟验证。

## 8. 下一步强制门槛

下一步是022_01确定性固定阶段可行性搜索，不是DQN训练。

主科学判据沿用021_18原始预注册口径：

- HWAM≥11077 kg/ha；
- WP_ET不低于同输入official expert与auto的较高值；
- N>0时PFP_N不低于official expert；
- I≤120 mm、N≤300 kg/ha；
- DAP90后N=0。

I≤90/N≤250不是021_18原始门槛，而是后来021_20示范学习smoke使用的更严格工程门槛，并受到021_18已经找到I75/N200候选的启发。本轮将它保留为“严格附加等级”，但不能冒充原始预注册条件。只有主科学判据通过，才允许写022_02；严格等级用于判断候选质量。

如果不存在，停止本结构，不得为了结果事后移动阶段点。若存在，再预注册SY2014 seed1短训练；seed1通过后至少补一个独立seed，失败则停止，不做无上限调参。

## 9. 文件

- `prompts/022_00_sy2014_stage_based_dqn_preregistered_design.md`
- `configs/sy2014_stage_dqn_022_00.yaml`
- `src/stage_based_dqn_core_022.py`
- `src/test_sy2014_stage_dqn_design_022_00.py`
- `benchmark_results/022_00/022_00_unit_test_results.csv`
- `benchmark_results/022_00/022_00_stage_provenance_and_masks.csv`
- `benchmark_results/022_00/022_00_oracle_distance_audit.csv`
- `benchmark_results/022_00/022_00_terminal_complete_return_handcheck.csv`
- `benchmark_results/022_00/022_00_summary.json`

## 10. 状态

- 022_00：completed
- 结构单元测试：24/24通过
- 固定阶段科学可行性：尚未验证
- 下一步：022_01 deterministic fixed-stage feasibility search
- Git commit/push：未执行
