# 012 阶段后：水氮联合优化目标函数重构方案

## 1. 为什么现在不建议继续直接调 PPO/DQN 参数

当前已经有几条比较清楚的证据：

| 证据 | 结果 | 含义 |
|---|---|---|
| windowed PPO，HLA2010 seed0/seed1 | 都使用 I120/N150，产量约 7854 kg/ha | 加窗口后解决了 DAP2 早期打满，但 PPO 仍倾向于用满预算 |
| windowed PPO，HLA2015 seed0 | 使用 I120/N150，产量约 7639 kg/ha | 行为可解释，但仍没有证明资源效率最优 |
| DQN 离散动作，HLA2010 seed0 | 只灌溉 30 mm，但仍施氮 150 kg/ha，产量约 7373 kg/ha | 换成离散动作后，水分行为改变，但氮仍然用满 |
| 固定 DQN 灌溉动作，扫描 N0/N50/N100/N150 | N0 到 N150 产量几乎不变，约 7354-7355 kg/ha | 当前情景下，N150 几乎没有产量边际收益 |

因此，现在的问题不只是：

- PPO 连续动作不好；
- 或者 DQN 训练步数不够；
- 或者某个超参数没调好。

更核心的问题是：

> 当前目标函数没有有效区分“降低氮胁迫指标”和“提高最终产量/收益”。

在当前 reward 下，算法可能认为“压低 NSTRES 是好事”，即使额外施氮并没有带来产量提升。

所以继续小幅调 PPO/DQN 参数意义不大，应该先重新定义优化目标。

## 2. 三套可选目标函数方案

### 方案 A：经济收益型 reward

核心思想：

> 直接让算法优化净收益，而不是只追求产量或胁迫指标。

可以写成：

```text
R = crop_price × Δyield
    - water_price × irrigation
    - nitrogen_price × nitrogen
    - optional_environment_penalty
```

如果按阶段奖励写：

```text
R_t = p_y × ΔGRNWT_t
      - c_i × I_t
      - c_n × N_t
      - c_s × stress_penalty_t
```

终止时可加入：

```text
R_terminal = p_y × final_GRNWT
```

或者只在终止时结算：

```text
R_terminal = p_y × final_GRNWT
             - c_i × total_I
             - c_n × total_N
```

#### 优点

- 最直接解决“没有产量收益却施氮”的问题；
- 导师容易理解；
- 可以解释为农户收益最大化；
- 如果 N150 不增产，施氮会被成本惩罚，自然变差。

#### 缺点

- 需要确定作物价格、水价、氮肥价格；
- 不同价格假设会影响结果；
- 如果价格设置不合理，算法结论也会被质疑。

#### 适合回答的问题

> 在考虑水氮成本后，RL 是否能学会少用无效氮肥？

## 3. 方案 B：约束内产量最大化

核心思想：

> 水和氮只作为硬约束，reward 主要看产量，不在 reward 里重复扣水氮成本。

可以写成：

```text
maximize final_GRNWT
subject to:
    total_I <= I_budget
    total_N <= N_budget
    single_I <= I_single_max
    single_N <= N_single_max
    action only allowed in agronomic windows
```

阶段 reward 可以简单化：

```text
R_t = a × ΔGRNWT_t + b × ΔTOPWT_t
```

终止 reward：

```text
R_terminal = final_GRNWT
```

#### 优点

- 逻辑清楚：这是“有限资源预算下最大化产量”；
- 不需要人为设定水价和氮价；
- 比较适合当前 I120/N150 这种预算约束场景；
- 和当前 windowed PPO 结果比较容易衔接。

#### 缺点

- 如果预算给得太高，算法仍可能把预算用满；
- 不能解决“同样产量下谁更省氮”的问题；
- 无法体现资源效率或环境代价。

#### 适合回答的问题

> 在固定水氮上限内，RL 能否找到最高产调度？

## 4. 方案 C：Constrained RL / Lagrangian 约束优化

核心思想：

> 把产量作为 reward，把水、氮、环境损失作为 cost constraint，而不是手工塞进一个 reward 权重。

形式上可以写成：

```text
maximize E[final_GRNWT]
subject to:
    E[total_I] <= I_limit
    E[total_N] <= N_limit
    E[N_loss] <= N_loss_limit
```

拉格朗日形式：

```text
L = reward
    - λ_i × max(0, total_I - I_limit)
    - λ_n × max(0, total_N - N_limit)
    - λ_loss × max(0, N_loss - N_loss_limit)
```

其中 λ 可以由算法自动调整，而不是手动固定。

#### 优点

- 方法上最正式；
- 适合“产量-资源-环境”多目标问题；
- 避免手工猜 reward 权重；
- 和 constrained PPO / PPO-Lagrangian / P3O 文献相关。

#### 缺点

- 实现复杂度最高；
- 需要额外实现 cost logging 和拉格朗日更新；
- 训练稳定性需要重新验证；
- 目前对开题/短期汇报来说工作量较大。

#### 适合回答的问题

> RL 能否在满足水氮/环境约束的同时最大化产量？

## 5. 当前最推荐的优先级

| 优先级 | 方案 | 理由 |
|---:|---|---|
| 1 | 经济收益型 reward | 最直接解决当前“无产量收益仍施氮”的问题，导师也最容易理解 |
| 2 | 约束内产量最大化 | 可作为当前结果的保守主线，适合说明有限资源下的产量调度 |
| 3 | Constrained RL / Lagrangian | 方法最正式，但实现复杂，适合作为后续算法改进方向 |

## 6. 建议给导师的汇报说法

可以这样说：

> 我们现在已经确认，PPO 和 DQN 的问题并不只是训练步数或单个超参数。DQN 离散动作可以改变灌溉行为，但仍然会施满氮；进一步固定灌溉扫描 N0-N150 后发现，N150 几乎没有产量边际收益。这说明当前 reward/目标函数没有区分“降低氮胁迫指标”和“提高最终产量/收益”。所以下一步不应继续盲目调 PPO 或 DQN，而应该先重新定义优化目标。我们准备比较三种目标函数：经济收益型、约束内产量最大化、Constrained RL。

## 7. 建议下一步小实验

如果导师同意继续让 RL 尝试做出更合理的氮决策，建议优先测试方案 A。

### 012_03 建议实验

只做 HLA2010，先不扩年份。

算法可以先用 DQN，因为 DQN 已经显示出离散动作能改变灌溉行为。

动作空间仍用：

| action | 含义 |
|---:|---|
| 0 | 不操作 |
| 1 | 灌溉 30 mm |
| 2 | 施氮 50 kg N/ha |
| 3 | 灌溉 30 mm + 施氮 50 kg N/ha |

reward 改成经济收益型：

```text
R_terminal = p_y × final_GRNWT
             - c_i × total_I
             - c_n × total_N
```

先做 2-3 组氮价敏感性：

| 情景 | 氮成本 |
|---|---|
| low_N_cost | 低氮成本 |
| medium_N_cost | 中等氮成本 |
| high_N_cost | 高氮成本 |

评价重点：

- DQN 是否还用满 N150；
- 如果少施氮，产量损失多少；
- 净收益是否提高；
- 管理行为是否更像“有收益才施氮”。

## 8. 当前结论边界

当前证据不能说：

> DQN 一定比 PPO 好。

当前证据只能说：

> DQN 离散动作改变了灌溉策略，但没有解决氮肥用满；因此关键问题已经从“PPO 参数调优”转向“水氮联合优化目标函数重构”。
