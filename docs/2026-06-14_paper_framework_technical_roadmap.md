# Paper Framework And Technical Roadmap

## 1. Proposed Paper Positioning

### Working Title

基于天气预报与胁迫诊断辅助的作物水氮管理强化学习优化框架

可选英文表达：

Forecast- and Stress-Aware Reinforcement Learning Framework for Water-Nitrogen Crop Management

### Core Claim

本研究不是直接提出一个万能的、完全自动的水氮联合最优 PPO 模型，而是构建一个更稳妥的框架：

1. 先用作物模型诊断站点年份中的水分胁迫和氮素胁迫；
2. 识别真正适合灌溉优化的代表性干旱年份；
3. 将天气预报信息和胁迫状态引入强化学习动作约束；
4. 在代表性干旱年份验证 PPO 能否产生非零、非饱和、具有产量收益且可解释的灌溉决策；
5. 讨论当前 PPO 策略稳定性和跨年份泛化仍有待验证。

更简洁的论文结论可以写成：

> 本研究构建了一个天气预报/胁迫诊断辅助的水氮管理强化学习优化框架。结果表明，在代表性干旱年份 HLA 2004 中，去除硬性规则托底后，PPO 能够自主产生后期补水策略，以较低灌溉量获得接近充分灌溉的产量；但策略对随机初始化仍敏感，说明真实站点水氮联合优化仍需要更稳健的训练与泛化设计。

## 2. Why This Framing Is Necessary

### Initial Ambition

最初目标是：

- 使用 gym-DSSAT / DSSAT；
- 在中国五个站点进行玉米水氮联合优化；
- 训练 PPO 同时决定灌溉和施肥；
- 最终实现高产、高氮回收率、低灌溉、低施肥。

### What We Learned

直接做这个目标太难，原因包括：

- 很多站点年份本身没有明显水分胁迫；
- 产量限制因素经常是氮，而不是水；
- daily continuous PPO 容易出现 DAP=1 大量灌水/施肥或打满 cap；
- 水氮管理的产量收益有明显延迟，reward credit assignment 很难；
- 导师要求过程合理性，不只是最终产量高；
- 不同站点土层数不同，直接跨站点迁移 PPO 模型会遇到 observation 维度不一致问题；
- hard minimum forecast gate 虽然农学解释好，但会让规则替 PPO 决策，导致 PPO 自主贡献为 0。

因此，论文框架需要从“直接端到端水氮联合最优 PPO”收缩为：

> 诊断驱动、天气预报辅助、阶段决策约束下的水氮管理强化学习框架。

## 3. Overall Technical Route

```text
原始站点数据
    |
    v
气象数据清洗与 WTH 生成
    |
    v
播种/收获期与初始土壤水氮校正
    |
    v
所有站点年份水氮胁迫诊断
    |
    v
筛选代表性年份
    |-- 氮限制案例：如 FQA 2008
    |-- 水分限制案例：HLA 2004, FQA 2016, SYA 2017
    |
    v
情景对比
    |-- Null
    |-- N-only
    |-- fixed irrigation
    |-- recorded expert
    |-- DSSAT auto attempt
    |-- rule / PPO
    |
    v
forecast/stress-aware stage action design
    |
    v
PPO 贡献验证
    |-- hard minimum gate: 规则有效但 PPO 贡献为 0
    |-- no-hard-minimum soft-stress PPO: PPO 有自主贡献
    |
    v
稳定性诊断
    |-- seed0 成功
    |-- seed1 原配置退化
    |-- stronger penalty 部分恢复
    |
    v
第二年份验证与论文收束
```

## 4. Completed Work And Evidence

### 4.1 Data Preparation

已完成：

- 多站点气象数据清洗；
- RAIN 缺失处理；
- QC 版 WTH 文件生成；
- 播种期、收获期观测数据整理；
- 初始土壤水分和初始氮素条件校正；
- dry / normal / wet 年份分类；
- all-year weather scenario pool 构建。

论文中对应作用：

> 说明模拟输入数据不是随意设置，而是经过气象、物候、土壤初始条件整理后用于 DSSAT/gym-DSSAT。

### 4.2 Water-Nitrogen Stress Diagnosis

已完成：

- NullAgent / fixed management smoke tests；
- water × nitrogen factorial diagnosis；
- all-year fixed management stress diagnosis；
- observed-year and all-year water-stress screening。

核心发现：

- 严格 null_zero 情况下，很多年份没有 SWFAC，因为无氮导致作物长不起来，水分需求低；
- 更合理的水分胁迫判断应使用 adequate-N rainfed reference，例如 N-only medium；
- FQA 2008 并不适合作为主要灌溉优化展示案例，它更偏氮限制；
- HLA 2004 是强水分限制案例；
- FQA 2016、SYA 2017 是后续可验证的水分限制年份。

论文中对应作用：

> 证明为什么不能随便选一年训练 PPO，必须先做水氮限制因子诊断。

### 4.3 Scenario Comparison

HLA 2004 已完成代表性情景对比：

| 情景 | 灌溉 | 氮肥 | GRNWT | 作用 |
| --- | ---: | ---: | ---: | --- |
| null_zero | 0 | 0 | 418.6 | 严格下限 |
| n_only_medium | 0 | 150 | 678.9 | 水分限制诊断基准 |
| fixed_I60_N150 | 60 | 150 | 5583.4 | 中等灌溉响应 |
| fixed_I120_N150 | 120 | 150 | 7062.5 | 充分灌溉参考 |
| expert_reference_recorded | 30 | 165 | 5150.2 | 单年实测专家管理 |

论文中对应作用：

> 证明 HLA 2004 是强干旱/强灌溉响应年份，适合作为水分优化代表案例。

### 4.4 Forecast Gate Rule Validation

008_11 已完成纯规则验证：

| 站点年份 | 规则灌溉 | GRNWT | 解释 |
| --- | ---: | ---: | --- |
| HLA 2004 | 120 | 6940.1 | 极旱年份，规则多次触发 |
| FQA 2008 | 30 | 7003.4 | 非主要水分案例，只少量触发 |
| FQA 2016 | 90 | 6641.8 | 中间型水分胁迫年份 |

论文中对应作用：

> forecast/stress-aware gate 本身可以作为方法贡献：它不是盲目灌溉，而是根据未来降雨和胁迫风险形成响应梯度。

### 4.5 PPO Contribution Diagnosis

008_12 发现：

- hard minimum gate 下，PPO 与纯规则结果完全一致；
- PPO 额外贡献为 0；
- 说明规则在替 PPO 决策。

论文中对应作用：

> 解释为什么不能只用 hard minimum gate，否则虽然结果好，但不是 PPO 学到的。

### 4.6 No-Hard-Minimum PPO Result

008_14 / 008_15 完成：

- HLA 2004；
- 固定 N150；
- PPO 只控制灌溉；
- gate 只允许/禁止灌溉，不再强制最低灌水；
- 加入 soft SWFAC stress penalty。

核心结果：

| 实验 | seed | 灌溉 | GRNWT | GRNWT / fixed_I120 | PPO 自主贡献 | 策略 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 008_15 | 0 | 80 | 6777.9 | 0.960 | 80 | S4/S5 各 40 mm |

论文中对应作用：

> 证明 PPO 在没有规则硬托底时，确实能做出非零、非饱和、有产量收益的灌溉决策。

### 4.7 Stability Diagnosis

008_16 / 008_17 完成：

| 实验 | seed | penalty | 灌溉 | GRNWT | GRNWT / fixed_I120 | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 008_15 | 0 | 1.0 / 0.25 | 80 | 6777.9 | 0.960 | 成功 |
| 008_16 | 1 | 1.0 / 0.25 | 0 | 762.0 | 0.108 | 退化 |
| 008_17 | 1 | 3.0 / 1.0 | 53.6 | 5579.5 | 0.790 | 部分恢复 |

论文中对应作用：

> 说明当前 PPO 不是完全稳定，策略对 seed 和 reward scaling 敏感；这是局限性，而不是需要掩盖的问题。

## 5. What Is The Actual Contribution?

### Contribution 1: Stress Diagnosis Before RL

不是所有站点年份都适合训练灌溉 PPO。我们先诊断水分胁迫和氮素胁迫，再选择代表年份。

这比直接扔给 PPO 更符合农学逻辑。

### Contribution 2: Forecast/Stress-Aware Stage Decision Interface

PPO 不再每天乱灌，而是在阶段尺度上决策，并由未来 7 天降雨和 SWFAC 胁迫信息决定灌溉是否允许。

这个设计回应导师对“为什么没胁迫也灌水”的质疑。

### Contribution 3: Separating Rule Contribution From PPO Contribution

008_11 / 008_12 / 008_15 清楚区分：

- 规则可以做什么；
- hard gate 是否替 PPO 做决策；
- PPO 在无 hard minimum 下是否真的有自主贡献。

这是论文里很有价值的方法诊断。

### Contribution 4: Demonstrating PPO Potential And Limitation

PPO 在 HLA 2004 seed0 下有明确有效策略，但 seed1 不稳定。

这让结论更可信：

> PPO 有潜力，但真实作物水氮管理中的稳定训练仍然是挑战。

## 6. What Is Still Missing?

### Must Have Before A Strong Draft

1. 第二水分胁迫年份验证
   - 推荐 FQA 2016；
   - 不调参；
   - 使用 008_17 stronger penalty 配置；
   - seed0 5k；
   - 只看是否出现非零、非饱和、可解释的灌溉行为。

2. 正式图表整理
   - HLA 2004 情景对比柱状图；
   - 008_11 forecast gate 响应梯度图；
   - 008_15 PPO 过程图；
   - 008_18 诊断链表；
   - 技术路线图。

3. 论文方法章节草稿
   - 数据；
   - DSSAT/gym-DSSAT 环境；
   - 胁迫诊断；
   - forecast gate；
   - stage PPO；
   - reward；
   - 评价指标。

### Nice To Have

1. FQA 2016 如果表现合理，再加 SYA 2017；
2. 多 seed 稳定性进一步修复；
3. 恢复部分氮肥决策；
4. 更正式的 forecast uncertainty 设计。

但这些不是当前初稿必须完成。

## 7. Recommended Paper Structure

### Abstract

说明问题、方法、代表性结果和局限。

### 1. Introduction

- 水氮管理重要性；
- DSSAT/crop model 与 RL 的潜力；
- 真实站点水氮优化难点；
- 本研究提出 forecast/stress-aware RL 框架。

### 2. Materials And Methods

#### 2.1 Study Sites And Data

- 五个中国生态站点；
- 气象、土壤、品种、管理记录；
- 播种/收获期；
- WTH 生成。

#### 2.2 DSSAT/gym-DSSAT Simulation Environment

- DSSAT 作物模型；
- gym-DSSAT interface；
- action/state/reward；
- HLA/FQA/SYA 年份设置。

#### 2.3 Water-Nitrogen Stress Diagnosis

- Null；
- N-only；
- fixed irrigation；
- SWFAC/NSTRES；
- 为什么筛选代表年份。

#### 2.4 Forecast/Stress-Aware Stage Decision Framework

- DAP stage；
- future 7-day rainfall；
- SWFAC trigger；
- no hard minimum gate；
- PPO controls irrigation amount。

#### 2.5 PPO Training And Evaluation

- PPO setup；
- seed；
- reward；
- evaluation metrics；
- baselines。

### 3. Results

#### 3.1 Stress Diagnosis Across Station-Years

说明很多年份不是水分限制，HLA 2004 是强水分限制案例。

#### 3.2 HLA 2004 Scenario Comparison

展示 null / N-only / fixed irrigation / expert。

#### 3.3 Forecast Gate Rule Response

展示 008_11：不同年份规则响应不同。

#### 3.4 PPO Contribution Under Different Designs

展示 008_12 到 008_15：

- hard gate PPO 无贡献；
- no-hard-min PPO 有贡献。

#### 3.5 PPO Stability And Reward Sensitivity

展示 008_16 / 008_17：

- seed0 成功；
- seed1 退化；
- stronger penalty 部分恢复。

#### 3.6 Second-Year Validation

待做：FQA 2016。

### 4. Discussion

- 为什么直接 PPO 不适合真实作物管理；
- 天气预报和胁迫诊断如何提升解释性；
- 与已有 RL irrigation / nitrogen / DSSAT 文献对比；
- seed instability 和 reward sensitivity；
- 未来改进。

### 5. Conclusion

核心结论应克制：

> 本研究证明了 forecast/stress-aware 强化学习框架在代表性干旱年份中具有产生有效灌溉决策的潜力，但跨 seed 稳定性和跨年份泛化仍需进一步研究。

## 8. Are We Close To A Draft?

### Already Close

- 数据处理方法；
- 胁迫诊断逻辑；
- HLA 2004 主案例；
- forecast gate 方法；
- PPO 贡献与局限性；
- 技术路线图。

### Not Yet Enough

还缺：

- 第二年份验证；
- 系统图表；
- 方法章节文字；
- 结果章节整理；
- 文献综述嵌入。

### Practical Estimate

如果目标是形成一版可给导师看的初稿：

- FQA 2016 验证：0.5 到 1 天；
- 图表整理：0.5 到 1 天；
- 初稿大纲/文字：1 到 2 天；
- 总计：约 2 到 4 天可以形成初稿材料。

如果目标是投稿级论文，还需要更多多 seed、多年份和语言打磨。

## 9. One-Sentence Roadmap For The Supervisor

本研究先通过 DSSAT/gym-DSSAT 对多站点玉米年份进行水氮胁迫诊断，筛选代表性水分限制年份；随后构建基于未来降雨和 SWFAC 胁迫的阶段决策约束，使 PPO 在农学合理的窗口内控制灌溉量；结果显示，在 HLA 2004 干旱年份中，PPO 能够在无规则硬托底下自主学习后期补水策略，以 80 mm 灌溉达到充分灌溉处理约 96% 的产量，但策略仍存在 seed 敏感性，需要在第二年份和更稳健训练设置中进一步验证。
