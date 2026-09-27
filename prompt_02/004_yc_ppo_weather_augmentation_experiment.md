# 004 YC PPO 天气增强受控实验

## 一、任务前提

本任务只有在上一任务 `003_06_yc_train_only_cli_and_wgen_pilot` 最终状态为：

```text
PASS_YC_WGEN_PILOT
```

时才允许进入正式 PPO 训练。

若 `003_06` 的最终状态不是 `PASS_YC_WGEN_PILOT`，必须：

1. 读取上一轮最终 Gate；
2. 明确指出阻塞点；
3. 不启动 PPO；
4. 最终状态记为：

```text
BLOCKED_UPSTREAM_WGEN_NOT_READY
```

不得绕过 WGEN / DSSAT smoke Gate。

---

# 二、研究问题

本轮只回答一个问题：

> 在 **PPO 算法、reward、action space、observation、训练步数和验证年份完全不变** 的情况下，增加由 YC train-only WGEN 生成的合成天气多样性，是否能改善 YC PPO 在真实未参与天气参数估计的 2014–2023 validation 年份上的表现与稳定性？

本任务不研究：

- PPO 算法改进；
- 网络结构改进；
- reward redesign；
- action mask 改进；
- constrained PPO；
- DQN；
- LC/SY/HL/FQ；
- 五站点最终比较。

本轮仅做：

```text
YC
historical-weather PPO
vs
weather-augmented PPO
```

---

# 三、科学口径

## 3.1 当前数据划分

保持：

```text
train weather = 2004–2013
validation weather = 2014–2023
independent test = not_available_or_not_verified
```

必须继续使用“validation”称呼 2014–2023。

禁止把 2000–2003 或其他曾参与人工比较/调参的年份追认为独立 test，除非另有充分证据。

---

## 3.2 天气增强的含义

本任务中的 weather augmentation 指：

> 在训练天气池中保留真实历史训练天气，同时加入由 **2004–2013 train-only CNYC.CLI** 生成的合成天气，以增加训练期 climate scenario diversity。

禁止表述为：

- 增加独立观测样本；
- 获得新的真实气象观测；
- 扩展独立样本量。

合成天气仍来自有限的 2004–2013 气候统计估计。

---

# 四、严格冻结的 PPO 条件

必须从当前 YC 正式训练配置中读取并冻结：

- PPO implementation
- policy network
- optimizer
- learning rate
- gamma
- GAE
- clipping
- batch size
- rollout length
- total training steps
- action space
- action mask
- observation space
- reward
- fertilizer budget
- irrigation budget
- episode termination
- DSSAT cultivar / soil / management
- evaluation mode

不得因为 weather augmentation 结果不好而修改任何上述项。

输出：

```text
results/yc_ppo_weather_augmentation/frozen_ppo_config.json
```

并保存原配置文件 SHA256。

---

# 五、实验 Arms

本轮主实验固定为两个 Arms。

## Arm H：Historical-only baseline

保持当前 YC PPO 训练方式：

```text
2004–2013 historical weather
```

每个 episode 从真实训练天气年份中按当前生产逻辑抽样。

不得偷偷加入 WGEN weather。

标签：

```text
HISTORICAL_ONLY
```

---

## Arm A：Historical + WGEN augmentation

训练天气包含：

```text
2004–2013 historical weather
+
frozen WGEN synthetic weather bank
```

标签：

```text
HISTORICAL_PLUS_WGEN
```

### 关键原则

必须保留真实历史天气，不能用 WGEN 完全替代。

默认 episode-level source mixture：

```text
50% historical
50% WGEN synthetic
```

如果现有技术实现无法安全做到 50/50：

1. 不得自行改成“全部 WGEN”；
2. 必须实现 source-aware sampler；
3. 或停止并报告技术阻塞。

50/50 是本轮预先冻结的首个 controlled augmentation design，不根据 PPO 结果调比例。

---

# 六、冻结 WGEN training bank

不要在 PPO 训练过程中无限在线生成不可审计的天气。

优先构建一个**固定、可复现的 synthetic weather bank**。

## 6.1 来源

必须来自：

```text
frozen CNYC.CLI
+
已经通过 003_06 的 WGEN runtime
```

不得使用：

- NASA POWER 直接替代训练天气；
- ERA5 直接替代训练天气；
- 历史年份复制；
- 独立高斯噪声；
- 217YCA deterministic scenarios；
- 任意乘法扰动。

---

## 6.2 weather seed 与 PPO seed 分离

建立：

```text
weather_bank_seed
ppo_seed
```

两个独立 seed namespace。

禁止：

```text
weather_bank_seed == ppo_seed
```

仅因为数值相同而耦合 RNG。

例如可使用：

```text
weather bank seeds = 1001, 1002, 1003, ...
ppo seeds = 原项目正式 seed set
```

实际 seed 列表必须在训练前写入 manifest。

---

## 6.3 synthetic bank 数量

先读取：

- 当前 full PPO training 的 episode 数量 / approximate episode count；
- WGEN 单天气生成成本；
- 存储成本。

本轮 synthetic bank：

```text
target = 100 unique synthetic weather years
minimum acceptable = 30
```

如果可以合理生成 100 套，则固定为 100。

如果因明确的 runtime / storage 成本只能生成 30–99 套：

- 在 PPO 启动前确定实际 N；
- 记录原因；
- 冻结 bank；
- 不得根据 PPO 结果追加天气。

若少于 30 套：

```text
BLOCKED_SYNTHETIC_BANK_TOO_SMALL
```

停止。

---

## 6.4 Bank QC

每个 synthetic weather 都必须继承 `003_06` 的天气 QC。

生成：

```text
results/yc_ppo_weather_augmentation/weather_bank_manifest.csv
```

字段至少：

- weather_id
- WGEN seed
- CLI SHA256
- weather SHA256
- year length
- QC status

以及：

```text
weather_bank_summary.json
```

训练前完成并冻结。

---

# 七、训练 source sampler

Arm A 每个 episode：

```text
P(source = historical) = 0.5
P(source = synthetic) = 0.5
```

source 选中后：

### historical
从 2004–2013 十个真实训练年份均匀抽样，除非当前正式实现已有明确、已冻结的其他抽样方式。

### synthetic
从 frozen synthetic bank 中均匀抽样。

必须记录训练期间实际抽样次数：

```text
historical episode count
synthetic episode count
weather_id frequency
```

输出：

```text
results/yc_ppo_weather_augmentation/training_weather_usage_<arm>_<seed>.csv
```

50/50 指概率设计，不要求实际有限 episode 次数恰好完全相等。

---

# 八、PPO seeds

必须使用**相同 PPO seed set**训练两个 Arms。

优先读取当前项目已正式使用的 YC PPO seed set。

若项目已有例如：

```text
0,1,2,3,4
```

则沿用，不重新选择。

若此前只正式使用单个 PPO seed：

本轮最低要求：

```text
5 PPO seeds
```

在训练前冻结并写入：

```text
ppo_seed_manifest.json
```

禁止：

- 先跑很多 seed 再挑好的；
- Historical 用一套 seed、Augmented 用另一套；
- 只报告最佳 seed。

---

# 九、两阶段运行，避免浪费算力

## Stage 0：integration smoke

正式 full training 前：

对：

```text
1 个 PPO seed
HISTORICAL_ONLY
HISTORICAL_PLUS_WGEN
```

分别跑一个短 smoke。

smoke 只验证：

- environment reset
- historical / synthetic source switching
- action / observation shape
- reward calculation
- weather loading
- no DSSAT crash
- logging
- checkpoint saving

smoke 结果不用于科学比较。

若失败：

```text
BLOCKED_PPO_AUGMENTATION_PIPELINE
```

---

## Stage 1：full controlled experiment

两个 Arms：

```text
相同 PPO seeds
相同 total training steps
相同 PPO config
相同计算预算
```

不得因为某个 Arm 学得慢而延长训练步数。

---

# 十、Validation 必须完全相同

所有训练完成模型统一评估在：

```text
2014–2023 real observed/official YC validation weather
```

禁止 validation 使用 WGEN。

所有模型：

- 同一 validation 年份；
- 同一 DSSAT cultivar；
- 同一 soil；
- 同一 management baseline conditions；
- 同一 deterministic/stochastic evaluation 设置；
- 同一指标定义。

validation 不参与训练天气生成、bias correction 或模型选择。

---

# 十一、评价指标

必须先从项目当前评价代码/文档中读取实际定义，不得擅自换公式。

至少报告：

### 农业表现

- grain yield
- irrigation amount
- N fertilizer amount
- WUE
- NUE

如果当前项目对 WUE/NUE 有多个定义：

必须先明确使用哪一个公式、单位、分母。

禁止统计口径混用。

---

### RL 表现

- validation reward
- training reward trajectory
- episode return variance

但：

> reward 高不能自动等价于农业综合表现更好。

必须把 reward 与农业指标分开报告。

---

### 管理行为

至少报告：

- fertilizer event count
- irrigation event count
- fertilizer timing
- irrigation timing
- per-event amount
- total N
- total irrigation

若当前 action 为联合 N × irrigation action，也要检查 joint action frequency。

---

### 稳定性

跨 PPO seeds 报告：

- mean
- median
- SD
- min/max
- all individual seeds

禁止只报告最优 seed。

---

# 十二、paired comparison

因为两个 Arms 使用相同 PPO seed set，优先按 seed 做 paired comparison：

```text
seed 0: H vs A
seed 1: H vs A
...
```

validation 年份也按：

```text
same seed × same validation year
```

配对。

输出 long-format：

```text
results/yc_ppo_weather_augmentation/validation_results_long.csv
```

至少字段：

- arm
- ppo_seed
- validation_year
- yield
- irrigation
- fertilizer
- WUE
- NUE
- reward
- fertilizer_events
- irrigation_events

---

# 十三、预注册的“成功”口径

本轮必须在读取 full-training 结果前，先写：

```text
results/yc_ppo_weather_augmentation/success_criteria.json
```

不得事后根据结果改。

## 13.1 首要判断

Weather augmentation 的核心目标不是追求某个单一最大 reward，而是：

> 在真实 2014–2023 validation weather 上，提高 PPO 的整体泛化表现或稳定性，同时不以明显农业代价换取表面 reward。

因此至少分别判断：

1. validation agricultural metrics
2. validation reward
3. seed stability
4. management behavior

---

## 13.2 不强行压成单一综合分数

禁止把：

```text
yield
NUE
WUE
fertilizer
irrigation
reward
```

未经已有科学依据直接加权成一个新 composite score。

优先报告多指标 trade-off 与 Pareto 情况。

---

## 13.3 seed-level success

如果项目已有经导师确认的 success definition，优先沿用，并记录来源。

如果没有已有正式阈值，本轮使用透明的 seed-level operational success：

一个 Augmented PPO seed 相对于相同 seed 的 Historical PPO，在 2014–2023 平均结果上：

```text
A. mean validation reward > historical
AND
B. mean yield 不低于 historical 超过 2%
AND
C. 至少一个资源效率指标（NUE 或 WUE）提高
AND
D. N fertilizer 与 irrigation 不出现同时明显上升
```

其中：

```text
yield tolerance = -2%
```

仅作为本轮预注册的工程容忍线，不宣称为农业学通用标准。

若用户/项目已有更正式阈值，则以已有阈值替换，并在训练前冻结。

---

## 13.4 experiment-level success rate

报告：

```text
successful PPO seeds / all PPO seeds
```

以及：

- 全部 seed 结果；
- median effect；
- worst seed；
- best seed。

不得只展示 successful seeds。

本项目允许随机种子表现不同，但必须报告成功概率/比例，而不是事后挑 seed。

---

# 十四、统计分析

样本单位必须说清楚。

禁止把：

```text
10 validation years × 5 PPO seeds
```

简单当作 50 个完全独立样本。

至少区分：

- PPO seed variation
- validation-year variation

优先输出：

1. 每 seed 的 10-year validation mean；
2. seed-level paired differences；
3. year × seed long table。

如进行显著性检验，必须说明层级结构和局限。

本轮不要求为了得到 p<0.05 而增加 seed。

---

# 十五、训练日志和模型冻结

每个：

```text
arm × ppo_seed
```

必须保存：

- config snapshot
- git commit
- model checkpoint
- training log
- TensorBoard/CSV log（若已有）
- weather usage log
- final model SHA256
- training start/end metadata

不得覆盖既有 YC 模型。

目录建议：

```text
results/yc_ppo_weather_augmentation/models/
  historical_only/
  historical_plus_wgen/
```

---

# 十六、不要做超出本轮的问题

禁止根据结果：

- 改 PPO 算法；
- 调 reward；
- 调 action mask；
- 调网络；
- 改 WGEN 参数；
- 改 50/50 mixture；
- 改 synthetic bank size；
- 删除表现差的 seed；
- 重新生成“更有利”的 weather bank。

如果 Augmented 没改善：

这本身就是实验结果。

下一步再讨论原因。

---

# 十七、基线策略

本轮**不重新跑完整四类农业 baseline 对比**，除非已有完全相同 validation 条件下的结果可以直接引用。

当前主要因果比较是：

```text
Historical-only PPO
vs
Historical+WGEN PPO
```

原因：

本轮只检验 weather augmentation。

已有：

- no management
- DSSAT automatic
- farmer
- expert

代码和结果不得删除。

若已有完全可比结果，可作为 context table；不得让 baseline 改变本轮两个 PPO Arms 的训练条件。

完整五站点 baseline 比较留到后续。

---

# 十八、最终判定

允许状态：

## `PASS_AUGMENTATION_EXPERIMENT_COMPLETE`

条件：

- upstream WGEN PASS；
- weather bank 冻结；
- 两 Arms 全部预定 PPO seeds 完成；
- validation 全部完成；
- success criteria 在结果读取前已冻结；
- 全 seed 报告完整；
- 无 cherry-picking。

注意：

`PASS_AUGMENTATION_EXPERIMENT_COMPLETE`

只表示实验完整，不表示 augmentation 一定更好。

---

## `BLOCKED_UPSTREAM_WGEN_NOT_READY`

003_06 未通过。

## `BLOCKED_SYNTHETIC_BANK_TOO_SMALL`

无法获得至少 30 套冻结 synthetic weather。

## `BLOCKED_PPO_AUGMENTATION_PIPELINE`

source sampler / DSSAT / environment integration smoke 失败。

## `BLOCKED_FULL_TRAINING`

预定 full runs 无法完成。

## `BLOCKED_VALIDATION`

训练完成但统一 validation 无法完成。

---

# 十九、必须生成的结果

## scripts

建议：

```text
scripts/run_yc_ppo_weather_augmentation.py
scripts/evaluate_yc_ppo_weather_augmentation.py
```

尽量复用现有 PPO runner，不复制整套算法。

---

## results

至少：

```text
results/yc_ppo_weather_augmentation/
  frozen_ppo_config.json
  weather_bank_manifest.csv
  weather_bank_summary.json
  ppo_seed_manifest.json
  success_criteria.json
  training_manifest.csv
  validation_results_long.csv
  validation_seed_summary.csv
  paired_arm_comparison.csv
  management_action_summary.csv
  experiment_summary.json
```

---

# 二十、中文报告 / PPT / 实验日志

必须生成：

```text
docs/yc_ppo_weather_augmentation_experiment.md
docs/yc_ppo_weather_augmentation_experiment.pptx
results/yc_ppo_weather_augmentation/experiment_log.md
```

主体全部中文。

报告至少包括：

1. 研究问题；
2. upstream WGEN 状态；
3. weather bank；
4. Historical vs Augmented 实验设计；
5. 50/50 source sampler；
6. PPO seed set；
7. frozen success criteria；
8. training curves；
9. 2014–2023 validation；
10. yield / N / irrigation / NUE / WUE；
11. management actions；
12. seed-level paired results；
13. success rate；
14. trade-offs；
15. limitations；
16. 下一步 YC 决策。

---

# 二十一、Git / GitHub

1. 开始前记录 branch、HEAD、git status。
2. 不纳入已有无关用户修改。
3. 大模型 checkpoint 若超出 Git 适合范围，不强行 commit；记录 SHA256 和本地路径。
4. 代码、配置、轻量结果、报告可提交。
5. 建议 commit：

```text
feat: evaluate YC PPO weather augmentation
```

6. 遵守 `AGENTS.md`。
7. 未经用户明确批准不得 push。
8. 最终回复提供待批准 push 命令。

---

# 二十二、Codex 最终回复格式

用中文依次汇报：

1. 003_06 是否为 `PASS_YC_WGEN_PILOT`。
2. frozen synthetic bank 数量、seed 范围和 SHA256 manifest。
3. Historical / Augmented 是否使用完全相同 PPO config 和 training steps。
4. PPO seed set。
5. 50/50 weather source sampler 是否实际生效，实际 episode source 比例是多少。
6. 两 Arms 是否全部完成 full training。
7. 2014–2023 validation 是否全部完成。
8. yield / fertilizer / irrigation / WUE / NUE / reward 的主要 paired differences。
9. 各 PPO seed 是否达到预注册 success criteria。
10. success rate。
11. 是否存在明显 trade-off。
12. 是否存在 cherry-picking 或缺失 run；若有必须明确说明。
13. Final experiment status。
14. 新增/修改文件。
15. Git commit / push 状态。
16. 下一步最小任务。
