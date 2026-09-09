# Codex 任务：单站点逐步数据增强 PPO 验证（先做 YC，成功后再进入下一站点）

## 0. 任务背景与总原则

当前研究目标仍然是使用 PPO 优化玉米水氮管理，并最终形成一套能够应用于五个站点（SY / LC / HLA / YC / FQ）的统一研究框架。

导师当前要求：

> **一个站点一个站点做。一个站点先做成功，再进入下一个站点。**

因此本任务**只处理第一个站点 YC / YCA**，不要自动扩展到其他站点。

本轮核心科学问题不是重新发明 PPO，也不是重新大规模调参，而是检验：

> **在保持 PPO 核心算法与决策框架基本不变的前提下，增加农业上合理的训练环境多样性（training-domain diversity / G×E-related augmentation），是否能够改善 YC 站点 PPO 的稳定性和相对基线表现。**

### 本轮禁止自动做的事情

1. 不要自动进入 HLA / FQ / SY / LC。
2. 不要因为结果不好就马上更换 PPO 为 DQN、SAC、TD3 或其他算法。
3. 不要同时大改 reward、网络结构、动作空间、约束、数据增强方式，避免无法判断究竟是哪一项起作用。
4. 不要为了得到漂亮结果而反复挑 seed、年份或删除失败结果。
5. 不要自行实施大规模 synthetic weather generator（WGEN/LARS-WG 等）。本轮先用低风险、可解释的数据扩展；如果仍失败，再单独决定是否进入 synthetic-weather 阶段。
6. 不要把“YC 单站成功”表述成“已经证明跨站点泛化”。

---

## 1. 第一阶段：先审计现有 YC PPO 工作流

在写新代码之前，先完整审计现有 YC PPO 相关实验。

重点定位：

- 当前/最近一次 YC 正式 PPO 配置；
- 当前 observation/state variables；
- action space；
- reward；
- PPO network architecture；
- learning rate 等主要 hyperparameters；
- irrigation / nitrogen budget；
- daily cap；
- minimum action interval；
- training years；
- validation/test years；
- 现有 split registry；
- 现有 farmer / expert / null / fixed / external rule-based controller 基线；
- 当前 PPO 在这些基线上的实际表现；
- 当前是否已经存在多 seed 结果；
- YC 是否已有 lowIC / originIC 或其他经过验证的 initial-condition profiles；
- YC 是否已有多个经过校准、可直接使用的 cultivar parameter sets。

优先复用现有代码与配置，不要重新造一套独立 framework。

输出：

```text
docs/yc_ppo_augmentation_audit.md
```

至少写清楚：

```text
1. 当前正式 YC PPO 配置来自哪个 experiment / config
2. 当前 PPO 训练和评价年份
3. 当前基线有哪些
4. 当前“超越基线”的项目内正式判据是什么
5. 当前 PPO 失败主要表现在哪里
6. 本轮哪些东西保持固定
7. 本轮唯一主动改变的核心因素：training-domain diversity
```

如果项目中存在多套互相矛盾的 YC PPO 配置，不要自行挑“最好看”的版本。

应优先选择：

> 最近一次正式、完整、可复现、用于和基线比较的 YC PPO 配置

并在 audit 文档里说明选择依据。

---

## 2. 本轮实验必须先固定的东西

本轮目的在于检验“数据/环境扩展是否有用”。

因此原则上保持以下内容不变：

```text
PPO algorithm
observation schema
network architecture
reward functional form
action space
site-specific physical constraints
training steps / rollout budget
evaluation metric
baseline definitions
```

如果某些 site-specific constraint 已经在 YC 的正式实验中使用，例如：

```text
IRRIGATION_BUDGET
NITROGEN_BUDGET
DAILY_IRRIGATION_CAP
DAILY_NITROGEN_CAP
MIN_INTERVAL_DAYS
```

本轮继续使用 YC 当前合理的正式配置。

不要为了“统一五站点”强行把 YC 的物理约束改成其他站点的数值。

> **统一框架 ≠ 所有站点强行使用完全相同的农业物理参数。**

本轮先验证：

> 在 YC 自己合理的环境和约束下，增加训练域多样性是否改善 PPO。

---

## 3. 数据增强原则：不是“随便加噪声”，而是扩展合理农业情景

本轮的数据扩展必须满足：

> **增加数量 + 保持每个 scenario 内部农业/气象合理性。**

禁止以下做法：

```text
rainfall × random_factor
temperature 独立乱加噪声
solar radiation 独立乱加噪声
把不同年份的月份/天数随意拼接
单独随机 P1/P2/P5/G2/G3/PHINT 后任意组合
逐土层独立随机初始水分导致不合理土壤剖面
```

原因：

- 降雨、温度、辐射等天气变量具有联合统计结构；
- 天气日序列有持续性和事件顺序；
- cultivar 参数之间存在生物学/校准约束；
- soil/IC profile 具有垂向结构。

因此本轮只做**低风险、已有数据支持的 augmentation**。

---

## 4. YC 第一版 augmentation：只做三类，缺哪类就跳过

### 4.1 Historical whole-year weather sampling（必须做）

使用 YC 已有完整历史天气年。

每个 PPO episode reset 时，从训练年份集合中：

```text
随机抽取一个完整 weather year
```

必须保持该年份：

```text
rainfall
Tmax
Tmin
solar radiation
及现有天气文件中的其他变量
```

作为一个完整、原始、未拆散的 daily sequence。

禁止：

- 把不同年份的月份拼接；
- 单独缩放 precipitation；
- 随机打乱日期。

优先复用项目当前 train/validation/test split。

**held-out validation/test 年份不能进入 augmentation training pool。**

### 4.2 Initial-condition diversity（有可靠输入时做）

先检查 YC 现有项目中是否存在经过验证的：

```text
originIC
lowIC
或其他正式 IC profiles
```

如果存在多个已经验证的完整 IC profile：

> 每个 episode 从“完整 IC profile 集合”中随机抽取一套。

不要逐土层独立随机 SH2O / SNO3 / SNH4。

如果只有一套可靠 IC，没有足够依据构造额外合理 IC：

> 本轮先不人工制造 IC 噪声，记录为 unavailable。

### 4.3 Genotype diversity（只有存在多个已校准 cultivar set 时做）

检查 YC / 当前项目中是否存在多个：

```text
经过校准
可以正常完成 DSSAT
产量/生育期合理
```

的 cultivar parameter sets。

如果存在，则每个 episode：

> 整套抽取一个 cultivar vector。

例如整套：

```text
P1, P2, P5, G2, G3, PHINT
```

一起抽。

禁止：

> 每个 coefficient 独立随机扰动后自由组合。

如果 YC 只有一个可信 cultivar：

> 本轮不做 G randomization，记录为 unavailable。

---

## 5. 本轮只设置两个核心实验组

为了避免实验爆炸，本轮不要做六层 augmentation ladder。

### Experiment A — Original PPO

完全复现当前 YC 正式 PPO：

```text
原训练环境
原 PPO 配置
原训练预算
原评价协议
```

目的：

> 作为严格 reference，确认新代码没有改变原始 PPO 结果。

### Experiment B — Augmented PPO

保持 PPO 其他设置与 A 相同，只增加：

```text
whole-year weather sampling
+ valid IC profile sampling（若有）
+ valid cultivar-set sampling（若有）
```

不要在 B 中同时改 reward / network / action constraints。

本轮真正要回答：

> **Experiment B 是否比 Experiment A 更稳定、更容易超过既定基线？**

---

## 6. 训练与计算资源控制

不要一开始跑大规模多 seed。

### Step 1：smoke run

每组先：

```text
1 seed
较短但足够确认训练流程正常的 training budget
```

检查：

- environment reset 是否真正抽到了不同 scenario；
- scenario sampling 日志是否正确；
- weather / IC / cultivar 没有串到 held-out evaluation；
- action delivery 正常；
- reward 正常；
- episode 能完整结束；
- DSSAT 输出没有异常 sentinel / crash。

如果 smoke 失败，停止，不进入正式训练。

### Step 2：正式 first-pass

若 smoke 正常：

```text
Original PPO: 1 formal seed
Augmented PPO: 同一个 formal seed
```

使用相同 training budget 和 evaluation protocol。

先看 augmented 是否出现明确改善信号。

### Step 3：只有出现改善趋势时才做 seed confirmation

如果 Augmented PPO 相对 Original PPO：

- 明显改善；
- 或开始超过主要 baseline；
- 或稳定性明显提高；

再补：

```text
3 fixed seeds total
```

例如：

```text
seed = 0, 1, 2
```

Original 与 Augmented 使用相同 seed 集合。

如果第一轮完全没有改善：

> 不自动扩大到 5/10 seeds，不自动增加百万级 training steps。

先停止并汇报。

---

## 7. 评价：沿用项目已有基线与判据，不临时改标准

本轮不要重新发明新的“成功指标”。

在第 1 阶段 audit 时，先找到项目当前用于判断 PPO 是否优于基线的正式方法。

至少输出：

```text
Yield
Total N
Total irrigation
Reward / objective（若项目已有正式定义）
NUE（若已有）
WUE（若已有）
```

比较：

```text
Original PPO
Augmented PPO
Null
Fixed
Farmer
Expert
External NSTRES / rule-based controller（若当前正式 benchmark 中存在）
```

关键要求：

1. 所有模型使用同一 evaluation years。
2. 所有 PPO 使用同一评价 protocol。
3. 不允许 Original 用一个年份、Augmented 用另一批年份。
4. 不允许跑完后才换 baseline。
5. 不允许只报告 PPO 最好的 seed。
6. 失败 seed 也必须保留。

---

## 8. 本轮“成功”如何定义

在 audit 中先写下现有正式 benchmark 判据。

### 8.1 数据增强有没有帮助

```text
Augmented PPO > Original PPO ?
```

看：

- 平均评价表现；
- seed 间方差；
- held-out year 稳定性；
- 资源投入；
- yield。

### 8.2 PPO 有没有开始稳定超过基线

回答：

```text
Augmented PPO 是否达到当前项目正式“超越基线”标准？
```

不要只写“reward 提升了”。

必须回到农业结果：

```text
yield
N
irrigation
以及当前项目正式综合评价指标
```

### 8.3 YC 是否可以判定为“当前框架下成功”

只有满足预先定义的 baseline-success criterion 后，才能写：

```text
YC = success
```

否则：

```text
YC = not yet successful
```

不要因为某一个 seed 或某一个年份漂亮就判 success。

---

## 9. Scenario sampling 必须完整记录

为每个 PPO training run 保存：

```text
episode_id
seed
site
weather_year
IC_profile_id
cultivar_id
```

建议路径：

```text
results/training_domain_manifest.csv
```

同时输出统计：

```text
每个 weather year 被抽到多少次
每个 IC profile 被抽到多少次
每个 cultivar set 被抽到多少次
```

确认 sampling 没有严重偏斜。

---

## 10. 最少需要的输出

### 10.1 Summary table

```text
results/yc_augmentation_summary.csv
```

至少包括：

```text
method
seed
evaluation_year
yield
total_n
total_irrigation
reward
NUE（若有）
WUE（若有）
success_vs_baseline
```

### 10.2 Original vs Augmented 对比图

至少包括：

```text
Yield comparison
N input comparison
Irrigation comparison
Reward/objective comparison
```

### 10.3 Seed / year stability

如果跑了 3 seeds：

画：

```text
每个 evaluation year × seed 的表现
```

重点看：

> Augmented PPO 是否减少“某些年份突然崩掉”的情况。

### 10.4 Training-domain sampling 图或表

展示：

```text
weather years
IC profiles
cultivar sets
```

的抽样覆盖情况。

---

## 11. 必须生成的实验报告

输出：

```text
docs/yc_ppo_augmentation_results.md
```

结构：

```text
1. Research question
2. Existing YC PPO configuration
3. Augmentation design
4. What was kept fixed
5. Training-domain composition
6. Original vs Augmented results
7. Comparison with baselines
8. Seed/year stability
9. Whether YC meets success criterion
10. Failure analysis if not successful
11. What should be tested next
```

---

## 12. 本轮停止条件

### 情况 A：YC 成功

如果 Augmented PPO 按预先定义标准稳定超过基线：

```text
STOP
```

不要自动开始 HLA。

最终报告：

> YC 已达到 success criterion，等待用户决定下一站点。

### 情况 B：YC 明显改善但仍未完全超过基线

输出：

```text
partial_improvement
```

明确：

- 改善了多少；
- 哪些年份仍失败；
- 哪些指标仍输；
- 是否值得继续增强。

然后 STOP。

### 情况 C：数据增强基本无效或更差

输出：

```text
augmentation_not_supported
```

不要自行继续：

```text
WGEN
大规模 synthetic weather
新算法
新 reward
新 network
```

先汇报：

> 在当前 YC 配置和计算预算下，没有证据支持“简单增加合理训练域多样性即可解决 PPO 性能问题”。

然后等待用户决定下一步。

---

## 13. 关于五站点“统一框架”的边界

本轮只做 YC，但代码结构要为后续站点复用做好准备。

统一框架应尽量统一：

```text
PPO algorithm
observation schema
network architecture
reward functional form
training/evaluation pipeline
augmentation interface
logging
benchmark protocol
```

允许 site-specific：

```text
soil
weather
cultivar
initial condition
resource budgets
agronomically justified action limits
```

不要为了“通用”强行让五站点共享完全相同的农业约束数值。

同时不要声称：

> “YC 单站训练成功 = PPO 已具有 cross-site generalization。”

本项目当前阶段更准确的目标是：

> **验证同一 PPO/augmentation framework 能否在不同站点分别训练成功。**

真正的 zero-shot cross-site policy generalization 是更严格的另一个研究问题，本轮不做。

---

## 14. Git 与目录要求

建立新的独立 experiment 目录，不覆盖旧 PPO 结果。

例如：

```text
experiments/<new_id>_yc_ppo_domain_augmentation/
```

保存：

```text
configs/
scripts/
results/
logs/
snapshots/
```

所有失败运行保留。

生成：

```text
docs/yc_ppo_augmentation_audit.md
docs/yc_ppo_augmentation_results.md
```

完成后 Git commit。

不要强制 push。

---

## 15. Codex 最终回复必须按这个格式

```text
A. YC 现有 PPO 审计结论
B. 本轮实际使用的数据增强内容
C. 哪些 augmentation 因缺乏可靠数据而跳过
D. Original PPO 结果
E. Augmented PPO 结果
F. 与各基线比较
G. 是否满足 YC success criterion
H. 数据增强假设：supported / partially supported / not supported
I. 所有关键文件路径
J. 本轮未做的事情
K. 下一步建议（只建议，不自动执行）
```

---

## 16. 最后强调

本轮不是要求“一次解决五站点泛化”。

只回答一个问题：

> **在 YC 这个站点，增加农业上合理的训练环境多样性，能不能把 PPO 从当前状态往“稳定超过基线”推进？**

先把 YC 做清楚。

YC 没有明确结论之前，不进入下一个站点。
