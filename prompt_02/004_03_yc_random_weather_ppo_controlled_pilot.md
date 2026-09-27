# 004_03：YC random-weather PPO controlled pilot

## 一、任务定位

上一阶段 `004_02` 已完成 YC episode-level 随机天气 ensemble 验证：

```text
synthetic_episode_count = 100
weather_seed_range = 1001–1100

generation_status = PASS
same_seed_reproducibility = PASS
different_seed_diversity = PASS

rainfall_status = PASS
temperature_status = PASS_WITH_NOTES
srad_status = PASS
dependence_structure_status = PASS

random_weather_episode_quality_status = PASS_WITH_NOTES
yc_random_weather_ready_for_ppo_pilot = YES
```

主要 note：

```text
synthetic TMIN mean ≈ fitting observed +0.58 °C
```

该 note 需要保留，但不阻断本轮 pilot。

全年 WGEN：

```text
DEFERRED / OPTIONAL ADDITIONAL VALIDATION
```

不再作为 episode-level random-weather PPO 的 blocker。

---

# 二、本轮核心科学问题

导师提出的当前假设是：

> YC PPO 表现不稳定/不理想，可能主要与训练天气多样性不足有关。

该命题目前仍是：

```text
HYPOTHESIS
```

不是既定事实。

本轮只检验：

> 在保持 PPO 算法、奖励、状态、动作、训练预算和作物配置不变的条件下，仅增加 WGEN 随机天气多样性，是否能改善 YC PPO 在未见天气条件下的表现或稳健性？

不修改 PPO 算法结构。

---

# 三、实验设计总览

进行一个小规模、严格受控的 paired pilot：

```text
Training regime A:
HISTORICAL_WEATHER

Training regime B:
RANDOM_WEATHER_WGEN
```

每个 regime：

```text
3 PPO seeds
```

预先固定：

```text
ppo_seed = 0, 1, 2
```

总训练模型数：

```text
2 regimes × 3 PPO seeds = 6 models
```

### 关键原则

```text
ppo_seed
≠ weather_seed
≠ weather schedule/order seed
≠ evaluation weather seeds
```

所有 seed 必须显式记录，不允许隐式共用。

---

# 四、冻结项

本轮不得改变：

```text
PPO algorithm
PPO hyperparameters
network architecture
reward function
observation space
action space
fertilizer/irrigation constraints
crop cultivar
soil
planting configuration
DSSAT binary
gym-dssat-pdi installed package
CNYC.CLI
WGEN parameters
wet-day threshold
```

当前 reward 保持：

```text
0.06 * final_grnwt - 0.04 * cumfert
```

状态保持现有 YC PPO canonical configuration。

不要为了提高结果临时改：

```text
learning rate
gamma
batch size
n_steps
clip range
reward weights
action mask
training timesteps
```

---

# 五、先读取 canonical YC PPO 配置

不要在 prompt 中猜训练预算。

从仓库中找到当前 YC PPO 正式训练入口和 canonical config，记录：

```text
training script
total_timesteps
n_steps
batch_size
learning_rate
gamma
gae_lambda
clip_range
ent_coef
vf_coef
policy architecture
normalization
reward configuration
action configuration
observation configuration
```

保存：

```text
results/yc_random_weather_ppo/004_03/config/canonical_yc_ppo_config.json
```

两种 regime 必须使用完全相同的 PPO 配置。

如发现已有 YC 训练配置存在多个版本：

优先使用此前正式 YC 对比实验使用的当前 canonical 配置，并在报告中说明依据。

不要自行调参。

---

# 六、Training regime A：historical-weather control

目标：

> 建立与 random-weather training 完全同预算的 historical-weather control。

使用：

```text
random_weather = False
```

训练天气来源保持现有 YC historical training design：

```text
2004–2013
```

但必须审计并显式记录：

```text
每个 episode 使用哪个 historical year
historical year sampling 是否与 ppo_seed 混用
```

### 重要

如果当前 historical-year sampling 隐式使用 `ppo_seed`：

不要继续混用。

建立独立的、预先保存的 historical-year schedule，例如：

```text
historical_weather_schedule_seed = fixed constant
```

并让 ppo_seed=0/1/2 三次训练使用相同的 historical-year schedule。

这样：

```text
PPO randomness varies
weather exposure schedule stays controlled
```

保存：

```text
training_weather_schedule_historical.csv
```

---

# 七、Training regime B：random-weather WGEN

使用：

```text
random_weather = True
WTHER = W
CNYC.CLI = frozen official candidate
```

训练 weather seed pool：

```text
1001–1080
```

这 80 个 seed 已包含在 `004_02` 验证 ensemble 中。

### 训练 weather seed 调度

不得让：

```text
weather_seed = ppo_seed
```

也不得用 PPO RNG 隐式产生 weather seed。

创建独立 weather schedule。

要求：

```text
weather_schedule_seed = fixed constant
```

使用该独立 schedule seed 对：

```text
1001–1080
```

生成可复现的 episode weather-seed sequence。

如果训练 episode 数 > 80：

按“每个 block 覆盖全部 80 个 seed 后重新固定洗牌”的方式继续。

如果训练 episode 数 < 80：

仍按预先生成的 schedule 使用前 N 个，不根据结果换 seed。

三个 PPO seed：

```text
0
1
2
```

必须使用完全相同的 weather-seed schedule。

保存：

```text
training_weather_schedule_random.csv
```

并记录每个 episode 实际注入 DSSAT PDI 的：

```text
rseed1_
```

---

# 八、evaluation weather 必须与训练隔离

随机天气 held-out evaluation：

```text
weather_seed = 1081–1100
```

共：

```text
20 unseen WGEN weather realizations
```

这些 seed：

```text
不得出现在 random-weather training schedule
```

---

# 九、第二套 evaluation：真实历史 validation weather

同时使用：

```text
2014–2023 observed YC weather
```

作为第二套 evaluation context。

重要：

此前审计注明：

```text
2014–2023 仅作比较；
既有审计未认证为全项目 pristine holdout
```

因此本轮报告必须继续使用：

```text
independent comparison period
```

或：

```text
held-out-from-WGEN-fitting observed period
```

不要未经证据把它升级描述成：

```text
pristine final test set
```

---

# 十、evaluation 必须完全一致

每个训练后的 PPO model 都评估：

```text
A. 20 held-out synthetic weather seeds: 1081–1100
B. observed weather years: 2014–2023
```

所以：

```text
6 trained models
×
same evaluation set
```

评估时：

```text
deterministic policy action = TRUE
```

除非项目现有正式评估协议明确要求其他方式；如有冲突，沿用既有 protocol 并记录。

所有模型必须：

```text
same initial management configuration
same action interpretation
same evaluation code
same metrics
```

---

# 十一、不要混淆“训练天气 seed”和“评估天气 seed”

结果目录和表格中必须显式包含：

```text
training_regime
ppo_seed
training_weather_source
training_weather_schedule_id

evaluation_weather_type
evaluation_weather_seed_or_year
```

禁止只记录一个：

```text
seed
```

字段。

---

# 十二、工程 smoke

在正式 6 个模型训练前，先做一个极短 smoke：

```text
RANDOM_WEATHER
ppo_seed = 0
```

只用于验证：

```text
episode reset 时 weather seed 确实按 schedule 切换
ppo_seed 没有覆盖 weather_seed
weather_seed 没有覆盖 ppo_seed
training loop 正常
model checkpoint 正常
```

smoke 不进入性能统计。

如果 smoke 失败：

```text
STOP
```

只修 seed plumbing / logging / obvious runtime bug。

不得趁机改 PPO 超参数。

---

# 十三、训练 budget 公平性

Historical 与 Random 两组必须完全一致：

```text
total_timesteps
checkpoint interval
evaluation interval
PPO hyperparameters
hardware/runtime
```

如 wall-clock 不同无所谓。

公平性以：

```text
same training timesteps / same update budget
```

为准。

---

# 十四、需要保存的训练证据

每个 model 保存：

```text
model checkpoint
training config
ppo_seed
weather schedule
training log
episode reward
episode yield if available
episode fertilizer
episode irrigation
episode length
weather seed / historical year per episode
```

不得只保存最终 model。

---

# 十五、评估核心指标

### Primary decision metric

使用项目现有 PPO objective 对应的：

```text
episode return / final reward
```

作为 pilot 的首要比较指标。

由于 reward 为：

```text
0.06 * final_grnwt - 0.04 * cumfert
```

必须同时拆开报告：

```text
yield
fertilizer
```

不能只看 reward。

### Secondary metrics

至少：

```text
yield
cumulative fertilizer
total irrigation
episode length
```

### NUE

只有项目已有明确、固定的 NUE 定义和计算链时才报告。

必须：

```text
reuse existing project NUE definition exactly
```

如果仓库当前没有唯一 canonical NUE definition：

```text
NUE = NOT_REPORTED
```

不要在本轮自己新建定义。

同理 WUE 等指标也不得临时发明统计口径。

---

# 十六、稳健性指标

本轮 random-weather augmentation 的核心不是只看平均值。

在 20 个 held-out WGEN evaluation seeds 上，对每个 model 计算：

```text
mean reward
median reward
SD reward
CV reward if meaningful
P10 reward
minimum reward

mean yield
P10 yield

mean fertilizer
P90 fertilizer

mean irrigation
```

核心观察：

```text
average performance
lower-tail performance
weather-to-weather variability
```

---

# 十七、paired comparison

对每个：

```text
ppo_seed = 0, 1, 2
```

建立 paired comparison：

```text
Historical PPO seed 0
vs
Random-weather PPO seed 0

Historical PPO seed 1
vs
Random-weather PPO seed 1

Historical PPO seed 2
vs
Random-weather PPO seed 2
```

因为两个模型使用：

```text
same PPO seed
same PPO config
same training budget
```

仅训练天气 regime 不同。

评估时，又在完全相同的天气集合上比较。

---

# 十八、pilot 预定义决策规则

本轮只是 pilot，不作为最终论文性能结论。

定义：

## `GO_SIGNAL`

同时满足：

```text
1. 在 held-out synthetic weather 上：
   random-weather PPO 的 mean reward 相对 historical control
   在至少 2/3 PPO seeds 上为正改善；

2. 三个 PPO seed 汇总后的 pooled mean reward difference > 0；

3. random-weather PPO 的 pooled mean yield
   不比 historical control 下降超过 2%；

4. random-weather PPO 的 pooled mean fertilizer
   不比 historical control 增加超过 5%；

5. held-out synthetic weather 的 P10 reward
   不比 historical control 恶化超过 5%；

6. observed 2014–2023 comparison 上
   不出现明显一致性退化：
   即三个 PPO seeds 中不能 3/3 都表现更差。
```

这些阈值仅作为：

```text
pilot go/no-go operational criteria
```

不是最终科学结论阈值。

---

## `MIXED_SIGNAL`

例如：

```text
平均 reward 改善但只有 1/3 seed 改善
或
平均 reward 改善但 yield / fertilizer guardrail 触发
或
synthetic eval 改善但 observed comparison 明显下降
```

此时：

```text
不要调参
不要挑最好 seed
```

下一步应先分析：

```text
为什么不同 PPO seeds 结果不同
```

---

## `NO_SIGNAL`

例如：

```text
random-weather PPO 在 0/3 或 1/3 PPO seeds 上改善
且 pooled mean reward <= historical control
```

此时结论只能是：

> 当前 pilot 没有支持“天气增强本身改善 YC PPO”的假设。

不能自动改 PPO 算法。

---

# 十九、不允许挑 seed

必须展示：

```text
ppo_seed 0
ppo_seed 1
ppo_seed 2
```

全部结果。

即使：

```text
一个特别好
一个特别差
```

都要保留。

禁止：

```text
best seed selection
post-hoc seed replacement
只汇报成功 seed
```

---

# 二十、可选 contextual baseline

如果仓库中已有稳定、无需新增实现的：

```text
Expert
Farmer
DSSAT automatic
No-management
```

evaluation pipeline，可以在同一评价天气集合上运行作为背景参照。

但：

```text
不是本轮必需 gate
```

不要因此扩大任务或修改这些 baseline。

本轮核心仍是：

```text
Historical-weather PPO
vs
Random-weather PPO
```

---

# 二十一、统计分析

由于只有：

```text
3 PPO seeds
```

不要做夸大的显著性结论。

优先报告：

```text
per-seed values
paired differences
mean
median
range
```

在 20 weather realizations 内可以做 weather-level paired summaries。

允许：

```text
bootstrap CI
paired descriptive difference
```

但不要用：

```text
p < 0.05
```

作为 pilot 是否成功的唯一依据。

---

# 二十二、结果目录

```text
results/yc_random_weather_ppo/004_03/
```

建议：

```text
config/
    canonical_yc_ppo_config.json
    experiment_design.json
    training_weather_schedule_historical.csv
    training_weather_schedule_random.csv
    evaluation_manifest.csv

smoke/

training/
    historical/
        ppo_seed_0/
        ppo_seed_1/
        ppo_seed_2/
    random_weather/
        ppo_seed_0/
        ppo_seed_1/
        ppo_seed_2/

evaluation/
    heldout_wgen/
    observed_2014_2023/

analysis/
figures/
experiment_log.md
summary.json
```

---

# 二十三、至少输出的机器可读结果

```text
training_run_manifest.csv
training_episode_summary.csv

evaluation_episode_level.csv

heldout_wgen_model_summary.csv
observed_weather_model_summary.csv

paired_ppo_seed_comparison.csv
pooled_regime_comparison.csv

pilot_decision.json
```

`evaluation_episode_level.csv` 至少包含：

```text
training_regime
ppo_seed
model_path

evaluation_weather_type
evaluation_weather_seed
evaluation_weather_year

reward
yield
fertilizer
irrigation
episode_days

NUE  # only if canonical definition exists
```

---

# 二十四、图

至少：

```text
1. held-out WGEN reward by model
2. held-out WGEN yield by model
3. held-out WGEN fertilizer by model
4. paired reward difference by PPO seed
5. reward distribution across 20 held-out weather seeds
6. observed 2014–2023 reward comparison
```

不制作 PPT。

---

# 二十五、中文报告

生成：

```text
docs/yc_random_weather_ppo_controlled_pilot.md
```

结构至少：

```text
1. Hypothesis
2. Why this is a controlled pilot
3. Frozen PPO configuration
4. Seed separation
5. Historical training regime
6. Random-weather training regime
7. Held-out weather design
8. Training completion
9. Held-out WGEN results
10. Observed 2014–2023 results
11. Yield/fertilizer/irrigation decomposition
12. Robustness / lower-tail analysis
13. PPO-seed variability
14. Pilot decision
15. What the result does and does not prove
16. Next step
17. Files changed
18. Tests
19. Git status
```

---

# 二十六、解释边界

如果 random-weather PPO 更好：

只能写：

> This controlled pilot provides preliminary support for the hypothesis that increasing weather diversity during training improves YC PPO performance/robustness under the tested conditions.

不能写：

```text
weather diversity is proven to be the cause of all previous YC instability
```

如果没有改善：

只能写：

> This pilot did not provide evidence that WGEN-based weather augmentation alone improves YC PPO under the current configuration.

不能自动推断：

```text
PPO algorithm is defective
```

---

# 二十七、运行环境原则

允许修改：

```text
repo-level training/evaluation scripts
config files
logging
seed plumbing
```

仅为支持：

```text
explicit weather schedule
explicit ppo seed separation
controlled evaluation
```

禁止：

```text
pip install
apt install
DSSAT rebuild
gym-dssat-pdi reinstall
binary patch
runtime replacement
```

如发现必须改 installed runtime 才能完成：

```text
STOP
```

报告 blocker，不擅自改。

---

# 二十八、测试

至少检查：

```text
ppo_seed independent from weather_seed
historical schedule independent from ppo_seed
random weather schedule independent from ppo_seed

training weather seeds = only 1001–1080
evaluation WGEN seeds = only 1081–1100
train/eval WGEN seed intersection = empty

same PPO config across regimes
same total_timesteps across regimes

all 6 model runs completed
all expected evaluation rows present

reward decomposition internally consistent
no post-hoc seed exclusion
```

---

# 二十九、Git

local commit：

```text
experiment: run YC random-weather PPO controlled pilot
```

只 stage 本任务相关文件。

工作区已有无关修改保持原样。

未经用户明确批准：

```text
git push = NO
```

---

# 三十、最终终端摘要

必须输出：

```text
=== YC RANDOM-WEATHER PPO CONTROLLED PILOT SUMMARY ===

hypothesis:
weather_augmentation_only_change: YES

canonical_training_script:
total_timesteps:
ppo_hyperparameters_identical: YES

ppo_seeds:
0,1,2

historical_training_period:
2004-2013

random_training_weather_seed_pool:
1001-1080

heldout_wgen_evaluation_seeds:
1081-1100

observed_comparison_period:
2014-2023

ppo_weather_seed_separated:
historical_schedule_separated_from_ppo_seed:
train_eval_weather_seed_overlap:

smoke_status:

historical_models_completed:
random_weather_models_completed:

heldout_wgen_historical_regime_mean_reward:
heldout_wgen_random_regime_mean_reward:
heldout_wgen_pooled_reward_difference_pct:

heldout_wgen_historical_regime_mean_yield:
heldout_wgen_random_regime_mean_yield:
heldout_wgen_pooled_yield_difference_pct:

heldout_wgen_historical_regime_mean_fertilizer:
heldout_wgen_random_regime_mean_fertilizer:
heldout_wgen_pooled_fertilizer_difference_pct:

heldout_wgen_historical_regime_P10_reward:
heldout_wgen_random_regime_P10_reward:
heldout_wgen_P10_reward_difference_pct:

ppo_seed_0_reward_direction:
ppo_seed_1_reward_direction:
ppo_seed_2_reward_direction:
ppo_seeds_with_positive_reward_change:

observed_2014_2023_direction_by_ppo_seed:

pilot_decision:
GO_SIGNAL / MIXED_SIGNAL / NO_SIGNAL

random_weather_training_ready_for_larger_experiment:
YES / NO / NEEDS_DIAGNOSIS

full_year_wgen_validation:
DEFERRED
full_year_wgen_is_blocker:
NO

PPO_algorithm_modified:
NO
reward_modified:
NO
runtime_modified:
NO
CNYC_CLI_modified:
NO
WGEN_refit:
NO
seed_cherry_picking:
NO
ppt_created:
NO

recommended_next_step:

report_md:
results_directory:

tests_status:
git_commit:
git_push: NO
github_backup_status:
```
