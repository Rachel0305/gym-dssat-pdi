# 004_05：YC 多 PPO seed 增量实验——检验天气增强是否改变策略 archetype 出现概率

## 一、任务定位

前序结果：

- `004_03`：Historical-weather PPO 与 Random-weather WGEN PPO 各用 `ppo_seed=0,1,2`。held-out WGEN pooled reward 从 0.830541 提高到 0.907329（+9.2456%），但逐 seed 方向为 `+66.65% / -3.93% / -13.15%`，因此判定 `MIXED_SIGNAL`。
- `004_04`：行为诊断确认 `H1` 与 `W0` 为 `IDENTICAL_POLICY_BEHAVIOR`。说明 W0 的低投入策略并不是 random-weather training 独有。

当前真正需要检验：

> Random-weather training 是否改变 PPO 收敛到不同管理策略 archetype 的概率，并提高“跨天气表现较好策略”的出现比例？

---

## 二、本轮目标

将 PPO seed 从：

```text
0–2
```

扩展到：

```text
0–7
```

每种训练天气 regime 最终共 8 个 PPO seeds：

```text
Historical-weather PPO: seed 0–7
Random-weather WGEN PPO: seed 0–7
```

总共 16 个模型。

已有并必须复用：

```text
H0,H1,H2
W0,W1,W2
```

只新增训练：

```text
Historical seed 3–7 = 5 models
Random-weather seed 3–7 = 5 models
total new models = 10
```

禁止重训 seed 0–2，除非 verified checkpoint 损坏或不可加载；若发生则 STOP 并报告。

---

## 三、预注册研究问题

### H1：行为分布假设

> Random-weather training 会改变不同 policy archetype 的出现频率。

### H2：性能概率假设

> Random-weather training 可能提高得到“跨天气表现较好策略”的概率。

注意：

```text
H1 成立 ≠ H2 必然成立
```

---

## 四、冻结 canonical 配置

继续沿用 `004_03 / 004_04` 已确认的正式 YC canonical：

```text
training script:
src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py
```

正式训练预算：

```text
requested total_timesteps = 100000
actual expected ≈ 100080
```

算法：

```text
MaskablePPO / MlpPolicy
```

沿用已冻结：

```text
reward
policy architecture
learning_rate
gamma
gae_lambda
n_steps
batch_size
n_epochs
clip_range
ent_coef
vf_coef
action space
action mask rules
observation space
crop
soil
cultivar
management constraints
```

旧 reward：

```text
0.06 * final_grnwt - 0.04 * cumfert
```

继续标记：

```text
SUPERSEDED / NOT_APPLICABLE
```

不得恢复。

---

## 五、唯一主要实验因素

两组除训练天气来源外必须完全一致。

### Regime H

```text
HISTORICAL_WEATHER
random_weather = False
historical period = 2004–2013
```

### Regime W

```text
RANDOM_WEATHER_WGEN
random_weather = True
WTHER = W
frozen CNYC.CLI
```

天气增强是唯一主要实验因素。

---

## 六、PPO seeds

已有：

```text
0,1,2
```

新增：

```text
3,4,5,6,7
```

不得更换 seed，不得因表现差而改用其他 seed。

---

## 七、训练天气 schedule 完全沿用 004_03

### Historical group

继续使用：

```text
historical year schedule seed = 64003
```

2004–2013 按 `004_03` 相同逻辑生成固定年份序列。

所有 PPO seeds 0–7 使用相同 historical-year sequence。

### Random-weather group

继续使用：

```text
weather schedule seed = 64004
weather seed pool = 1001–1080
```

沿用 `004_03` 相同 schedule。

所有 PPO seeds 0–7 使用相同 weather-seed sequence。

继续核对实际 DSSAT PDI：

```text
rseed1_
```

与预定 schedule 一致。

---

## 八、evaluation sets 不变

所有 16 个模型使用相同 evaluation。

### Held-out WGEN

```text
weather_seed = 1081–1100
20 realizations
```

### Observed comparison

```text
2014–2023
10 observed years
```

继续称：

```text
independent comparison period
```

不要称 pristine final test set。

evaluation：

```text
deterministic=True
```

---

## 九、复用已有结果

已有 seed 0–2 的 verified checkpoints 与 evaluation 结果必须复用。

最终汇总前检查：

```text
checkpoint SHA256
canonical config
evaluation protocol
weather schedules
```

一致则：

```text
REUSE
```

不要为了统一时间重训。

---

## 十、资源保护

新增 10 个模型必须：

```text
串行训练
```

记录每个模型：

```text
wall time
peak RSS
requested steps
actual steps
completed episodes
checkpoint paths
```

若超既有 memory guard：

```text
安全停止
```

不要为节省内存改模型参数后继续。

---

## 十一、训练前 preflight

不重复长 smoke。

只做轻量检查：

```text
seed 3 historical:
WTHER=M
ppo_seed=3

seed 3 random:
WTHER=W
ppo_seed=3
weather schedule 与 ppo_seed 分离
```

通过后正式训练。

---

## 十二、训练 QC

新增 10 个模型全部检查：

```text
correct PPO seed
correct weather regime
correct weather schedule
correct canonical reward
correct action mask
correct obs/action definitions
requested total_timesteps = 100000
actual steps consistent with rollout boundary
checkpoints loadable
no NaN
no abnormal termination
runtime unchanged
```

若明确工程故障，可同 seed retry，但：

```text
不得换 seed
不得改超参数
不得改天气 schedule
不得改 reward
```

所有 retry 留档。

---

## 十三、archetype 分类原则

非常重要：

```text
archetype classification ≠ performance ranking
```

不能用 reward 高低定义 archetype。

archetype 必须基于 policy behavior。

---

## 十四、复用 004_04 的 fixed same-state probe

`004_04` 已冻结：

```text
600 policy probe states
probe sampling seed = 404006
```

必须直接复用同一份：

```text
policy_probe_states.csv
```

不要重抽。

对新增 10 个模型，在相同：

```text
state
action mask
```

上做 deterministic action prediction，不 step DSSAT。

---

## 十五、参考 policy prototypes

已有行为 prototypes：

```text
H0 = 高灌溉高氮
H1/W0 = 低季节投入、高 no-op
H2 = 中等/混合投入、高 no-op
W1 = 低灌溉高氮、高 no-op
W2 = 低灌溉高氮、高 no-op（投入更高）
```

不要强迫新模型属于旧类别。

---

## 十六、same-state prototype similarity

每个新模型与：

```text
H0
H1/W0 prototype
H2
W1
W2
```

比较 600 probe states 的：

```text
overall action agreement
irrigation-component agreement
fertilizer-component agreement
```

预注册匹配规则：

```text
如果 top overall agreement >= 0.95
且 top1 - top2 >= 0.03
→ prototype_match = top prototype

如果 top >= 0.95
但 top1 - top2 < 0.03
→ AMBIGUOUS_EXISTING_ARCHETYPE

如果 all < 0.95
→ NEW_OR_OTHER_ARCHETYPE
```

不得根据 performance 改阈值。

---

## 十七、rollout behavior 交叉验证 archetype

每个模型从 evaluation rollouts 计算：

```text
mean seasonal irrigation
mean seasonal fertilizer
irrigation event count
fertilizer event count
no-op fraction
management timing distribution
```

如果 same-state prototype match 与 rollout behavior 明显矛盾：

```text
REVIEW_REQUIRED
```

不要人工硬归类。

允许描述性标签：

```text
HIGH_INPUT
VERY_LOW_INPUT
MODERATE_MIXED
LOW_I_HIGH_N
OTHER_NEW
AMBIGUOUS
```

标签必须来自 same-state probe + rollout behavior，不得来自 reward。

---

## 十八、核心分析 1：archetype frequency

分别统计：

```text
Historical seeds 0–7
Random-weather seeds 0–7
```

每个 archetype：

```text
count
proportion
```

由于：

```text
n=8/group
```

只做描述性频率和不确定性展示，不夸大显著性。

---

## 十九、核心分析 2：paired seed transition

对每个：

```text
ppo_seed = 0…7
```

记录：

```text
Historical archetype
→
Random-weather archetype
```

保存：

```text
paired_seed_archetype_transition.csv
```

直接回答：

> 同一个 PPO seed，加入 random-weather 后是否更容易切换到不同 policy archetype？

---

## 二十、性能评估

每个模型分别报告：

### Held-out WGEN

```text
mean reward
median reward
P10 reward
minimum reward
mean yield
P10 yield
mean irrigation
mean fertilizer
SWFAC penalty
canonical reward components
```

### Observed 2014–2023

同样报告。

---

## 二十一、预定义 paired-seed success criterion

为了估计“天气增强成功比例”，预先定义：

对于同一 seed 的：

```text
W_s vs H_s
```

必须同时满足：

### Held-out WGEN

```text
mean reward difference > 0
```

### Observed comparison

```text
mean reward difference >= -5%
```

### Yield guardrail

两个 evaluation domain 的 pooled mean yield：

```text
W_s 不得比 H_s 下降 > 5%
```

### Resource guardrail

```text
mean fertilizer increase <= 20%
mean irrigation increase <= 20%
```

满足全部：

```text
paired_seed_weather_augmentation_success = YES
```

否则：

```text
NO
```

这是本轮 operational criterion，不是论文最终普适标准。

---

## 二十二、success fraction

在 8 个 paired PPO seeds 中统计：

```text
success_count
success_fraction = x / 8
```

不做过度精确概率外推。

---

## 二十三、archetype 与 success 必须分开

最终分别回答：

```text
A. Weather augmentation 是否改变 archetype frequency？

B. Weather augmentation 是否提高 paired-seed success fraction？

C. 各 archetype 在 synthetic / observed 两类天气上的表现分别如何？
```

不能把：

```text
某 archetype 出现更多
```

直接等同于：

```text
天气增强有效
```

---

## 二十四、跨 archetype 性能分析

按 archetype 汇总：

```text
held-out WGEN reward
observed reward
yield
N
I
SWFAC penalty
```

重点检查：

```text
VERY_LOW_INPUT
```

是否仍表现出：

```text
WGEN 上较有利
observed 某些年份风险更高
```

不要仅从 H1/W0 外推。

---

## 二十五、禁止 cherry-pick

必须包含：

```text
all seeds 0–7
```

禁止：

```text
排除差 seed
更换 seed
只汇报成功模型
只汇报某个 archetype
```

工程失败只允许同 seed 修复后 retry。

---

## 二十六、最终判定框架

### Finding 1：archetype distribution

```text
CLEAR_SHIFT
POSSIBLE_SHIFT
NO_CLEAR_SHIFT
```

解释规则：

```text
CLEAR_SHIFT:
有可解释的频率变化，并与 paired transition 方向一致

POSSIBLE_SHIFT:
有变化，但 n=8 仍不足或 NEW/OTHER 较多

NO_CLEAR_SHIFT:
两组分布高度相似或无稳定方向
```

不要机械依赖 p-value。

### Finding 2：paired-seed performance

统计：

```text
success_fraction = x/8
```

预注册 operational rule：

```text
CONSISTENT_POSITIVE_SIGNAL:
success >= 6/8

MIXED_SIGNAL:
success = 3/8 to 5/8

NO_POSITIVE_SIGNAL:
success <= 2/8
```

### Finding 3：下一步

若：

```text
CLEAR_SHIFT + CONSISTENT_POSITIVE_SIGNAL
```

→ 推荐正式扩大多 seed 实验。

若：

```text
CLEAR_SHIFT + MIXED_SIGNAL
```

→ 天气增强改变收敛分布，但收益仍不一致。

若：

```text
NO_CLEAR_SHIFT
```

→ 当前证据不支持明显改变“收敛到更好策略”的概率。

---

## 二十七、不改算法

无论结果：

```text
不修改 PPO
不修改 reward
不修改 action mask
不增加天气变量进 observation
不调超参数
```

本轮只回答：

> 数据增强是否改变策略收敛分布和成功比例。

---

## 二十八、004_04 已知限制

已知：

```text
WGEN step-level weather field coverage = 89.85%
```

本轮不需要重新做复杂 weather-response 诊断。

它不阻塞：

```text
多 seed 训练
policy probe
archetype classification
episode performance
reward components
resource trajectories
```

---

## 二十九、结果目录

使用：

```text
results/yc_random_weather_ppo/004_05/
```

建议：

```text
config/
training/
evaluation/
policy_probe/
archetype_analysis/
performance_analysis/
figures/
experiment_log.md
summary.json
```

已有 seed 0–2 不复制大型 checkpoint，只在 manifest 中引用原 verified artifact。

---

## 三十、至少输出

```text
all_model_manifest_0_7.csv

new_training_run_manifest.csv
new_training_qc.csv

all_evaluation_episode_level_0_7.csv
all_model_performance_summary_0_7.csv

policy_probe_actions_new_models.csv
prototype_similarity.csv
policy_archetype_assignment.csv

archetype_frequency_by_regime.csv
paired_seed_archetype_transition.csv

paired_seed_performance_comparison.csv
paired_seed_success_criterion.csv

archetype_performance_summary.csv

multi_seed_decision.json
```

---

## 三十一、图表

至少：

```text
1. archetype counts by training regime
2. paired seed archetype transitions
3. held-out WGEN reward per paired seed
4. observed reward per paired seed
5. seed-level reward difference H→W
6. archetype vs held-out reward
7. archetype vs observed reward
8. seasonal N/I by archetype
```

不制作 PPT。

---

## 三十二、中文报告

生成：

```text
docs/yc_random_weather_multi_seed_archetype_experiment.md
```

至少包含：

```text
1. Scientific question
2. Why expand to 8 seeds
3. Frozen canonical setup
4. Reused models 0–2
5. New models 3–7
6. Seed/weather separation
7. Archetype definition
8. Same-state probe method
9. Archetype assignments
10. Archetype frequency by regime
11. Paired seed transitions
12. Held-out WGEN performance
13. Observed 2014–2023 performance
14. Paired-seed success rate
15. Archetype-performance relationship
16. What weather augmentation changed
17. Whether weather augmentation appears useful
18. Limitations
19. Recommended next step
20. Files/tests/git
```

---

## 三十三、解释边界

若 random-weather success 比例较高，只能写：

> Across the eight tested PPO seeds, random-weather augmentation increased the frequency of the observed behavior pattern and produced a higher paired-seed success fraction under the predefined evaluation criteria.

不能写：

```text
random-weather augmentation universally improves PPO
```

若无明显差异：

> Across the eight tested seeds, the experiment did not provide clear evidence that WGEN weather augmentation materially changes the probability of converging to a better policy under the current setup.

---

## 三十四、测试

至少检查：

```text
existing H0-H2/W0-W2 checkpoint hashes match 004_03

only seeds 3–7 newly trained

all 10 new models completed

same canonical config

same total_timesteps

historical schedule identical across seeds
random schedule identical across seeds

ppo_seed != weather schedule seed

training random seeds only 1001–1080
evaluation WGEN only 1081–1100
intersection empty

600 fixed probe states reused exactly
probe hash unchanged

no model excluded

archetype assignment performed before performance interpretation

reward components reconcile
```

---

## 三十五、Git

本地 commit：

```text
experiment: expand YC weather augmentation to eight PPO seeds
```

只 stage 本任务相关文件。

不要 stage 大型 checkpoints（如项目既有规范不提交模型）。

不清理无关工作区。

未经用户明确批准：

```text
git push = NO
```

---

## 三十六、最终终端摘要

必须输出：

```text
=== YC WEATHER AUGMENTATION MULTI-SEED EXPERIMENT SUMMARY ===

scientific_question:
Does random-weather training change policy-archetype frequency
and increase paired-seed success probability?

ppo_seeds_final:
0-7

existing_models_reused:
6

new_models_trained:
10

historical_models_total:
8

random_weather_models_total:
8

canonical_config_identical:
YES

weather_augmentation_only_change:
YES

training_weather_seed_pool:
1001-1080

heldout_wgen_seeds:
1081-1100

observed_comparison:
2014-2023

train_eval_weather_overlap:
NONE

fixed_policy_probe_states:
600

probe_state_hash_match_004_04:
YES

archetype_counts_historical:
<counts>

archetype_counts_random_weather:
<counts>

paired_seed_archetype_transitions:
<summary>

archetype_distribution_finding:
CLEAR_SHIFT / POSSIBLE_SHIFT / NO_CLEAR_SHIFT

paired_seed_success_count:
x/8

paired_seed_success_fraction:
x.xx

paired_seed_performance_finding:
CONSISTENT_POSITIVE_SIGNAL / MIXED_SIGNAL / NO_POSITIVE_SIGNAL

heldout_wgen_mean_reward_historical:
heldout_wgen_mean_reward_random:

observed_mean_reward_historical:
observed_mean_reward_random:

heldout_mean_yield_historical:
heldout_mean_yield_random:

observed_mean_yield_historical:
observed_mean_yield_random:

weather_augmentation_interpretation:

weather_augmentation_usefulness:
SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED

recommended_next_step:

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

report_md:
results_directory:

tests_status:
git_commit:
git_push: NO
github_backup_status:
```
