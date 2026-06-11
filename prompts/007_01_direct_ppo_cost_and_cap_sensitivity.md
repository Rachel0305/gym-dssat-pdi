# 007_01_direct_ppo_cost_and_cap_sensitivity

请阅读：

```text
AGENTS.md

TASK_LEAVE_ONE_YEAR_STRATEGY.md

prompts/006_17_all_year_direct_action_safe_ppo_simple_baseline.md

docs/2026-06-06_all_year_direct_action_safe_ppo_report.md
```

如果文件日期不同，请搜索：

```text
all_year_direct_action_safe_ppo_report
ppo_direct_eval_summary
ppo_decision_reasonableness_diagnosis
```

---

# 一、任务背景

006_17 已完成。

当前结果表明：

```text
5/5 station PPO models training successful

45/45 evaluations successful

四联图生成成功
```

但同时：

```text
45/45 cases 被标记为 cap_saturated
```

主要表现：

```text
total_irrigation ≈ season_irrigation_cap

total_n ≈ season_n_cap
```

说明：

```text
当前 reward + action scale 约束
不足以诱导 PPO 学习节水节氮策略。
```

因此：

```text
006_17 只能作为 baseline，
不能作为最终展示策略。
```

---

# 二、本阶段目标

本阶段目标：

```text
寻找一组：

water_cost
nitrogen_cost
season_irrigation_cap
season_n_cap

使 PPO：

1. 不再打满 cap；
2. 不退化为全 0 动作；
3. 仍保持合理 TOPWT / GRNWT；
4. 动作出现在胁迫发生前后；
5. 能作为导师展示版本。
```

---

# 三、本阶段禁止事项

不要执行：

```text
RF
Random Forest

Behavior Cloning

Imitation Learning

Offline Schedule Search

Expert Replay

Expert Dataset Augmentation

Constrained PPO Fine-Tuning

Rainfall Scaling

Reward Structure 重写

Episode-level Profit Reward

Phenology Window Action Design

Event Action Design

Multi-objective PPO
```

不要修改：

```text
my_data

weather_clean

weather_clean_qc

WTH 文件

土壤文件

品种参数文件
```

---

# 四、本阶段唯一允许修改的内容

只允许修改：

```text
water_cost

nitrogen_cost

season_irrigation_cap

season_n_cap
```

其它 PPO 配置保持不变。

---

# 五、敏感性实验设计

## A 当前基线

```text
water_cost = 0.1

nitrogen_cost = 0.25

season_irrigation_cap = 160

season_n_cap = 250
```

标签：

```text
A_baseline
```

---

## B 成本增强

```text
water_cost = 0.3

nitrogen_cost = 0.5

season_irrigation_cap = 160

season_n_cap = 250
```

标签：

```text
B_cost_high
```

---

## C 收紧 cap

```text
water_cost = 0.1

nitrogen_cost = 0.25

season_irrigation_cap = 100

season_n_cap = 180
```

标签：

```text
C_cap_low
```

---

## D 双重约束

```text
water_cost = 0.3

nitrogen_cost = 0.5

season_irrigation_cap = 100

season_n_cap = 180
```

标签：

```text
D_cost_high_cap_low
```

---

# 六、训练要求

沿用：

```text
006_17
```

中的：

```text
all-year train years

all-year eval years

seed = 0

action safety

action scaling
```

保持一致。

不要重新设计训练集。

不要重新划分年份。

---

# 七、评估输出

输出：

```text
Leave_One_experiments/
all_year_direct_ppo_cost_cap_sensitivity/
```

目录结构：

```text
configs/

models/

evaluation/

figures/

reports/
```

---

# 八、生成敏感性分析表

输出：

```text
ppo_cost_cap_sensitivity_summary.csv
```

字段至少包括：

```text
scenario

station_code

year

total_irrigation

total_n

final_topwt

final_grnwt

mean_swfac

max_swfac

mean_nstres

max_nstres

profit_simple

irrigation_event_count

n_event_count

decision_reasonableness_label
```

---

# 九、生成关键比较图

至少生成：

```text
1.
total_irrigation comparison

2.
total_n comparison

3.
final_grnwt comparison

4.
profit_simple comparison

5.
swfac stress days comparison

6.
nstres stress days comparison

7.
cap saturation rate comparison
```

---

# 十、定义优良策略标准

如果满足：

```text
不是 cap_saturated

total_irrigation < 0.8 × cap

total_n < 0.8 × cap

final_grnwt
不低于 baseline 的 90%

TOPWT 正常增长

动作集中于
SWFAC 或 NSTRES 升高前后

不存在明显异常年份
```

则标记：

```text
recommended
```

否则：

```text
not_recommended
```

---

# 十一、推荐导师展示版本

请最终输出：

```text
recommended_configuration.csv
```

内容包括：

```text
最佳参数组合

推荐原因

优点

缺点

是否适合作为论文主结果
```

---

# 十二、方法来源与文献支撑

Markdown 报告和 PPT 必须新增：

```text
方法来源与文献支撑
```

章节。

至少引用并说明：

```text
Gautron et al. 2022
gym-DSSAT

Wu et al. 2022
Optimizing Nitrogen Management with Deep Reinforcement Learning and Crop Simulations

Tao et al. 2022
Optimizing Crop Management with Reinforcement Learning and Imitation Learning

Kallenberg et al. 2023
Nitrogen Management with Reinforcement Learning and Crop Growth Models

Saikai et al. 2023
Deep Reinforcement Learning for Irrigation Scheduling
```

必须明确区分：

```text
哪些属于文献支持的方法；

哪些属于本研究的实验结果。
```

---

# 十三、报告输出

生成：

```text
docs/
2026-06-XX_direct_ppo_cost_and_cap_sensitivity_report.md
```

以及：

```text
docs/
2026-06-XX_direct_ppo_cost_and_cap_sensitivity_report.pptx
```

并同步保存到：

```text
Leave_One_experiments/
all_year_direct_ppo_cost_cap_sensitivity/
reports/
```

---

# 十四、完成后请汇报

请重点回答：

```text
1.
哪组参数最能避免 cap_saturated？

2.
哪组参数获得最高 final_grnwt？

3.
哪组参数获得最高 profit_simple？

4.
哪组参数最符合农学逻辑？

5.
哪组参数最适合作为导师展示版本？

6.
下一步是否还需要
phenology window action design？
```
