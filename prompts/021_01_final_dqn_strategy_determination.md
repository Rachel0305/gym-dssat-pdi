# 021_01 Final DQN Strategy Determination

## 任务目标

基于现有五站点实验结果，直接开展最小必要验证，确定一套可用于五站点统一比较的 DQN 水氮管理配置。

当前主问题是：

> 为什么不同站点的 DQN 表现不同？如何用同一套 DQN 框架获得稳定、合理、兼顾产量、节水、节氮和稳定性的管理策略？

本任务不是继续搭框架，不是继续整理文件，也不是做大规模全组合敏感性分析。

## 当前统一参考配置

以当前冻结配置作为对照：

```text
Algorithm: DQN
n_steps: 5

Action:
I = [0, 15, 30] mm
N = [0, 50, 100] kg/ha

Budget:
I <= 120 mm
N <= 300 kg/ha

Daily cap:
I <= 30 mm
N <= 100 kg/ha

Decision interval:
7 DAP

Reward:
terminal yield gain relative to local null
- 1.0 * irrigation
- 5.0 * nitrogen
```

不得擅自修改 IC，不得修改 DSSAT 输入来制造优化空间，不得加入氮淋洗惩罚。

# 1. HLA

HLA 作为当前成功案例，不做大规模调参。

执行：

1. 复用已有正式结果。
2. 不重复训练已有完整结果。
3. 统一整理以下指标：
   - yield
   - irrigation
   - nitrogen
   - reward
   - WUE
   - NUE（若数据可用）
   - 跨年份稳定性
   - 跨 seed 稳定性
4. 比较：
   - null
   - recorded
   - DSSAT auto
   - official expert
   - DQN
5. 若证据充分，标记：

```text
status = keep_current_config
```

# 2. YC

YC 当前主要问题：

```text
DQN 接近 DSSAT auto，但低于 recorded / official expert；
DQN 经常选择 N = 0。
```

执行最小必要诊断：

```text
nitrogen_cost = [2, 5, 8]
```

要求：

1. 使用 YC2014。
2. 其他设置全部保持冻结配置不变。
3. 先运行 seed0、5K steps smoke test。
4. smoke 正常后运行 seed0/1/2、50K steps。
5. 每 5K 保存 checkpoint。
6. 输出：
   - yield
   - irrigation
   - nitrogen
   - reward
   - WUE
   - NUE（若数据可用）
   - 非零施氮次数
   - budget utilization
   - 与 null / recorded / DSSAT auto / official expert 的差值
7. 判断：
   - nitrogen_cost=5 是否导致过度节氮；
   - nitrogen_cost 降低后是否增加合理施氮；
   - 是否提高产量；
   - 是否仍保持氮投入低于 expert；
   - 若 cost 改变后策略仍为 N=0，记录该结果，不要擅自修改 IC 或 action。

# 3. FQ

FQ 当前主要问题：

```text
DQN 产量高于 official expert、接近 DSSAT auto，
但灌溉量高于 DSSAT auto。
```

执行最小必要诊断：

```text
water_cost = [0.5, 1.0, 2.0]
```

要求：

1. 使用 FQ2016。
2. 其他设置全部保持冻结配置不变。
3. 先运行 seed0、5K steps smoke test。
4. smoke 正常后运行 seed0/1/2、50K steps。
5. 每 5K 保存 checkpoint。
6. 输出：
   - yield
   - irrigation
   - nitrogen
   - reward
   - WUE
   - 灌溉次数
   - budget utilization
   - 与 DSSAT auto 的产量差和灌溉差
7. 判断：
   - water_cost=1 是否过低；
   - 提高 water_cost 后能否减少灌溉；
   - 节水是否造成明显减产；
   - 是否存在相近产量但显著减少灌溉的设置。

# 4. LC

LC 当前主要问题：

```text
不同 seed 在相近产量下出现完全不同的施氮量。
```

这里先检查稳定性，不调 reward。

执行：

1. 使用 LC2010 已修复正式输入。
2. 使用冻结 n_steps=5 配置。
3. 运行 seed0/1/2。
4. 每个 seed 50K steps。
5. 每 5K 保存 checkpoint。
6. 使用统一 checkpoint 选择规则。
7. 输出：
   - yield
   - irrigation
   - nitrogen
   - reward
   - action frequency
   - budget utilization
   - seed mean ± SD
8. 判断：
   - 是否仍存在同产量、不同资源投入；
   - 是否存在某个 seed 打满 N300；
   - 是否属于训练不稳定；
   - 是否需要后续再做 nitrogen_cost 诊断。

不要在本任务中直接改 reward。

# 5. SY

SY 当前主要问题：

```text
IC provenance 存在冲突，历史文档与实际训练输入不一致。
```

执行：

1. 不启动正式长训练。
2. 查明 SY2014 实际使用的：
   - IC level
   - MZX
   - soil
   - weather
   - cultivar
   - treatment
3. 对比：
   - 文档记录
   - 源码
   - 落盘输入
   - 实际运行文件
4. 输出权威输入链路。
5. 若无法确认，标记：

```text
status = blocked_input_provenance
```

6. 只有确认输入后，才生成后续正式训练配置。

不得为了增加优化空间修改 IC。

# 6. 统一配置确定规则

本任务完成后，根据 HLA、YC、FQ、LC、SY 结果，生成候选统一配置，但不得按站点分别使用不同 reward。

优先判断以下参数：

```text
action_space
water_cost
nitrogen_cost
irrigation_budget
nitrogen_budget
decision_interval
n_steps
checkpoint_selection_rule
```

生成：

```text
configs/final_dqn_candidate.yaml
```

要求：

1. 只能使用一个统一配置。
2. 每个参数必须注明证据来源。
3. 若证据不足，标记 `pending`，不要擅自定值。
4. 不允许为了让五站点结果更漂亮而按站点分别调参。
5. action space 暂不进行 9/16/25 大扫描，除非 YC/FQ/LC 诊断明确表明当前动作粒度是主要限制。
6. 氮淋洗惩罚保持关闭。

# 7. 结果文件

至少生成：

```text
benchmark_results/021_01/
station_diagnosis.csv
yc_nitrogen_cost_summary.csv
fq_water_cost_summary.csv
lc_seed_stability_summary.csv
sy_input_provenance.csv
unified_parameter_evidence.csv
final_dqn_candidate.yaml
```

# 8. 图表

至少生成：

1. 五站点当前表现总览。
2. YC nitrogen cost 对 yield / N input 的影响。
3. FQ water cost 对 yield / irrigation 的影响。
4. LC 跨 seed 产量与资源投入。
5. 候选统一配置证据图。

所有图：

```text
PNG 300 dpi
SVG
```

# 9. 实验记录

生成：

```text
docs/YYYY-MM-DD_021_01_final_dqn_strategy_determination.md
docs/YYYY-MM-DD_021_01_final_dqn_strategy_determination.pptx
```

Markdown 至少包含：

1. 任务目标
2. 当前统一配置
3. 五站点现状
4. HLA 结论
5. YC nitrogen cost 结果
6. FQ water cost 结果
7. LC seed 稳定性结果
8. SY input provenance
9. 候选统一配置
10. 未解决问题
11. 下一步正式五站点验证
12. Methods Source
13. References
14. Git 状态

PPT 至少包含：

1. 当前问题
2. 五站点差异
3. YC 结果
4. FQ 结果
5. LC 结果
6. SY 输入状态
7. 候选统一配置
8. 下一步

# 10. 禁止事项

1. 不继续重构 Benchmark Framework。
2. 不做 81 组全组合搜索。
3. 不修改 IC 制造优化空间。
4. 不修改 DSSAT 输入追求更高产量。
5. 不加入氮淋洗惩罚。
6. 不按站点分别设置正式 reward。
7. 不重复训练已有完整结果。
8. 不只生成代码而不运行诊断实验。
9. 不把 smoke test 当正式结果。
10. 不伪造 NUE、WUE 或输入来源。
11. 不让 Codex 自行扩大实验范围。

# 11. Git

完成后：

```bash
git status
git diff --stat
```

只提交本任务相关：

```text
代码
配置
CSV
图表
Markdown
PPT
```

不提交：

```text
大型模型
replay buffer
DSSAT 临时目录
cache
无关文件
```

提交信息：

```text
Diagnose and determine unified DQN strategy
```

然后尝试：

```bash
git push
```

将 commit hash 和 push 状态写入实验记录。

# 12. 完成标准

只有以下内容完成后，任务才能标记为 completed：

1. HLA 已有结果完成统一评价。
2. YC nitrogen_cost 诊断完成。
3. FQ water_cost 诊断完成。
4. LC seed0/1/2 冻结正式复核完成。
5. SY input provenance 已确认或明确 blocked。
6. 已生成统一参数证据表。
7. 已生成 `final_dqn_candidate.yaml`。
8. 已生成 Markdown。
9. 已生成 PPT。
10. 已完成 Git commit。
11. 已尝试 Git push。

若有部分未完成，标记：

```text
partial
```

并明确列出未完成项，不得伪造完成。

# 13. 最终终端输出

打印：

```text
021_01 Final DQN Strategy Determination Completed

HLA:
...

YC:
...

FQ:
...

LC:
...

SY:
...

Unified candidate:
...

Generated files:
...

Failed or blocked items:
...

Git commit:
...

Git push:
...
```
