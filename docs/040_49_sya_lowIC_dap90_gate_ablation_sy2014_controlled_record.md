# 040_49_sya_lowIC_dap90_gate_ablation_sy2014_controlled 记录

## 任务目的

修正040_48的对照缺陷：040_48把“多年训练+gate开”和“单年训练+gate关”放在一起比，
混入了训练年份范围这个额外变量，无法单独归因于gate。本任务改为同时训练两个模型，
两者都只在SY2014一年上训练、其余奖励/档位/超参数/seed完全一致，唯一区别是
DAP<=90/195mm这条gate开还是关，从而干净地检验gate本身的影响。

## 边界

- 两个模型都只用SY2014一年训练，seed相同，只有gate开关不同。
- 不改reward、不改离散动作档位、不改PPO超参数。
- 不涉及DAP<=30/DAP<=60分段配额（040_26源码未核实，本任务不动）。
- 040_48的单臂结果因存在训练年份混淆，不作为有效结论，本任务取代它。
- 这是对动作合法性mask的结构性修改，按项目预注册习惯应视为新的预注册任务。

## 结论先说

- 训练步数：`100000`（两个arm相同）。
- control（gate开）灌溉序列：`DAP1:I45; DAP8:I30; DAP31:I30; DAP38:I30; DAP61:I45; DAP91:I45`
- treatment（gate关）灌溉序列：`DAP1:I45; DAP8:I30`
- 两者完全相同：`False`
- control终产量：`10895.26123046875` kg/ha；treatment终产量：`7635.7708740234375` kg/ha
- control总灌溉/总施氮：`225.0` mm / `200.0` kg/ha；treatment总灌溉/总施氮：`75.0` mm / `200.0` kg/ha

## 附：040_48已作废的单臂结果（仅作背景，不可比）

- 040_40 checkpoint100000, trained on MANY years, gate ON -- not a clean comparison for this task.
- 灌溉序列：`DAP1:I45; DAP8:I30; DAP31:I45; DAP38:I30; DAP61:I45; DAP91:I45`；终产量：`10210.0989` kg/ha

## 解释边界

- 若control和treatment灌溉序列相同，说明在“只训练单年”这个前提下gate本身
  不改变时机选择，嫌疑应转回policy本身或reward对时机的敏感度。
- 若两者不同，说明gate至少在单年单seed设定下是有影响的成因之一；仍不能直接推广到
  多年多seed的正式训练设置，需要另开预注册任务在多年多seed下重复验证。
- control本身的表现（相对于040_44多年训练的checkpoint）也提供了额外信息：
  如果control（单年训练+gate开）本身就明显弱于040_44多年训练的checkpoint，说明
  单年训练本身确实会显著拖累效果——这正是040_48暴露出的混淆来源，在这里可以直接量化。
