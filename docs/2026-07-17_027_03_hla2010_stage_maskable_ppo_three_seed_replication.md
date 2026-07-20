# 027_03 HLA2010 阶段型 MaskablePPO 三seed复核记录

## 结论

状态：`completed / B_three_seed_primary_not_replicated`。

HLA2010 三seed工程执行全部通过，但预注册选中checkpoint只有seed0通过primary，seed1/2均未通过，计数为1/3，低于至少2/3的门槛。因此HLA2010当前不能判定为跨seed稳定成功，也不允许进入HLA站内跨年迁移。

本轮严格执行停止规则：不增加seed、不延长训练、不改reward、不重新选择checkpoint。

## 冻结实验设计

- 三个seed使用相同HLA2010输入、24维scaler、六阶段、9动作及I120/N300预算。
- MaskablePPO网络与超参数完全相同，仅随机seed不同。
- 每个seed训练240阶段步，评估0/60/120/180/240。
- checkpoint0不入选；60/120/180/240中按确定性整季reward最大选择，并列选更早者。
- primary：yield≥7853.665161、WP_ET≥1.64、PFP_N可定义且≥26.2。
- recorded只独立比较，不参与奖励或选模。

## 三seed预注册选中结果

| seed | 选中checkpoint | 产量 kg/ha | I mm | N kg/ha | WP_ET | PFP_N | reward | primary | recorded全可比项 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 180 | 7853.67 | 90 | 50 | 1.69 | 157.1 | 2.1772 | 是 | 否 |
| 1 | 60 | 7853.67 | 120 | 150 | 1.63 | 52.4 | 1.6472 | 否 | 否 |
| 2 | 60 | 7853.67 | 90 | 300 | 1.63 | 26.2 | 0.9272 | 否 | 否 |

三个seed都追平产量gate。seed1/2失败的直接判据均为WP_ET=1.63，低于预注册primary门槛1.64；不是产量不足，也不是PFP_N不可定义。

## 一个必须保留的现象

seed1和seed2的随机初始化checkpoint0都通过primary，但checkpoint0在实验前已明确禁止入选。训练后：

- seed1的最高reward策略从I45/N300转向I120/N150，资源成本按当前reward下降，但WP_ET从1.70降至1.63；
- seed2的训练后策略保持N300，并把灌溉转为I90或I120，WP_ET从1.69降至1.63。

这说明训练并非简单地“什么都没学”，而是reward最大化得到的资源权衡未在seed1/2同时满足独立的WP_ET门槛。该观察提示reward与primary指标并不完全同义，但本轮没有授权修改reward，因此只能记录，不能据此现场调参。

## 工程检查

- seed0选中模型哈希复核一致；
- seed1/2均精确240步、40训练季、4次learn调用、5个评估季；
- 所有评估均六阶段，非法/masked动作0；
- reward分项闭合，模型及指标有限；
- 每个seed五个checkpoint哈希互不相同；
- 源输入哈希未改变；
- 全程串行，无OOM。

## 科学边界

- HLA seed0是明确正向案例，但不能代表HLA整体稳定成功。
- 1/3不能表述为“多数seed成功”或“可跨年迁移”。
- 0/3全面超过recorded；其中seed0只差WP_ET 0.01，但不能放宽门槛。
- 不得挑选seed1/2的checkpoint0，也不得改用非预注册的其他checkpoint凑2/3。

## 输出

- `benchmark_results/027_03/027_03_result.json`
- `benchmark_results/027_03/027_03_hla2010_three_seed_selected_summary.csv`
- `benchmark_results/027_03/seed1/` 与 `seed2/` 的checkpoint、动作、训练季和结果文件
- `benchmark_results/027_03/027_03_hla2010_three_seed_checkpoint_comparison.png`
- `benchmark_results/027_03/027_03_hla2010_three_seed_checkpoint_comparison.svg`
- `src/run_hla2010_stage_maskable_ppo_seed1_seed2_027_03.py`

## 下一步

按027_00顺序，停止HLA跨年迁移，下一站进入YC2014 Phase A：只审计输入、四基线、站点专属scaler与完整季节no-op smoke，暂不训练YC PPO。HLA reward与WP_ET目标的关系可作为后续统一方法复盘问题，但不能在进入YC前临时修改HLA结果。

