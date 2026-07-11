# 020_12 YC/FQ 冻结 n-step DQN 跨站点验证最终记录

## 完成范围

- YC2014 与 FQ2016 均使用同一冻结训练框架重新训练50K；不是直接迁移HLA模型权重。
- 两站均使用各自同输入null产量，奖励、9动作、I120/N300、7 DAP间隔、n_steps=5和其余DQN超参数不变。
- 当前只有seed0，不能据此宣称跨seed稳定。

## 与基线的统一比较

| site | scenario_family | final_grain_kg_ha | irrigation_mm | nitrogen_kg_ha | unified_reward | yield_diff_vs_auto_kg_ha | yield_diff_vs_extension_expert_kg_ha |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | null | 7066.000 | 0.000 | 0.000 | 0.000 | -946.000 | -874.000 |
| FQ | recorded | 7933.000 | 75.000 | 144.000 | 72.000 | -79.000 | -7.000 |
| FQ | dssat_auto | 8012.000 | 59.900 | 0.000 | 886.100 | 0.000 | 72.000 |
| FQ | extension_expert | 7940.000 | 198.800 | 247.000 | -559.800 | -72.000 | 0.000 |
| FQ | frozen_nstep5_dqn | 7985.000 | 120.000 | 50.000 | 548.564 | -27.000 | 45.000 |
| YC | null | 7825.000 | 0.000 | 0.000 | 0.000 | -888.000 | -1592.000 |
| YC | recorded | 9418.000 | 120.000 | 374.000 | -397.000 | 705.000 | 1.000 |
| YC | dssat_auto | 8713.000 | 86.500 | 0.000 | 801.500 | 0.000 | -704.000 |
| YC | extension_expert | 9417.000 | 228.800 | 247.000 | 128.200 | 704.000 | 0.000 |
| YC | frozen_nstep5_dqn | 8659.000 | 60.000 | 0.000 | 774.153 | -54.000 | -758.000 |

## 与旧 n-step=1 DQN 的对照

| site | framework | seed | checkpoint_step | final_grain_kg_ha | irrigation_mm | nitrogen_kg_ha | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YC | previous_nstep1 | 0 | 25000 | 8657.000 | 120.000 | 0.000 | 711.924 |
| YC | previous_nstep1 | 1 | 50000 | 8657.000 | 75.000 | 0.000 | 756.584 |
| FQ | previous_nstep1 | 0 | 25000 | 7779.000 | 60.000 | 0.000 | 652.609 |
| FQ | previous_nstep1 | 1 | 30000 | 7995.000 | 60.000 | 0.000 | 869.176 |
| FQ | frozen_nstep5 | 0 | 50000 | 7985.000 | 120.000 | 50.000 | 548.564 |
| YC | frozen_nstep5 | 0 | 35000 | 8659.000 | 60.000 | 0.000 | 774.153 |

## 结论

- YC2014：冻结n-step5在35K选中GWAD 8659、I60、N0。比null高834 kg/ha，较auto低54 kg/ha但少26.5 mm水；明显低于recorded和官方推广expert的约9417–9418 kg/ha。因此这是低投入、近auto候选，不是全面优于五情景的成功案例。
- FQ2016：冻结n-step5在50K选中GWAD 7985、I120、N50。比null高919 kg/ha，也略高于recorded与官方推广expert，但较auto低27 kg/ha，同时比auto多60.1 mm水和50 kg/ha氮；尚未达到产量与水氮效率同时超过auto的目标。
- 同一冻结框架在两个目标站点都能学出显著优于null的策略，说明训练链路可跨站点使用；但两个seed0结果均未同时超过所有强基线，跨站点优越性尚未成立。
- 下一步应先做seed1最小稳定性复核：YC检验低投入近auto能否复现，FQ检验是否仍落后auto。只有复现后才值得扩seed2或跨年份，不再修改奖励/约束。

## QA

- YC保存模型re-audit：10/10 checkpoint产量、水氮、奖励精确复现；按pre-action operation_dap复核后10/10运行审计通过。
- FQ：10/10 checkpoint运行审计通过。
- HLA冻结配置仍为本次唯一框架源；本轮没有重新训练HLA，也没有覆盖旧结果。