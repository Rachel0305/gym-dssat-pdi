# 026_04 SY2014 阶段型 MaskablePPO seed2 三 seed 确认记录

## 1. 目的

保持 026_02/026_03 的环境、奖励、算法、训练步数、检查点和选模规则完全不变，仅新增 seed2，检验是否达到三 seed 的 primary 结果与训练轨迹持续性复现。

## 2. 工程检查

全部工程检查通过：

- SB3 / sb3-contrib 均为2.8.0；
- 精确完成240阶段步、40训练季；
- 固定检查点0/60/120/180/240完整；
- 每次评估均为六阶段；
- masked action请求为0；
- 指标和模型参数均为有限值。

## 3. seed2 固定检查点

| 步数 | 动作序列 | 产量 kg/ha | I mm | N kg/ha | WP_ET | PFP_N | reward | primary | strict |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 6,6,4,5,2,2 | 11190.98 | 105 | 300 | 2.29 | 37.3 | 5.7980 | 是 | 否 |
| 60 | 4,4,4,7,1,1 | 11201.44 | 90 | 250 | 2.24 | 44.8 | 6.0734 | 否 | 否 |
| 120 | 4,5,4,1,1,1 | 10816.41 | 105 | 150 | 2.16 | 72.1 | 4.5534 | 否 | 否 |
| 180 | 2,2,8,7,1,0 | 11153.39 | 120 | 200 | 2.21 | 55.8 | 6.2454 | 否 | 否 |
| 240 | 3,3,3,5,1,1 | 11208.40 | 60 | 200 | 2.33 | 56.0 | 6.3604 | 是 | 是 |

预注册规则在训练后60/120/180/240中选确定性reward最大者，因而seed2选中checkpoint240。该策略primary=true、strict=true。

但seed2后半程120/180/240仅1/3 primary，未达到至少2/3的持续性门槛。训练轨迹表现为中期下降、最终恢复，不能用最终成功掩盖中间不稳定。

## 4. 三 seed 选模结果

| seed | 选中步数 | 产量 kg/ha | I mm | N kg/ha | WP_ET | PFP_N | primary | strict | 后半程primary |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0 | 120 | 11204.88 | 60 | 200 | 2.31 | 56.0 | 是 | 是 | 3/3 |
| 1 | 60 | 11202.68 | 105 | 150 | 2.26 | 74.7 | 是 | 否 | 2/3 |
| 2 | 240 | 11208.40 | 60 | 200 | 2.33 | 56.0 | 是 | 是 | 1/3 |

三 seed 按预注册选模均为 primary，且seed0/seed2达到strict；但只有seed0/seed1满足后半程至少2/3 primary。因此预注册判定为：

`B_two_of_three_primary_only`

## 5. 解释

1. 不能说“三 seed 全部失败”：三 seed 选中的策略全部达到主要科学门槛，这是明确的正向结果。
2. 也不能说“三 seed 稳定复现”：seed2只有最终checkpoint恢复达标，中期策略波动明显。
3. seed2说明阶段型PPO能够从中期低产策略恢复到严格成功策略，但尚未证明训练过程稳定收敛。
4. 三个seed得到不同动作序列，但选中策略产量都约11203--11208 kg/ha，显示存在多个性能接近的水氮时序解。
5. strict不是本轮主要判据；seed1体现水氮权衡，而非简单的整体资源效率失败。

## 6. 当前停止点

按预注册，`next_step_allowed=false`：

- 不自动延长seed2；
- 不增加seed；
- 不调整reward或PPO参数；
- 不选择额外中间checkpoint；
- 不立即扩展站点。

下一步需要在方法层面明确论文采用哪一种稳定性定义：

1. 标准RL选模复现：三seed按固定reward规则选出的模型均primary；或
2. 训练轨迹持续性：要求每个seed后半程多数checkpoint都primary。

当前证据满足前者，不完全满足后者。这是评价协议选择问题，不应通过现场调参解决。

## 7. 输出

- `prompts/026_04_sy2014_stage_maskable_ppo_seed2_three_seed_confirmation.md`
- `src/run_sy2014_stage_maskable_ppo_seed2_confirm_026_04.py`
- `benchmark_results/026_04/026_04_result.json`
- `benchmark_results/026_04/026_04_seed2_checkpoint_summary.csv`
- `benchmark_results/026_04/026_04_seed2_checkpoint_stage_actions.csv`
- `benchmark_results/026_04/026_04_seed2_training_episode_summary.csv`
- `benchmark_results/026_04/026_04_three_seed_comparison.csv`
- checkpoint ZIP本地保留，不提交Git。
