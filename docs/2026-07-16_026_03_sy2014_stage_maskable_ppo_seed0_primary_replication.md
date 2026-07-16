# 026_03 SY2014 阶段型 MaskablePPO seed0 primary 复核记录

## 1. 目的

在不修改 026_02 算法、环境、奖励、动作空间、网络和训练步数的条件下，仅将 seed1 改为 seed0，检验相对 expert/auto 的 primary 综合优势能否跨 seed 复现。

本任务在运行前明确：primary 是主要科学判据；I<=90/N<=250 strict 是更保守的二级资源档位。026_02 原有 strict B 分支保持不变，不作追溯重判。

## 2. 预注册选择与持续性规则

- checkpoint 固定为 0/60/120/180/240；
- checkpoint0 只作随机初始化参照，不允许入选；
- 在 60/120/180/240 中按确定性季节 reward 最大选择，reward 并列取更早者；
- seed0 选中 checkpoint 必须 primary；
- seed0 后半程 120/180/240 至少 2/3 primary；
- 已有 seed1 按同一规则也必须满足以上两项。

## 3. 工程结果

全部工程检查通过：版本、检查点、240 步、40 个训练季、六阶段评估、有限数值和模型参数均正常；masked action 请求为 0。

## 4. seed0 学习曲线

| 步数 | 动作序列 | 产量 kg/ha | I mm | N kg/ha | WP_ET | PFP_N | reward | primary | strict |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 8,5,6,5,2,0 | 11182.38 | 120 | 300 | 2.24 | 37.3 | 5.7744 | 否 | 否 |
| 60 | 8,8,7,2,1,0 | 11181.15 | 120 | 300 | 2.21 | 37.3 | 5.7731 | 否 | 否 |
| 120 | 0,0,7,7,1,1 | 11204.88 | 60 | 200 | 2.31 | 56.0 | 6.3569 | 是 | 是 |
| 180 | 0,0,7,7,1,1 | 11204.88 | 60 | 200 | 2.31 | 56.0 | 6.3569 | 是 | 是 |
| 240 | 0,0,7,7,1,1 | 11204.88 | 60 | 200 | 2.31 | 56.0 | 6.3569 | 是 | 是 |

seed0 的随机初始化和 60-step 策略均不达标；从 120 步开始，策略转为 I60/N200，并在 180、240 步保持完全相同。因此 seed0 证据不是“随机初始网络偶然命中”。预注册选择为 checkpoint120，primary=true、strict=true；后半程 primary 为 3/3。

## 5. 跨 seed 对照

| seed | 预注册选中步数 | 产量 kg/ha | I mm | N kg/ha | WP_ET | PFP_N | primary | strict | 后半程 primary |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0 | 120 | 11204.88 | 60 | 200 | 2.31 | 56.0 | 是 | 是 | 3/3 |
| 1 | 60 | 11202.68 | 105 | 150 | 2.26 | 74.7 | 是 | 否 | 2/3 |

四项科学检查全部通过，判定：

`A_cross_seed_primary_signal`

## 6. 结论边界

1. 阶段型 MaskablePPO 已在两个独立 seed 中获得满足主要科学门槛的策略，并且后半程不是单一尖峰。
2. seed0 提供了从不达标到连续三个 checkpoint 达标的直接学习证据。
3. 两个 seed 学到的动作序列和资源组合不同，说明目前证明的是“目标性能可复现出现”，不是“唯一管理时序可复现”。
4. 还不能称为正式、跨 seed 稳定成功：当前只有两个 seed、一个年份，并使用同一年份进行训练和确定性评估。
5. 下一步允许新增一个独立 seed2 作为三 seed 确认，或者冻结选模协议后进行独立年份迁移；不得修改当前配置后再把结果混为同一组证据。

## 7. 输出

- `prompts/026_03_sy2014_stage_maskable_ppo_seed0_primary_replication.md`
- `src/run_sy2014_stage_maskable_ppo_seed0_primary_026_03.py`
- `benchmark_results/026_03/026_03_result.json`
- `benchmark_results/026_03/026_03_seed0_checkpoint_summary.csv`
- `benchmark_results/026_03/026_03_seed0_checkpoint_stage_actions.csv`
- `benchmark_results/026_03/026_03_seed0_training_episode_summary.csv`
- `benchmark_results/026_03/026_03_cross_seed_comparison.csv`
- checkpoint ZIP 仅本地保留，不提交 Git。
