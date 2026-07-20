# 027_02 HLA2010 阶段型 MaskablePPO seed0 240 步实验记录

## 结论

状态：`completed / A_seed0_primary_signal`。

HLA2010 seed0 在完全冻结的阶段型 MaskablePPO 配置下出现了持续 primary 正向信号。预注册规则选中 checkpoint180；该模型在训练年追平 DSSAT auto 和官方推广 expert 的产量，同时明显减少灌溉和施氮，并提高 WP_ET 与 PFP_N。

这仍然只是一个 seed 的训练年结果，不能称为 HLA 已跨 seed 稳定成功，也不能直接开始跨年迁移。下一步只能按预注册协议复核 seed1/2。

## 执行范围

- 站点年份：HLA2010。
- 算法：SB3-Contrib MaskablePPO。
- seed：0。
- 24维 HLA 专属 scaler。
- 六阶段：DAP 1/30/50/65/85/110。
- 9动作、I≤120 mm、N≤300 kg/ha、晚期禁氮 mask。
- 网络 `[32,32]`，learning rate 3e-4，gamma=1，GAE lambda=1，n_steps=60，batch=30，epoch=5。
- 240阶段步，checkpoint 0/60/120/180/240。
- 无参数扫描、无跨年、无其他 seed 或站点。

## attempt 记录

attempt1 的外层 shell 等待时间误设为1秒，启动命令被终止。没有形成可解释训练结果，保留于 `benchmark_results/027_02/027_02_attempt_note.json`，不参与任何科学判断。

权威 attempt2 使用未改变的预注册配置，在新目录 `benchmark_results/027_02_attempt2/` 完成。

## 工程验证

全部检查通过：

- 027_01 readiness 分支、精确 null/gate 与24维 scaler 一致；
- 训练前 no-op smoke 为六阶段、零水、零氮、null产量、总奖励0；
- 版本为 SB3/sb3-contrib 2.8.0；
- 精确240步、40训练季、4次learn调用、5个确定性评估季；
- 所有checkpoint均为六阶段，非法/masked动作0；
- reward分项闭合、模型参数有限；
- 五个模型哈希互不相同；
- 源输入哈希未改变。

## checkpoint 结果

| checkpoint | 动作序列 | 产量 kg/ha | I mm | N kg/ha | WP_ET | PFP_N | reward | primary |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 0,8,4,5,5,1 | 7853.67 | 120 | 250 | 1.63 | 31.4 | 1.1472 | 否 |
| 60 | 1,8,8,2,4,0 | 7853.67 | 120 | 250 | 1.63 | 31.4 | 1.1472 | 否 |
| 120 | 3,0,4,2,2,2 | 7853.67 | 105 | 100 | 1.69 | 78.5 | 1.9122 | 是 |
| 180 | 3,0,0,2,2,2 | 7853.67 | 90 | 50 | 1.69 | 157.1 | 2.1772 | 是 |
| 240 | 3,0,0,2,2,2 | 7853.67 | 90 | 50 | 1.69 | 157.1 | 2.1772 | 是 |

180与240奖励完全相同，按训练前锁定的“并列选更早者”规则选择180。选中模型哈希：

`6715fffb4bbe251cf0cdad13121331d9e6f875a3e6629c26423500e502e042da`

## 选中策略

动作序列 `3,0,0,2,2,2` 对应：

- DAP1：施氮50 kg/ha；
- DAP30、50：不操作；
- DAP65、85、110：各灌溉30 mm；
- 全季合计 I90/N50。

与 auto/expert gate 比较：

- 产量：7853.67 kg/ha，追平 auto/expert；
- WP_ET：1.69，高于 auto 1.64 与 expert 1.63；
- PFP_N：157.1，高于 expert 26.2；
- 因而 primary 通过。

## recorded 边界

选中策略的产量和 PFP_N 高于 recorded，但 WP_ET=1.69，略低于 recorded 的1.70。因此 `recorded_all_comparable_pass=false`，不能包装为已经全面超过 recorded、auto、expert 三个基准。

## 科学解释边界

- checkpoint0 已追平产量门槛，说明“产量追平”本身不能单独证明学习成功。
- 训练后的明确变化是资源投入由 I120/N250 逐步下降到 I90/N50，同时保持产量平台，且120/180/240连续三个后期checkpoint均通过primary。
- 这构成 seed0 的学习信号，但跨 seed 复现尚未完成。
- 不允许据此修改 recorded 门槛、选择 checkpoint240 或直接开展跨年迁移。

## 输出

- `benchmark_results/027_02_attempt2/027_02_result.json`
- `benchmark_results/027_02_attempt2/027_02_hla2010_seed0_checkpoint_summary.csv`
- `benchmark_results/027_02_attempt2/027_02_hla2010_seed0_checkpoint_stage_actions.csv`
- `benchmark_results/027_02_attempt2/027_02_hla2010_seed0_training_episode_summary.csv`
- `benchmark_results/027_02_attempt2/027_02_hla2010_seed0_training_stage_actions.csv`
- `benchmark_results/027_02_attempt2/027_02_hla2010_seed0_checkpoint_and_training_curve.png`
- `benchmark_results/027_02_attempt2/027_02_hla2010_seed0_checkpoint_and_training_curve.svg`
- `src/run_hla2010_stage_maskable_ppo_seed0_027_02.py`

## 下一步

只允许另立 027_03，以完全相同配置串行运行 seed1/2。至少2/3 seed 的预注册选中checkpoint通过primary，才允许冻结三个模型并进入HLA站内跨年迁移。

