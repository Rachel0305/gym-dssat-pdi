# 021_20 SY2014 最小 DQfD-style 5K seed0 A/B smoke

## 1. 目标

在不修改 SY2014 IC=2、DSSAT 输入、奖励函数、动作空间、预算和基础 DQN 超参数的前提下，检查 021_18 的 I75/N200 oracle 示范能否通过“TD 学习 + 示范约束”改善 DQN，而不是像 021_19 简单行为克隆那样过度操作并打满 I120/N300。

本任务只做离线单元测试和 5K seed0 A/B smoke，不启动长训练、多 seed 或其他站点实验。

## 2. 方法来源与边界

参考 Hester et al. (2018), *Deep Q-learning from Demonstrations* 的核心思想：

- 1-step TD loss；
- n-step TD loss；
- 仅用于示范样本的 large-margin supervised loss；
- L2 正则；
- 示范 transition 永久保留。

本任务是项目本地的“最小 DQfD-style”实现，不是完整 DQfD 复现。它不实现论文中的优先经验回放、示范优先级加成和统一混合 replay sampler；示范数据保存在独立、永久的示范张量中，每次在线更新额外采样一个示范 batch。论文使用的 n=10 在本项目中适配为冻结主线的 n=5。

## 3. 冻结变量

- 站点年份：SY2014，IC=2；
- PDI/DSSAT 4.8.0 和 021_14 同源输入；
- 9 个离散动作：I∈{0,15,30} mm，N∈{0,50,100} kg/ha；
- 季节预算 I≤120 mm、N≤300 kg/ha；
- 单次上限 I30/N100；DAP1–120；共享最小操作间隔 7 d；
- reward：每步 `-1*I-5*N`，终止时 `+max(0, HWAM-null_yield)`；不缩放 reward；
- DQN 网络、学习率、batch、buffer、gamma、target update、n-step=5 等沿用 020_11；
- seed=0；计划总时程 50K，但 callback 在 5K 停止，保证探索率仍按全局 50K 日程计算；
- 不加载任何历史 checkpoint。

## 4. 唯一实验变量

- Control：修复探索率调度后的标准 SB3 DQN；
- Treatment：相同 DQN，增加独立永久示范数据和最小 DQfD-style 示范更新。

除示范机制外两组所有设置一致。

## 5. 预注册参数

- demonstration batch size：32；
- 离线示范预训练更新：100 次；
- online 每次标准 DQN gradient step 后，追加 1 次示范更新；
- large-margin：0.8；
- `lambda_n_step=1.0`；
- `lambda_margin=1.0`；
- `lambda_l2=1e-5`；
- n-step horizon：5；
- gradient clip：10；
- 不根据 smoke 结果临时调整上述参数。

## 6. 示范 transition

重新在同一环境回放 021_18 oracle：DAP22/79 action1，DAP29/56 action4，DAP42 action7，其余 action0。保存：

- observation；
- action；
- reward；
- next_observation；
- done；
- 5-step return、5-step next observation、5-step done、实际折扣步数；
- 实际执行水氮及裁剪审计。

必须复现 HWAM=11205 kg/ha、I75/N200、5 个非零事件且无裁剪，否则停止。

## 7. 单元测试（训练前必须通过）

1. 示范 transition 数量、前后状态、done 和 n-step 边界对齐；
2. large-margin loss：当 expert Q 比其他动作至少高 0.8 时为 0，否则为正；
3. toy batch 一次优化后，组合 loss 有限，expert action margin 方向改善；
4. 示范数据保存于独立永久结构，不进入会被覆盖的普通 replay ring；
5. 5K 时 control/treatment 探索率均约为 0.729，不能提前降到 0.05。

任一核心单元测试失败，则不跑 DSSAT 5K A/B。

## 8. 5K A/B 与过程记录

两组各运行一次 5K，单进程顺序执行，避免 OOM。一次 `learn()` 使用 50K 全局日程，callback 每 1K 保存模型；训练完成后对 1K/2K/3K/4K/5K checkpoint 分别进行独立确定性评估，保存：

- HWAM、CWAM、I、N、总 reward；
- 灌溉和施氮 DAP；
- 施氮是否晚于 DAP90；
- 9 动作计数；
- epsilon、训练更新数；
- control 与 treatment 的 checkpoint 轨迹。

不得只挑最终 5K，也不得用单个漂亮 checkpoint 宣称稳定。

## 9. 预注册判定

Treatment 的 5K smoke 只判“是否值得继续”，不判正式成功。满足以下条件才允许建议后续多 seed/长训练：

1. 至少一个 1K–5K checkpoint 同时达到 HWAM≥11077、I≤90、N≤250；
2. 该候选不在 DAP90 后施氮；
3. 相同 checkpoint 下的 Treatment 资源使用或产量至少一项严格优于 Control，且另一项不明显恶化；
4. 未出现非有限 loss/Q/reward；
5. 过程记录完整。

若未通过，按预注册停止：不扫 margin、不扫 lambda、不增加 epoch、不直接扩 seed 或训练步数。

## 10. 输出

- `src/minimal_dqfd_style.py`
- `src/run_sy2014_minimal_dqfd_style_5k_ab_021_20.py`
- `benchmark_results/021_20/` 下的示范数据、单元测试、checkpoints、逐 checkpoint CSV、summary JSON；
- `docs/2026-07-15_021_20_sy2014_minimal_dqfd_style_5k_ab_smoke.md`；
- 失败日志和修复记录不得删除或覆盖。

## 11. 资源纪律

- 只使用容器 `b2fd6726c8c1` 和 `/opt/gym_dssat_pdi/bin/python`；
- 先单元测试，后顺序运行 control/treatment；
- 不并行训练，不安装新包，不运行长训练；
- 所有新结果使用新目录，禁止覆盖旧结果。

