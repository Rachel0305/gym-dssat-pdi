# 011_04 HLA 2010 official reward PPO training smoke

## 背景

011_01-011_03 已确认：

- `references/rewards.py` 官方 reward 可以加载；
- HLA 静态 MZX 已通过 Jinja 占位符修复；
- `mode='all'` 能正确渲染为 `IRRIG=L, FERTI=L`；
- gym/PDI action 已能进入 DSSAT `MgmtEvent.OUT` 并改变产量。

因此可以进入极短 PPO training smoke。

## 本轮目标

只做训练流程 smoke，不做正式训练。

验证：

1. SB3 PPO 能用官方 reward 标量化版本正常训练；
2. 训练后 deterministic eval 能输出动作；
3. PPO eval 的动作能进入 DSSAT 管理事件；
4. 保存模型、eval CSV、`MgmtEvent.OUT` 摘要和输入快照。

## 运行设置

- year: HLA 2010
- IC: 1
- cultivar: 最新 HY0006
- reward: `references/rewards.py`
- scalar reward: `sum(all_reward)`
- management: Jinja 渲染后的 `IRRIG=L, FERTI=L`
- timesteps: first attempted 100; timed out at 360 s before eval output.
- revised smoke timesteps: 5
- seed: 0
- PPO smoke hyperparameters: `n_steps=5`, `batch_size=5`, `n_epochs=1`.

## 判断标准

通过 smoke 只需要：

- 训练进程 exit code 为 0；
- eval CSV 能生成；
- eval 中 action 不报错；
- `pdi_tmp_snapshot_eval/fileX.MZX` 为 `IRRIG=L, FERTI=L`；
- `event_summary.json` 能生成。

不把 100 steps 的产量、动作策略当作正式结论。

## 100-step attempt result

100-step smoke used too much wall time under linked PDI/DSSAT and timed out at 360 seconds before eval output was generated.

This does not mean PPO failed scientifically; it means the smoke was still too expensive. Revised plan: use 5 timesteps with tiny rollout/update settings only to validate the training/evaluation plumbing.

## Final smoke result

After replacing the wrapper with `LazyScalarGymDssatWrapper(gymnasium.Env)` and avoiding the extra reset in wrapper initialization, 5-step PPO smoke completed successfully.

This validates the technical pipeline only:

- SB3 PPO initializes;
- `learn()` completes;
- model saves;
- deterministic eval runs;
- eval actions enter DSSAT linked management.

The eval policy is not agronomically meaningful yet: it applies very large daily water and nitrogen actions. The next step must add action-frequency and seasonal-budget safeguards before longer training.

## 安全规则

- 不长训练；
- 不覆盖旧实验；
- 不修改 Docker/site-packages；
- 如果超时或 OOM，立即停止，不继续加大步数。
