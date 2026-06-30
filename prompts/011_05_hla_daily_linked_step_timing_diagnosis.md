# 011_05 HLA daily linked step timing diagnosis

## 背景

011_04 中，Jinja 占位符修复后 action channel 已经真实进入 DSSAT：

- `IRRIG=L`
- `FERTI=L`
- `MgmtEvent.OUT` 能记录 gym action；
- forced action 能改变最终产量。

但 100-step 和 5-step PPO smoke 都超时，说明日尺度 linked interaction 的计算成本可能很高。

用户仍希望保留日交互，因此本轮不放弃日交互，而是先诊断慢在哪里。

## 目标

不训练 PPO，只运行手写 daily `env.reset()` / `env.step()`，记录耗时。

检查：

1. `env` 创建耗时；
2. `reset()` 耗时；
3. 前 10 个 `step()` 的逐步耗时；
4. 每步返回的 DAP 是否按日推进；
5. 每步 action 是否能继续进入 linked PDI；
6. 是否存在某一步异常慢、卡死或 done 过早。

## 运行设置

- year: HLA 2010
- IC: 1
- cultivar: 最新 HY0006
- template: 已插入 Jinja 占位符的 copied MZX
- mode: all
- management after render: `IRRIG=L, FERTI=L`
- steps: 10
- action:
  - DAP 1: `anfer=165`
  - DAP 2-10: zero action

## 输出

保存到：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/daily_step_timing`

文件：

- `daily_step_timing.csv`
- `summary.json`
- raw `pdi_tmp_snapshot`
- README

## 判断

如果单步 `env.step()` 本身就需要几十秒，则 SB3 日尺度训练很难直接可行，需要优化交互方式或减少步数。

如果手写 `env.step()` 很快，但 PPO smoke 慢，则问题在 SB3 wrapper/rollout/eval 流程。

