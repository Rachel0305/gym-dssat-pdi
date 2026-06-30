# 015_07 YC2014 DQN headroom 诊断记录

## 目的

本轮不训练 DQN，只用同一 PDI/gym-DSSAT 环境前向模拟一小批候选水氮调度，判断在当前 I≤120 mm、N≤300 kg/ha、单次 I≤30 mm、单次 N≤100 kg/ha、最小间隔 7 天的约束内，是否存在明确超过 recorded expert 9418 kg/ha 的产量空间。

## 结果

- 候选数量：9
- 超过 recorded expert + 50 kg/ha 的候选数量：0
- 最高产量候选：s03_I_seed1_like_N_seed0_250，GWAD=9418.0 kg/ha，I=120.0 mm，N=250.0 kg/ha，proxy=8048.0
- 最高 proxy 候选：s03_I_seed1_like_N_seed0_250，GWAD=9418.0 kg/ha，I=120.0 mm，N=250.0 kg/ha，proxy=8048.0

## 决策解释

如果本轮没有找到超过 9468 kg/ha 的候选，则不建议在 YC2014 上盲目加长 DQN 训练来追求更高产；当前可汇报结论应是 DQN 少氮追平专家，而不是产量超越专家。如果找到明确候选，则下一轮再围绕该候选的时点和动作空间做 DQN 长训练。

## 输出文件

- 全部日值：`DSSAT_auto_validation/yc2014_dqn_headroom_probe_015_07/015_07_yc2014_dqn_headroom_daily.csv`
- 候选汇总：`DSSAT_auto_validation/yc2014_dqn_headroom_probe_015_07/015_07_yc2014_dqn_headroom_candidates.csv`
- top 候选：`DSSAT_auto_validation/yc2014_dqn_headroom_probe_015_07/015_07_yc2014_dqn_headroom_top_schedules.csv`
- 图：`DSSAT_auto_validation/yc2014_dqn_headroom_probe_015_07/figures/yc2014_headroom_yield_vs_input.png`