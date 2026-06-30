# 012_11 HLA2015 DQN 训练动作覆盖诊断 prompt

## 背景

012_10 显示，HLA2015 economic DQN seed0 从 5K 延长到 20K 后有改善，但仍没有学到 fixed I60/N0 这类合理策略。

当前问题已经不是单纯“要不要继续加训练步数”，而是需要诊断：

> 失败 seed 在训练过程中是否真的探索过关键窗口内的灌溉动作？

如果失败 seed 几乎没有在关键灌溉窗口探索到 action=1 或 action=3，那么问题主要是探索覆盖不足。

如果失败 seed 探索过足够多的灌溉动作，但最终仍没有学到，那么问题可能是 reward 信用分配、Q值学习不稳或动作价值估计失败。

## 本轮目标

在不改变算法、不改变 reward、不改变窗口、不改变训练步数的前提下，给训练过程加日志，记录每一步：

- 当前 DAP；
- raw action index；
- raw amir/anfer；
- safe amir/anfer；
- 是否处于灌溉窗口；
- 是否处于施氮窗口；
- reward；
- grnwt/topwt；
- swfac/nstres。

然后分别跑 HLA2015 economic DQN seed0、seed1、seed2，各 5000 步。

## 实验设置

- 年份：HLA2015
- 算法：DQN
- seed：0、1、2
- 训练步数：5000
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

- 动作空间：
  - 0：不操作
  - 1：灌溉 30 mm
  - 2：施氮 50 kg/ha
  - 3：灌溉 30 mm + 施氮 50 kg/ha
- 灌溉窗口：DAP 20–35、45–65、70–95
- 施氮窗口：DAP 25–40、55–70
- 灌溉预算：120 mm
- 施氮预算：150 kg/ha

## 关键诊断指标

1. 全训练阶段 action 分布：
   - action 0/1/2/3 各出现多少次。

2. 灌溉窗口内 action 分布：
   - 在灌溉窗口内，action=1 或 action=3 出现多少次。

3. 安全动作执行率：
   - raw action 想灌溉，但 safe_amir=0 的比例；
   - raw action 想施氮，但 safe_anfer=0 的比例。

4. 有效灌溉 episode 覆盖：
   - 每个 episode 中是否至少执行过一次 safe_amir>0。

5. 成功 seed 与失败 seed 的差异：
   - seed1 为什么能学到 I60/N0；
   - seed0/seed2 是否因为关键窗口内探索不足而失败。

## 禁止事项

- 不改 reward；
- 不改 DQN 参数；
- 不延长训练；
- 不保存大模型文件；
- 不并行开多个训练；
- 不把这一步直接当作新优化结果。

## 输出

- 每个 seed 的训练 action log；
- 每个 seed 的 action coverage summary；
- 三个 seed 的合并对比表；
- 中文实验记录。

