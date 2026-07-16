# 022_17 SY2014 DAP110变化动作受控DSSAT回放

## 1. 目的

022_15/16证明继续离线优化会改变DAP110支持集argmax，但没有证明这些变化在真实DSSAT季节中是改善还是退化。本任务固定DAP110以前的五阶段管理，仅交换最终action0/action1，以完整季节回报判断变化后果。

## 2. 预注册设计

- 状态集合：022_15或022_16任一条件发生DAP110变化的全部场景并集，共9个；
- 每场景固定原前5阶段动作；
- DAP110分别执行action0和action1；
- 共18个确定性季节；
- 完整回报差容差固定为1；
- 预算裁剪导致两个动作实际执行相同的场景标为alias，不计入方向；
- DQN训练0步，不修改reward、IC、预算或动作空间。

分支定义：仅改善为A，仅退化为B，改善与退化并存为C，全部中性/别名或实现失败为D。

## 3. 执行记录

首次外层命令等待上限错误设为10秒，工具返回超时，但容器内任务并未停止。检查确认同一进程仍在运行，因此没有删除目录或重新启动第二份任务，而是等待原进程自然完成。该问题不影响18季的唯一执行，也没有产生重复DSSAT季节。

## 4. 动作对照结果

| 场景 | DAP110前I | action1实际增加I | Δ产量(a1−a0) | ΔG0(a1−a0) | 判读 |
|---|---:|---:|---:|---:|---|
| W120 critical N300 | 120 | 0 | 0.000 | 0.000 | 预算别名 |
| W60 critical N250 mid | 60 | 15 | 0.000 | -15.000 | action0优 |
| W75 critical N250 mid | 75 | 15 | 0.000 | -15.000 | action0优 |
| W75 uniform N200 spread | 75 | 15 | +3.317 | -11.683 | action0优 |
| W90 all6 N150 | 75 | 15 | +3.231 | -11.769 | action0优 |
| W90 critical N150 | 90 | 15 | +2.717 | -12.283 | action0优 |
| W90 critical N200 early | 90 | 15 | 0.000 | -15.000 | action0优 |
| W90 critical N250 | 90 | 15 | 0.000 | -15.000 | action0优 |
| W90 critical N300 | 90 | 15 | 0.000 | -15.000 | action0优 |

除1个预算别名外，8/8可执行差异场景中action1的完整回报均低于action0。部分场景晚期15mm灌溉使产量增加约2.7–3.3 kg/ha，但不足以抵消15的灌溉成本。

## 5. 模型选择变化的真实后果

按seed×scenario×训练条件逐项计算，共21次变化：

- 退化：17次；
- 改善：2次；
- 预算别名：2次；
- 中性：0次。

两次改善均来自同一个真实方向：seed1在`W60_critical__N250_mid`中由旧action1改为action0，完整回报提高15。17次退化均为从action0改成action1；两次alias均为已用满I120后请求action1但实际执行0mm。

## 6. 判定

**C分支：改善与退化并存。**

可以确认：

- DAP110的多数argmax漂移不是无害数值变化，而是真实选择了回报更低的晚期灌溉；
- 少数变化确实修正了旧checkpoint原本错误的action1选择；
- “所有argmax必须相对旧checkpoint零变化”的guardrail过于机械，因为它会把真实改善也判失败；
- 但不能简单删除该guardrail，因为本次绝大多数变化（17/19非别名变化）确实是退化；
- 022_16仍然成立：这些退化在MC-only和MC+pairwise中都出现，pairwise不是必要原因。

## 7. 对当前主线的含义

022_15的pairwise局部修复是有效的：三个seed均纠正DAP65排序。失败的真正矛盾进一步收紧为：继续优化MC回归时，共享网络会在其他状态中产生多数有害、少数有益的排序重组。

因此下一步不能：

- 直接删除DAP110 guardrail；
- 挑选中间checkpoint；
- 调lambda或继续增加训练步数；
- 直接把问题归因于“小数据过拟合”。

如果继续，guardrail应由“相对旧网络完全不变”升级为“在预注册受控状态上不允许回报下降”。技术上可考虑基于起点网络的保守锚定/蒸馏约束，只允许有受控证据支持的局部排序改变；但这属于新的结构性实验，必须另立预注册任务，不能在本轮现场实现。

## 8. 输出

- `prompts/022_17_sy2014_dap110_changed_action_controlled_replay.md`
- `src/run_sy2014_dap110_changed_action_controlled_replay_022_17.py`
- `benchmark_results/022_17/022_17_season_summary.csv`
- `benchmark_results/022_17/022_17_stage_actions.csv`
- `benchmark_results/022_17/022_17_daily_values.csv`
- `benchmark_results/022_17/022_17_controlled_action_pairs.csv`
- `benchmark_results/022_17/022_17_model_change_consequences.csv`
- `benchmark_results/022_17/022_17_result.json`
- `benchmark_results/022_17/022_17_dap110_controlled_reward_difference.png/.svg`

当前未执行Git commit或push，022_13 warm-start仍未启动。
