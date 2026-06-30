# 015_07 YC2014 DQN 更优策略空间诊断 prompt

## 背景

015_06 已经整理出 YC2014 的正式四情景对照：

- recorded expert：GWAD≈9418 kg/ha，I≈120 mm，N≈374 kg/ha。
- DQN seed0 best checkpoint：GWAD≈9418 kg/ha，I≈120 mm，N≈250 kg/ha。
- DQN seed1 best checkpoint：GWAD≈9418 kg/ha，I≈120 mm，N≈300 kg/ha。

因此，当前 DQN 不是“产量明显超过专家”，而是“在产量追平专家时减少氮投入”。如果导师要求 DQN 的产量也更高，不能直接盲目加大训练步数，因为如果当前动作空间/预算约束中不存在超过 9418 kg/ha 的可行调度，训练再久也无法证明产量超越。

## 目标

先做低成本 headroom 诊断，回答：

1. 在当前统一 DQN 设置下（I≤120 mm，N≤300 kg/ha，动作 I∈{0,15,30}，N∈{0,50,100}，最小操作间隔 7 天），是否存在比 recorded expert 更高产的候选调度？
2. 如果存在，候选调度的水氮事件是什么，是否可解释，是否可以作为下一轮 DQN 长训练/动作空间扩展的目标？
3. 如果不存在，说明 YC2014 当前约束下 DQN 的合理叙事应是“少氮追平专家”，而不是“产量超过专家”；下一步应换年份/站点或调整约束，而不是继续盲目增加步数。

## 低成本执行原则

- 不训练模型，不启动长时间 RL。
- 只做有限数量 DSSAT/PDI 前向模拟或优先从已有结果中筛选。
- 使用指定 Docker 容器 `b2fd6726c8c1` 和虚拟环境 `/opt/gym_dssat_pdi/bin/python`。
- 保持 IC=1、品种参数、天气、土壤、管理模板与 YC2014 当前 DQN 实验一致。
- 保持 `IRRIG=L / FERTI=L`，确保水氮事件由脚本显式写入/执行。
- 不覆盖旧结果；输出到新目录：

```text
DSSAT_auto_validation/yc2014_dqn_headroom_probe_015_07/
```

## 候选调度设计

第一层：已有策略候选复核

- recorded expert
- DSSAT auto
- DQN seed0 best checkpoint
- DQN seed1 best checkpoint

第二层：围绕 DQN seed0 和 expert 的小扰动候选

- 灌溉总量固定或接近 120 mm，尝试 4 次 ×30 mm 或 3 次 ×40 mm 等不超过单次 cap 的近似组合。
- 氮肥总量限定在 250、300 kg/ha 两档。
- 施氮时点围绕 DAP 0、43、DQN seed0 的施氮事件附近，允许 ±7 天扰动。
- 灌溉时点围绕 DQN seed0、DSSAT auto、recorded expert 的灌溉事件附近，允许 ±7 天扰动。
- 候选总数先控制在 30–80 个以内，避免算力浪费。

## 输出

1. `015_07_yc2014_dqn_headroom_candidates.csv`
   - schedule_id
   - irrigation_events
   - fertilizer_events
   - total_irrigation
   - total_n
   - final_gwad
   - final_cwad
   - max_swfac
   - max_nstres
   - reward_proxy = final_gwad - 1.0×total_irrigation - 5.0×total_n
   - note

2. `015_07_yc2014_dqn_headroom_top_schedules.csv`
   - 按 final_gwad 排名前 10
   - 按 reward_proxy 排名前 10

3. 图：
   - `yc2014_headroom_yield_vs_input.png`
   - 横轴可用 N 或总投入，纵轴 GWAD，标出 recorded expert 和 DQN seed0。

4. 记录：

```text
docs/2026-06-30_015_07_yc2014_dqn_headroom_probe_record.md
```

记录必须用中文，写清：

- 是否发现超过 9418 kg/ha 的可行候选；
- 如果没有，为什么不建议继续在 YC2014 盲目加步数；
- 如果有，下一步建议是加长 DQN、扩动作空间，还是迁移到其他站点年份。

## 决策规则

- 若最高候选 GWAD ≤ recorded expert + 50 kg/ha：认为当前约束下没有明确产量超越空间，YC2014 的正式叙事改为“少氮追平专家”。
- 若最高候选 GWAD > recorded expert + 50 kg/ha 且调度合理：再启动下一轮 DQN 训练，优先围绕该候选的时间窗口/动作空间改进。
- 若 reward_proxy 明显高但 GWAD 不高：作为“资源效率更优”结果，不包装成“产量更优”。
