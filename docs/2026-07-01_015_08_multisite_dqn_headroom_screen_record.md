# 015_08 多站点 DQN headroom 筛选记录

日期：2026-07-01

## 目的

基于已有证据，不新增训练、不新增 DSSAT 运行，对当前几个站点年份的 DQN 结果做一次低成本筛选，明确：

1. 哪些站点年份已经有稳定成功案例价值；
2. 哪些站点年份还有继续推进正式 DQN 长训练的空间；
3. 哪些站点年份不建议继续追求“更高产”，应当保留为对照或成功案例。

## 使用的现有证据

- `docs/2026-06-30_014_14_success_examples_four_scenario_figures_record.md`
- `docs/2026-06-30_014_12_yc_fq_dqn_transfer_smoke_record.md`
- `docs/2026-06-30_015_06_yc2014_formal_four_scenario_record.md`
- `docs/2026-06-30_015_07_yc2014_dqn_headroom_probe_record.md`

## 筛选结论

| site | year | scenario_status | current_best_label | yield_kg_ha | irrigation_mm | fertilizer_kg_ha | max_water_stress | max_nitrogen_stress | stability_flag | headroom_flag | next_action |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| FQ | 2016 | formal_candidate | DQN seed1 window | 8012.4 | 30.0 | 300.0 | 0.000 | 0.0122 | medium | low | 作为正式长训练候选优先保留，但不再盲目加大步长追高产 |
| HLA | 2010 | success_case | DQN seed1 | 7853.7 | 120.0 | 300.0 | 0.416 | 0.0158 | high | very_low | 保留为成功案例和跨 seed 复现正例 |
| HLA | 2015 | success_case | DQN seed1 | 7651.9 | 120.0 | 300.0 | 0.000 | 0.0145 | high | very_low | 保留为成功案例和跨 seed 复现正例 |
| YC | 2014 | plateau_case | DQN seed0 / seed1 best | 9418.0 | 120.0 | 250.0 | 0.000 | 0.0129 | high | very_low | 保留为高产平顶案例，不建议继续以“更高产”为主要目标 |

## 具体解释

### FQ2016

- 已经能跑通，并且是目前最像“还有可解释优化空间”的候选。
- 适合作为正式 DQN 长训练的优先候选，但前提是保持同一套奖励与约束逻辑，不要先扩展成多套互相不一致的目标。

### HLA2010 / HLA2015

- 都已经证明可以稳定给出“明显优于 null、且有意义的水氮操作”。
- 更像成功案例和正例，不建议把主要精力继续放在“追求更高产”上。

### YC2014

- 这是最典型的平顶案例：DQN 已经追平专家产量，同时少用氮。
- 015_07 的 headroom probe 已经说明，在当前动作上限与预算下，继续加大训练并不能保证超过专家。
- 因此它更适合作为成功案例和方法展示，不作为继续追高产的首选。

## 下一步建议

1. 若要继续正式 DQN，优先放在 FQ2016。
2. HLA2010 / HLA2015 作为稳定成功案例保留，用于论文中的正例与可解释性展示。
3. YC2014 保留为“同产量更少氮”的平顶案例，不再作为追求更高产的主线。
4. 后续如果要做算法改进，应先明确目标是“更稳定”还是“更高产”，不要把两者混成一个。

## 备注

本记录只做筛选，不重新训练，不重新跑 DSSAT。
