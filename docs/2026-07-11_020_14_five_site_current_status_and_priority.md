# 020_14 五站点当前状态与下一步优先级

日期：2026-07-11

## 1. 目的

本记录用于回答当前主线问题：

1. 五个站点是否都有继续水氮优化的潜力？
2. 是否已经能达到导师提出的目标：DQN 策略尽量同时超过 recorded/farmer practice、官方推广 expert 和 DSSAT auto 的产量与水氮利用效率？
3. 是否需要修改初始条件、奖励函数参数，或加入氮淋洗惩罚？
4. 下一步应该优先做什么，避免重复训练和重复审计。

本记录不是新训练结果，只整理已经完成的证据。

## 2. 证据范围

主要依据：

| 编号 | 证据文件 | 用途 |
|---|---|---|
| 020_11 | `DSSAT_auto_validation/HLA_2004/hla_five_scenario_nstep_020_11/020_11_hla_five_scenario_summary.csv` | HLA 五情景、三 seed、五年份完整结果 |
| 020_13 | `DSSAT_auto_validation/frozen_nstep_cross_site_020_12/020_13_yc_fq_seed0_seed1_summary.csv` | YC2014/FQ2016 冻结 n-step 框架 seed0/seed1 复核 |
| 020_13 | `DSSAT_auto_validation/frozen_nstep_cross_site_020_12/020_13_yc_fq_cross_seed_dqn_comparison.csv` | YC/FQ 跨 seed DQN 对比 |
| 018_01 | `docs/2026-07-07_018_01_multisite_dqn_potential_reward_ic_audit.md` | 五站点旧阶段潜力、IC、reward 审计 |
| 019_09 | `docs/2026-07-10_019_09_five_site_evidence_status_refresh.md` | 五站点已有证据刷新 |
| 017_08/017_09 | `docs/2026-07-06_017_08_sy_local_dqn_train_cross_year_transfer_record.md`; `docs/2026-07-06_017_09_sy2014_dqn_resource_space_record.md` | SY 本地训练和 SY2014 四情景证据 |
| 017_10/017_12 | `docs/2026-07-06_017_10_lc_pdi_initialization_rescue_record.md`; `docs/2026-07-07_017_12_lc2010_seed0_seed1_smoke_comparison.md` | LC 输入链路修复和 LC2010 seed0/seed1 smoke |

## 3. 统一框架状态

当前最可信的统一框架是 HLA 020_11 冻结的 n-step DQN：

| 项目 | 当前冻结设置 |
|---|---|
| 算法 | Stable-Baselines3 DQN |
| 动作 | 9 个离散动作，I in {0, 15, 30} mm，N in {0, 50, 100} kg/ha |
| 预算 | I <= 120 mm，N <= 300 kg/ha |
| 单次上限 | I <= 30 mm，N <= 100 kg/ha |
| 操作窗口 | DAP 1-120，至少 7 DAP 间隔 |
| n_steps | 5 |
| 训练 | 50K steps，每 5K 保存 checkpoint |
| reward | `max(0, GWAD_final - local_null_GWAD) - 1.0 * irrigation - 5.0 * nitrogen` |
| baseline | 每个站点年份使用自己的 same-input null 作为 local null |

这个框架的意义是：公式和训练逻辑统一，但 baseline 使用本地 null，避免把某个站点的产量尺度硬搬到其他站点。

## 4. 五站点状态总表

| 站点 | 当前最强证据 | 是否有优化潜力 | 是否已接近导师目标 | 主要限制 | 建议状态 |
|---|---|---|---|---|---|
| HLA 海伦 | 020_11：2007/2010/2015/2016/2022，三 seed，五情景齐全 | 明确有 | 最接近。多年份可达到接近或略高于 auto 的产量，并显著少用氮；部分年份节水 | 多数结果是接近或小幅超过 auto，不是大幅增产；需要正式 WUE/NUE 表述 | 主线成功候选，优先写入汇报 |
| YC 禹城 | 020_13：YC2014 seed0/seed1 冻结框架 | 有 | 暂未全面达到。DQN 明显优于 null，接近 auto，但低于官方推广 expert | 产量略低于 auto 和 expert；I 用量跨 seed 不完全一致 | 第二梯队，需要继续作为跨站点验证，不宜先声称成功 |
| FQ 封丘 | 020_13：FQ2016 seed0/seed1 冻结框架 | 局部有 | 部分达到。DQN 高于官方推广 expert，但低于 auto，且用水高于 auto | 年份敏感；auto 很强；N 动作跨 seed 不完全一致 | 有潜力但不能作为主成功站点，适合做边界案例 |
| LC 栾城 | 017_10/017_12：输入链路已修复，LC2010 seed0/seed1 smoke | 有，但证据层级低于 HLA/YC/FQ | 暂未达到正式结论 | 之前有土壤 ID/日期链路问题；seed1 会打满 N300；只有 smoke，不是冻结长训练 | 先补齐输入链路和冻结框架复核，再谈正式训练 |
| SY 沈阳 | 017_08/017_09：SY2014 本地训练高产 | 有高产响应 | 不满足“节水节氮同时超过”。DQN 高产但 I120/N300 打满 | IC 敏感，2014 用 IC=2；auto 弱，不能单靠超过 auto 证明有效 | 诊断/扩展站点，先记录 IC 合理性和资源效率问题 |

## 5. 最新关键数字

### 5.1 HLA

HLA 当前是最完整的结果集。020_11 中 DQN 结果按三 seed 展开，和 null、recorded farmer、DSSAT auto、official extension expert 同图同表比较。

代表性结果：

| 年份 | DQN 最好产量范围 | 相对 auto | 相对 official expert | 资源特征 |
|---|---:|---|---|---|
| 2007 | 7979.7-7986.7 | 约持平或略高 | 高约 39.7-46.7 kg/ha | N=0；I=60-120，低于 official expert |
| 2010 | 7853.7 | 略低 0.335 kg/ha | 略低 0.335 kg/ha | N=0；I=75-120，低于 auto 190.4 和 official expert 266.1 |
| 2015 | 7635.4-7652.7 | seed1/2 略高 auto | 高于 official expert 32.4-49.7 | N=0；I=45-120 |
| 2016 | 7527.6-7538.5 | seed0/2 约持平 | 高约 98.6-109.5 | seed1 出现 N=50，其余 N=0 |
| 2022 | 7925.3-7934.7 | 略低 auto | 高约 122.3-131.7 | N=0；I=60-120 |

判断：HLA 可以作为“DQN 在多年份保持接近 auto 产量、同时显著减少氮投入，并常常少用水”的主案例。若导师要求“大幅高产同时节水节氮”，HLA 还需要谨慎表述为资源效率优势，而非大幅增产优势。

### 5.2 YC2014

020_13 冻结框架结果：

| 情景 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha |
|---|---:|---:|---:|
| null | 7825 | 0 | 0 |
| recorded | 9418 | 120 | 374 |
| DSSAT auto | 8713 | 86.5 | 0 |
| official extension expert | 9417 | 228.8 | 247 |
| DQN seed0 | 8659 | 60 | 0 |
| DQN seed1 | 8676 | 90 | 0 |

判断：YC2014 在冻结框架下跨 seed 产量稳定，且 N=0 稳定；但目前低于 auto，也明显低于 recorded/official expert。它说明统一框架能跨站点跑通，但不能作为“全面优于 expert/auto”的成功案例。

### 5.3 FQ2016

020_13 冻结框架结果：

| 情景 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha |
|---|---:|---:|---:|
| null | 7066 | 0 | 0 |
| recorded | 7933 | 75 | 144 |
| DSSAT auto | 8012 | 59.9 | 0 |
| official extension expert | 7940 | 198.8 | 247 |
| DQN seed0 | 7985 | 120 | 50 |
| DQN seed1 | 7985 | 105 | 0 |

判断：FQ2016 跨 seed 产量完全一致，且高于 official expert 45 kg/ha；但低于 auto 27 kg/ha，且用水高于 auto。它是“接近强 baseline”的证据，不是全面胜出证据。

### 5.4 LC2010

017_12 smoke 结果：

| seed | checkpoint | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | 相对 auto |
|---|---:|---:|---:|---:|---:|
| seed0 | 5000 | 8739 | 90 | 0 | +1 |
| seed1 | 5000 | 8739 | 120 | 300 | +1 |

判断：LC2010 有真实优化空间，且 seed0 很漂亮；但 seed1 在同产量下打满 N300，说明资源效率策略不稳定。LC 还不能进入“正式成功案例”，应先做冻结框架复核。

### 5.5 SY2014

017_09 结果：

| 情景 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha |
|---|---:|---:|---:|
| null | 2769 | 0 | 0 |
| recorded | 9593 | 0 | 293 |
| DSSAT auto | 2724 | 33.4 | 0 |
| DQN | 11216 | 120 | 300 |

判断：SY2014 证明有强烈产量响应，但 DQN 是高投入高产，不满足节水节氮目标。SY 还存在 IC=2 的特殊处理，需要在论文/汇报里单独说明，不适合现在作为主成功案例。

## 6. 是否需要改初始条件

当前结论：不要为了制造 DQN 优势而改 IC。

可以改 IC 的情况：

| 情况 | 是否允许 | 说明 |
|---|---|---|
| 土壤 ID、日期、IC 开关等输入链路错误 | 允许 | 例如 LC 的土壤 ID/日期链路修复 |
| 品种校准阶段已经证明某个 IC 才能正确复现实测 | 允许 | 例如 SY2014 后来使用 IC=2 |
| 为了让低优化空间站点出现更强胁迫而人为降低初始水氮 | 不建议作为主线 | 只能作为 IC 敏感性情景，不能混入主实验 |

下一步如果处理 LC/SY，重点是“输入合理性修复”，不是“为了结果调 IC”。

## 7. 是否需要改奖励函数参数

当前统一奖励函数仍保留：

```text
reward = max(0, GWAD_final - local_null_GWAD)
         - 1.0 * irrigation
         - 5.0 * nitrogen
```

当前不建议马上改正式 reward，原因：

1. HLA 020_11 已经形成一套相对完整的冻结框架证据。
2. YC/FQ 020_13 证明同一框架可以跨站点运行，但还没有全面胜出。
3. 如果此时改 reward，会把“框架泛化”与“reward 敏感性”混在一起。

建议：先完成当前 reward 下的五站点分级；只有当导师明确要求更强节氮或环境目标时，再做统一 reward 敏感性，而不是按站点单独调参。

## 8. 是否加入氮淋洗惩罚

当前结论：暂不加入正式主线 reward。

原因：

1. 019 阶段已确认技术链路可探索，但 leaching reward 仍更适合作为敏感性或环境扩展。
2. 当前主问题还没完全解决：五站点是否都有优化空间、冻结框架是否可泛化。
3. 加入淋洗会改变研究目标函数，必须由导师确认。

建议表述：

> 氮淋洗惩罚是下一阶段的环境目标扩展项，不是当前正式 DQN 主线的必要条件。当前主线先回答产量、灌溉、施氮三者的联合优化能力。

## 9. 下一步优先级

### P0：先整理成导师可看的证据表

不训练，只用已有数据：

1. HLA：输出“五年份三 seed 五情景”的简表，突出资源效率。
2. YC/FQ：输出“冻结框架 seed0/seed1”的简表，明确还不是全面胜出。
3. LC/SY：列为“有潜力但证据不足/输入敏感”的诊断站点。

这是下一次和导师沟通最需要的材料。

### P1：LC 输入链路与冻结框架复核

LC2010 seed0 很好，但 seed1 打满资源。下一步应：

1. 确认 LC 源输入包已经永久修复土壤 ID 和日期问题；
2. 用 020_11 冻结 n-step 框架重跑 LC2010 seed0/seed1 的正式 50K；
3. 若 seed1 仍打满 N300，再进入统一 N cost 或动作窗口敏感性。

### P2：YC/FQ 不再盲目长训练

YC/FQ 020_13 已经说明：

- YC：跨 seed 稳定接近 auto，但低于 official expert；
- FQ：跨 seed 产量稳定，高于 official expert，但低于 auto，且用水高于 auto。

因此不建议直接继续加 seed2 或更长训练。更合理的是先和导师确认：

1. “接近 auto 但更省氮/部分省水”是否可接受；
2. official extension expert 是否作为正式 expert baseline；
3. 是否需要 reward 敏感性来强化节氮目标。

### P3：SY 暂作扩展诊断，不作主成功案例

SY2014 目前是高产高投入。下一步若继续：

1. 先固定 IC=2 的合理性说明；
2. 再做统一 N cost 敏感性；
3. 不建议直接作为“节水节氮成功案例”汇报。

## 10. 当前可向导师汇报的安全结论

可以说：

> 现在 HLA 已经形成最完整的多年份、多 seed、五情景证据，显示 DQN 可以在多个年份接近或达到 DSSAT auto 的产量，同时显著降低氮投入，并在部分年份降低灌溉。YC 和 FQ 在同一冻结框架下完成了 seed0/seed1 复核，说明框架能跨站点运行，但尚未在这两个站点全面超过 auto 和 official expert。LC 和 SY 仍属于诊断站点：LC 有潜力但 seed 资源策略不稳定，SY 有高产响应但资源投入过高且 IC 敏感。因此下一步应先做站点分级和 LC 冻结框架复核，而不是马上修改 reward 或盲目长训练。

不能说：

> 五个站点都已经实现 DQN 同时超过 expert 和 DSSAT auto。

也不能说：

> 只要继续训练或调整 IC，就一定能让所有站点成功。

## 11. 建议路线

当前最稳妥路线：

1. HLA 作为主成功案例整理汇报。
2. YC/FQ 作为统一冻结框架的跨站点复核案例，强调“接近强 baseline，但还不是全面胜出”。
3. LC 作为下一优先级正式复核站点。
4. SY 作为高产但资源效率不足的边界案例。
5. reward 参数和氮淋洗先不进入正式主线，保留为导师确认后的扩展方向。
