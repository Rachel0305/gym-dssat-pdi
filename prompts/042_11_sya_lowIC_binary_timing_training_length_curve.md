# 042_11 SYA lowIC binary-timing PPO 训练长度曲线

## 背景

042_10 的 2K smoke 跑通了 binary-timing PPO：

- 灌溉剂量固定为 45 mm；
- 施氮剂量固定为 80 kg/ha；
- 动作空间从 9/16 档剂量组合压缩成 4 个时机动作；
- 继承 040_36 的 lowIC、reward 和安全约束。

但是 smoke 出现一个关键现象：

- 1000 步时验证年份动作序列有 9 套，出现跨年差异；
- 2000 步时动作序列坍成 1 套模板。

因此不能直接跑 100K。需要先审计训练长度与动作多样性的关系。

## 目标

固定 binary-timing PPO 框架，只改变训练长度观察点：

- 1000
- 2000
- 5000
- 10000
- 25000

本任务不是选择最终 checkpoint，而是记录：

- endpoint 指标；
- 动作签名多样性；
- 灌溉/施肥总量跨年差异；
- 灌溉/施肥事件 DAP 跨年差异；
- 是否出现模板化坍缩。

## 固定不变

- 输入目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 年份划分：沿用 032/040 主线 half split
- 算法：MaskablePPO
- reward：继承 040_36/040_28
- safety：继承 040_36
- 动作档位：灌溉 `[0,45]`；施氮 `[0,80]`

## 判定

本任务不宣称成功训练，只产生诊断分支：

- A：`early_diversity_then_late_collapse`
  - 早期 checkpoint 有动作多样性，但后期 checkpoint 变模板。
  - 下一步应设计早停/响应性 checkpoint selection guardrail，或改 PPO 更新稳定性。

- B：`diversity_persists`
  - 多个 checkpoint 保持动作多样性，且 endpoint 指标可接受。
  - 下一步才允许进入正式候选选择。

- C：`no_useful_diversity`
  - 所有 checkpoint 都模板化或 no-op。
  - 下一步不应继续该 binary-timing PPO 路线。

## 纪律

- 不根据结果补充新的训练长度；
- 不把本任务中某个 checkpoint 直接宣布为最终策略；
- 若结果显示早期好、后期坏，不能事后硬挑早期 checkpoint，必须另写 selection guardrail。
