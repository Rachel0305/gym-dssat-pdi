# 042_02 SYA lowIC 全归一化天气/预报 observation + teacher warm-start + 氮胁迫响应 PPO

## 背景

`041_06` 的敏感性审计显示，当前 PPO 更像是在学习：

- DAP；
- 作物生长量；
- 已累计灌溉/施氮量；
- 大致阶段模板。

而不是明显响应：

- 降雨；
- 温度；
- 土壤水分；
- SWFAC/NSTRES 胁迫变化。

`042_00` 直接加入 rain/tmin/7 天天气窗口后，裸 PPO 训练效果变差，说明“直接增加 observation 信息”不足以解决问题。

## 本任务目的

建立一个最小、可审计的 `042_02` 版本：

1. 检查并显式归一化所有 PPO 输入变量；
2. 加入 rain/tmin/past7/future7 weather features；
3. 加入与已有 SWFAC guardrail 对称的 NSTRES 过程惩罚；
4. 复用 `041_04` balanced teacher warm-start；
5. 先 smoke，再 100K 正式训练。

## 输入数据边界

- 数据源固定为：
  - `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点固定为：
  - `SYA`
- 训练/验证对象沿用 teacher warm-start 路线：
  - 2014--2023 selected teacher trajectories
- 本任务不修改 MZX、WTH、SOL、CUL、jinja2 原始输入文件。

## observation 设计

原始 25 维 observation 顺序按 `041_06` 审计结果固定为：

1. cumsumfert
2. dap
3. dtt
4. ep
5. grnwt
6. istage
7. nstres
8. rtdep
9. srad
10--18. sw_1 ... sw_9
19. swfac
20. tmax
21. topwt
22. totir
23. vstage
24. wtdep
25. xlai

新增 5 维：

26. rain_today_mm
27. tmin_today_c
28. rain_past7_mm
29. rain_future7_mm
30. tmean_future7_c

全部输入使用固定物理尺度缩放到大致 0--1 附近，不使用验证结果反推 scaler。

## reward 改动

沿用 `040_40` reward 和约束：

- yield/resource reward；
- stress relief bonus；
- SWFAC process guardrail；
- late irrigation reserve mask；
- terminal yield guardrail。

本任务只新增：

```text
nstres_penalty_unscaled = 50 * max(0, NSTRES_after_step - 0.05)
reward = reward_04040 - nstres_penalty_unscaled * reward_scale
```

参数来源：

- `threshold=0.05`：与现有 SWFAC guardrail 阈值一致；
- `coef=50`：与现有 SWFAC guardrail 系数一致；
- 不扫描、不根据结果调整。

## 训练

- 算法：MaskablePPO；
- BC warm-start：复用 `041_04` balanced nonzero sampling；
- seed：0；
- 默认 PPO fine-tune：100000 steps；
- checkpoint：25000, 50000, 75000, 100000。

## 成功/失败判据

本任务主要回答两件事：

1. normalization + weather features 是否让 observation 链路正确、稳定；
2. teacher warm-start 后，PPO 是否比 `041_04/040_40` 更能在不同年份给出天气/胁迫响应差异。

不得仅凭本任务宣称“天气预报一定有效”。

如果 100K 结果仍然表现为固定模板、对天气/胁迫无响应，则停止该分支，转向更强的策略结构或明确的天气响应策略学习设计。
