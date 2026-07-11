# 020_11 HLA 五情景补齐与 n-step 框架冻结

## 先核对历史

- `020_09/020_10` 已完成 HLA2010 n-step 三个 seed 在 2007、2015、2016、2022 的迁移，不重新训练。
- 旧文件中的 `expert_2007_shifted` 实际是 2007 年农民/试验田管理记录平移，不能再称为官方 expert。
- 推文 expert 已在 `018_03` 对 HLA2010 实现，但 2007、2015、2016、2022 尚未补齐。
- 2016、2022 尚缺农民记录平移情景。

## 正式五情景定义

1. Null
2. Recorded farmer practice（2007 日程平移）
3. DSSAT auto
4. Official extension expert（推文东北春玉米区中值，固定 DAP 实现）
5. n-step DQN（内部区分 seed0/seed1/seed2）

## 执行要求

- 年份：HLA 2007、2010、2015、2016、2022。
- 对缺失的 recorded farmer 与 official extension expert 使用该年份与 null/auto/DQN 相同的 WTH、SOIL、CUL、IC=1 MZX 输入回放。
- 推文 expert：固定 DAP 0/30/50/65/85/110；总施氮 300 kg/ha；总灌溉约 266.25 mm；单次灌溉超过 PDI 上限时拆至次日。
- recorded farmer：DAP0 施氮 165 kg/ha；DAP49/70/95 各灌溉 10 mm。
- 不改奖励、动作、预算或已有模型；不覆盖旧结果。
- 降雨必须从同年份 null/WTH 按 DAP 附加，不能使用 018_03 中全零的 `rain_obs`。
- 实际管理总量以 `MgmtEvent.OUT` 为核对依据。

## 输出

- 所有年份统一日值长表。
- 所有年份统一管理事件表。
- 所有年份统一汇总表；每年 7 行（4 条基线 + 3 个 DQN seed）。
- 每年一张包含全部 seed 的高对比度五情景过程图，PNG/SVG/PDF。
- 输入哈希审计、两套专家日程表、中文实验记录。
- 冻结 HLA n-step=5 配置为共享代码和 JSON 清单，后续 YC/FQ 必须从该共享配置读取。

## 图形契约

- 核心结论：区分农民记录与官方推广 expert，并在相同站点年份输入下审计三 seed 的产量、资源投入、胁迫与统一奖励。
- 证据链：降雨 → 水/氮胁迫 → 灌溉/施氮事件 → 籽粒/生物量 → 累积奖励。
- 类型：quantitative grid。
- 后端：Python/matplotlib。
- 导出：可编辑 SVG/PDF + 高分辨率 PNG；白底、黑色 null、红色虚线 recorded、深黄色 auto、蓝色 official expert、绿色系 DQN。
