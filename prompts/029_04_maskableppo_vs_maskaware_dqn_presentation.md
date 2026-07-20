# 029_04 MaskablePPO vs mask-aware DQN 汇报 PPT

## 目标

基于 029_00—029_04 已冻结结果，制作与 `docs/2026-07-19_028_16_existing_results_advisor_presentation_plain_full_daily.pptx` 相同的朴素学术版式汇报文件，用于回答是否需要把当前阶段型 MaskablePPO 主线切换为 DQN。

## 输入

- PPO 17 年结果与 028_16 汇报 PPT；
- DQN 17 年、51 seed 配对结果；
- DQN 17 年五情景终值表、阶段措施表、逐日表与八联图；
- 029_04 站点级与逐年算法比较表。

## 内容要求

1. 说明公平比较范围与算法差异；
2. 给出 29/51 vs 23/51、10/17 vs 8/17 的总结果；
3. 给出五站点汇总；
4. 每个站点年提供 DQN 精确终值/措施/seed 证据页；
5. 每个站点年提供完整八联日值图，不裁切为措施子图；
6. 明确 DQN 的优势年份、PPO 的优势年份和持平年份；
7. 结论不得扩大为算法普遍优劣，只回答当前配置是否值得切换。

## 样式与 QA

- 复用 028_16 的字体、页边距、线条、灰色表头、页码和单图布局；
- 不增加装饰性主题；
- 使用 artifact-tool 编辑/导出；
- 渲染全部页面并检查标题换行、表格溢出、图片裁切和页码；
- 运行 slides_test.py。

## 输出

- `docs/2026-07-19_029_04_maskableppo_vs_maskaware_dqn_comparison.pptx`
- `docs/2026-07-19_029_04_maskableppo_vs_maskaware_dqn_presentation_record.md`
