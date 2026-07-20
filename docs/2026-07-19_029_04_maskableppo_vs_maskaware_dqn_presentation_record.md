# 029_04 PPO–DQN 对照汇报 PPT 制作记录

状态：`completed`

## 目标

把 029_00—029_04 的冻结比较结果整理为与 028_16 PPO 汇报文件一致的朴素学术版式，包含 DQN 的逐年精确终值表、阶段措施、seed 判定及完整五情景八联日值图。

## 模板与方法

- 版式来源：`docs/2026-07-19_028_16_existing_results_advisor_presentation_plain_full_daily.pptx`；
- 完整审计源文件 44 页；
- 使用 artifact-tool 的 template-following 流程复制源页框架并原位替换内容；
- 保留白底、黑色标题、灰色表头、细分隔线、图像框、页脚和页码位置；
- 没有叠加新主题或装饰样式。

## 内容结构

共 40 页：

1. 标题；
2. 公平比较口径；
3. 总体算法结论；
4. 17 年逐年 winner-seed 图；
5. 五站点汇总表；
6. 17 个站点年各 2 页：精确终值/措施/seed 证据页 + 完整八联日值图；
7. 最终主线建议。

## 数据来源

- PPO：028_13/028_16 冻结结果；
- DQN：029_02 五站训练、029_03 冻结跨年、029_04 日值证据包；
- 所有终值表均从 DQN 五情景 summary CSV 重新填充，不沿用或猜测 PPO 候选数值；
- N=0 时 PFP_N 和对应差值显示为 NA。

## QA

- template plan 校验：`pass`，0 issue；
- 最终 PPT：40 页；
- 全部 40 页使用 artifact-tool 渲染为 PNG 并检查；
- 对标题、方法页、总结图、站点汇总页、代表数据页和结论页进行了高分辨率抽查；
- `slides_test.py`：`Test passed. No overflow detected.`；
- DQN 17 年完整日值图均使用 `contain` 式完整呈现，没有裁成措施子图。

## 非科学性失败记录

1. starter deck 首次生成时，辅助脚本调用系统 `python3` 制作 contact sheet 失败；starter PPTX 已成功生成。随后去掉 contact-sheet 参数原样重跑，未影响内容。
2. 第一次编辑时使用 starter inspect 的元素 aid；PPTX 再导入后 aid 会重建，导致找不到元素。改为按继承框架的精确位置匹配原有元素后原位改写。
3. 第一次文本写入把 Text 对象本身传给 `replace`，导致文本未更新；改为直接设置继承文本框的文本内容，保留框架样式。
4. 初版 17 年比较图沿用热图窄框而被裁切；在模板计划中显式登记 `rewrite-and-reposition`，扩大继承图框并用 `contain` 后复核完整。
5. 初版对 N=0 的 PFP_N 差值错误地把空字符串数值化为 0；修复为空值显式 NA，并重新生成和渲染全部页面。

## 输出

`docs/2026-07-19_029_04_maskableppo_vs_maskaware_dqn_comparison.pptx`

## 结论

PPT 与底层 CSV/图表的结论一致：当前配置不支持全面切换到 DQN；PPO 保留主线，DQN 作为正式算法对照和局部站点候选。
