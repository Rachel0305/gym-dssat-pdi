# Results 3.2 Bootstrap统计方法审计与稳健性检查

日期2026-10-31。编号按prompts中任务编号确定：已有最高134；文件名中的500k是训练预算，不作为任务编号。
复用2014—2023五站H/A各8seed，不训练、不模拟、不生成天气、不修改旧数据。检查800年度唯一键、完整性、模型身份、修订条件、训练与天气有效性审计；相同seed数字不直接视为配对。
独立复现076平均产量4000次bootstrap与080年际SD差10000次bootstrap，包括原随机种子、跨站/指标RNG消耗顺序、离散percentile索引。每seed十年SD使用ddof=1，再比较组内8个SD的均值。固定年份，以训练seed为单位，不将年度当独立训练。
仅两项敏感性检查：逐次从H或A删除一个seed（16次/站，不同时删两组），报告delta_sd范围及方向，重点HL；Welch-Satterthwaite 95%区间对照。未校正多重比较，只作探索性推断，不将跨零当无效，不用统计稳健性冒充天气质量认证或因果机制验证。
输出results/section32_bootstrap_robustness_20261031唯一目录中的审计报告、五站CSV、Methods、3.2定稿及脚本；项目记录另存docs/2026-10-31_section32_bootstrap_audit.md与.pptx，所有旧文件不覆盖。开头回答统计正确性、8seed推断边界、HL稳健性、其他站点变化及READY_TO_FREEZE/READY_WITH_CAVEATS/BLOCKED_BY_STATISTICAL_ERROR。分类A/B/C。无实质错误即结束，不延伸研究。
使用独立Git任务分支仅备份本任务及必要小型复现输入，保护现有未提交修改，不强推。用户本请求授权GitHub备份；远程失败如实记录。
