# 3.2图表质量与一致性记录

状态：FIGURES_READY_WITH_CAVEATS。本轮只读取既有冻结统计结果，未重算bootstrap或执行新检验。
图形合同：天气增强未显示统一改善，平均产量、年际波动与资源权衡分别由A、B、C提供互补证据。quantitative grid；Python matplotlib；183mm双栏宽，主图160mm高，英文8pt主体，PNG400dpi，PDF TrueType及SVG文本保留。未指定目标期刊，尚需按目标期刊最终版式检查。
数据：135审计五站均产/SD区间，076其他三指标区间，079 WP_ET区间，126组均值与年度端点。逐项差值对齐检查通过；H/A不混入forecast。30条表格记录，50条年度差值。SY/HL/LC/YC/FQ正差年数1/5/3/2/7；HL最大年度增益21.3408，最大年度损失−1208.8058 kg/ha。正差年数非独立二项重复。
Panel C使用组均值百分比，不同单位指标不混成综合分；以零为中心、对称全范围色标，未截断极值，无星号。历史分母均非零；表格缺失Welch区间标NA。年际SD仅组内policy跨年样本SD均值，不误写seed间SD。
正文只增加图表引用，保留原结论与数字。Table Sx包含正文重点提及的HL WP_ET区间及所有五站SD、Welch值。所有年度值以CSV保存，Figure Sx绘制全部十年及相同纵轴，无平滑、无逐年检验。
限制：READY_WITH_CAVEATS，未校正多重比较、8seed小样本与固定评价年份；保留审计中的天气数据来源和非认证pristine final test限制。HL的bootstrap排零不写稳健显著性。图形完整不补齐天气QC来源缺口。不以跨零证明无效，不以SD降低推断适应能力。


## 图表与正文对应
第一段A；第二段C与Table Sx；第三段Figure Sx；第四段B；第五段总体总结。

# Figure captions

**Figure X | Effects of weather augmentation on management performance and interannual yield variability across five sites.** H denotes historical-weather training and A denotes weather-augmented training. All differences are A minus H. **A**, Difference in group-average ten-year mean yield, with exploratory 95% percentile bootstrap confidence intervals. **B**, Difference in the group mean of policy-specific interannual yield sample standard deviations (SD; ddof=1), with bootstrap intervals (solid blue) and Welch intervals (dashed orange) vertically offset for readability. Each SD is calculated across the same ten evaluation years, not across training seeds. The shared point marks the estimated difference. For HL, the positive point-estimate direction survives deletion of any one seed, but zero exclusion depends on interval method. **C**, Descriptive percentage changes, calculated as 100×(A−H)/H from group means. PFP_N is grain yield per unit N applied; WP_ET is grain yield per unit evapotranspiration. Cell colours encode numerical direction rather than management benefit. Each regime includes eight training seeds evaluated over 2014–2023. Bootstrap resampling units are policies, independently within regimes. All 95% intervals are unadjusted for multiple comparisons and reflect training-seed uncertainty conditional on the fixed evaluation years and weather schedules; they do not establish equivalence, causal mechanisms or performance in arbitrary future years. No significance symbols are used.

**Figure Sx | Year-specific yield differences between weather-augmented and historical training.** Points show the difference between the annual group means of eight policies per regime (A−H), for all evaluation years from 2014 to 2023. Lines connect the observed annual estimates without smoothing. The five sites share the same vertical scale; dotted lines indicate zero difference. Years are repeated evaluations of the policies, not independent training replicates. No annual significance tests were performed.

**Table Sx | Management performance and interannual variability by site and training regime.** Values report H and A group means, absolute differences (A−H), percentage differences relative to H, and archived exploratory 95% bootstrap intervals for absolute differences. Welch intervals are supplied only for interannual SD, as audited. Yield, irrigation, N application, PFP_N and WP_ET are averaged over valid years within each policy, then across the eight policies. PFP_N is the mean of annual yield/N ratios, not the ratio of mean yield to mean N; zero-N annual values remain undefined rather than zero. WP_ET is the mean of annual yield/(10×ETCP) ratios (kg m−3). Interannual SD is computed separately for each policy over ten years (ddof=1) and then averaged across policies. Missing interval fields mean not supplied for that statistic, not zero uncertainty. Intervals are unadjusted for multiple comparisons.


PFP_N有效策略数：FQ历史组7/8（seed5零施氮）、YC增强组7/8（seed6零施氮）；其余组8/8。所有其他核心指标均8/8。未定义不补零；因此PFP_N百分比是各组有效策略均值的描述性比较，缺失模式不同，不能直接当作全部8策略整体氮效率改善。
