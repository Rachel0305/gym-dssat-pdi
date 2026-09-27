# YC 随机天气组会汇报摘要

- 生成：由2004–2013 frozen fitting weather 拟合 CNYC.CLI，经历史 DSSAT/WGEN runtime 按 seed/context 生成；本轮只读取 004_17 已归档的逐日状态天气。
- 设计：training seeds1001–1080 (80套)，held-out seeds1081–1100 (20套，严格不入训练)；Observed 2014–2023 (10年)。
- 每日变量：DATE、DAP、RAIN、SRAD、TMAX、TMIN；WGEN DATE 是运行时参考日期，crop year 见 manifest；observed DATE 来自正式 WTH。
- Training 整季降雨中位数 466.1 mm (P5–P95 275.4–768.3)；held-out 515.2 mm；observed 198.7–709.0 mm。
- Training Tmax 均值中位数 30.9°C，SRAD 均值中位数 16.5 MJ/m²/day；held-out 分别 31.4°C / 16.0；observed 范围分别 30.1–32.8°C / 16.5–20.7。
- FULL80 vs EXACT78：SENSITIVITY_CHANGES_IDENTIFIED；触发预设敏感性判据 6 项。2014、2016、2019 整季降雨在 FULL80 范围内、但在 EXACT78 范围外，不能称 tail identification 完全稳健。
- 2014/2019 与2017/2023的天气并列展示，不解释产量因果。
- 1003、1036 为 provisional，98/100 历史逐日 hash 精确匹配；100/100 归档文件完整。
- 仍需004_18B解决两条 provenance，随后004_19做正式 climate coverage audit，才能决定是否扩天气池。

## 组会建议图

Figure 1 设计与核验数；Figure 2 降雨；Figure 3 干旱/极值；Figure 6 生育阶段降雨；Figure 8 observed 百分位；Figure 9 关键年份。

Descriptive climate-position analysis; formal coverage verdict pending provenance closure / sensitivity review.
