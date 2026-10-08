# 041 封丘 D222 降雨规则与静态 WGEN 输入恢复

## 结论

依据用户本轮明确的数据处理决定，FQA 2005–2013 年拟合天气已在独立目录构建；`FQA_FITTING_WEATHER=PASS`、`FQA_STATIC_CLI=PASS`。这是**静态输入门禁**：未运行 DSSAT、WGEN 随机天气或 PPO，随机天气的实际读取、不同 seed 多样性和分布 QC 均为 `NOT_RUN`。机器结论见 [final_gate.json](../results/fqa_weather_resume_041/final_gate.json)。

本轮接受国家生态科学数据中心 CERN D222 的 `20-20合计(mm)` 作为封丘降雨主值。原表有数值日完全照录，包括 2012-09-02 的 240.0 mm 和 2013-05-26 的 204.0 mm；不以外部源量级不一致为由改写。D222 空白按 0 mm 转换用于拟合，这一操作明确属于项目建模规则，**转换后的零不称为原表实测零**。此前“2013 年 157 天缺外源确认”的严格审计计数在本规则下不再作为 FQA 来源门槛，但历史报告原样保留。

## 输入与结果

- 拟合期固定为 2005-01-01 至 2013-12-31，共 **3,287** 个连续日期；没有使用 2014–2023 年作拟合。
- 降雨从项目内完整 D222 工作簿读取，**395** 个有值日照录、**2,892** 个空白日转为 0。原格状态、原值和每个变量的来源见 [daily_provenance.csv](../results/fqa_weather_resume_041/daily_provenance.csv)；完整四变量输入见 [fitting_weather.csv](../results/fqa_weather_resume_041/fitting_weather.csv)。
- TMAX、TMIN 各有 **83** 个原缺失日，SRAD 有 **93** 个原缺失日和 **1** 个超出既有物理范围的异常日。这些位置采用 008 已通过筛选的训练期 NASA POWER 偏差校正候选；有效本站值不替换。原始文件、候选文件、旧 CLI 计算脚本和坐标来源文件的路径与 SHA-256 在 `final_gate.json`。
- 逐日检查通过：日期连续、四变量有限、RAIN/SRAD 非负、TMAX ≥ TMIN；每个 D222 有值日最终 RAIN 与原值相同，每个原空白日最终 RAIN 为 0，候选值仅用于指定的温度/辐射缺失或异常位置。
- 年度雨量与雨日见 [annual_summary.csv](../results/fqa_weather_resume_041/annual_summary.csv)。特别是 2011–2013 年雨量分别为 **1,614.4 / 1,378.2 / 1,144.4 mm**，这是按本轮转换规则计算的 D222 年量，不是独立雨量计校验结果；本轮依用户决定保留其数值。

复用项目 [YC CLI 计算脚本](../scripts/build_dssat_cli.py)的月统计与固定列宽序列化逻辑，生成 [CNFQ.CLI](../results/fqa_weather_resume_041/CNFQ.CLI)及 [12 个月参数](../results/fqa_weather_resume_041/monthly_wgen_statistics.csv)。该脚本对 YC 的 `2004/10 年` 有硬编码；本轮仅在新 FQA CLI 文本中精确替换为 `2005/9 年`，并单独核对结果，没有修改旧脚本。站点头取 FQ 既有 WTH 的 `35.020°N / 114.542°E / 300 m`。月参数回算与静态 CLI 格式检查均通过；静态格式通过不代表 PDI 运行时已读取此 CLI。

## 执行记录与下一阶段

第一次运行因新脚本遗漏旧 008 审计中的 SRAD >60 异常上限而在写产物前停止；已按原门槛修正，过程见 [attempt_01_failure.md](../results/fqa_weather_resume_041/attempt_01_failure.md)。最终脚本是 [build_static_fitting.py](../results/fqa_weather_resume_041/build_static_fitting.py)。原始 D222、T2、D32、旧 WTH、历史审计和实验结果未修改。

下一阶段对 FQA 做**一个站点、一个 CLI、一个固定 weather seed、一个作物季**的隔离 DSSAT/WGEN 运行时 smoke，检查实际读取 `CNFQ.CLI`、四变量逐日捕获、坐标传递与资源占用。通过后再做有限随机天气分布 QC；正式 8-seed PPO 仍需单独门禁。HLA 另行处理 2004 年长连续空白的既有补源选择和原生 FIELD 坐标问题，本轮没有修改 HLA 输入或训练。
