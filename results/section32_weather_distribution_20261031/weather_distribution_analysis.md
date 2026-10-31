# 天气分布与Discussion证据

状态：DISCUSSION_EVIDENCE_PARTIAL

Q1. 五站是否扩大训练天气分布？按站点和指标回答见下表；范围与SD并不总同向。SY/LC是选择库窗口，HL/FQ是已归档完整窗口，YC只代表EXACT78子集，不能统一宣称完整实际训练分布已恢复。
Q2. 是否改善评价天气覆盖？下表给四指标min/max覆盖；另存P5–P95。覆盖只是固定10年的单变量经验支持，不等于联合分布或独立气候总体覆盖。
Q3. 能否解释缺乏一致收益？可检验新增窗口与评价输入的错位，但不能从覆盖推断收益因果；完整训练暴露及来源限制使解释仍为部分证据。
Q4. 能否解释站点表现差异？五站增强方法、样本池及来源不同，不能据五点归纳普遍规律或判定天气分布决定产量。
Q5. 可以开始写Discussion吗？可以写有边界的事实与限制；状态为DISCUSSION_EVIDENCE_PARTIAL，不阻止讨论，也不声称机制闭合。

## 预先固定的分析窗口
本轮用既有DAP0–30、31–60、61–90阶段并集，固定播种日至DAP90共91天。它是固定管理窗口，不是全生育季。所有H/A/T在同站点同一相对窗口，不随策略成熟或收获长度改变；不将全年指标替代生育期。播种DOY来自原runtime build_env_config的物候中位数。未观察天气分布后调阈值；干日沿用既有RAIN<=0定义。温度用平均TMAX；四变量均参与增强且YC已有SRAD问题，所以保留SRAD均值，未新增尾部挖掘。

## 五站证据

### SY（产量A−H -481.6 kg/ha）
seasonal_precipitation:范围176.0→159.1, SD62.4→50.9; maximum_dry_spell:范围19.0→11.0, SD5.3→3.8; seasonal_temperature:范围2.1→2.1, SD0.7→0.7; srad_mean:范围3.0→2.7, SD0.8→0.7
seasonal_precipitation:H7/10→A7/10; maximum_dry_spell:H10/10→A9/10; seasonal_temperature:H5/10→A9/10; srad_mean:H6/10→A7/10

### HL（产量A−H -324.0 kg/ha）
seasonal_precipitation:范围490.1→508.8, SD139.3→95.3; maximum_dry_spell:范围83.0→12.0, SD24.8→2.6; seasonal_temperature:范围4.0→4.1, SD1.1→0.9; srad_mean:范围3.1→3.7, SD1.0→0.9
seasonal_precipitation:H9/10→A10/10; maximum_dry_spell:H6/10→A10/10; seasonal_temperature:H10/10→A10/10; srad_mean:H9/10→A8/10

### LC（产量A−H -329.0 kg/ha）
seasonal_precipitation:范围375.8→324.5, SD132.7→108.3; maximum_dry_spell:范围13.0→19.0, SD3.8→4.4; seasonal_temperature:范围2.9→2.3, SD1.0→0.7; srad_mean:范围3.2→1.2, SD0.9→0.4
seasonal_precipitation:H9/10→A9/10; maximum_dry_spell:H9/10→A10/10; seasonal_temperature:H6/10→A8/10; srad_mean:H3/10→A3/10

### YC（产量A−H -247.2 kg/ha）
seasonal_precipitation:范围631.4→824.6, SD175.5→158.7; maximum_dry_spell:范围82.0→30.0, SD24.0→4.8; seasonal_temperature:范围3.5→3.4, SD1.0→0.7; srad_mean:范围1.2→3.2, SD0.4→0.6
seasonal_precipitation:H9/10→A7/10; maximum_dry_spell:H10/10→A10/10; seasonal_temperature:H8/10→A10/10; srad_mean:H3/10→A7/10

### FQ（产量A−H +60.7 kg/ha）
seasonal_precipitation:范围818.5→954.3, SD249.3→191.7; maximum_dry_spell:范围9.0→33.0, SD3.3→7.3; seasonal_temperature:范围2.1→3.3, SD0.7→0.7; srad_mean:范围3.9→4.1, SD1.2→0.7
seasonal_precipitation:H9/10→A10/10; maximum_dry_spell:H7/10→A10/10; seasonal_temperature:H6/10→A10/10; srad_mean:H8/10→A8/10

## 来源与冲突处理
本轮发现004_17、18B、19晚于004_16恢复gate失败。旧“无法恢复”不能覆盖后来EXACT78证据；004_19整季指标因长度99–110天不等，不直接混入本轮固定91天图。1003/1036仍按18B unresolved处理；旧manifest的MISMATCH不自行升级为确定逐日不符。FULL80候选不纳入正式A。
SY/LC旧WGEN停止记录后有054–059新的日历/注入链。057每站20个源日历映射90份WTH，不把90当90独立气候；本轮验证所有8seed选择manifest与源SHA，选中窗口不等于已证明每个窗口的91天全被PPO访问。055旧状态CSV晚标一天，不使用，改读057最终WTH。
HL/FQ读取所有8seed逐episode CSV及hash，完整91天才进入分布；截断窗口单列exclusions，不能声称覆盖所有训练日。相同窗口去重；共享训练天气不计8倍独立样本。
H来自实际渲染使用的multisite冻存WTH，频次未完整重建，因此H按年份等权；A同时提供已选回合频次/窗口摘要，不将其与H称为严格匹配暴露权重比较。
FQ2018 T使用125已修订评价快照；该启发式TMIN修订是评价条件，不是独立真值。H/其他T保留既有填补或混合来源；异常天数见数据，不以“物理无异常”替代源数据QC。

## 事实、关联与假设
事实：图表和CSV中的单变量范围、SD和覆盖计数。关联：同站增强窗口覆盖变化与既有产量变化并列，仅描述。未验证机制：PPO是否利用信息、训练优化难度、奖励限制、管理时序是否造成减产；本轮未读取critic/网络或开展新模拟。

## 是否继续
本轮到来源、描述分布及解释边界为止。可开始Discussion事实段；先核对未闭合的实际暴露/晚季与来源再作更强主张。若后续授权管理日志分析，可单独检查灌溉施氮与胁迫时序，本轮不启动。