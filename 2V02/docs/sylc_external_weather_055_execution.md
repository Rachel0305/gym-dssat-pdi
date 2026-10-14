# 055 全年WGEN路径与小样本天气质量检查

2026-10-12，已编写并执行 `prompt_02/055_sy_full_year_wgen_path_and_weather_quality.md`。先SY、后LC，单进程/CPU线程1。暂停论文，未训练PPO，未变更正式奖励、观测、16格动作、掩码、品种、土壤或冻结结果，未Git push。

## 路径结果

使用054已验证的gym-DSSAT运行接口和原CLI，在独立诊断FileX副本中将模拟/初始土壤日期设为1月1日，保留播种日期，设置报告收获日为12月31日。诊断FileX的作物/土壤结果不得用于管理对比，正式训练/验证模板没有修改。DSSAT实际INFO.OUT报告版本4.8.0.024-stable；不是仅依据上游源码断言运行成功。

SY和LC实际运行均输出2005-01-01至2005-12-31连续365天；各站2008闰年试验输出2008-01-01至2008-12-31连续366天。每条天气日期以原始DSSAT状态yrdoy解码，已核验完整日期列表，无缺日/重复日。相同weather seed1001两次全序列哈希一致，seed1002与其不同。运行CLI SHA256与候选一致，WTHER=W，PDI rseed1核验通过。该配置是现有官方日循环内的全年诊断路径，不是独立WeatherMan工具，不声称旧的作物季生成序列与从1月1日生成的天气相同。

## 全年小样本结果

各站weather seed1001–1010共10个独立全年样本，每站3650天，不计重复复现或闰年测试。实际生成样本的四变量均无缺失、负降水/辐射或TMAX<TMIN，日期完整，物理筛查通过；未clip或改写任何生成值。单次约3秒，最大进程树RSS约178MB，未触发1536MB/300秒限制。

|指标|SY拟合期9年|SY生成10样本|LC拟合期14年|LC生成10样本|
|---|---:|---:|---:|---:|
|平均年降水mm|735.36|763.74|538.31|527.82|
|平均年湿日数（RAIN>0）|95.33|96.20|76.57|79.50|
|年平均SRAD MJ/m²/day|13.58|13.68|13.42|13.48|
|年平均TMAX ℃|14.38|14.37|17.79|17.64|
|年平均TMIN ℃|3.05|2.98|5.86|5.80|
|平均年度最长无雨段 天|27.89|24.80|43.64|45.30|
|平均年度SRAD/TMAX相关|0.58|0.62|0.63|0.61|

SY年降水均值差约+3.86%，LC约−1.95%；这只是描述，不是预设统计通过阈值。保存12个月雨量/湿日/温度辐射均值方差极值、年度干湿持续段、变量相关、分位数/极值对比。SY生成样本最大单日降水最高137.48mm，拟合样本最大101.3mm；LC生成最长无雨段最大84天，拟合最大71天。随机生成可超出有限拟合样本极值，本任务未把这种超出直接标为物理失败，也没有宣称这些尾部已完成气候合理性证明。

10样本不足以证明分布等价、极端气候代表性或完整训练weather seed1001–1080的质量。LC扩展拟合期仍保留历史补充来源/迁站/混合数据限制。当前结论为`PASS_FULL_YEAR_ENGINEERING_AND_PHYSICAL_SCREEN_WITH_DISTRIBUTION_NOTES`，只允许继续设计并执行固定天气注入小规模验证，不能据此启动8seed100K。

## 失败与修复

LC attempt001：运行输出目录从054换到055后，CLI路径错误地随输出目录变化，FileNotFoundError；保留失败快照与gate。修复为固定054/LC_v2输入路径，未改CLI。

LC attempt002：原LC模板已有HARVEST DETAILS，又插入新段，DSSAT选择旧段，于10月2日终止，仅275天。删除诊断副本的旧收获段后再写单一年底收获段；attempt003完成365天及复现。原冻结模板不变。

旧尝试初始化失败还出现环境析构时NoneType异常，失败证据保留，不将其误写为天气物理失败。成功尝试均完成close并保存快照。脚本修正前版本在`backups/sylc_054_scripts/`保留。

## 下一阶段

将全年天气按WTH要求在新目录序列化，核验舍入误差与实际DSSAT每日天气一致性，先做跨年（含闰年）固定天气注入smoke，再做SY 2K PPO smoke，后做LC。保留冻结农业站2014–2023验证及四管理基线。不使用本次延长收获日期的诊断FileX训练PPO。完整训练天气队列需再做全候选QC，不能靠本次10样本外推放行。PPO及正式训练本次均未运行。

## 产物

`results/sylc_external_weather_055/`：SY `path_gate.json`，LC `LC_path_gate_attempt003.json`，两站闰年gate，两站ensemble gate、年度/月度/分位数CSV，运行快照/原始状态/日期精确天气CSV。可复现脚本：`run_sy_full_year_path_055.py`、`screen_sy_full_year_ensemble_055.py`、`verify_sy_leap_and_lc_full_year_055.py`及受备份保护的`run_sylc_wgen_runtime_smoke_054.py`。

官方源码参考：[CSM v4.8.0.24](https://raw.githubusercontent.com/DSSAT/dssat-csm-os/v4.8.0.24/CSM_Main/CSM.for)、[WGEN v4.8.0.24](https://raw.githubusercontent.com/DSSAT/dssat-csm-os/v4.8.0.24/Weather/WGEN.for)。本次运行证据解决了此前“无已验证全年入口”的工程问题，仅对这里保存的运行配置成立。
