# 054 SY/LC天气增强前置审计与DSSAT运行验证

2026-10-12。已编写并执行 `prompt_02/054_sylc_external_weather_readiness_and_runtime_smoke.md`，暂停论文。未修改奖励、16格动作、掩码、PPO、冻结天气、模型或原结果；未Git push。

## 结果

|站点|候选拟合期|数据及统计检查|实际WGEN季节smoke|全年生成检查|PPO smoke/八seed训练|
|---|---|---|---|---|---|
|SY|2005–2013|此前候选、CLI及独立统计复算通过|3次通过；相同seed天气相同，不同seed天气不同|未运行|未运行|
|LC|2000–2013|5114天物理QC、12个月×15项独立复算、CLI固定宽度检查、一月主估计及逐一剔年检查通过|3次通过；相同seed天气相同，不同seed天气不同|未运行|未运行|

SY weather seed1001重复两次，各138个状态/步骤；seed1002为129。LC seed1001重复两次，各95个状态/步骤；seed1002为94。不同天气产生不同作物季长度，不据此认定步数错误。CLI SHA256分别为SY `e4fce2601338ba689e294af167bd35e236925a0e9b41c873e2741ce5972b7a0f`、LC `6f73f8ea7fbe04b0ce042e2d632ec32d7648f1a72c59a79bb3dd8c1b60c2face`。实际运行文件哈希与候选一致，FileX WTHER=W，PDI rseed1=指定weather seed。各次RAIN/SRAD/TMAX/TMIN无缺失、负雨量/辐射或温度倒置，均含降水。峰值进程树RSS约126–129MB，单次约1.1–1.4秒，无超限。

保存CLI/FileX/PDI yaml/DSSAT输出快照、原始逐日状态、天气CSV、QC和运行gate；所有状态时间的实际`yrdoy`保留，当前CSV由模拟开始日连续排序映射，后续需按原始`yrdoy`独立复核日历语义，不能把季节序列称为全年天气。

## LC来源审计修正与限制

第一版把原XLS空白全部作为缺口：SRAD408、TMAX393、TMIN393，均值补缺后2002年1–2月、2003年1月的辐射方差接近零；`std==0`未捕获浮点约1e-15的常数段。此版本保留在`LC/`，不得用于正式增强。第一版CLI坐标也没有精确继承WTH，已在第二版从原文件头读取37.890/114.694/50，原脚本备份保留。

第二版`LC_v2/`按原XLS缺口逐项比较最终WTH与历史cleaned的舍入值，只有仍与旧均值相同或015已明确标记来源不确定的记录才修复。其余后续WTH补充值保留并逐项标记来源记录不完整，不能声称获得独立逐行实测凭证。原用户确认的历史下载补充背景继续适用。

第二版2000–2013修复SRAD18项、TMAX9项、TMIN2项；SRAD中原2005–2013窗口13项不变，扩展年份新增5项。辐射缺口用拟合期同月排除原XLS缺口的农业站值均值，温度缺口用邻站实测；423个邻站降水缺失日期回退同日原农业站WTH，不补为零。逐年月方差以1e-8近零阈值复核，无常数月。阈值是数值常数识别口径，不是DSSAT官方质量阈值。推算rsds使用次数0。

扩大的是LC生成器拟合窗口，PPO训练2005–2013、验证2014–2023保持不变；验证期未参与统计或供体。迁站和混合来源保留为方法限制，本次没有推断其影响为零。

## 尚未满足的门槛

季节smoke只证明CLI可被实际DSSAT WGEN使用、种子可复现、已观测季节天气物理检查通过。未验证全年降水/辐射分布、跨变量相关和尾部、多个训练年份的注入闭合、完整PPO天气调度或PPO smoke。

项目既有`results/yc_wgen_cli_pilot/004_01/full_year_path_audit/path_decision.json`记录无已验证自动全年WGEN入口；官方模型天气随作物日循环结束，现有运行不会自然导出完整全年。该记录作为项目内已有证据，本次未据旧记录声称当前系统安装情况已核实。没有安装或访问项目外WeatherMan。下一步需在项目内验证全年生成入口，或明确建立并验证以作物季为范围的增强合同；不能把未做的全年检查标为PASS。本阶段按prompt不启动PPO smoke或八seed正式队列。

## 文件

- Prompt：`prompt_02/054_sylc_external_weather_readiness_and_runtime_smoke.md`
- LC原始来源审计与候选：`results/sylc_external_weather_054/LC_v2/`
- SY/LC实际运行：`results/sylc_external_weather_054/{SY,LC}/runtime/`
- 运行gate：`results/sylc_external_weather_054/{SY,LC}/runtime_gate.json`
- 执行脚本：`scripts/audit_lc_long_window_054.py`、`scripts/run_sylc_wgen_runtime_smoke_054.py`
- 修正前脚本：`backups/sylc_054_scripts/`

本次FQ seed7训练/图表已完成，不重复运行。
