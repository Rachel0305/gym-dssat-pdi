# YC 随机天气 QC 与 DSSAT 单季作物 Smoke

**结论：天气 QC 为 `PASS_WITH_NOTES`，三组 DSSAT 作物模拟均正常结束；但 FileX 经纬度和海拔没有进入 DSSAT runtime 字段，坐标最终被设为 0。因此 `DSSAT_CROP_SMOKE_READY=NO`，`YC_RANDOM_WEATHER_DSSAT_SMOKE_STATUS=BLOCKED_BEFORE_PPO`。本轮没有运行 PPO。**

## 1. 任务范围

只检查 YC 单站、2008 年 treatment 1。Gate A 对比 WGEN seed 101–105 与冻结 2004–2013 训练期天气；Gate B 串行运行历史 2008 天气控制及随机天气 seed 101、104。没有重新拟合 WGEN，没有改 wet-day 定义、PPO、管理、其他站点或原始输入文件。

运行环境为项目 Docker 容器 `nifty_taussig`，DSSAT `4.8.0.024`。所有 crop smoke 都是单进程串行运行，以控制内存和计算负荷。

## 2. 输入与来源

| 输入 | SHA256 | 核验 |
| --- | --- | --- |
| `results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI` | `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929` | 通过 |
| `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv` | `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34` | 通过 |
| 历史天气 `my_data/CNYC0801.WTH` | `39B814537927E012B7EB04A238F88DE9DF8C92C011FDB9854F0B7A121E3B7003` | 通过，历史控制运行时逐字节一致 |

YC 站点坐标以历史 WTH 站点头和任务提供的站点元数据分别核对：纬度 `36.830`、经度 `116.570`、海拔 `22 m`。没有把 CLI 里的站点信息当作 FileX 坐标。

## 3. 固定比较窗口

按 101–105 五组天气的实际日期交集自动确定共同窗口：`2008-06-01` 至 `2008-09-24`，共 `116` 天。历史训练期逐年取相同月日；窗口不跨闰日。共同窗口承担累计量及分布对比。

各 seed 的完整模拟区间分别为 116–120 天。完整期间的降雨累计量保存在 `weather_qc/full_simulation_period_weather.csv`，标记为 `NOT DIRECTLY COMPARABLE FOR CUMULATIVE TOTALS`，没有把不等长累计量当作公平 seed 对比。

## 4. 训练期历史参考

历史参考仅使用冻结的 2004–2013 训练天气，不含 2014–2023 validation 天气。共同窗口历史降雨总量为 `289.7–682.6 mm`，湿日为 `23–45 天`，最长连续干日为 `7–42 天`。十个训练年各自的逐日天气统计及 mean、SD、median、min、max、P05、P25、P75、P95 位于 `weather_qc/historical_common_window_reference.csv`。

这些是训练气候的描述性范围，不是用来调参或挑选 seed 的接收阈值。

## 5. 降雨结构

湿日定义保持 `RAIN > 0.0 mm`。共同窗口 seed 101–105 的降雨总量依次为 `507.7、509.5、466.4、290.1、483.1 mm`。五组均落在历史年度 min–max 内。湿日数依次为 `36、35、37、22、30`，历史为 23–45 天，4/5 seed 在历史范围内；seed 104 比历史最低值少 1 天。最长连续干日依次为 `14、21、12、19、12 天`，均落在历史范围内。

降雨日分布并未退化，历史期与生成序列均有干湿连段变化；最大日雨量没有超过训练期共同窗口的历史极值范围。月尺度有 seed 局部偏干：seed 104 的 7、8 月降雨低于相应历史 P05，seed 102 的 6 月也低于 P05；同一 seed 的其他月份不同，整体季节累计仍处于历史年度范围。此处作为描述性备注，不据单月偏离判错或改 CLI。

## 6. 温度 QC

所有生成日均满足 `TMAX >= TMIN`。训练期全体日值 TMAX 的经验 P90 为 `35.4°C`，TMIN 的 P05 为 `13.119°C`。生成 seed 的平均 TMAX 为 `30.65–32.11°C`，4/5 落在历史年度平均值 min–max (`28.97–31.65°C`)；seed 104 高出历史年度平均值上界约 `0.45°C`。平均 TMIN 为 `20.02–21.08°C`，全部落在历史年度范围 (`19.34–21.20°C`)。

各 seed 高于训练期 pooled TMAX P90 的日数为 `15、11、17、29、17`；seed 104 的 29 天超过历史年度最高 24 天。该尾部偏高只出现在一个 seed，记为暖端备注，不用它重设 WGEN 参数。生成 TMIN 最低值约 `6.10°C`，略低于训练期观测最低值约 `6.7°C`，但没有触发物理范围检查。

## 7. SRAD QC

SRAD 均非负。seed 平均值为 `16.04–17.05 MJ m⁻² d⁻¹`，历史年度均值范围为 `15.87–17.40 MJ m⁻² d⁻¹`，五组均重叠。生成值具备逐日和 seed 间变化，没有发现持续偏低、偏高或无变异迹象。

## 8. 天气 QC 判定

`weather_qc_status=PASS_WITH_NOTES`。物理异常数为 0，五个天气序列互不相同；累计雨量、最长干段和平均 SRAD 与训练期年度范围有重叠。需要保留两项描述性注意：seed 104 湿日比历史最小值少 1 天、且出现偏高的暖端日数；seed 104 的季节雨量仍位于历史年度范围。没有识别到所有 seed 同方向的明显中心偏移。

七张单图保存在 `results/yc_wgen_cli_pilot/003_06_06/weather_qc/figures/`，涵盖共同窗口降雨、湿日数、最长干段、TMAX、TMIN、SRAD 和月降雨。阈值没有用于再拟合。

## 9. Runtime warning 审计

警告总数按离散事件计数，不按日志非空行计数。三组正式运行共 `42` 个事件：A 类 `0`，B 类 `6`，C 类 `36`。逐项依据见 `results/yc_wgen_cli_pilot/003_06_06/runtime_warning_audit/runtime_warning_audit.csv`，原始快照及 SHA256 清单见 `warning_evidence/` 和 `warning_summary.json`。

| 事项 | 观察 | 分类与判断 |
| --- | --- | --- |
| PHOTO `L -> C` | 三组运行都报告 DSSAT 为 MZCER 兼容性将 PHOTO 从 L 改为 C，历史控制也有 | B：已有兼容回退，统一冻结，不随意改模型设置；共 3 次 |
| 纬度、经度、海拔读取 | 隔离运行 FileX 保留 `36.83000 / 116.57000 / 22.0`，但 `WARNING.OUT` 报读取失败；`DSSAT48.INP` 仍是占位值，`Summary.OUT` 坐标栏为空 | C：DSSAT 字段传递日志最终报纬度、经度和海拔设为零；不能确认模型实际坐标，须先修复并重跑，不能据 crop 输出放行 PPO |
| STONES / ADCOEF | YC 土壤剖面对应 `SLCF`、`SADC` 值为缺失；`INFO.OUT` 报使用默认值。运行细节显示 STONES/SLCF=`0.0%`、ADCOEF=`0.0` | B：历史和随机天气运行使用相同土壤文件与默认值，影响土壤/根系及硝态氮吸附传输的不确定性；不凭空补值，作为既有基线局限披露 |
| Cultivar | 当前及复制的上一轮 `WARNING.OUT` 均未发现 cultivar-specific warning；`MZ/ZD0985` 存在于 `MZCER048.CUL` 并进入 `DSSAT48.INP` | 没有观察到 cultivar warning；不推断品种参数校准已经充分 |

坐标写入发生在隔离 FileX 中，Gym 运行快照也保留了文字值；丢失发生在 DSSAT 解析/传递阶段。历史 WTH 头也记录站点坐标，但这不能证明 DSSAT 的 FileX 字段传递已成功。对照 DSSAT 的 FileX 格式复核后，继续使用独立模板和标准列宽复跑仍未改变 runtime 结果，因此根因尚未确定。原始 MZX、WTH、SOL、CUL 和 CLI 均未改写。

## 10. 历史天气控制

采用 `random_weather=False`、`WTHER=M`，历史 `CNYC0801.WTH` 与指定 SHA256 一致。YC 2008 treatment 1、品种 `ZD0985`、土壤 `YC99001200`、灌溉和施肥计划及 DSSAT 版本与随机天气组相同。最终对照目录为 `crop_smoke/historical_control_retry4/`，运行 123 个每日 step 后正常结束。

## 11. 随机天气作物 Smoke

seed 101 与 104 均以 `random_weather=True`、`WTHER=W` 运行；正确 CLI SHA256 在运行期复核。随机天气运行各 120 和 116 个每日 step 后正常结束。三组在 WTHER 归一化后的 FileX SHA256 相同，说明坐标、treatment、cultivar、soil 和其他 FileX 内容相同。

## 12. 物候与产量

| 情景 | 播种 | 出苗 | 抽雄/开花 | 成熟/收获 | 成熟 DAP | 籽粒产量 kg/ha | 生物量 kg/ha | 最大 LAI |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 历史天气控制 | 06-18 | 06-23 | 08-14 | 10-01 | 105 | 7,578 | 17,752 | 4.80 |
| WGEN 101 | 06-18 | 06-23 | 08-14 | 09-28 | 102 | 7,326 | 17,387 | 4.61 |
| WGEN 104 | 06-18 | 06-26 | 08-11 | 09-24 | 98 | 7,948 | 16,402 | 3.97 |

三个结果均有合理的出苗—开花—成熟—收获顺序，产量和生物量有限且满足 `biomass >= yield >= 0`。这只是 crop-output smoke，不是产量性能结论，也不能用两个 seed 推断 WGEN 的一般产量效应。

## 13. 水分与氮状态

所有运行保留了同一管理事件：灌溉 `120 mm`，施氮 `303 kg N/ha`（96+207 kg/ha）。Summary.OUT 的累计 ET 为历史控制 `405 mm`、seed 101 `400 mm`、seed 104 `372 mm`；作物 N uptake 为 `219、213、205 kg N/ha`。对应 crop 期间降雨累计为 `442、508、290 mm`，因天气终止日不同，不作为等长累计比较。

直接从 `PlantGro.OUT` 读取 WSPD、WSGD、NSTD，从 `SoilWat.OUT` 读取 SWTD、SWXD。所有值有限，因子落在 0–1 区间。WSPD 最大值三组均为 0；WSGD 最大值分别为 `0、0.226、0`；NSTD 最大值为 `0.013、0.013、0.012`。SoilWat 中 profile water SWTD 范围为历史 `262–411 mm`、seed 101 `256–439 mm`、seed 104 `183–448 mm`。这些是模型输出范围检查，不表示 PPO 管理效果。

## 14. 最终分类与 PPO 准入

- `weather_qc_status`: `PASS_WITH_NOTES`
- `historical_control_status`: `PASS`
- `random_weather_crop_seeds_run`: `101, 104`
- `phenology_status`: `PASS`
- `yield_biomass_status`: `PASS`
- `water_status`: `PASS_WITH_NOTES`
- `nitrogen_status`: `PASS_WITH_NOTES`
- `crop_smoke_status`: `PASS`
- `warnings_blocking_ppo`: `YES`，FileX 坐标虽在隔离文件中存在，DSSAT 实际输入仍保留占位符，transfer warning 报坐标变量设为零
- `DSSAT_CROP_SMOKE_READY`: `NO`
- `YC_RANDOM_WEATHER_DSSAT_SMOKE_STATUS`: `BLOCKED_BEFORE_PPO`

## 15. 剩余问题与下一步

先定位 DSSAT 4.8.0.024 对 YC FileX `XCRD/YCRD/ELEV` 的读取/传递路径，确认运行时 `DSSAT48.INP`、Summary.OUT 与 warning 均反映正确坐标。再用当前冻结 CLI、训练天气、同一处理和相同管理串行重跑历史控制及 seed 101、104。土壤缺失的 STONES/ADCOEF 不得猜填；若后续取得权威 YC 实测值，再按独立的数据校核流程决定是否补充。上述门槛通过前，不开始 random-weather PPO pilot。

## 16. 边界核验

`validation_weather_used_for_fitting=NO`；`wet_day_definition_changed=NO`；`wgen_refit=NO`；`ppo_training_run=NO`；`other_sites_modified=NO`。没有将本轮生成结果用于调整正式 CLI 或接受阈值。

## 17. 实验记录与产物

完整命令、失败尝试、修复过程和决策见 `results/yc_wgen_cli_pilot/003_06_06/experiment_log.md`。天气 QC 文件在 `results/yc_wgen_cli_pilot/003_06_06/weather_qc/`；warning 分类与证据在 `results/yc_wgen_cli_pilot/003_06_06/runtime_warning_audit/`；crop 汇总、日状态、DSSAT 输出、provenance 和 smoke 汇总在 `results/yc_wgen_cli_pilot/003_06_06/crop_smoke/`。

本任务 prompt 原文副本为 `prompts/003_06_06_yc_wgen_weather_qc_and_dssat_smoke.md`。

## 18. 参考资料

- DSSAT User’s Guide Vol. 4：FileX `XCRD`、`YCRD`、`ELEV` 定义与定宽格式。[官方手册 PDF](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol4.pdf)
- DSSAT User’s Guide Vol. 2：土壤 coarse fraction `STONES/SLCF` 与 DSSAT 输出变量单位。[官方手册 PDF](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol2.pdf)
- DSSAT `DATA.CDE`：WSPD、WSGD、NSTD、SWTD、SWXD 输出定义。[DSSAT 官方源代码数据字典](https://github.com/DSSAT/dssat-csm-os/blob/develop/Data/DATA.CDE)
- 土壤库参数复核：`SADC` 土壤吸附系数的定义说明。[DSSAT 官方网站论文 PDF](https://dssat.net/wp-content/uploads/2012/05/Romero-2012-Reanalysis-of-a-global-soil-database-for-crop-and-environmental-modeling.pdf)

## 19. 文件与 Git 状态

本任务新增中文报告、PPT、实验记录、prompt 副本、QC/汇总脚本、CSV/JSON、图表与 warning/runtime 证据。未覆盖受保护原始 `.WTH`、`.SOL`、`.CUL`、`.MZX`、冻结 CLI、训练天气、既有训练脚本或模型结果。

提交前仓库已有大量无关工作区变更。本任务只暂存 003_06_06 结果及本任务的报告和 prompt 文件，不清理、不还原、不纳入其他变更。GitHub push 为 `NO`；GitHub backup 等待用户明确批准。

## 20. 验证状态

天气 QC 脚本完整运行，hash、共同窗口、物理约束与 seed 唯一性检查通过；固定宽度坐标及 Summary.OUT 锚点解析做了单元断言；三组 DSSAT crop smoke 均正常结束；事件、水氮和物候汇总已从原始 OUT 快照二次提取。未运行 formal PPO 或 pytest 全套回归。
