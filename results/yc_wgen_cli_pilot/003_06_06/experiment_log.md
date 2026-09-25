# YC WGEN 天气 QC 与 DSSAT Crop Smoke 实验记录

日期：2026-09-25（北京时间）

## 目标与冻结边界

完成 YC WGEN seed 101–105 的固定日历窗口完整天气 QC，再以 2008 YC treatment 1 执行历史测量天气 control 和 WGEN seed 101、104。保持正式 `CNYC.CLI`、训练期天气、`RAIN > 0.0 mm`、土壤、cultivar、管理和 DSSAT `4.8.0.024` 不变。validation weather、WGEN refit、PPO 和其他站点均不涉及。

## Gate A 运行

在项目 Docker `nifty_taussig` 中运行：

```text
docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python results/yc_wgen_cli_pilot/003_06_06/weather_qc/analyze_weather_qc.py
```

脚本复核 frozen CLI 和 train weather SHA256，按五个 weather period 交集计算 2008-06-01 至 2008-09-24 共 116 天，并为 2004–2013 每年取完全相同的月日。所有 5 个 seed 序列唯一；物理异常 0 项。QC 为 `PASS_WITH_NOTES`。备注集中在 seed 104 湿日数比训练期历史最小值少 1 天，以及 seed 104 热日尾部高于历史年度极值；没有一组 seed 共同偏向同一侧。累计降雨和月尺度偏离均按 descriptive reference 报告，没有调 CLI、参数或阈值。

## Gate B 失败尝试与定位

1. `historical_control` 首次尝试在模型启动前失败：`ModuleNotFoundError: No module named 'scripts'`，0 daily steps。修复运行脚本导入路径后改用独立 retry 目录。
2. `historical_control_retry1` 又在模型启动前失败：`NameError: _runtime_class is not defined`，0 daily steps。补齐现有 pilot helper imports 后重试。
3. `historical_control_retry1` 后续 DSSAT 运行本身正常结束，但首版 Summary.OUT 提取器没有识别 `TRNO`，把产量列整体错位。保留其输出，不使用这次错误汇总；改为以 `SOIL_ID` 锚定 Summary.OUT 列，并用 `HWAM/CWAM/PDAT/ADAT/MDAT/IRCM/NICM/NUCM` 重新解析。对 retry1 原始 Summary.OUT 的断言通过：yield 7578、biomass 17752、irrigation 120、fertilizer 303、N uptake 219。
4. `historical_control_retry2` 使用新的解析器通过，但坐标列填充没有被 DSSAT 读取。保留完整输出和 warning。
5. `historical_control_retry3` 根据 FileX 表头列位调整隔离模板，warning 仍存在。轻量列位断言纠正了 `@L` 表头与数据行的 1 列偏移，未改原始 MZX。
6. `historical_control_retry4` 按 DSSAT FileX XCRD/YCRD/ELEV 定宽字段格式复核后通过 crop smoke，但实际 runtime 仍保留 `DSSAT48.INP` 坐标占位符，Summary.OUT 坐标为空，WARNING.OUT 中 `CYCRDin/CXCRDin/CELEVin` 最终设为零。由此把问题归类为 C 类 PPO blocker，不再用未验证输入继续尝试“修复”。

前几次坐标列单元测试也发现测试断言本身误用了原始模板行索引；按处理后的 template 对照标准列宽重跑后，固定宽度断言通过。即使模板层检查通过，DSSAT 运行时仍未接受这些值，因此不把模板内容正确等同于模型输入正确。

## Gate B 最终 Smoke

```text
docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python results/yc_wgen_cli_pilot/003_06_06/crop_smoke/run_dssat_crop_smoke.py --runs historical_control_retry4
docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python results/yc_wgen_cli_pilot/003_06_06/crop_smoke/run_dssat_crop_smoke.py --runs seed_101 seed_104
```

三组均在 123、120、116 个日 step 后正常结束。历史控制 WTHER=M 且历史 WTH hash 匹配；随机组 WTHER=W 并确认 weather seed 101/104。三组 WTHER 归一化后 FileX SHA256 相同。管理事件均为灌水 120 mm、施氮 303 kg N/ha。

| run | anthesis | maturity/harvest | yield kg/ha | biomass kg/ha | irrigation mm | fertilizer kg N/ha | N uptake kg/ha |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| historical control | 2008-08-14 | 2008-10-01 | 7578 | 17752 | 120 | 303 | 219 |
| seed 101 | 2008-08-14 | 2008-09-28 | 7326 | 17387 | 120 | 303 | 213 |
| seed 104 | 2008-08-11 | 2008-09-24 | 7948 | 16402 | 120 | 303 | 205 |

`PlantGro.OUT` 的 WSPD、WSGD、NSTD 与 `SoilWat.OUT` 的 SWTD、SWXD 均有限；stress factor 处于 0–1 范围，输出范围另见 `crop_smoke/dssat_water_n_stress_summary.csv`。Yield、biomass、物候次序与管理事件检查通过。它们只证明 crop-output smoke 可完成，不构成 WGEN 性能或 PPO 效果结论。

## Warning 审计决策

在历史 control、seed 101、seed 104 三个最终运行中共计 42 个离散事件，按不重复/INFO 口径逐项分为 A=0、B=6、C=36。PHOTO L→C 与缺失 STONES/ADCOEF 默认在历史 control 中同样存在，归 B 并保持输入不变。soil 默认实况从 INFO.OUT 确认 STONES/SLCF=0.0%、ADCOEF=0.0；由于没有可靠 YC 原始值，不补造。

FileX 隔离副本包含站点坐标，但 `DSSAT48.INP` 保留占位值、Summary.OUT 坐标列为空，warning 的变量转移最终置零，故纬度/经度/海拔事件归 C。三类变量在历史与两个随机运行中重复出现，共 36 个 C 类事件。历史 WTH 的站点头有坐标，但目前不能证实模型的 FileX 字段成功接收，因此在修复后需重跑三组。当前决定：`crop_smoke_status=PASS`，`DSSAT_CROP_SMOKE_READY=NO`，PPO 仍阻断。

上一轮 checklist 所列 cultivar warning 在当前三组及复制的 seed 101 历史 `WARNING.OUT` 中均未观察到。品种 `MZ/ZD0985` 在 CUL 与 DSSAT 输入中解析存在。

## PPT 生成与版面校验

使用提供的 `@oai/artifact-tool` 创建 10 页可编辑 PPT，天气和作物图表保留为 5 个原生柱图、warning/crop/water-N/readiness 表保留为 4 个原生表格。首次构建因缺少当前进程 `RUNTIME_NODE_MODULES` 环境变量在字体工具初始化阶段终止，尚未导出；补齐进程级变量后继续。finalizer 随后提示字体策略不接受普通字体放入 `scriptFonts`，改为把 Arial 与 Microsoft YaHei 都列作设计字体；并按要求将 6、7、9、10 页登记为原生表格校验页。初次布局扫描发现物候页相邻日期标签重叠，调整标签垂直位置后扫描 warning 为 0。

最终文件 `docs/yc_wgen_weather_qc_and_dssat_smoke.pptx` SHA256 为 `4624B85143CA07575F9DDD7D3EF4163ED4A39B19ED3847DD3E0C0E41005D638E`。包完整性通过（10 页、5 图表）；原生图表工作簿快照、4 个可编辑表格与 Artifact Tool 导入验证通过；版面扫描 0 finding / 0 warning。最终 PPTX 重新逐页渲染并检查 10 张图，未见裁切、遮挡或剩余标签重叠。构建预览、回执及前一候选版本留在未暂存的 `.codex-ppt-build/`，不作为正式交付。

## 交付与 Git 决策

天气 CSV、七幅图、warning 证据、DSSAT OUT 快照、物候/产量/水氮汇总及中文报告/PPT 均留在本任务目录和 docs。完整 hash 及事件频次见 `runtime_warning_audit/warning_summary.json`。

工作区中存在本任务开始前的其他脏文件和未跟踪产物。本轮不清理或暂存这些内容，只提交本任务文件。要求的本地 commit message 为 `test: validate YC WGEN weather and DSSAT crop smoke`；不执行 `git push`，GitHub backup 等用户明确批准。
