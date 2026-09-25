# YC DSSAT 坐标传递排查记录

## 1. 任务范围

仅调查 YC 的 `LAT=36.830`、`LONG=116.570`、`ELEV=22 m` 从 FileX 到 DSSAT runtime 的传递。本轮未改 WGEN、CLI、天气拟合、PPO、reward、action/observation、其他站点或 DSSAT 数值模型；未制作 PPT。

## 2. 上一轮 blocker

003_06_06 已完成历史天气 control、WGEN seed 101 和 seed 104 crop smoke，但三组都有坐标读取/传递 warning，`Summary.OUT` 坐标列为空。因此上一轮的 crop smoke 通过不等于坐标门禁通过，PPO 仍不允许启动。

## 3. 坐标数据流追踪

追踪表见 [`coordinate_trace.csv`](../results/yc_wgen_cli_pilot/003_06_07/coordinate_trace.csv)，完整的仓库代码入口与边界见 [`coordinate_code_path.md`](../results/yc_wgen_cli_pilot/003_06_07/coordinate_code_path.md)。原始 YC FileX 使用 `-99` 占位；隔离后的三组 runtime `fileX.MZX` 均为 `XCRD=116.570`、`YCRD=36.830`、`ELEV=22.0`。`XCRD` 映射 longitude，`YCRD` 映射 latitude。

## 4. 代码级根因与假设判断

已证实的丢失区间为正确的 runtime FileX 与 `DSSAT48.INP *FIELDS` 坐标记录之间：INP field row 三组均为 `XCRD=-999`、`YCRD=-99`、`ELEV=-99`。文件中 `Yucheng` 土壤/站点描述行另有 `36.830 116.570`，但它不等于 `*FIELDS` 的坐标记录，不能当作替代验证。

| 假设 | 判断 | 依据 |
| --- | --- | --- |
| LAT/LONG/ELEV 未被 parser 读取 | SUPPORTED（运行现象）；代码原因未证实 | warning 报坐标读取失败，parser 对象不可见 |
| X/Y 方向反了 | REJECTED（当前证据） | FileX 数值与 DSSAT 标准字段方向相符；报错对应三类值丢失而非互换 |
| field name 与 template name 不一致 | NOT_APPLICABLE | 项目目录内没有安装包 parser 源码，不能对仓库外实现作映射判断 |
| parser 值未传入 DSSAT48.INP renderer | SUPPORTED（行为层） | FileX 正确，INP field row 仍为占位值 |
| renderer key mismatch | NOT_APPLICABLE | 缺少安装包源码及内部对象 trace，无法检查其 renderer key |
| 默认零/占位值覆盖 | SUPPORTED（行为层） | INP 占位及 runtime 置零可见；具体覆盖代码位置不可见 |
| placeholder substitution 未触发 | SUPPORTED（行为层） | 三组真实 DSSAT48.INP field row 都未替换 |
| FileX section/line parsing 范围错误 | SUPPORTED（候选行为） | 文件行在 `*FIELDS` 的 `@L XCRD/YCRD/ELEV` 后，DSSAT 仍报读错；具体行解析代码未知 |
| fixed-width parser index off-by-one | REJECTED（仓库 runner 写入侧）；NOT_APPLICABLE（安装包侧） | 当前 runner 固定宽度往返解析为正确坐标；安装包 parser 不在仓库内 |
| runtime wrapper 忽略坐标 | NOT_APPLICABLE | wrapper 实现未在项目内，不能判定是否有意忽略 |

## 5. 最小修复

本轮没有提交运行时修复。仓库内只能控制模板/runner；真正生成 DSSAT 输入的 Gym-DSSAT/PDI 实现在项目外安装包目录，且项目目录规则禁止本轮读取或改写它。手工改动保留快照或运行结束后的 INP 不能证明 DSSAT 启动前已收到坐标，故不作为修复。

新增 [`debug_yc_coordinate_propagation.py`](../scripts/debug_yc_coordinate_propagation.py) 作为只读、无 DSSAT 调用的证据预检；它检测 INP field row 的真实坐标并生成门禁结果。回归测试验证 FileX parse/map/render 与 INP 占位阻断，不声称覆盖不可见的 PDI renderer。

## 6. DSSAT48.INP 验证

见 [`dssat48_coordinate_check.json`](../results/yc_wgen_cli_pilot/003_06_07/dssat48_coordinate_check.json)：seed 101 的 `DSSAT48.INP` 真实 field record 为 `LAT=-99`、`LONG=-999`、`ELEV=-99`；companion `DSSAT48.INH` 同样保留占位值，`pass=false`。历史 control、seed 101 和 seed 104 的 field rows 一致。此值不通过，因此按任务门禁停止，不进入新的 crop simulation。

## 7. Runtime 验证

仅复核上一轮已经留存的运行证据，没有在本轮启动 DSSAT。三组 `WARNING.OUT` 均记录 latitude、longitude、elevation read error，并分别记录 `CYCRDin`、`CXCRDin`、`CELEVin` transfer error；后续值被置零。三组 `Summary.OUT` 的 `XLAT/LONG/ELEV` 坐标区为空。PDI parsed internal coordinates 没有仓库内可观测 trace，记录为 `NOT_OBSERVED`，不能当成已验证值。

## 8. Warning 比较

[`warning_comparison.csv`](../results/yc_wgen_cli_pilot/003_06_07/warning_comparison.csv) 记录修复前三组各 12 个坐标 read/transfer 事件和各 2 个非坐标基线事件（PHOTO L→C、STONES/ADCOEF defaults），共 42 个离散事件。修复后列为空并标注未运行，不能宣称任何 warning 已消失。

## 9. 作物输出修复前后比较

[`coordinate_fix_crop_output_comparison.csv`](../results/yc_wgen_cli_pilot/003_06_07/coordinate_fix_crop_output_comparison.csv) 保留历史、101、104 修复前的物候、产量、生物量、灌溉、施氮、N uptake 及水氮 stress；修复后列为空，`crop_output_changed_after_fix=NOT_ASSESSED`。由于没有已验证 fix，也没有坐标通过的 DSSAT48.INP，本轮未重跑三组，不能比较修复后产量差值。

## 10. 回归测试

新增 `tests/test_yc_coordinate_propagation_probe.py`，覆盖 FileX 坐标读取与 X/Y 映射、固定宽度渲染、LAT/LONG/ELEV 保持、DSSAT48.INP/INH field 坐标读取、占位值门禁拒绝，以及合成 INP field row 的坐标替换后读取检查。该合成测试只验证格式/门禁，不代表 PDI renderer 已修复或集成。完整测试命令与通过数记录于 `experiment_log.md`；测试不触发 DSSAT。

## 11. PPO 准入决定

`ppo_coordinate_gate=BLOCKED_BEFORE_PPO`，`yc_random_weather_ppo_ready=NO`。关键的 INP/runtime/Summary 条件未满足；本轮没有 PPO training。

## 12. 未解决事项

需在项目内提供/复制 Gym-DSSAT/PDI 对应版本的 `dssat_pdi.py` 与 INP renderer 源码，或由项目所有者明确允许在 `/opt/gym_dssat_pdi` 中定位代码后，才能确定安装包级 parser/mapping/substitution 的具体失效函数并修复。此后需先运行小型 non-crop/preflight 确认 INP，再依序重跑 control、101、104。

## 13. 本轮文件

本轮新增坐标诊断脚本、针对性 pytest、代码路径说明、CSV/JSON 证据与本报告。未修改原始 FileX、CLI、WTH、WGEN 主 runner、PPO 文件或其他站点文件。

## 14. Git 状态

工作区预先存在大量与本任务无关的修改/未跟踪文件，本轮不清理、不暂存。仅将本任务新建文件加入本地提交；提交后其他既有 dirty paths 仍保留。不推送。GitHub backup pending explicit user approval。
