# YC DSSAT 坐标运行时源码追踪与修复门禁

## 结论

本轮确认 YC 坐标在 FileX 输入和 Gym-DSSAT/PDI 的真实 Jinja 渲染后均正确：`XCRD=116.57000`、`YCRD=36.83000`、`ELEV=22.0`。但此前三次真实 DSSAT 输出的 `DSSAT48.INP` 和 `DSSAT48.INH` 字段坐标仍为 `-999/-99/-99`，`Summary.OUT` 的坐标列为空，`WARNING.OUT` 同时报告坐标读取和 PDI transfer 错误。

运行时 Python 是 `gym-dssat-pdi==0.0.5`；DSSAT 模型版本是 `4.8.0.024`。安装包不含 FileX 坐标解析器或 INP/INH 坐标 renderer，真正消费 FileX 的是无调试符号的 `/opt/dssat_pdi/dscsm048`。因此本轮没有在 repo 中实施 native runtime 修复，也没有修改容器、可执行文件或已安装 package。行为根因分类为 **`PARSER_FIELD_NOT_READ`**；对编译版本内部具体语句的判断仍待匹配的 Fortran 源码验证。

`DssatPdi._make_fileX_template` 的实际 Jinja context 只包含 `wther/ferti/irrig/plant`。使用当前已安装方法对 YC 模板进行隔离渲染，生成 SHA256 `3587215024992d0c87ce725cd0752bcda51497d1f9cd84907a6493a5797576fb`，与先前历史对照的 `fileX.MZX` 完全相同；本次探针没有实例化环境或启动 DSSAT。坐标丢失边界因而位于 Python FileX 输出之后、native 输入解析/字段传递/输入文件生成路径之内。

## 运行时与源码证据

- 完整 distribution、路径和哈希：[`runtime_source_inventory.json`](../results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_inventory.json)。
- 最小只读快照与来源哈希：[`runtime_source_snapshot_manifest.json`](../results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot_manifest.json)。`runtime_source_snapshot/` 仅供本地检查，禁止 stage/commit。
- Python 到 `run_dssat` 的方法/变量链及上游源码定位：[`coordinate_runtime_code_trace.md`](../results/yc_wgen_cli_pilot/003_06_07_01/coordinate_runtime_code_trace.md)。
- 实际值、不可见的 parser 内存值、INP/INH 与 runtime 输出：[`coordinate_value_trace.json`](../results/yc_wgen_cli_pilot/003_06_07_01/coordinate_value_trace.json)。

DSSAT 官方当前公开 `InputModule/ipexp.for` 把对应坐标读取定位在 `IPFLD`：字段记录中的 `CXCRD/CYCRD/CELEV` 经格式 `80` 读出，再解析为 `XCRD/YCRD/ELEV` 并写入 PDI `FIELD` 数据项。它提供了精确的 upstream 文件/子程序/变量，但不是容器内 `4.8.0.024` 二进制对应源码，报告没有把这份当前源码误称为已安装版本源码。[官方 `IPFLD` 源码](https://github.com/DSSAT/dssat-csm-os/blob/develop/InputModule/ipexp.for#L1007-L1236)

## 上游补丁提案

建议在与 `/opt/dssat_pdi/dscsm048` 完全匹配的 DSSAT-PDI Fortran 源码中修改 `dssat-csm-os/InputModule/ipexp.for` 的 `IPFLD`，围绕 `HFNDCH/IFIND/LN/LNFLD/CHARTEST/CXCRD/CYCRD/CELEV/XCRD/YCRD/ELEV/ERRNUM/CKELEV`：

1. 读取前初始化坐标字符 buffer；显式验证坐标二级 header 和 `LN==LNFLD` 的记录存在，缺失则报告文件和行号并终止输入生成，不继续以未初始化值静默回退。
2. 按 `XCRD/YCRD/ELEV` header 与字段格式读取原值，逐项检查 I/O 状态、空值和数值范围；不得因解析错误继续把占位符写入 runtime。
3. 成功时保持 `CXCRD -> XCRD -> FIELD/CXCRD`、`CYCRD -> YCRD -> FIELD/CYCRD`、`CELEV -> ELEV -> FIELD/CELEV` 的传播，并确认 DSSAT48.INP 与 INH writer 使用这些 FIELD 值。
4. 加 YC 与非 YC 的 parser/transfer/INP regression fixtures，再在受控、匹配版本的构建中核对 INP/INH、Summary.OUT 和 warning。禁止把 YC 数字写进共享 Fortran parser。

这是基于已定位上游函数与运行时失败边界的可审查提案，不是声称已证明 4.8.0.024 的具体源代码缺陷。仅更换 repo-side FileX 模板或事后改写已生成的 INP 无法可靠修复在 native 模拟启动前发生的 parser/renderer 路径；本轮不实施这种绕行。

## 门禁与实验结果

[`dssat48_coordinate_gate.json`](../results/yc_wgen_cli_pilot/003_06_07_01/dssat48_coordinate_gate.json) 为 **FAIL**。FileX 输入门禁 PASS，但 parser output 未观察到，历史实际 INP/INH 门禁 FAIL。根据任务要求，在这个门禁失败时停止，未重跑历史 control、seed 101 或 seed 104，也未启动 PPO。

[`coordinate_fix_crop_output_comparison.csv`](../results/yc_wgen_cli_pilot/003_06_07_01/coordinate_fix_crop_output_comparison.csv) 保留修复前作物指标；修复后列和绝对/相对差异为空，因为没有修复后运行，不能推断修复对产量或水氮指标的影响。坐标 warning 仍未解决：此前三组运行共 36 个 class-C 坐标事件（读取+PDI transfer），另有 6 个 class-B 非坐标事件；修复后未复测，见 [`warning_comparison.csv`](../results/yc_wgen_cli_pilot/003_06_07_01/warning_comparison.csv)。

本轮唯一 repo 代码调整是将旧诊断辅助函数的 round-trip 断言改成按传入坐标校验，并补了非 YC FileX/INP fixture，避免通用诊断 renderer 隐含 YC 硬编码。`python -m pytest -q tests/test_yc_coordinate_propagation_probe.py`：**10 passed**。这些是离线结构/fixture 测试，不等价于 native runtime 集成门禁。

## PPO 准入状态

**`BLOCKED_BEFORE_PPO`，`YC_RANDOM_WEATHER_PPO_READY=NO`。** 下一步应取得 DSSAT-PDI 4.8.0.024 精确匹配的 Fortran 源码，在隔离环境审查并实现上述 `IPFLD` 修复，先对 parser 输出和 INP/INH 做不运行作物的门禁；门禁通过后再按历史 control、seed 101、seed 104 顺序串行重跑。WGEN、CLI、天气拟合、湿日定义、管理和 PPO 均未修改。
