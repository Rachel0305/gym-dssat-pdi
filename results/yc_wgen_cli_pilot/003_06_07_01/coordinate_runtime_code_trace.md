# YC 坐标运行时源码追踪

## 环境身份

- Python distribution：`gym-dssat-pdi==0.0.5`，实际 import 模块 `gym_dssat_pdi.envs.dssat_pdi`，路径见 `runtime_source_inventory.json`。
- DSSAT 模型运行时：`4.8.0.024`，来自此前真实运行生成的 `Summary.OUT`；这不是 Python distribution 版本。
- `/opt/dssat_pdi/run_dssat` 是 29 字节启动脚本，最终执行 `/opt/dssat_pdi/dscsm048 "$@"`。原生文件无可用符号，授权路径中没有与该二进制对应的 Fortran 源码。

## 当前安装包中的实际调用链

1. `gym_dssat_pdi/envs/dssat_pdi.py:DssatPdi.__init__`（快照第 34 行起）读取传入的 FileX 模板，加载配置，然后依次调用 `_make_tmp_folder()`、`_make_fileX_template()`、`_write_fileX_template()` 和 `_get_sockets_()`。构造函数最后才等待 native runtime 状态。
2. `DssatPdi._make_fileX_template`（第 229-232 行）传给 Jinja 的字典只有 `wther/ferti/irrig/plant` 四个管理选项；没有 `XCRD/YCRD/ELEV` 或 `LAT/LONG` 坐标映射。YC 坐标已是 YC 专用 FileX 模板中的固定输入文本，不是该 Python context 注入的值。
3. `gym_dssat_pdi/envs/utils/utils.py:_fill_template_from_string`（第 46 行起）只是 Jinja 渲染；`_load_fileX_template`（第 154 行起）只读入模板文本。实际安装的 `env_config.yml`、`dssat_pdi.jinja2` 没有坐标映射键。
4. `_write_fileX_template`（第 234-235 行）把渲染文本写成临时 `fileX.MZX`。`_get_sockets_`（第 327-330 行）启动 socket、写 PDI YAML，再调用 `_launch_client`。
5. `_launch_client`（第 258-279 行）以子进程方式执行 `/usr/bin/env run_dssat C fileX.MZX <experiment_number>`。Python 没有在启动前把 FileX 解析成 DSSAT48.INP 的步骤；后续 FileX 解析和 INP/INH 生成属于 native DSSAT/PDI 路径。

因此，在不启动 native DSSAT 的限制下，Python 侧可以验证 FileX 渲染，但不能观测 native parser 的内存字段或其生成 INP 的 renderer 输入。直接构造完整 `DssatPdi` 对象会进入 `_get_sockets_` 并启动 native 进程，不适合作为“不跑 DSSAT”的门禁探针。

## Native 字段路径与证据边界

已安装 Python 包不含 `IPFLD` 或 DSSAT48.INP 坐标 renderer 源码。为精确定位上游对应符号，核对 DSSAT 官方公开源码 `InputModule/ipexp.for` 中的 `IPEXP`/`IPFLD`：

- `IPEXP` 调用 `IPFLD` 并传入 `XCRD/YCRD/ELEV`；`IPFLD` 在 `*FIELD` 下读取第一层记录，再通过 `HFNDCH='SLAS'` 和 `HFIND` 定位第二层字段头。
- `IPFLD` 以格式 `80` 读取 `CXCRD/CYCRD/CELEV`（格式定义 `I3,2(A15,1X),A9,...`），再分别解析为数值 `XCRD/YCRD/ELEV`。
- 有效坐标分支向 PDI `FIELD` 数据区写入 `CYCRD`、`CXCRD`、`CELEV`；无效/读取失败分支写入默认值。按 DSSAT 字段定义，`XCRD` 是 X/经度方向，`YCRD` 是 Y/纬度方向。
- 该公开 `develop` 源码用于标明上游应检查的精确文件、子程序与变量；它不是容器中 4.8.0.024 binary 的可验证源码，不能据此声称已经定位到该编译产物的具体错误语句。

来源： [DSSAT 官方 `InputModule/ipexp.for`（IPEXP/IPFLD）](https://github.com/DSSAT/dssat-csm-os/blob/develop/InputModule/ipexp.for#L1007-L1236)，[DSSAT FileX *FIELDS 坐标格式说明](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol4.pdf)。

## 根因判断

证据支持的失效边界是：YC rendered FileX 坐标正确，native 运行却报告 latitude/longitude/elevation read errors 和 `CYCRDin/CXCRDin/CELEVin` transfer errors；真实 DSSAT48.INP/INH 的字段坐标仍为 `-999/-99/-99`，Summary.OUT 坐标列为空。因此按任务分类记为 **`PARSER_FIELD_NOT_READ`（运行时行为分类）**。Python renderer、气象 CLI/WGEN 和天气拟合没有证据显示为坐标丢失点。

`IPFLD` 内部逐个 parser buffer 的数值本轮没有被观测；对 4.8.0.024 的具体子原因仍未证实。此限制不是“没有追源码”：已定位 Python 到 native 的真实调用边界，并给出官方上游对应文件/子程序/变量及下述可审查补丁方案。

## 最小上游补丁方案（未应用）

目标文件：`dssat-csm-os/InputModule/ipexp.for`；子程序：`IPFLD`；关键变量：`HFNDCH/IFIND/LN/LNFLD/CHARTEST/CXCRD/CYCRD/CELEV/XCRD/YCRD/ELEV/ERRNUM/CKELEV`。

- 现状（公开上游对应实现）：`HFIND` 查找 `SLAS` 二级头；只有 `IFIND==1` 时读值，但坐标 buffer 在未找到头时没有显式初始化/失败出口；后续仍解析 buffer，并可能写默认坐标。
- 建议修改：在字段坐标读取前将 `CXCRD/CYCRD/CELEV` 初始化为空；显式确认 `HFIND` 命中包含 `XCRD/YCRD/ELEV` 的二级头并读到 `LN==LNFLD` 的对应记录；每次数值解析检查 `IOSTAT` 和非空值；缺字段时报告 `IPFLD` 文件名/行号并停止生成带静默默认值的 INP。成功时保持 `CXCRD -> XCRD -> FIELD/CXCRD`、`CYCRD -> YCRD -> FIELD/CYCRD`、`CELEV -> ELEV -> FIELD/CELEV`，不得站点硬编码；对任意站点值做单测，并以生成的 INP/INH 做端到端验证。
- 验收：同一份通用 parser 对 YC 与非 YC fixture 都保留数值；INP/INH 的 X/Y/ELEV 与 FileX 一致，Summary.OUT 有对应坐标，WARNING.OUT 无坐标读取/传递错误。

这只是针对公开上游对应实现的补丁提案。需拿到与 4.8.0.024 `dscsm048` 完全匹配的源码/tag 后核对并在隔离环境编译验证，不能把它直接移植到容器二进制或用 post-run 改写 INP 冒充修复。
