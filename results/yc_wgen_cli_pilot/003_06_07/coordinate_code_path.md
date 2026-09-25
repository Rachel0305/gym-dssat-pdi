# YC 坐标代码路径追踪

## 已验证的仓库内路径

| 环节 | 文件/函数/行 | 输入 | 处理与输出 | 证据/结论 |
| --- | --- | --- | --- | --- |
| 原始 YC FileX | `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0801.MZX`，`*FIELDS` 后 `@L ... XCRD ... YCRD ... ELEV` 行及 field 1 行 | XCRD/YCRD/ELEV 为 `-99` | 站点源模板提供占位值 | 原始文件未修改 |
| FileX 坐标写入 | `results/yc_wgen_cli_pilot/003_06_06/crop_smoke/run_dssat_crop_smoke.py:77-109`，`_coords_in_isolated_filex` | YC metadata LONG=116.570、LAT=36.830、ELEV=22 | 按 `XCRD->LONG`、`YCRD->LAT`、`ELEV->ELEV` 写入固定宽度字段 | 该历史 runner 的实现及 `runtime_snapshot/fileX.MZX` 可核对；本轮没有改写它 |
| 文件读取映射 | `results/yc_wgen_cli_pilot/003_06_06/crop_smoke/run_dssat_crop_smoke.py:113-122`，`_filex_coordinates` | runtime FileX 第一个 field row 的空白分隔值 | token 1/2/3 读取 XCRD/YCRD/ELEV | 三组快照解析为 LONG=116.570、LAT=36.830、ELEV=22 |
| PDI 环境创建 | 同一 runner `:282-305`，`gym.make(... fileX_template_path=...)` 及后续 FileX 检查 | 上述 FileX 模板 | 调用 `gym_dssat_pdi:GymDssatPdi-v0`；仓库 runner 只验证生成后的 FileX，没有暴露 parser 内部坐标对象 | internal parsed variables 在仓库记录中不可观测 |
| WGEN 主 runner | `scripts/run_yc_wgen_seed_pilot.py:55-74`、`:396-403`、`:501-512` | WGEN FileX 和 Gym-DSSAT/PDI 环境参数 | one-treatment FileX / runtime class 入口；调用安装的 `gym_dssat_pdi` | 该脚本没有坐标 parser、映射或 `DSSAT48.INP` 坐标替换逻辑 |
| DSSAT field input | 上一轮三组 `runtime_snapshot/DSSAT48.INP` 的 `*FIELDS` coordinate row | FileX 中正确坐标 | 生成记录仍为 XCRD=-999、YCRD=-99、ELEV=-99 | 丢失已被定位在正确 FileX 之后、field input 实际接收之前；同文件 `Yucheng` 土壤/站点文字行中的 36.830/116.570 不是 `*FIELDS` 坐标记录 |
| Companion input | 上一轮三组 `runtime_snapshot/DSSAT48.INH` 的 `*FIELDS/@L XCRD YCRD ELEV` rows | 同上 | 对应记录同样保留 -999/-99/-99 | 作为第二份 DSSAT 输入证据；坐标 preflight 同时检查 INP 与 INH |
| DSSAT runtime | 三组 `WARNING.OUT`、`Summary.OUT` | field coordinate placeholders | latitude/longitude/elevation read 与 `CYCRDin/CXCRDin/CELEVin` transfer 报错；summary 坐标列空 | 最终 runtime 坐标被 warning 记录置零 |

## 安装包源码边界

仓库中未发现 `gym_dssat_pdi.envs.dssat_pdi` 的实现文件；项目调用它的运行入口配置为容器内 `/opt/gym_dssat_pdi`。项目 `AGENTS.md` 限定只读写当前仓库，因此本轮没有读取、复制或修改该仓库外路径，也没有 Docker exec 去访问安装包。由此不能诚实地给出内部 parser、坐标变量映射、DSSAT48.INP renderer 的具体函数/行号，亦不能判定是 PDI 不解析、键名不匹配还是后续默认值覆盖。

## 根因判断边界

**证实的丢失区间**：runtime FileX 正确 -> `DSSAT48.INP *FIELDS` 坐标仍为占位值 -> DSSAT coordinate read/transfer warning -> Summary.OUT 空坐标。

**尚未证实的根因**：安装包内解析或写入实现。`dssat48_coordinate_check.json` 因字段记录不符返回 `pass=false`。本轮不通过直接改写一个模拟后的 `DSSAT48.INP` 来伪造 runtime propagation，也不启动新 crop run。
