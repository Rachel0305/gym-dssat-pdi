# DSSAT v4.8.0.24 坐标补丁核验

## 结论

已在官方源码中确认 `IPFLD` 坐标读取条件错误，并生成了仅针对该错误的补丁。4 项静态/fixture 测试通过，补丁可应用于精确 tag 的原始文件。

**运行验证状态：`BLOCKED_AT_ISOLATED_BUILD`。** 宿主机和当前容器都没有 CMake 与 Fortran 编译器。未生成独立 binary，因此没有运行坐标门禁、历史 control 或 seed 101/104；本报告不把静态验证表述为 DSSAT runtime 结果。

## 上游身份与源码证据

仓库为 [DSSAT/dssat-csm-os](https://github.com/DSSAT/dssat-csm-os)，tag 为 `v4.8.0.24`，peeled source commit 为 `caaa55c6bee21aa894b325b67dad7ccacb05295b`。tag 下的 [CMakeLists.txt](https://raw.githubusercontent.com/DSSAT/dssat-csm-os/v4.8.0.24/CMakeLists.txt) 标记 major=4、minor=8、model=0、build=24。

官方 [v4.8.0.24 `InputModule/ipexp.for`](https://raw.githubusercontent.com/DSSAT/dssat-csm-os/v4.8.0.24/InputModule/ipexp.for) 中，XCRD、YCRD、ELEV 三次内部 READ 都把 `IOSTAT=0` 当作失败并替换成 sentinel。其坐标 gate 又附加 `.AND. ERRNUM .NE. 0`；这里的 ERRNUM 已被最近一次高程读取覆盖，成功时等于 0，导致有效坐标不能进入 transfer 分支。

只把 [v4.8.5.0 `ipexp.for`](https://raw.githubusercontent.com/DSSAT/dssat-csm-os/v4.8.5.0/InputModule/ipexp.for) 用作后续行为对照。补丁没有合入它在同一文件中的其它修改，包括 PMWD/PMALB。

本轮确认的是源码逻辑缺陷。因为没有 patched runtime，尚未完成“该缺陷导致当前 YC runtime 坐标丢失”的修复后实证。

## 补丁与静态验证

补丁位于 `results/yc_wgen_cli_pilot/003_06_07_02/patch/ipfld_coordinate_fix.patch`，只涉及 `InputModule/ipexp.for` 一个文件和四个语义条件：

- 三处失败分支改为 `IF(ERRNUM .NE. 0)`，读取失败才写 sentinel 并发 warning。
- 从坐标转交 gate 移除依赖高程最后一次读取状态的 `.AND. ERRNUM .NE. 0`。

补丁对原始 tag 文件的 `git apply --check` 通过，并已在隔离副本应用。`tests/test_dssat_v48024_coordinate_patch.py` 有 4 个静态/fixture 测试，覆盖 YC `116.570/36.830/22`、非 YC `-93.65/42.03/310`、畸形/空值、全零和超范围坐标。它们不等同于 Fortran 编译测试。

完整 DSSAT 源码没有下载或纳入版本控制。所选上游快照、tag/commit、文件 SHA256、差异审计见 `results/yc_wgen_cli_pilot/003_06_07_02/validation/source_diff_audit.json`。

## 构建与接口阻断

官方 v4.8.0.24 CMake 工程是 Fortran 工程，目标名由版本标记生成，预期为 `dscsm048`。建议在完整、隔离的源码副本和预装工具的专用 Linux 环境中使用单并行度构建：

```text
cmake -S <full-v4.8.0.24-source-copy> -B <project-results-build-dir> -DCMAKE_BUILD_TYPE=Release
cmake --build <project-results-build-dir> --parallel 1
```

本轮没有执行以上构建命令。当前容器工具检查的原始错误为：

```text
exec: "cmake": executable file not found in $PATH
exec: "gfortran": executable file not found in $PATH
```

此外，官方 CMake 源文件清单和 CSM 主入口没有 PDI/ZMQ 标记。基于这两处源码检查，**推断**普通上游可执行文件不能直接视为 Gym-PDI runtime；应先定位当前已安装 runtime 的精确 PDI 源码集成，再做接口兼容性审计。没有为此修改容器、安装软件或碰触 `/opt/dssat_pdi/dscsm048` 与 `/opt/gym_dssat_pdi/`。

## 门禁与实验范围

`coordinate_gate.json` 记录为 `NOT_RUN_BLOCKED_AT_ISOLATED_BUILD`。因此：

- DSSAT48.INP/INH、Summary.OUT 坐标和 warning 消失情况：未检查。
- 历史 control、seed 101、seed 104：未运行，也没有作物结果比较 CSV。
- `PPO_COORDINATE_GATE_PASS`：未达到。
- `YC_RANDOM_WEATHER_PPO_READY`：`NO`，不得据此启动 PPO。

本轮没有修改原始 runtime、已安装 Python 包、WGEN、CLI、天气、PPO 或其它站点；没有创建 PPT，也没有进行 Git push。

## 机器记录

- 实验日志：`results/yc_wgen_cli_pilot/003_06_07_02/experiment_log.md`
- 构建信息：`results/yc_wgen_cli_pilot/003_06_07_02/build/build_metadata.json`
- 源码差异审计：`results/yc_wgen_cli_pilot/003_06_07_02/validation/source_diff_audit.json`
- 坐标门禁状态：`results/yc_wgen_cli_pilot/003_06_07_02/validation/coordinate_gate.json`
