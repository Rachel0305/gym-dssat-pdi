# 003_06_07_02 实验记录

日期：2026-09-25（Asia/Shanghai）

## 目标

基于 DSSAT 官方 `v4.8.0.24`，核验 `InputModule/ipexp.for::IPFLD` 的坐标读取逻辑，制作最小坐标补丁并评估是否可在隔离环境构建。不得修改已安装 DSSAT/PDI runtime，不运行 WGEN、天气生成、作物模拟或 PPO。

## 操作与证据

1. 使用官方仓库 tag 引用核实版本身份：`v4.8.0.24` 的 tag object 为 `1a2bcfeb4b19e07769881df70c039d66d7e92609`，peeled commit 为 `caaa55c6bee21aa894b325b67dad7ccacb05295b`。本地 `CMakeLists.txt` 版本标记为 major=4、minor=8、model=0、build=24。
2. 保存官方源文件的选定快照：`ipexp.for`、`CMakeLists.txt` 和 `CSM.for`；另保存 `v4.8.5.0` 的 `ipexp.for` 作为只读对照。没有拉取完整上游仓库，因为当前环境无法构建，避免无用下载和磁盘/内存占用。
3. 在旧版 `IPFLD` 中确认三处 `READ(...,IOSTAT=ERRNUM)` 后使用 `IF(ERRNUM .EQ. 0)`，成功读取会错误地触发 sentinel 与 warning。字段坐标有效性 gate 还要求 `ERRNUM .NE. 0`，但此时 ERRNUM 来自紧邻的高程读取；正常高程读取会令其为 0，因而阻断有效坐标传递。
4. 对照 `v4.8.5.0`，只回移上述三处条件及坐标 gate 修正，没有复制 PMWD、PMALB 或其它版本变化。补丁涉及一个源文件、四个语义条件修改。补丁在隔离的原文副本上通过 `git apply --check`，并成功应用。
5. 运行 `python -m unittest discover -s tests -p 'test_dssat_v48024_coordinate_patch.py' -v`，4 项静态/fixture 用例全部通过，覆盖 YC、非 YC、错误/空值、全零及超范围坐标。该测试不是 Fortran 编译或 DSSAT runtime 测试。
6. 检查宿主机及 `nifty_taussig` 容器工具链。宿主机没有发现 CMake、Fortran 编译器、Make 或 Ninja。容器只有 `/usr/bin/gcc`；执行 `docker exec nifty_taussig cmake --version` 返回 `exec: "cmake": executable file not found in $PATH`，执行 `docker exec nifty_taussig gfortran --version` 返回 `exec: "gfortran": executable file not found in $PATH`。没有执行构建命令，也没有尝试安装依赖。
7. 官方 tag 的 CMake 构建清单和 CSM 主入口未发现 PDI/ZMQ 标记。由此推断，普通上游构建与当前 Gym-PDI runtime 的协议兼容性不能默认成立；需先识别并审计当前 runtime 对应的 PDI 源码集成。

## 结果与停止决定

- 源码层面的 `IPFLD` 条件错误：已确认。
- 最小补丁：已生成，静态验证通过。
- 隔离二进制构建：`BLOCKED_AT_ISOLATED_BUILD`，缺 CMake 和 Fortran 编译器；PDI 接口兼容性也未建立。
- 坐标门禁、历史 control、seed 101/104、Warning.OUT 对比：均未运行。按任务要求不越过该门禁，不产生作物输出对比 CSV。
- PPO：未运行。
- 已安装 binary/package、原始 FileX、CLI、天气及 PPO 配置：未修改。

## 下一步前置条件

准备已批准、隔离且预装 CMake、GNU Fortran、Make/Ninja 的 Linux 构建环境；同时提供或定位与 `gym-dssat-pdi==0.0.5` runtime 对应的 PDI 源码集成。随后在完整 `v4.8.0.24` 源码副本中应用本补丁，先验证 PDI 协议和历史 control 坐标门禁。只有门禁通过后，才依序决定是否运行 seed 101 和 104。
