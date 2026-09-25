# 003_06_07_01 实验记录

日期：2026-09-25（Asia/Shanghai）

## 目标

只读定位实际 Gym-DSSAT/PDI runtime 的源码与 YC 坐标传递边界；遵守资源约束，不在未通过 INP 坐标门禁时启动 DSSAT、seed 重跑或 PPO。

## 执行记录

1. 在运行容器 `nifty_taussig` 使用 `/opt/gym_dssat_pdi/bin/python` 自动定位 import：distribution `gym-dssat-pdi==0.0.5`，模块 `gym_dssat_pdi.envs.dssat_pdi`。DSSAT 模型版本 `4.8.0.024` 由既有 `Summary.OUT` 记录；两种版本号没有混用。
2. 只读检查实际 Python source、模板、配置和启动器。`DssatPdi._make_fileX_template` 的 context 只有 `wther/ferti/irrig/plant`；`_launch_client` 执行 `run_dssat`，wrapper 最终调用 `dscsm048`。Python package 中未找到 coordinate parser 或 DSSAT48.INP/INH renderer。
3. 逐文件复制 5 个相关 Python/模板/配置文件和 1 个 29-byte launcher 到 `runtime_source_snapshot/`，并比对 SHA256。未复制 10.63 MB 原生 binary；未对容器和安装文件作任何写入。快照不 stage、不 commit。
4. 用 `DssatPdi.__new__` 建立最小对象并直接调用实际安装的 `_make_fileX_template`，未调用构造函数、socket、子进程或 DSSAT。输出坐标行为正确，完整 FileX SHA256 与历史运行快照完全一致。
5. 复核 historical control、seed 101、seed 104 三组既有产物：FileX 行含 `XCRD=116.57000, YCRD=36.83000, ELEV=22.0`；INP/INH 的 field row 仍是 `-999/-99/-99`；WARNING.OUT 有 read 与 transfer error；Summary.OUT 坐标为空。因第一门禁 FAIL，本轮不启动任何新的 crop run。
6. 通过官方 DSSAT 当前公开源码定位 `InputModule/ipexp.for::IPFLD`，形成上游补丁提案。明确其不是安装二进制 4.8.0.024 的对应源码，具体内部 parser buffer 值仍未观察。
7. 调整 repo 中诊断 helper 的通用 round-trip 断言，删除隐藏的 YC 常量比较；增加非 YC FileX renderer 和 INP field parser fixtures。测试命令通过：10 passed。

## 错误与处理

- 初次生成源码哈希清单时把 `module_file` 字符串当作 `Path` 使用，出现 `AttributeError`；改为显式 `Path(module_file)` 后只读哈希成功，未影响任何文件。
- 第一次 `DssatPdi.__new__` 探针结束时，部分构造对象的 `__del__` 访问未初始化生命周期字段并打印异常。为避免干扰，复跑时设置 `closed=True` 和 `_f_out=None`；最终探针无异常，输出哈希仍与历史文件一致。

## 决策

- 根因行为分类：`PARSER_FIELD_NOT_READ`，因为 FileX 正确而 native read/transfer 告警及 INP/INH 占位符确认丢失发生在 native 解析/传递边界。
- 不把当前官方 `develop` 版本源码当作已安装 binary 源码；具体缺陷语句需要匹配 4.8.0.024 的源代码后确认。
- 无安全 repo-side pre-INP override；不修改 native package，不事后改写 INP，不重跑 crop，不做 PPO。
- `coordinate_warning_resolved=NO`；旧 warning 总计 42 个离散事件，其中 class-C 坐标事件 36、class-B 非坐标事件 6。修复后未运行，after 证据留空。

## 产物与边界

- 源码身份：`runtime_source_inventory.json`
- 快照哈希：`runtime_source_snapshot_manifest.json`
- 代码链：`coordinate_runtime_code_trace.md`
- 实际值链：`coordinate_value_trace.json`
- 第一门禁：`dssat48_coordinate_gate.json`（FAIL）
- 作物前后比较：`coordinate_fix_crop_output_comparison.csv`（after/delta 空）
- warning 前后比较：`warning_comparison.csv`（after 未复测）
- 容器 package/executable 修改：NO
- 本轮 DSSAT/WGEN/CLI/weather fitting/PPO run：NO
- PPT：NO
