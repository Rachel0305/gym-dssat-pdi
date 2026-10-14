# 003_06_05_02 YC corrected CLI + WGEN seed pilot 实验记录

## 任务边界

- 单站点：YC；单季：2008 treatment 1；DSSAT 4.8.0.024。
- 天气拟合唯一输入：冻结 2004–2013 CSV，SHA256 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`，3653 行，2004-01-01 至 2013-12-31。
- 不使用 2014–2023 fitting，不用 validation fitting；不修改 wet-day 定义（`RAIN > 0.0 mm`）；不运行 PPO；不触碰其他站点。
- 旧的 003_06_04 CLI 保留，SHA256 `5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0`。

## 执行记录

1. **回归测试与输入预检**：先执行 CLI 构建和 seed pilot 原有测试，`23 passed`。检查 frozen weather、旧错误 CLI 和 parsefix 候选哈希均符合任务记录。
2. **Gate A 正式 CLI**：用 corrected formatter 从冻结 2004–2013 天气生成 `final/CNYC.CLI`。结果 SHA256 `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`；与 parsefix 候选字节完全一致；12 月份、固定宽度、数值与 schema 检查通过。
3. **门禁字段读取修复**：第一次 seed 编排在运行任何 DSSAT 前退出，因为编排脚本寻找 `schema_pass`，而实际 schema JSON 的成功字段叫 `passed`。读取条件改为实际字段后才启动 pilot；这次中止没有产生 runtime 运行或改写天气结果。
4. **Gate B 六次串行 WGEN**：101a、101b、102、103、104、105 全部 `PASS` 并捕获日天气。最大进程树 RSS `131.54 MB`；运行间无并发。正式 CLI、冻结天气和旧 CLI 的运行前后哈希均未改变。
5. **天气比较与 QC**：101a/b 的 DATE、DOY、RAIN、SRAD、TMAX、TMIN 完全一致；101–105 有 5 条不同序列，10/10 pairwise 不同。六次天气 QC 全部通过。
6. **分析脚本导入修复**：首次单独运行分析脚本时，它尝试从脚本所在 results 目录导入 repo runner 而失败；脚本改为内置所需的轻量 WTHER 读取逻辑后重跑分析。没有重跑 DSSAT，也没有修改既有 runner。
7. **Runtime 提示**：没有 CLI lookup、WGENIN 5010、parse 或 seed-interface 错误。六个 `WARNING.OUT` 均提示 FILEX 的经纬度/海拔缺失、PHOTO 方法兼容转换、`CYCRDin/CXCRDin/CELEVin` 字段未传递；`INFO.OUT` 还说明 STONES、ADCOEF 使用默认值。作为 WGEN pilot 这些提示不阻止天气生成，但必须在 crop-output smoke 中核实影响。
8. **PPT 校验修复**：首轮 finalizer 拒绝 Han script fallback 字体声明；移除校验器不支持的 fallback 配置，保留运行时解析的 Arial 后重新导出。最终 7 页、3 个 native tables、1 个 native chart、package/layout 校验通过；逐页检查导出预览。
9. **最终回归测试**：运行原 CLI/seed pilot 测试及新增天气比较/QC 单测，共 27 项；结果写入本任务报告，最终复核命令见报告“测试”章节。

## 六次逐日序列结果

| Run | Seed | Days | Weather SHA256 | Runtime | QC |
|---|---:|---:|---|---|---|
| seed_101_run_a | 101 | 120 | `D9CD9926AE4382BA24783E498A928ED25F5EA92F1C60D1B3C7034448AAE5C29D` | PASS | PASS |
| seed_101_run_b | 101 | 120 | `D9CD9926AE4382BA24783E498A928ED25F5EA92F1C60D1B3C7034448AAE5C29D` | PASS | PASS |
| seed_102 | 102 | 117 | `240C34531DC83727D010317BC006BD44DE267418896AD237085A03F103C4AA43` | PASS | PASS |
| seed_103 | 103 | 117 | `DC5C2EAA31F9930CAB3EC3312AF9AF95315D8ACC57A4AA1BD5BD15CB248E20C2` | PASS | PASS |
| seed_104 | 104 | 116 | `7E6D92914D38E0C6B4C9A90F9FA4438F8DFA51AFC7963AA64F147A29FEE74E67` | PASS | PASS |
| seed_105 | 105 | 118 | `94E881EBFFFFA8224F5E1BF386E5589C74CAB3CB3EF8A9B52F422C3B569B3405` | PASS | PASS |

## 判定

`WGEN_SEED_PILOT_PASS`。完整机器可读证据在本目录的 `pilot_summary.json`、`reproducibility_check.json`、`seed_diversity_check.json`、`weather_summary_by_seed.csv`、`physical_sanity_check.json` 和 `validation/`。推荐下一步 `003_06_06_yc_wgen_weather_qc_and_dssat_smoke`；不得把当前结果当作 crop-output 通过或 PPO 长训许可。

工作区原有与本任务无关的修改和未跟踪文件未清理。本任务按明确文件清单本地提交，不推送 GitHub；备份推送待用户明确批准。
