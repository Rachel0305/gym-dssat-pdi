# YC 完整全年 WGEN 路径源码追踪

## 范围与版本

本次源码依据为 DSSAT 官方 `dssat-csm-os` 仓库 `v4.8.0.24`。仓库内已有的 CSM 主程序摘录 SHA256 为 `A173EEDB3BF98D9FB152D3538A73B90B56723492A694E4A3B11B5AD3EB5CA0F6`，与此前 `source_diff_audit.json` 记录一致；该审计记录指向官方标签解析提交 `caaa55c6bee21aa894b325b67dad7ccacb05295b`。该摘录是选取源码，不是本机运行二进制或完整源码树。

## 调用链

1. `CSM_Main/CSM.for` 的每个日循环步依次执行 `RATE`、`INTEGR`、`OUTPUT`，每步都把控制交给 `LAND`。`LAND` 是向作物/土壤/天气组件分发动态阶段的模型入口。
2. 官方 `Weather/weathr.for` 在 `RATE` 阶段按 `MEWTH` 选择天气方式；`MEWTH='W'` 或 `'S'` 时调用 `WGEN`。WGEN 是天气模块调用的子程序，不是被该源码证明为独立 CLI 可执行程序。
3. 官方 `Weather/WGEN.for` 在季节初始化时经 `WGENIN` 读取天气参数；`MEWTH='W'` 时从输入文件的 `*WGEN` 区段装载 12×14 月参数。正式 YC `CNYC.CLI` 的 CSM-WGEN 读取路径因此可由当前既有源码机制解释，但这不证明 WeatherMan 可直接导入/再生成该文件。
4. WGEN 在每日 `RATE` 调用中执行 `WGENPM` 和 `WGENGN`，产生当天 `RAIN`、`SRAD`、`TMAX`、`TMIN` 等数值；`RSEED1` 传入随机数种子，非正时源码默认首种子为 2510。日逐次调用说明它可服务于模型日模拟，不等于存在一条脱离 CSM 生命周期的全年批量导出 CLI。
5. `CSM.for` 在季节初始化调用 `LAND` 后进入 `DAY_LOOP: DO WHILE (YRDOY .GT. YREND)`。同一 `YREND` 在 RATE、INTEGR、OUTPUT 阶段继续传入 `LAND`；CSM 源码变量注释将其定义为“季节结束日期（通常是收获日）”。退出日循环后进入 `SEASEND`；全部运行结束后才进入 `ENDRUN`。因此 `SEASEND/ENDRUN` 是结束流程，不是把短作物季节自动补齐到 12 月 31 日的机制。
6. `WEATHR` 在 `OUTPUT` 阶段调用 `OpWeath`，官方 v4.8.0.24 `Data/OUTPUT.CDE` 列有 `Weather.OUT`。它记录已执行模拟日的天气结果，时段仍受 CSM 日循环限制，不能据此推断它会生成完整全年数据。

## 对 YC 116–120 天 blocker 的解释

现有 YC WGEN 运行只保存作物单季输出。根据 CSM 主循环，天气调用随模拟日推进，停止点由 `YREND` 控制；变量注释说明该点通常为收获日期。故已观察到的约 116–120 天是作物季节终止边界，不是 WGEN 算法内建的 120 天上限。要获得年历全年，需有经验证的独立生成工具，或证明 CSM 可在不受作物季节终止约束时持续到年末；本次没有证明后一种配置。

## 全年配置判定

- 官方源显示 CSM 支持多个运行模式，也显示 WGEN 接受每日日期和随机种子；但审计到的 CSM 驱动仍按 `YREND` 划分季节，并调用 `LAND`。没有在官方 v4.8.0.24 证据中找到 WGEN-only/no-crop/fallow 配置，可保证从 1 月 1 日跑到 12 月 31 日并作为独立 WTH/CSV 导出。
- Sequence/seasonal simulation 模式确实存在，但其存在本身不能证明该日循环不依赖 crop/land lifecycle；本次不把它们当作全年 WGEN 解决方案。
- 现阶段 `dssat_csm_full_year_path_found=NO_VERIFIED_PATH`，不是断言 DSSAT 绝不可能实现全年生成。

## 官方来源

- [DSSAT CSM v4.8.0.24 CSM.for](https://raw.githubusercontent.com/DSSAT/dssat-csm-os/v4.8.0.24/CSM_Main/CSM.for)：日循环、YREND、SEASEND/ENDRUN。
- [DSSAT CSM v4.8.0.24 WGEN.for](https://raw.githubusercontent.com/DSSAT/dssat-csm-os/v4.8.0.24/Weather/WGEN.for)：WGEN 参数读取、逐日生成、随机种子接口。
- [DSSAT CSM v4.8.0.24 weathr.for](https://github.com/DSSAT/dssat-csm-os/blob/v4.8.0.24/Weather/weathr.for)：天气方式选择、调用 WGEN、逐日 Weather.OUT 输出。
- [DSSAT CSM v4.8.0.24 OUTPUT.CDE](https://raw.githubusercontent.com/DSSAT/dssat-csm-os/v4.8.0.24/Data/OUTPUT.CDE)：Weather.OUT 定义。
- [DSSAT 官方 WeatherMan FAQ](https://dssat.net/5165/)：WeatherMan 从气候数据生成日天气并导出；CSM 亦提供内部 WGEN/SIMMETEO 方式。
- [DSSAT User's Guide Vol. 3（WeatherMan 参考手册，旧版 DSSAT v3）](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf)：手册称 WeatherMan 可生成完整年份，记录随机种子并导出生成的日值。该旧版资料只能支持功能概念，不能证明本机版本的 UI/API/导入兼容性。
