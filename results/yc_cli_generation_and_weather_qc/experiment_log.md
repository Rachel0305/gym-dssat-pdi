# YC `.CLI` / WGEN 天气 QC 实验记录

## 运行边界

- 日期：2026-09-23，Asia/Shanghai。
- 站点：仅 YC/YCA；年份：2004–2013 训练期。
- 起始 HEAD：`c0c1f9f`；分支：`codex/sya-forecast-freeze-2026-08-16`。
- 起始时已存在的无关 tracked 修改：2 个 `configs/068_*_smoke.yaml`、1 个 proposal Markdown、`src/054_02` 绘图脚本、`src/mask_aware_dqn_029.py`。均未改动/暂存。
- 目标结果目录在开始时不存在；原始 Excel 以 Excel COM `ReadOnly=true` 打开，所有工作簿以 `Close($false)` 关闭。
- 未写入/覆盖生产 WTH、FileX、SOL、CUL、旧 CLI 或 PPO 配置；未运行 DSSAT、WeatherMan、WGEN 或 PPO。

## 执行记录

1. 复核两个 003_02 任务文件、lowIC 输入目录、`055_00` 配置、safe-render 路径和前轮 split/CLI provenance。确认训练=2004–2013、验证=2014–2023、独立 test 未认证；确认旧 `CNYC.CLI` 统计窗口含 2014，不可用。
2. 核对 FileX 字段：`CNYC0801.MZX` 处理行包含 `CNYC0801` 与 `CNYC1401`；safe renderer 使用按年份改写的 `CNYCyy01` 并复制同名天气文件；`experiment_number=1`、`random_weather=false`。静态预期气候文件名为 `CNYC.CLI`，但没有实际运行 WGEN。
3. 扫描 2004–2013 全部 WTH：十个文件均为完整自然年，无日期缺失/重复或基础物理错误。逐日对 `YCA_weather_cleaned.csv` 的四变量匹配全部落在 0.051 舍入容差内。
4. 从 `data_check_by_year_before_fill.csv` 读取逐年源缺测数：十年均含 SRAD/TMAX/TMIN 的源缺值；最终缺值填补规则是站点-月份均值。2004 数量为 SRAD 41、TMAX 47、TMIN 47。
5. 只读 Excel COM 筛选三份原始表的 YCA 2004：T2 两温度列各 47 空白，D32 SRAD 41 空白，降雨列 348 空白、4 个显式零、18 个数值。WTH 最长干段 DOY 001–257 的 257 日对应原始降雨空白；预处理将其转零。依据“数据覆盖/缺值语义未被充分独立证实”，Gate B 判定失败。
6. 写出机器可读结果、中文报告与中文过程记录；停止于 Gate B。未生成 CLI、realization、manifest hash 或 smoke 输出；seed 101–105 仅登记为计划值；PPO=0。

## 失败尝试、纠正与决策

- 一次内联 PowerShell 审计代码因 try/catch 括号解析失败，在任何 Excel 打开之前退出；未读写源数据。改为可复用 `.ps1`。
- 第一版 Excel 审计遍历所有工作表行，运行缓慢，并因 PowerShell 脚本对中文文件名的代码页解码失败而未完成第三本工作簿；打开的表均为只读且未保存。改用 ASCII 通配符匹配降雨工作簿，并在 Excel 内存视图筛选 `YCA / 2004` 后读取可见区域，审计在约 13 秒内完成。
- 第一版格式审计把所有绝对值 `>=90` 当成 sentinel，误将 `90–160 mm` 的日降雨标为 sentinel。核对原始数据后改为检查常见精确 sentinel 数值（±99、±999、±9999）；最终十年均无此类 sentinel。高降雨保留为有效记录并单独提供极端值计数，不自动剔除。
- Git 全量状态枚举既有大型研究工作区时收到若干 Windows 路径过长警告；没有执行删除、移动或清理。后续仅针对本任务路径核验和暂存。
- PPTX 首次结构校验缺少每个原生表格对应的声明策略；补充 slide 2/3/6/7 校验参数后继续。第一次 Artifact Tool 重载验证缺 `RUNTIME_NODE_MODULES`；改用已配置 workspace dependency 路径后通过。
- 初版 PPTX 的 slide 6 表格几何警告显示超出页脚；调低行高后生成最终版。首个有警告文件和后续通过的中间版本均保存在私有构建目录，没有删除。最终标准路径文件 8 页全部渲染并逐页检查；包校验无发现、版面 warning=0、原生表格 4 张、原生图表 2 个。未在桌面 PowerPoint 打开。
- 最终汇报文件 SHA256：`b2bcd93ab6ef1281362356a91bd28a9d50ff2736195b13e3d1480b60e1e02454`；结构摘要见 `pptx_validation_summary.json`。

## 判定

最终状态：`BLOCKED_WEATHER_INTEGRITY`。阻塞点不是 `.WTH` 格式，而是用于拟合的逐日源气象记录含缺测/填补，尤其 2004 的长段降雨空白。先解决/批准源数据处理原则，再进入 `.CLI` 参数估计；本轮没有绕过门槛。
