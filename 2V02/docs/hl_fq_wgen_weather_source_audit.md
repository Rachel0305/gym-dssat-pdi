# HL / FQ WGEN 随机天气实验：气象数据源审计

**审计日期：** 2026-10-04（Asia/Shanghai）  
**当前判定：** `WEATHER_SOURCE_AUDIT = BLOCKED`；`STOP_BEFORE_WGEN = TRUE`  
**适用范围：** HLA 2004–2013、FQA 2005–2013；2014–2023 未读取、未用于补偏或拟合。

## 用户请求与附件说明的边界

用户请求是阅读并执行 `prompt_02/008_hl_fq_wgen_random_weather_8seed.md`。该文件作为实验流程与门禁要求执行；其中关于“YC 已有成功 WGEN-PPO 路线”的陈述先按本地代码、配置和 QC 证据核验，不把附件文字本身当作实验事实。

项目 `AGENTS.md` 要求节省算力、避免 OOM、只在项目目录读写、保护原始气象与已有结果，并禁止未获确认的 GitHub push。本阶段只做源审计；未运行 DSSAT、WGEN 或 PPO，未改写原始 `.xls`、`.WTH`、配置、reward 或 action mask。

## 结论摘要

两站已有 `.WTH` 文件日期连续且可被 DSSAT 读取，但其拟合期 provenance 和极值质量不满足本任务要求，不能直接拟合 WGEN。训练期原始 `.xls` 的温度、辐射缺值/异常可以由通过重叠期筛选的 NASA POWER 候选值按偏差校正后仅填缺失或物理异常行；NASA 降水未通过门槛。附近官方 GHCN-Daily 雨量站可填大部分空白，但对所有通过筛选的站点取可用值后，仍有 HLA 111 天、FQA 157 天在 2013 年没有可核实雨量来源。

因此没有可靠的完整 `DATE/RAIN/SRAD/TMAX/TMIN` 拟合序列。当前两站均判 `BLOCKED`，不生成站点 `.CLI`、随机天气池或 PPO 训练输入。逐字段与文件哈希见 [weather_source_audit_final.json](../results/hl_fq_random_weather_8seed/source_audit/weather_source_audit_final.json)。

## 本地源数据与已有 WTH

| 站点 | 拟合期 | 期望天数 | 原始文件 | 主要发现 |
|---|---|---:|---|---|
| HLA | 2004-01-01–2013-12-31 | 3,653 | `D32.xls`、`T2.xls`、`HLLCYCFQ降雨数据.xls` | RAIN 2,691 天空白；TMAX 有 31 天缺测和 1 个物理异常值，最高 154.3°C（2010-06-02）；TMIN 缺 31 天；SRAD 缺 29 天。 |
| FQA | 2005-01-01–2013-12-31 | 3,287 | 同上 | RAIN 2,892 天空白；TMAX/TMIN 各缺 83 天；SRAD 缺 93 天，另有 306.055（2013-03-31）异常值。 |

原始 Excel SHA-256：`D32.xls`=`FD885AFF6690CC5D415A3A31A4D409BD032A9BC3502827F5161AA60FDAA983E8`；`T2.xls`=`D95CEEE849DF1717F3F2C9E301048467372C3974623D68CE59B7DADFFED54B01`；`HLLCYCFQ降雨数据.xls`=`6E797937ABD021F061CF8A89711A6B24EE462C1D33CE361328DDE5741E66430`。

现有 `.WTH` 每年完整覆盖 365/366 天、没有日期空洞或负降水/负辐射，但 HLA 2010 WTH 保留 TMAX=154.3°C；FQA 的原始 SRAD 异常与现有 WTH 不一致，且既有清洗记录显示温度/辐射曾按站点-月份均值填补。WTH 文件哈希和站点头信息保存在 machine-readable audit 中。这些事实说明“DSSAT 能运行”不能证明该系列适合估计 WGEN 气候参数。

## NASA POWER 候选：温度与辐射通过，降水拒绝

只查询各自训练拟合期和站点坐标，没有查询或使用 2014–2023。按有效本地重叠日计算平均偏差，再将该偏差应用于缺值或物理异常行；其余有效站点观测不替换。

| 站点 | 变量 | 有效重叠天数 | 偏差校正 MAE | RMSE | r | 候选填补行 | 门槛 |
|---|---|---:|---:|---:|---:|---:|---|
| HLA | TMAX | 3,621 | 1.515°C | 2.095°C | 0.992 | 32 | PASS |
| HLA | TMIN | 3,622 | 2.008°C | 2.614°C | 0.988 | 31 | PASS |
| HLA | SRAD | 3,624 | 1.278 MJ/m²/day | 1.772 | 0.969 | 29 | PASS |
| FQA | TMAX | 3,204 | 1.839°C | 2.484°C | 0.976 | 83 | PASS |
| FQA | TMIN | 3,204 | 1.644°C | 2.131°C | 0.979 | 83 | PASS |
| FQA | SRAD | 3,193 | 1.629 MJ/m²/day | 2.342 | 0.940 | 94 | PASS |

RAIN 不采用 NASA POWER：本地空白按零的筛选口径下，HLA / FQA 湿干日一致率只有 48.8% / 68.1%，均低于附件规定的 80%。逐日候选明细保存为 `nasa_candidate_gapfill.csv`，所有行均保持 `promoted_to_fitting_weather=false`。

## 官方 GHCN-Daily 降水候选

官方 GHCN-Daily `.dly` 数据按 0.1 mm 解码，并排除缺测值和质量标记异常。附件给出的湿/干一致率阈值为 80%。本轮另加两项保守筛选：至少 3,000 个有值日、CMA/NMIC 来源标记 `s` 占比至少 90%；这两项是本轮审计的补充条件，不是附件新增的冻结实验参数。格式和来源标记说明参见 [NOAA GHCN-Daily README](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)。

| 站点 | 最近通过筛选的站 | 距离 | GHCN 有值日 | 湿/干一致率（空白按无雨） | 仅本地有记录日的一致率 | CMA `s` 占比 |
|---|---|---:|---:|---:|---:|---:|
| HLA | HAILUN `CHM00050756` | 5.38 km | 3,442 | 89.3% | 85.1%（935 对日） | 95.5% |
| FQA | ZHENGZHOU `CHM00057083` | 87.91 km | 3,028 | 86.9% | 83.2%（388 对日） | 96.5% |

解析复核：GHCN `.dly` 的质量标记固定占一个字符，空标记是一个空格。初版解析器误与空字符串比较，曾将有效记录全部丢弃；已改为按空白字符判断，并用缓存的原始站点文件重跑完整候选筛选。以上计数来自修正后的解析；原始 `.dly` 与本地雨量表均未改写。

本地完整期分别有 2,691 / 2,892 个空白雨量日。对 300 km 搜索半径内通过筛选的官方站逐日取最近可用记录（只填本地空白、不覆盖本地数值），可形成 2,580 / 2,735 个雨量候选填补值；但仍分别缺 111 / 157 天，全部落在 2013 年。HLA 的补充站最多距站点 168.91 km；FQA 最多 286.5 km。按空白记零构成的事件一致率不能证明 2013 年这些空白是无雨，也不能把未观测值安全改成 0。

候选完整序列的累计量也保留了空间代表性提醒：HLA 本地值加候选补值的总量为邻近 Hailun 站总量的 1.028 倍；FQA 的复合候选为 Zhengzhou 站总量的 1.602 倍，且该站距 FQA 约 88 km。记录值与邻站累计差异不单独推翻 80% 事件门槛，但在报告中保留为显著的空间差异说明。

NASA 降水未通过后，官方中国气象数据网目录中的地面日值资料是可调查的下一来源；目录说明日值产品覆盖温度、降水等要素。[数据目录](https://m.data.cma.cn/data/cdcindex/cid/6d1b5efbdcbf9a58.html)和[下载实名规则](https://data.nsmc.org.cn/DataPortal/cn/support/faq.html)显示下载需实名注册。项目中没有该数据，且本轮未使用任何账户信息或取得 CMA 原始日值文件。

## YC WGEN-PPO 实际路线核验

本地存在经修正并实际用于 PPO 的参考路线，附件对此项前提基本成立，但它的已完成模型规模是 3 个 PPO seeds，不是 8 个：

1. `results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI` 是冻结 CLI，SHA-256=`65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`。
2. `results/yc_random_weather_ppo/004_03/config/weather_runtime_correction.json` 记录 WGEN 的实际开关是 task-local FileX 副本中的 `METHODS.WTHER=W`；只设 `random_weather=true` 和注入种子而保留 `WTHER=M` 曾导致失败，旧运行明确排除在有效结论外。
3. 修正后在每个 episode 前将 scheduled weather seed 注入 PDI `rseed1_`，以 DSSAT 逐日 `RAIN/SRAD/TMAX/TMIN` runtime sequence hash 验证天气；不以输入 WTH 文件 hash 替代。
4. `experiment_design.json` 冻结训练天气种子 1001–1080、held-out 种子 1081–1100，并与 PPO seed 分离；每个 80-seed block 覆盖共同天气池。
5. `experiment_log.md` 记录 3 个 WGEN PPO seeds 均请求 100,000、实际 100,080 步，三个 seed 的 962 个 episode 年份/天气 seed/daily hash 序列一致，训练和 held-out seed 无交集。
6. `results/yc_random_weather_episode_validation/004_02/summary.json` 的气候 QC 是共同 106 天 crop-season 窗口，全年 WGEN 验证被明确标记为 deferred；不得把这一段历史 QC 说成全年完整天气 QA。

因此 WGEN-PPO 的执行方法和 seed contract 可复用；HL/FQ 的目标站点 `.CLI` 和实际 runtime QA 尚未生成。本任务在数据门禁处停止，没有将 YC 结果移作 HL/FQ 证据。

## 决策与恢复条件

- HLA：`BLOCKED`。在训练期外部源候选覆盖后仍有 111 个 2013 RAIN 日期无经验证来源。
- FQA：`BLOCKED`。在训练期外部源候选覆盖后仍有 157 个 2013 RAIN 日期无经验证来源。
- 总结：`BLOCKED_BY_DATA_PROVENANCE`；`STOP_BEFORE_WGEN=true`。
- 恢复需要：把经授权取得的 2013 官方逐日雨量源文件放入项目目录，或提供可核实的站点记录/数据字典，证明相关空白日期可被解释为 0 mm。之后重新跑来源与物理 QC，完整日期链通过后再进入 CLI/WGEN。

不应把目前未解决的 2013 空白填为零，也不应启动 PPO 以获得形式上的 seed 结果。
