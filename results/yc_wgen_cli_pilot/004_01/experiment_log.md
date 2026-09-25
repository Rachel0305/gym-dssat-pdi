# YC WGEN 参数方法审计与合成气候验证实验记录

## 范围与资源控制

- 仅审计 YC 单站 WGEN 参数和完整年生成路径；未运行 PPO，也未启动批量天气生成。
- 未修改 DSSAT runtime、冻存天气、冻存 `CNYC.CLI`、湿日阈值或已有脚本。
- Gate A 唯一输入为冻结的 2004-2013 天气 CSV；2014-2023 验证期未用于拟合。

## 执行记录

1. 冻结天气 SHA256 为 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`；冻存 CLI SHA256 为 `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`。
2. 核对 `build_dssat_cli.py`、既有交叉检查、DSSAT v4.8.0.24 `Weather/WGEN.for`、DSSAT Vol. 3 Appendix B 与 Richardson & Wright (1984) WGEN PAR Appendix D。
3. 新增独立审计器。首次运行因 RTOT 月总量已经聚合成浮点数，却被误当序列再次求和而报 `TypeError`；修正后复跑通过。
4. 14 参数 × 12 个月的 168 个原始值全部一致；冻结 CLI 168 个字段的序列化差异均被小数位舍入解释。未解释差异为 0。
5. WGEN PAR 源码按 `RAIN>0.00` 识别湿日，但同报告正文写 `>=0.01 inch`。冻结数据有 48 个日值位于 `0<RAIN<0.254 mm`。本轮遵守任务约束，不改湿日阈值，方法状态记录为 `PASS_WITH_NOTE`。
6. Gate B 未运行：官方 WGEN 是作物模型的逐日内部子程序；现存 YC seed 结果为 116-120 日 crop-season 片段。WeatherMan 安装位置/版本在先前工具审计中因项目边界限制未知。本轮没有以 crop-season 片段冒充全年，也没有自写 WGEN。
7. 相关回归测试共 `25 passed, 0 failed`，新增审计测试 2 项通过。开发阶段第一次类型错误已记载并修复。

## 门禁判定

- 参数方法：`PASS_WITH_NOTE`。
- 全年生成工程有效性：`NOT_ASSESSED / BLOCKED_GENERATION_PATH`。
- 合成气候有效性：`NOT_ASSESSED`；降雨、温度、辐射、变量依赖、复现性、多样性和物理 QC 均未运行。
- PPO readiness：`NO`。解除阻断后先完成一个完整日历年官方 WGEN smoke，再评估 100 年 ensemble。

## 证据文件

- `parameter_audit/wgen_parameter_audit.csv`
- `parameter_audit/parameter_independent_crosscheck.csv`
- `parameter_audit/parameter_method_summary.json`
- `validation/generation_gate.json`
- `validation/synthetic_weather_validation_summary.json`
- 中文报告：`docs/yc_wgen_parameter_audit_and_synthetic_weather_validation.md`
