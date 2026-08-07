# 046_02 SYA originIC 配置化 MaskablePPO 重建记录

## 状态

- `completed`
- 输入 profile：`originIC`
- 解析后输入目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`
- SY 模板 SHA256：`20b071bc49549cbf561be3aae81caa355aa0582564db524e17d68ab6a274418a`

## 框架保持不变

- 沿用 042_15 的 binary-timing MaskablePPO、奖励和 safety constraints。
- 灌溉动作：`[0.0, 45.0]` mm；施氮动作：`[0.0, 80.0]` kg/ha。
- 训练步数：`2000`；checkpoint：`[1000, 2000]`。
- 本轮唯一实验变量是 input profile：originIC。

## recorded 模板预审计

- 模板来源：`benchmark_results/027_05/027_05_daily_values.csv`
- 模板总灌溉：`0.0` mm；总施氮：`586.0` kg/ha。
- 该项是静态站点模板复用，不是逐年真实 recorded farmer；后续表图须保留此边界。

## 规范化输出

- `evaluation/046_02_training_checkpoint_inventory.csv`（复制自引擎兼容输出 `evaluation/042_10_training_checkpoint_inventory.csv`）
- `evaluation/046_02_checkpoint_validation_summary.csv`（复制自引擎兼容输出 `evaluation/042_10_checkpoint_validation_summary.csv`）
- `evaluation/046_02_validation_summary_by_station_checkpoint.csv`（复制自引擎兼容输出 `evaluation/042_10_validation_summary_by_station_checkpoint.csv`）
- `046_02_engine_result.json`（复制自引擎兼容输出 `042_10_result.json`）
