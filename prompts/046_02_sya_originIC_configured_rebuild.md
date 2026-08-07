# 046_02 SYA originIC 配置化重建

## 目的

在不修改 042_15 PPO 的奖励、网络、动作约束、训练/验证年份划分和 checkpoint 协议的条件下，只把输入从 `lowIC` 切换为 `originIC`，重新建立可复查的四情景基线和 PPO 对照。

本任务首先验证 recorded farmer template 在 originIC 下是否仍然出现不合理的极低籽粒产量；不预设 originIC 一定会修复该问题。

## 唯一配置入口

只编辑：`configs/046_02_sya_originIC_binary_timing_ppo.json`。

- `input_profile`: 只能填 `originIC` 或 `lowIC`；脚本不接受手写的任意输入目录，避免改错路径。
- 训练步数、checkpoint、动作档位均在该文件中统一设置。
- reward 与安全约束沿用 042_15；本轮不调 reward、不调掩码。

## 顺序

1. 运行 `--dry-run`，确认打印的 `resolved_input_root` 为 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`，且 `input_family` 为 `originIC`。
2. 先重建 originIC 的 null / recorded farmer template / DSSAT auto / official expert；检查 recorded 的逐年籽粒产量、请求与实际水氮量。
3. 只有四基线均成功且 rendered input 检查通过时，运行 PPO 100K 训练与 25K/50K/75K/100K checkpoint 验证。
4. 最后运行报告脚本，统一读取本轮 baseline 和 PPO 输出，绝不混用 042_15 的 lowIC 表格。

## 判读边界

- `recorded_farmer_template` 是按站点复用的静态模板，而不是逐年真实田间记录；报告必须保留该边界。
- 本轮的主产量为 DSSAT `GRNWT/HWAM` 对应的籽粒产量；`TOPWT/CWAM` 地上部生物量仅作过程图辅助。
- 若 originIC 下 recorded 仍显著偏低，优先审计 recorded 模板的静态水肥表、管理模式和年份适配性；不能把问题直接归咎于 PPO。

## 外部 auto-N

原生 DSSAT `FERTI=A` 在 046_00 的阈值变体中仍未触发氮肥。本轮另建立一个清晰命名的对照：DSSAT 原生自动灌溉 + gym-DSSAT 外部氮胁迫触发施肥；规则固定为 `NSTRES >= 0.5` 时 25 kg/ha，7 天最小间隔、季节上限 250 kg/ha、DAP90 后禁氮。
