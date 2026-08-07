# 046_03 SYA originIC 四情景基线重建

## 目的

使用与 046_02 完全相同的 `originIC` 输入目录和 2014--2023 验证年份，重建：

1. null；
2. recorded farmer template；
3. DSSAT native auto；
4. official extension expert。

本轮不训练 PPO。先确认 recorded 基线在 originIC 下的籽粒产量、实际水氮量和 rendered 输入管理模式，再决定是否值得运行 originIC PPO。

## 输入控制

唯一配置文件：`configs/046_02_sya_originIC_binary_timing_ppo.json`。

脚本只接受 `originIC` / `lowIC` 两个受控 profile，并将解析后的目录、SY 模板 SHA256、静态 recorded 模板总水氮写入 manifest。

## 成功条件

- 10 个年份 × 4 情景全部运行成功；
- 每一条 summary 都包含 `grain_yield_kg_ha`、`biomass_kg_ha`、实际水氮、WP_ET、PFP_N；
- 结果明确记录 recorded farmer template 是静态模板复用，不当作逐年真实农户记录。

