# HLA WGEN PPO 100K 八种子 GitHub 备份

日期：2026-10-09。备份范围固定为 053 HLA GPCC_RAW 情景的 PPO seed 0–7 训练和 2014–2023 历史天气验证。每个 seed 的精确 100,000 步 checkpoint 用于验证；实际训练结束步数为 100,080。八份训练归档审计均为 `PASS_100K_ARCHIVE_ONLY`。

## 结果

八 seed 十年平均产量的均值 ± seed 间 SD 为 6,308.42 ± 579.49 kg/ha，平均灌溉 154.69 ± 91.93 mm，施氮 191.50 ± 73.65 kg/ha，WP_ET 为 1.40 ± 0.05 kg/m³。相同十年冻结专家基线为产量 6,797 kg/ha、灌溉 266 mm、施氮 297 kg/ha、WP_ET 1.48 kg/m³。PPO 投入较低，同时产量和 WP_ET 也较低；8 seed 管理行为分化明显，不宣称天气增强带来稳定的全面优势。详细逐 seed 结果和中断续跑记录见 `docs/hla_wgen_training_053_record.md`。

## 备份内容

- 053 源脚本、prompt、实验记录、GPCC_RAW 气候拟合输入和 `CNHL.CLI`。
- 八 seed 的训练配置、审计、episode manifest、资源记录、25K/50K/75K/100K checkpoint、实际终点模型、**实际使用的逐 episode 随机天气 CSV**及轻量 runtime seed 证据。
- 天气池的 80 个训练和 20 个留出天气候选及筛查结果。
- 80 个年度验证 `Summary.OUT` 与 metadata，八套单 seed 图表、整组图表及五情景指标表。
- `github_sha256_manifest.csv` 对备份文件提供 SHA-256；天气逐 episode 哈希还可与训练 `episode_manifest.csv` 逐条核对。

不加入重复的每 seed 完整计划调度 JSON、临时 runtime 缓存、DSSAT 完整快照和其他站点的结果。完整运行快照仍在本地 `results/hla_wgen_8seed_053/validation/`。GitHub 文件对 053 禁用文本换行符转换，以便检出后核对字节哈希。

## 解释边界

2013 年 HLA 降水缺口使用 GPCC_RAW 建模情景，不视为气象站原始观测。原 native FIELD 坐标问题未作源码级修复，应随结果记录；2014–2023 验证使用冻结历史天气。八个 PPO seed 使用相同天气 episode 调度，反映训练初始化与优化随机性，不是八套独立天气采样。
