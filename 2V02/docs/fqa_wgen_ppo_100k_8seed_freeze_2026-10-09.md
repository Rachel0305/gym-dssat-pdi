# FQA WGEN PPO 100K 八种子结果冻结记录

冻结日期：2026-10-09

Git 标签：`fqa-wgen-ppo-100k-8seed-2026-10-09`
原始结果提交：`8c63a9ccb70e8c8907c907a429ee1c2716e87975`

## 冻结范围

本标签固定 FQA PPO seed 0–7 的精确 100,000 步策略，以及 2014–2023 十年确定性验证结果。seed0 复用 051 正式训练归档；seed1–7 来自 052 顺序训练。八个 seed 的训练天气 SHA-256 序列逐 episode 相同，训练归档和最终完整性门槛均通过。

冻结结果包括每个 seed 的 25K、50K、75K、100K checkpoint 和实际 100,080 步最终模型，共 40 个小型 ZIP（总计约 6.88 MiB）；实际使用的 7,648 个逐 episode 天气 CSV；80 个年度验证 `Summary.OUT` 和对应 metadata；184 张单 seed 图、2 张 cohort 图、逐年与总体表格、审计、脚本、prompt 与试验记录。`github_sha256_manifest.csv` 列出 GitHub 备份文件及 SHA-256。完整 DSSAT runtime snapshots 与 cache 仍留在本地实验目录。

机器可读的冻结索引为 `results/fqa_multiyear_wgen_ppo_052_8seed/freeze_backup_2026-10-09.json`。其中记录每个 checkpoint 的审计哈希，以及最终门槛和关键 cohort CSV 的文件哈希。40 个模型 ZIP 已逐文件重算 SHA-256，并与训练归档中的记录匹配。

仓库 `.gitattributes` 对 052 天气、表格及验证输出关闭换行符转换，保证从 GitHub 检出后仍可按归档 SHA-256 核对原始字节。

## 固定结论

八 seed 十年平均产量为 7,234.3 ± 86.6 kg/ha，灌溉为 110.6 ± 90.6 mm，施氮为 145.0 ± 82.6 kg/ha，WP_ET 为 2.140 ± 0.082 kg/m³。平均产量高于 Null、DSSAT auto + external N，略高于 recorded farmer template，低于 official expert。低投入 seed 和频繁小灌的高投入 seed 并存；不将本结果写成稳定、全面的水氮管理优势。所有均值含 FQ 2018 年；该年 PPO 与冻结基线各情景 grain yield 均为 0，原天气输入按实验合同保留。

详细逐 seed 指标、失败尝试和口径见 `docs/fqa_multiyear_wgen_ppo_052_8seed_record.md`。冻结后若修订方法、数据或解释，应在新目录和新标签下另存版本，保留本标签对应文件不变。

## 固定路径

- 完整性门槛：`results/fqa_multiyear_wgen_ppo_052_8seed/final_gate.json`
- 精确 seed0 checkpoint 与天气：`results/fqa_multiyear_wgen_ppo_051/attempt_01/`
- seed1–7 模型、天气与审计：`results/fqa_multiyear_wgen_ppo_052_8seed/seed_XX/attempt_01/`
- 单 seed 五情景图：`results/fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ/best_seed_seedX/figures/`
- 八 seed 汇总：`results/fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ/cohort_8seed_summary/`
- 逐年 DSSAT 原始快照（本地保存）：`results/fqa_multiyear_wgen_ppo_052_8seed/validation/`
- GitHub SHA-256 清单：`results/fqa_multiyear_wgen_ppo_052_8seed/github_sha256_manifest.csv`
