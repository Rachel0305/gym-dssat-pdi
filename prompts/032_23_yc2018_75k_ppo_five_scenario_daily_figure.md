# 032_23 YC2018 75K PPO 五情景日过程图

## 目的

为 032_22 五站点 100K 初筛结果中的 YC 站点生成一张可汇报的五情景日过程图，优先选择当前表现最好的 YC checkpoint（seed0, 75K）和代表性验证年份 YC2018。

## 输入

- PPO 候选：
  - 模型：`benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/models/YCA/YCA_half_split_stress_aware_maskableppo_seed0_ckpt75000.zip`
  - 年份：YC2018
  - 只做冻结确定性评估，不训练。
- 四情景基线：
  - null / recorded_farmer / official_extension_expert：来自 `benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/YCA/2018/`
  - dssat_auto：来自 `benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/YCA/2018/dssat_auto`

## 执行要求

1. 先对 YC2018 seed0 checkpoint 75K PPO 做一次确定性评估，并在环境关闭前复制 DSSAT 临时目录，保存为 PPO candidate snapshot。
2. 复用 027_05 五情景日过程图样式，输出：
   - 五情景 daily CSV；
   - 五情景 summary CSV；
   - daily evidence checks；
   - PNG/SVG 图；
   - 中文实验记录 MD。
3. 不得重新训练，不得修改 reward，不得替换模型。
4. 图中必须包含天气、土壤水、WSPD/NSTD、灌溉施肥、grain/biomass、累计统一 reward。
5. 如果任何情景缺少必需 DSSAT 输出文件，停止并记录，不得编造。

## 预期结论边界

本任务只回答“YC2018 当前 75K PPO 候选的日过程和五情景对照是否可视化、措施是否看起来合理”，不声称 YC 全站点或所有年份稳定成功。
