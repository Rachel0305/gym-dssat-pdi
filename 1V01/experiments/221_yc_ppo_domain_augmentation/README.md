# 221YCA YC PPO domain augmentation

这是 YC/YCA 单站点、单因素 training-domain diversity 实验。

- A：冻结的 `055_00_yca_lowIC_expanded_action_maskableppo`；不重跑、不覆盖。
- B：新建 `221YCA`，仅使用 YC 10 个训练源年 × 6 个完整天气情景的平衡采样。
- PPO、raw observation、16-action grid、040_36/042_10 reward 与 safety、100K budget、validation 年份和基线均固定。
- IC 固定 lowIC；不混合 originIC。cultivar 固定 ZD0985；不做 genotype randomization。
- 执行顺序：dry-run → 2K smoke → 单 seed 100K formal；无改善时停止，不自动做 3 seed。

训练引擎输出保留在 `benchmark_results/221YCA_yc_ppo_domain_augmentation*`，本目录的
`results/` 保存汇总、采样覆盖和图表，`snapshots/` 保存轻量 provenance，不复制大模型文件。
