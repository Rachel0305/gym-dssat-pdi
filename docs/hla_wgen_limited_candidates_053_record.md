# 053 HLA 五方案限量 WGEN 候选记录

日期：2026-10-09。范围：HLA 2007 年、WGEN seed101，每条 029 CLI 情景各一个作物季候选；未运行 PPO。

使用 `results/hla_wgen_8seed_053/run_limited_candidate.py` 逐情景单进程运行，复用 032 已验证的 HLA 隔离 FileX 与天气捕获方法。脚本在运行前校验每份 CLI 的冻结 SHA-256，不覆盖既有输出。每条情景保存输入 provenance、完整运行结果、实际逐日天气 CSV、日志及原生运行快照。机器审计与每条摘要见 `results/hla_wgen_8seed_053/limited_candidate_audit.json`；复核脚本为 `audit_limited_candidates.py`。

| 拟合情景 | 捕获日数 | 季节降水 (mm) | 日最大降水 (mm) | 逐日天气 SHA-256 前 12 位 |
| --- | ---: | ---: | ---: | --- |
| GPCC_RAW | 135 | 635.44 | 50.94 | E8B3D1BDA90E |
| CPC_RAW | 133 | 650.75 | 50.66 | 30DF3E5A292A |
| GPCC_BIASCORR | 135 | 634.93 | 50.97 | 61C24DC62F57 |
| CPC_BIASCORR | 133 | 650.74 | 50.64 | 93DA1BA83ED9 |
| ENSEMBLE_BIASCORR | 135 | 635.37 | 50.57 | B2122E2E92D3 |

五条均捕获完整 RAIN/SRAD/TMAX/TMIN，运行 seed101 与 CLI 哈希一致，天气文件哈希与审计一致，基本物理筛查通过；进程树峰值 RSS 约 125–126 MiB。候选长度差异来自作物季终点随天气变化，不能直接把 133 日与 135 日降水总量当作相同日窗的气候比较。

**结论：限量单季捕获通过，正式训练未放行。** 还需要多年份和多 seed 候选池的分布与极端值质量检查；033/034/037 的 native FIELD 坐标转写及物理影响仍未解决。此前用户选择五方案限量候选，本轮未从五份中选择唯一正式 PPO 训练输入。
