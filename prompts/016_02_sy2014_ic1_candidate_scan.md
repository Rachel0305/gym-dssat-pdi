# 016_02 沈阳站 2014 专属 IC=1 候选扫描

目标：

1. 在不改原始输入包的前提下，为沈阳站 2014 反推一个“可用的专属 IC=1 初始条件候选”；
2. 只做低成本 no-op 前向模拟，不训练 PPO/DQN；
3. 扫描 9 个组合：
   - 水分：把当前 SH2O 向该层 DUL 推近，3 档
   - 氮：当前 SNH4/SNO3 同步缩放，3 档
4. 输出每个组合的最终 GRNWT/TOPWT、最大 SWFAC/NSTRES、热图、CSV 和中文实验记录。

执行要求：

- 必须使用容器 `b2fd6726c8c1`
- 必须使用 `/opt/gym_dssat_pdi/bin/python`
- 不覆盖旧结果
- 中文记录到 `docs/2026-07-02_016_02_sy2014_ic1_candidate_scan_record.md`
