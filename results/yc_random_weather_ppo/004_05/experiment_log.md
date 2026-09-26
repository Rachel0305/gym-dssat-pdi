# 004_05 实验记录

- 首轮 preflight 识别 004_03 的 `training_qc_verified.json` 为旧失败尝试；正式依据是 `training_qc_verified_retry1.json`（6/6 通过），原失败文件保留未覆盖。
- seed 3 Historical 432-step smoke 本身完成、reward 分解为零误差且峰值 RSS 433 MB；首轮烟测 QC 因历史路径未填充 `filex_wther` 字段而拒绝放行。已改为直接读取实际 FileX 模板 METHODS/WTHER 行核验 M；原 smoke 工件保留。
- 10 个正式模型均完成 100080 步且 checkpoint 可加载。首轮聚合 training QC 将未记录在 episode 列的 Historical WTHER 和不同 episode 数误当失败；修正为解析实际模板并比较 schedule 共同前缀，首轮 QC JSON 另存为 `new_training_qc_initial_attempt_unverified.json`。
- 固定状态 probe 首次因旧 004_04 动作记录未覆盖全部 16 个 legal action 而在 H3 安全停止；未写出部分结果。已改为从冻结 canonical 4x4 action grid 解码，并逐项与 004_04 已观测 action 映射交叉核验。
- 复用 004_03 seed 0–2 六个 SHA 已验证 checkpoint，不重训。
- 新增正式训练仅限 Historical/Random-weather seeds 3–7；串行，每模型 100000 requested timesteps，RSS guard 6000 MB。
- 固定使用训练 weather seeds 1001–1080、evaluation WGEN seeds 1081–1100、observed years 2014–2023；reward、PPO、mask、runtime、CLI 和 WGEN fit 未改。
- 首轮 analysis component audit 发现新模型 step capture 读取了不存在的 info key，四个分量误记为零。首轮 summary、step/episode tables、8 张图及报告已复制到 `backups/yc_random_weather_004_05_initial_unverified_components/`；evaluation QC 在归档前已用已有 trace 重验，现含 component check。修正版从 frozen canonical episode formula 重建 aggregate components，stress-relief 标为 reward identity residual，step-level 原始分量标为 unavailable。未重训或重跑 DSSAT evaluation。
- 复用 004_04 seed 404006 的 600 个 probe states 与原 masks；按 probe 行为先分类，rollout 管理行为只作交叉核验。
- 失败/不完整 attempts 保留原目录，不覆盖；没有因结果更换 seed 或排除模型。

- Archetype finding：`POSSIBLE_SHIFT`；paired performance：`NO_POSITIVE_SIGNAL`；success `0/8`。
- Reward component aggregate reconstruction 最大误差：`2.220446049250313e-16`；stress-relief 为恒等式 residual，不解释为独立观测的原始分量。
- Git push：`NO`；要求的本地 commit subject：`experiment: expand YC weather augmentation to eight PPO seeds`。