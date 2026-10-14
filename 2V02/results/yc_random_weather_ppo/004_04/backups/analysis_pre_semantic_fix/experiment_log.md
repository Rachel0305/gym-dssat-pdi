# YC PPO 策略行为诊断实验记录

- 任务类型：evaluation-only / DIAGNOSTIC ONLY；未执行 PPO 训练、微调、调参或 seed 扩增。
- 模型：仅使用 004_03/training_verified 下 H0-H2、W0-W2 的 checkpoint_100000.zip。
- 评估集：held-out WGEN seed 1081-1100；observed YCA 2014-2023；deterministic=True。
- 资源约束：模型逐个串行评估；进程树 RSS 守卫 4000 MB；不安装依赖、不改 runtime。
- QC 烟测：W0 / seed 1081 的 runtime weather SHA256、reward、yield、N、I、episode days 与 004_03 一致。
- 过程修正：初次 smoke 完成后只在 JSON 终端打印处遇到 NumPy int64 序列化异常；结果已先写盘且全部复现检查通过。runner 后续加入 JSON 类型规范化。
- 天气来源：rendered FileX .WTH 在 WTHER=W 模式下不等同实际随机天气；使用与 004_03 runtime hash 同口径的 daily_weather_from_states 捕获，observed 使用 YCA cleaned weather。天气变量只用于 POST_HOC_DIAGNOSTIC_ONLY 分析。
- Reward：逐步保留 canonical action_info；拆分 scaled yield、water/N cost、stress relief 与 SWFAC guardrail 并核对。旧公式标记 SUPERSEDED / NOT_APPLICABLE。
- Same-state probe：先固定并保存 trajectory union 分层抽取的 state+mask，再逐模型预测；不推进 DSSAT，不提取概率分布。
- 结果解释不做 best-seed 选择、不作 observed 十年以外的泛化断言。
