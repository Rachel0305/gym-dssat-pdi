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
- 运行失败与修正：首次 smoke 仅在终端 JSON 序列化遇到 NumPy int64 类型错误，数据文件已落盘；规范化 JSON 类型后复现通过。H0 首轮观察天气评估遇到 weather path 配置错误，后续重试遇到 episode 元数据字段名不一致；修正路径为 weather_clean/all_sites_weather_cleaned.csv、字段为 historical_year_context 后，以独立 attempt tag 重跑，完整保留前次日志/产物。分析重算首次因 case manifest merge key 大小写不一致退出，修正统一键名后重新生成分析、probe 与报告；没有为该修正重跑 DSSAT episodes。
- 全量回放：六模型各完成 20 个 WGEN seed + 10 个 observed 年，共 180 episode、18,408 step；逐模型串行，无训练。运行期间峰值 RSS 约 898 MB，低于 4,000 MB 守卫。
- 天气 QC：120 个 WGEN episode 的 runtime hash 全部一致；逐步天气字段覆盖 89.85%，末段缺失值保持缺失，因此 weather-response 标记 PARTIAL。
- 诊断纠正：奖励成本项按回报方向转换为节省/惩罚效应；行为标签使用 season-level totals、事件数和 no-op；WGEN 若无负向差值则选取优势最小的两个案例，不称为下降。
- 结果解释不做 best-seed 选择、不作 observed 十年以外的泛化断言。
