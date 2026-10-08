# FQA/HLA WGEN 训练逐 episode 天气归档合同

本合同用于未来新建的 WGEN PPO runner；043 只以隔离零动作 episode 验证了捕获和落盘，**尚未接入正式训练入口**。

1. 每个 episode 启动前，冻结站点、训练/留出分组、年份、处理、PPO seed、weather seed、CLI 与渲染 FileX 的 SHA-256、PDI 二进制/包身份及进程/episode 序号。weather seed 不是 realization 的唯一身份。
2. 沿 YC 既有路线在 PDI socket 建立前注入 `rseed1_`，运行时捕获每日 state 的 `RAIN/SRAD/TMAX/TMIN` 和实际日期；从**本次实际运行**写出规范化 `DATE,DOY,RAIN,SRAD,TMAX,TMIN` CSV。禁止用预先生成的另一条序列或输入 WTH 代替。
3. episode 结束后、开始下一个 episode 前，以只创建模式写入独立文件，计算磁盘文件 SHA-256，并记录行数、首末日期、四变量完整性与物理筛查、运行时 CLI/FileX/rseed1、DSSAT 日志路径。归档未写成或复读哈希不一致时停止本 runner；保留失败文件和日志，不吞掉失败继续训练。
4. manifest 一行对应一个**实际 episode realization**，主键含站点、run_id、PPO seed、episode 序号；同一个年份×weather seed 若在不同进程或缓存上下文产生不同哈希，两条都保留并标记 `SAME_SEED_DIFFERENT_REALIZATION`，不得覆盖或“修正”为期望哈希。
5. 每个正式训练/留出 episode 都应有归档行，归档行数与运行 episode 数闭合。训练 weather pool `1001–1080` 与留出 `1081–1100` 沿 YC contract 分开，PPO seed 与 weather seed 分开；归档和 QA 通过后才解释政策结果。
6. 即使未来另有提前生成的 80/20 候选池，仍须保存训练与评估中**实际使用**的逐日天气并逐 episode 核对。043 的五个种子只证明小样本运行和归档链，不代表完整池或气候分布通过。

043 的实现参照：[capture_one.py](capture_one.py) 在 `_get_state` 捕获 daily state、在每次运行目录写出 CSV 和 SHA-256；[evaluate_archive.py](evaluate_archive.py) 复读验证每一份 archive。未来 PPO 入口须把这两个动作嵌入自己的 episode 生命周期，并在正式训练前通过至少一个训练与一个留出 episode 的归档 smoke。
