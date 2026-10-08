# 042 封丘 WGEN 单季运行与随机天气归档门禁

仅针对 FQA，沿 YC 已完成实验的实际运行链：隔离 FileX 副本将处理 1 的 `METHODS.WTHER` 设为 `W`，传入 041 的 `CNFQ.CLI`，在 PDI 启动前注入固定 `rseed1_`。运行 2007 年一个作物季、零动作、单进程，最多 240 日、120 秒、进程树 RSS 768 MB；同配置新进程重复一次。禁止 PPO、修改旧 FileX/WTH、扩成 80/20 pool 或启动 8 seeds。

必须把**实际运行时**每日 `DATE,RAIN,SRAD,TMAX,TMIN` 全部保存为 CSV，并记录规范化字节 SHA-256、CLI/FileX/土壤/品种哈希、年份、处理、weather seed、PDI 配置及运行快照。对两个新进程逐日比较，不以 seed 编号或“运行成功”代替天气复现证据。如果程序只输出 seed 而不能捕获逐日天气，本门禁失败。

检查运行目录确实读取 `CNFQ.CLI`、FileX 确实为 `WTHER=W`、`rseed1_` 与请求一致、四变量完整且物理筛查通过，并检查原生 `FIELD` 坐标及警告。运行成功与坐标有效分别判定；坐标缺失时不得宣称正式天气池和 PPO 已放行。

向后续 FQA 和 HLA 天气池施加同一归档规则：每个用于训练或留出的 realization 都要保存完整逐日天气及实际 runtime 哈希，并记录年份、CLI、FileX、weather seed、进程/episode 上下文。重现核查以逐日值或规范化哈希为准；同一 seed 在不同上下文产生不同天气时保留两份并显式标记。训练前先通过这项存档门禁，不得重演 YC 早期未保存天气的问题。

产物放入 `results/fqa_runtime_wgen_smoke_042/`，报告放入 `docs/fqa_runtime_wgen_smoke_042.md`。原始文件和历史结果只读。
