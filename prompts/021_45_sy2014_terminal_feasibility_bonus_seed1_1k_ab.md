# 021_45 SY2014 终端可行性 bonus：seed1 1K 单变量 A/B

## 背景

021_43 证明当前奖励 `max(0, Y-Y_null)-I-5N` 在 online seed1/2 上主动偏好低于官方 expert 产量门槛的 N150 策略。021_44 仅离线重排，证明加入由最大季节资源成本推导的终端常数 bonus：

`B = 1×I_budget + 5×N_budget = 1×120 + 5×300 = 1620`

可使三个 seed 的最高分候选均转向官方 expert 产量门槛可行域。本任务只验证该候选奖励在真实在线训练中的边际作用。

## 单变量与冻结项

- Control：直接复用 021_42 online seed1 的 1K 结果，不重跑。
- Treatment：从与 021_42 相同的 021_34 冻结网络出发，online action seed=1、sample seed=21036；仅在终止时，若最终产量不低于本站点年份官方 expert 产量，给基础奖励加 1620。
- 保持不变：IC=2、DSSAT 输入、动作空间、I120/N300 预算、7 天间隔、观测标准化、replay、采样、exploration、网络、优化器、demo n-step 屏蔽、agent TD1/TDn、demo margin、L2、检查点。
- 不修改旧奖励实现；使用项目内新增外层 adapter。

## 阈值注册与失败保护

- 阈值必须从站点—年份注册表读取；SY2014 映射为 11077 kg/ha。
- 注册项必须指向 021_18 的 `official_extension_expert` 基准行，并在运行前核对源 CSV 数值。
- 缺少站点—年份映射、来源文件或来源数值不一致时立即报错；禁止默认复用 SY2014 数值。

## 训练前离线单元测试

必须先通过：

1. 终产量 11076：bonus=0；
2. 终产量 11077：bonus=1620；
3. 终产量 11078：bonus=1620；
4. 非终止步：bonus=0；
5. 1620 必须由 `1×120+5×300` 计算得到；
6. 同在可行域时，N200 与 N300 的奖励差仍为 500；
7. SY2014 阈值与来源 CSV 的官方 expert 数值一致；
8. 未注册站点年份必须失败。

任何一项失败都不得启动 DSSAT/训练。

## 运行规模

- 仅 SY2014 online seed1；
- 1K steps；
- checkpoints：250/500/750/1000；
- 不自动运行 seed2；不自动扩大到 5K。

## 预注册判据

Treatment 进入分支 A 必须同时满足：

- 4 个在线 checkpoint 中至少 3 个通过 expert efficiency gate；
- 1000-step checkpoint 通过；
- 4 个在线 checkpoint 最低产量 ≥ recorded 9613 kg/ha；
- 通过数严格多于复用的 seed1 Control（Control 为 3/4，但 1000-step 失败）。

否则：有部分改善为 B；没有改善为 C。不得事后放宽。

## 必须保存

- 单元测试 JSON；
- 训练 interactions、update log；
- 每个训练季的终产量、基础累计奖励、bonus、候选累计奖励、I/N；
- 5 个 checkpoint 的确定性日值和汇总；
- Control/Treatment 对比 CSV；
- 产量—累计奖励散点图（标出 11077 门槛）及 checkpoint 轨迹图，PNG+SVG；
- validation、summary、中文实验记录。

## 解释边界

- 硬门槛存在不连续性；本轮只能观察 1K seed1 是否出现明显异常，不能证明其长期稳定。
- 若训练季没有落在门槛附近，不能声称已排除门槛附近 Q 值震荡。
- `+1620` 不改变可行域内部资源成本的绝对分差；它可能改变训练动力学，但不能在实验前预设方向。

