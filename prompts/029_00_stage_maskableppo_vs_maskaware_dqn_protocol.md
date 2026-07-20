# 029_00 阶段型 MaskablePPO 与 mask-aware DQN 公平对照协议

## 1. 任务目标

在不改变 DSSAT 输入、IC、观测、scaler、奖励、动作表、阶段窗口、季节预算、训练/验证年份、随机种子、环境交互步数、checkpoint 位置和选模规则的前提下，只替换强化学习算法，比较当前阶段型 MaskablePPO 与新增的 mask-aware DQN。

本任务回答：在当前已冻结问题定义下，DQN 是否能在训练锚点、跨 seed 稳定性和同站跨年迁移三个层面优于 MaskablePPO。

不允许用历史日尺度 DQN 结果替代本对照；不允许重复训练已有 PPO；不允许为 DQN 逐站、逐年或看结果后调参。

## 2. 已冻结的共同条件

- 五个训练锚点：SY2014、HLA2010、YC2014、FQ2016、LC2010。
- 同站跨年验证范围：完全复用 `028_13_site_year_overview.csv` 的 17 个已筛选站点年。
- 每个训练锚点 seed：0、1、2。
- 环境交互预算：每 seed 240 个阶段决策步。
- checkpoints：0、60、120、180、240。
- 网络隐藏层：`[32, 32]`。
- learning rate：`3e-4`。
- gamma：`1.0`。
- 观测及 scaler：复用对应 PPO 训练锚点已经冻结的站点专属实现与文件。
- 动作：9 个组合，`I in {0,15,30} mm` x `N in {0,50,100} kg/ha`。
- 动作掩码、阶段 DAP、季节预算 `I<=120 mm, N<=300 kg/ha`、`DAP>90` 禁氮：与对应 PPO 环境逐位一致。
- reward：逐站复用 PPO 的本地 null、auto、official expert 阈值和相同公式，不加入新项。
- deterministic evaluation：每个 checkpoint 运行一个完整季节。
- checkpoint selection：只按完整季节总 reward 最大选择；精确并列取更早 checkpoint；不得按产量、WP_ET、PFP_N 或导师规则重选。
- 评价规则：使用与 028_13 相同的四基线、全精度终值和“产量、WP_ET、可比 PFP_N 至少一项严格超过四基线最大值”导师规则；同时独立报告另外两项差距，不把“存在候选”写成“跨 seed 稳定”。

## 3. 不能伪装成相同的算法专属参数

PPO 的 GAE、rollout epochs、clip range 与 DQN 的 replay、target network、epsilon 没有一一对应关系。公平性主轴是相同环境交互次数和相同评估/选模规则，而不是把无意义的参数名强行设成相同。

本轮 DQN 专属参数必须一次预注册，五站共用：

- 自定义 PyTorch mask-aware DQN，不修改 `site-packages`。
- replay capacity：10000 transitions。
- learning starts：60 个环境步。
- batch size：30。
- train frequency：每个环境步一次。
- gradient steps：1。
- target network hard update：每 60 个全局环境步一次。
- epsilon：从 1.0 线性降到 0.05，衰减区间为总预算的 70%，之后保持 0.05。
- loss：Smooth L1 / Huber。
- optimizer：Adam。
- gradient norm clipping：10。
- double DQN：关闭；本轮比较标准单 target-network DQN，不在结果后切换算法变体。

这些参数不得在看到正式训练结果后修改。若 smoke 暴露工程错误，修复必须记录并重新从 smoke 开始；若只是科学结果不佳，不得现场调参。

## 4. mask-aware DQN 的硬性正确性要求

标准 SB3 DQN 不原生支持本项目动态动作掩码。本任务禁止把非法动作投影/裁剪成合法动作，因为会造成动作混叠并改变 MDP。

DQN 必须在以下三处使用完全相同的 mask：

1. epsilon 随机探索：只从当前合法动作均匀抽样；
2. greedy/确定性选择：非法动作 Q 值在 argmax 前置为负无穷；
3. TD target：下一状态最大 Q 只能在下一状态合法动作中计算。

Replay transition 必须显式保存 `mask` 和 `next_mask`；终止转移的 bootstrap 必须为零。

## 5. 先单元测试和 smoke，后正式训练

### 5.1 纯离线单元测试

- 随机探索从不选非法动作；
- greedy 从不选非法动作，即使非法动作原始 Q 最大；
- TD target 忽略非法动作；
- terminal target 不 bootstrap；
- epsilon 在 0/60/120/168/240 步与预注册曲线一致；
- target hash 只在 60 的整数倍更新；
- checkpoint 保存/加载后，相同观测和 mask 的 greedy 动作逐位一致；
- 固定 seed 的 toy 环境重复运行得到相同 loss 和模型 hash。

### 5.2 SY2014 工程 smoke

只运行 seed0、最多 60 个阶段步。检查：

- 所有动作合法；
- replay 中当前/下一 mask 与环境日志一致；
- episode、资源和 reward 记账闭合；
- 模型参数、loss、Q 值有限；
- 60 步 checkpoint 可保存、加载和确定性复评估；
- 源输入哈希不变。

smoke 只判工程正确性，不判 DQN 科学优劣。

## 6. 正式训练和停止线

只有 5.1 与 5.2 全部通过才进入正式训练。

为节省算力，严格串行执行五站；每站先 seed0，再 seed1/2。任何工程失败停止该站。科学失败不修改参数，仍保留结果用于算法比较。

正式训练后冻结每个 seed 的预注册选中 checkpoint，并在与 PPO 完全相同的同站筛选年份上做零训练迁移。不得在验证年重新训练、重选训练年 checkpoint 或使用验证年调参。

## 7. 主要比较指标

逐站点年、逐 seed 同时输出：

- yield、WP_ET、PFP_N、I、N；
- 相对四基线最大值的绝对和百分比差；
- 导师 any-metric 规则是否通过；
- 训练锚点选中 checkpoint 与轨迹持续性；
- 同站跨年固定迁移通过率；
- 非法动作数、预算违规数、DAP>90 施氮数；
- 环境交互步数、optimizer update 数和 DSSAT 季节数。

算法层面预先定义：

- “DQN 优于 PPO”只有在同一站点年、同一 seed/选择协议下，DQN 的导师规则通过率或三项指标的预注册比较占优时才成立；
- 若一方提高某一指标但降低另一指标，报告 Pareto 权衡，不强行判单一赢家；
- 总体比较使用站点年配对表，不用各自挑出的代表图代替全体结果。

## 8. 输出与记录

- 新代码仅放 `src/`，新结果仅放 `benchmark_results/029_*`。
- 生成 unit/smoke CSV、checkpoint 曲线、逐站点年配对 CSV、PNG/SVG、Markdown 记录和导师 PPT 补充页。
- 记录所有失败尝试、错误、修复和停止分支。
- 不覆盖 026/027/028 结果，不删除历史 DQN，不修改 reward/IC/DSSAT 输入。
- 不提交模型 zip、DSSAT runtime、缓存和临时文件。
- Git 提交/推送必须另行检查工作区并取得用户明确授权；本 prompt 本身不授权 push。

## 9. 状态术语

- `engineering_pass`：mask、replay、target、checkpoint 和记账全部正确。
- `anchor_candidate`：训练锚点某 seed 的预注册选中 checkpoint 通过导师规则。
- `cross_seed_initially_stable`：同一训练锚点至少 2/3 seed 通过。
- `fixed_crossyear_transfer_pass`：冻结训练年权重在验证年通过；验证年零训练。
- `algorithm_winner_not_identified`：指标存在权衡或配对证据不足，不得硬判赢家。
