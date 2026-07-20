# 029_00 阶段型 MaskablePPO 与 mask-aware DQN 对照：预注册与工程 smoke 记录

状态：`completed`（公平比较协议、工程 smoke、五站训练锚点、17年冻结跨年迁移、全套日值证据及 PPO/DQN 总比较均已完成）

## 背景与目标

用户要求在当前阶段型 MaskablePPO 的 DSSAT 输入、IC、观测/scaler、reward、动作、mask、阶段窗口、预算、训练/验证年份和评价规则全部保持不变的前提下，只替换为 DQN，判断是否有必要后续改用 DQN。

## 审计结果

- 当前 PPO 证据覆盖五个本地训练锚点：SY2014、HLA2010、YC2014、FQ2016、LC2010。
- 当前同站跨年汇总覆盖 17 个已筛选站点年。
- 当前统一的是阶段型 MaskablePPO 框架和核心训练超参数，不是单一模型权重跨五站。
- 标准 SB3 DQN 不原生支持本项目动态动作 mask。若把非法动作投影成合法动作，会造成动作混叠并改变实验问题，因此不可用于本次公平比较。
- PPO 专属参数和 DQN 专属参数不能机械设为相同；本次以相同环境交互步数、checkpoint、评价和选模协议作为主要公平性控制。

## 新增实现

- `prompts/029_00_stage_maskableppo_vs_maskaware_dqn_protocol.md`
- `src/mask_aware_dqn_029.py`
- `src/test_mask_aware_dqn_029.py`
- `src/smoke_sy2014_mask_aware_dqn_029_01.py`

mask-aware DQN 在三条路径使用动态 mask：

1. epsilon 随机探索仅在合法动作中抽样；
2. greedy/确定性 argmax 排除非法动作；
3. replay 保存下一状态 mask，TD target 只在下一状态合法动作中取最大 Q。

## 单元测试

命令：

```text
python src/test_mask_aware_dqn_029.py
```

结果：`passed`。

覆盖：非法动作 greedy 排除、随机探索合法性、终止转移零 bootstrap、epsilon 边界、target 同步、checkpoint 保存/加载及确定性动作一致性。

## SY2014 60步工程 smoke

正式执行环境：项目 Docker `nifty_taussig`，`sb3_contrib=2.8.0`、`stable_baselines3=2.8.0`。

命令：

```text
docker exec -w /workspace nifty_taussig python src/smoke_sy2014_mask_aware_dqn_029_01.py
```

结果目录：`benchmark_results/029_01_sy2014_mask_aware_dqn_smoke/`。

工程检查全部通过：

- 精确 60 个阶段环境步；
- 精确 10 个训练季；
- 60/60 采集动作合法；
- 环境非法动作尝试为 0；
- replay 60 条且保存当前/下一 mask；
- learning-start 时发生 1 次 optimizer update；
- 60 步发生 1 次 target sync；
- loss、梯度、Q 和评价指标均有限；
- checkpoint 恢复步数、online 参数哈希、确定性动作序列和 reward 一致。

smoke 的 checkpoint0 与 checkpoint60 结果只用于确认链路，不能用于算法优劣结论。按预注册配置，60步时刚达到 learning starts，只发生1次优化更新，因此 checkpoint60 的科学表现不具有正式比较意义。

## 失败与修复记录

1. 第一次启动等待上限误设为 1 秒，Python 在导入完成前被终端终止；无残留进程、无结果目录，不是科学失败。
2. 宿主机运行失败：`ModuleNotFoundError: sb3_contrib`。没有安装或修改系统依赖；改用已经验证依赖版本的项目 Docker 原样执行。
3. Docker 第一次同样因 1 秒等待上限被终止；无残留进程和科学输出。随后使用正常等待时间原样执行并通过。

## 后续完成情况

- 17 年冻结 DQN 同站跨年验证：已完成，见 029_03 记录；
- PPO 与 DQN 的 17 年、51 seed 配对比较：已完成，见 029_04 记录；
- DQN 五情景终值图、八联日值图、阶段措施图及日值 CSV：已完成；
- 与 028_16 同一版式体系的 PPO/DQN 对照 PPT：已完成；
- Git：仅检查工作区，本任务未获得提交或推送授权，不执行提交与推送。

## 五站训练锚点正式结果

随后按 029_00 冻结配置完成五个锚点、每点 seed0/1/2、每 seed 240 阶段步，共 15 次 DQN 正式训练。全部通过以下工程检查：精确 checkpoints、精确环境步、预期季节数、181 次 optimizer update、4 次 target hard update、零非法动作、有限 loss/Q/梯度、五个 checkpoint online 参数哈希互异。

结果目录：`benchmark_results/029_02_five_site_stage_mask_aware_dqn/`。

按与 028_13 相同的导师规则重新计算“产量、WP_ET、可比 PFP_N 至少一项严格超过四基线最大值”，训练锚点配对结果为：

|site-year|PPO winner seeds|DQN winner seeds|PPO >=2/3|DQN >=2/3|
|---|---:|---:|---|---|
|SY2014|3|3|是|是|
|HLA2010|2|0|是|否|
|YC2014|2|2|是|是|
|FQ2016|2|2|是|是|
|LC2010|1|2|否（原仅1 seed证据）|是|

当前锚点证据显示：DQN 在 SY/YC/FQ 与 PPO 的 winner-seed 数相同，在 LC 更高，在 HLA 明显更差。因此仅从锚点不能判定 DQN 总体优于 PPO；至少已经排除了“换成DQN会在所有站点一致改善”的说法。

该比较仍未包含冻结权重跨年迁移，不能据此决定后续主算法。下一步必须完成17年配对迁移，再以站点年通过率、指标差和策略合理性共同判断。

## 最终算法判断

17 年配对完成后，PPO 为 29/51 个 winner seed、10/17 个年份达到至少 2/3 seed；DQN 为 23/51、8/17。DQN 仅在 FQ2023 和 LC2010 的 winner-seed 数更高，PPO 在 6 个年份更高，其余 9 个年份持平。当前证据不支持把主算法从阶段型 MaskablePPO 全面切换为 DQN；DQN 保留为正式算法对照和 LC/FQ2023 的后续候选。
