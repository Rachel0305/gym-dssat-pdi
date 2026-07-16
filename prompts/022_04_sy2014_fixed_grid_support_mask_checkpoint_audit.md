# 022_04 SY2014 固定网格行为支持 mask checkpoint 审计

## 目标

零训练检验 022_03 的失败是否主要来自 DQN 对“该阶段从未在 48 组网格经验中出现的动作”产生过高 Q 外推。

## 冻结边界

- 不训练、不更新参数、不改 reward、IC、观测或 DSSAT 输入。
- 加载 022_03 season15/30/45/60 原始 checkpoint。
- Control：复用 022_03 原始阶段合法动作 mask。
- Treatment：在 Control 基础上，额外限制为 48 组网格在相应 DAP 实际出现过的动作支持集。
- 支持集必须由 `022_03_fixed_grid_transition_manifest.csv` 自动计算，不得手写、不得按成功标签筛选。
- Control 直接复用 022_03 已保存的确定性完整季节评估、阶段动作和逐日表，不重复运行 DSSAT；Treatment 对每个 checkpoint 新做一次确定性完整季节评估。

## 预运行检查

1. 四个 checkpoint 文件存在且可加载；
2. 支持集由全部 288 条 transition 得出；
3. 每个阶段同时包含成功和失败场景的经验来源；
4. 支持集为：DAP1 `{0,1,3,4}`、DAP30 `{4,5,7,8}`、DAP50 `{4,5,7,8}`、DAP65 `{1,2,4,5,7,8}`、DAP85 `{1,2,4,5}`、DAP110 `{0,1}`；
5. Treatment 的每个动作都同时满足原始阶段合法性和行为支持约束。

任一失败则停止。

## 判定

- **A_support_extrapolation_confirmed**：Treatment 至少 3/4 checkpoint 通过主判据、season60 通过，且相较各自 Control 不降低产量门槛表现；允许另立 022_05，在训练阶段从一开始应用同一支持 mask。
- **B_partial_support_effect**：至少一个 Treatment 通过主判据但不满足 A；只报告，不自动训练。
- **C_support_mask_not_sufficient**：0/4 Treatment 通过；支持外推不是充分解释，停止该分支。
- **D_implementation_failure**：checkpoint、支持集、回放或指标不一致。

不得因 Treatment 接近阈值而修改支持集或科学门槛。

## 输出

- 支持集 JSON/CSV；
- 8 次评估汇总与阶段动作、逐日表；
- Control/Treatment 对比图 PNG/SVG；
- 中文记录。
