# 033_00 初始土壤水分敏感性与 WSPD 审计 prompt

## 背景

近期五情景日过程图和 032_22 PPO 日值输出显示，多个站点年份的 `water_stress_index_wspd` 很低，甚至长期为 0。为了判断这是否与 DSSAT 初始土壤水分 `SH2O` 设置偏湿有关，需要做一个小规模、非训练、非调参的机制审计。

## 目标

只改变 DSSAT `*INITIAL CONDITIONS` 中的初始土壤水分 `SH2O`，观察水分胁迫指数 `WSPD` 是否随初始水分降低而升高。

## 实验设计

选择两个代表案例：

1. `LCA 2019`
   - 理由：032_24 五情景图中 null/PPO 的 WSPD 均为 0，是“水分胁迫低”的直接代表。
2. `HLA 2015`
   - 理由：032_22 中 HLA 站点 PPO 输出普遍 WSPD 为 0，适合作为北方站点代表。

每个案例只跑 null/no-op 管理，不训练模型，不使用 PPO/DQN。

追加检查：如果派生模板中 `*TREATMENTS` 的 `IC` 因子为 0，则 `*INITIAL CONDITIONS` 表不会真正参与处理。为避免误判，本任务同时保留两条分支：

- `current_ic_factor`：保持当前渲染模板的 IC 因子不变；
- `force_ic1`：只在派生模板中把 treatment 的 IC 因子设为 1，用于验证 SH2O 本身是否能触发 WSPD 响应。

初始水分设定使用统一公式：

```text
SH2O = SLLL + f × (SDUL - SLLL)
```

测试三个 `f`：

- `0.55`：当前主实验采用的中等偏干设定；
- `0.30`：偏干；
- `0.15`：明显偏干。

## 约束

- 不修改 `my_data/UFGA8201-*.jinja2` 原始模板；
- 不修改 `.SOL` 原始土壤文件；
- 只在 `benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/` 下生成派生渲染模板和输出；
- 只做 DSSAT 前向回放，不训练任何强化学习模型；
- 输出中文实验记录 MD。

## 输出

1. 派生模板中每层 `ICBL/SH2O` 设定表；
2. 每个案例、每个初始水分比例的：
   - final grain yield；
   - max/mean WSPD；
   - WSPD>0 天数；
   - max/mean NSTD；
   - rain total；
   - irrigation total；
   - nitrogen total；
3. WSPD 对初始水分比例的对照图；
4. 中文实验记录。

## 判读规则

- 如果 `current_ic_factor` 分支中降低 SH2O 没有效果，而 `force_ic1` 分支中有效果，说明主流程可能没有启用初始条件因子，需要单独审计 IC 因子链条；
- 如果 `force_ic1` 分支降低 `SH2O` 后 WSPD 明显升高，说明 SH2O 本身对 WSPD 有控制作用；
- 如果 `force_ic1` 降到 `f=0.15` 仍然 WSPD 很低，则说明低 WSPD 更可能来自该站点土壤蓄水能力、天气过程、DSSAT 水分胁迫变量定义或作物需水过程，而不是单纯初始水分偏湿。
