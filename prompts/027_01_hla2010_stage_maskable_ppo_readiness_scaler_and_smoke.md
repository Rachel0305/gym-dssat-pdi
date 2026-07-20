# 027_01 HLA2010阶段型MaskablePPO输入、scaler与环境smoke

## 1. 目的与边界

在0 PPO训练条件下，为HLA站内阶段型MaskablePPO建立可复现入口。只完成输入、四基线、本地24维scaler、本地reward参数和完整季节环境smoke。

不复用SY权重、不联合训练、不启动seed0正式240步训练。

## 2. 固定输入与基线

- 训练年：HLA2010；
- 输入入口：`run_hla_five_scenario_completion_020_11.py` 的 `YEAR_INPUTS[2010]`；
- 四基线优先复用 `020_11_hla_five_scenario_summary.csv` 及对应Summary.OUT，但逐行核对source_file和年份；
- 原始观测24维：9个前置非土壤变量、8个运行时SW层、7个后置变量；
- 重要原始输入只读，运行使用新目录副本，执行前后核对SHA-256。

## 3. 六个决策阶段

HLA属于“东北及长城沿线春玉米区”，按官方推广方案固定为：

```text
DAP 1 / 30 / 50 / 65 / 85 / 110
```

DAP1是官方播种期DAP0在gym环境中的可执行替代。smoke必须确认2010全季到达六点；跳过任一点则停止，不现场移动DAP。

## 4. HLA专属固定scaler

禁止使用SY的25维scaler或补零。重新前向采集HLA2010四条管理轨迹（null、recorded、auto、official expert）在六个决策点的完整24维原始观测，预期24个状态。只用训练年，不用验证年。

- population mean/std；`std<1e-6`时scale=1并标记near_constant；
- 保存24个标签、mean、scale、near_constant；
- 非常量维标准化均值绝对值<1e-4，std偏离1<1e-4；
- 重构最大误差<2e-3；
- 采集过程不得改变四条管理轨迹。

记录中必须标为“training-year four-baseline scaler”，不得称跨站点通用scaler。

## 5. 本地reward参数

从HLA2010四基线自动读取并保存JSON，禁止硬编码SY数值：

- `local_null_yield`；
- `local_feasibility_yield=max(auto_yield, official_expert_yield)`；
- water_cost=1、nitrogen_cost=5、feasibility_bonus=1620；
- I预算120、N预算300。

recorded仅用于最终比较，不进入reward。

## 6. 新增HLA环境入口

优先复制/组合026已验证逻辑，不修改SY工作脚本：

- observation_space=(24,)；9动作和mask同026；
- 六阶段间自动no-op；晚期禁氮规则同026；
- terminal reward使用第5节本地参数；
- 终止观测为24维零向量；
- 保存请求/实际动作、预算和reward分项。

## 7. 全no-op smoke

- 六次决策全为action0；训练步数0；
- 每步观测24维且有限，no-op始终有效，无效动作0；
- 最终yield与020_11 null误差≤2 kg/ha；
- I=0、N=0，Summary.OUT匹配；
- 原始输入哈希不变。

## 8. 输出

- `benchmark_results/027_01/027_01_hla2010_four_baselines.csv`
- `benchmark_results/027_01/027_01_hla2010_scaler_source_states.csv`
- `benchmark_results/027_01/027_01_hla2010_observation_scaler.csv`
- `benchmark_results/027_01/027_01_hla2010_smoke_stage_actions.csv`
- `benchmark_results/027_01/027_01_result.json`
- 对应代码和 `docs/2026-07-17_027_01_hla2010_stage_maskable_ppo_readiness_scaler_and_smoke.md`

## 9. 分支与停止

- A：四基线、scaler、reward参数和完整季节smoke全部通过；允许另写027_02 seed0训练prompt。
- B：输入完整但阶段/scaler/smoke失败；停止并只诊断具体工程原因。
- C：输入或四基线provenance不完整；停止，不训练。

本任务不得自动进入027_02。

