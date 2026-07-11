# 020_01 五站现有 DQN checkpoint 多目标 Pareto 筛选

## 任务性质

这是已有证据的统一评价，不是重新进行优化空间审计，也不是新训练。

执行前必须复核：

- `019_01` 已完成五站证据与缺口审计；
- `018_10` 已确认 HLA/YC/FQ 正式 seed 结果的来源映射；
- `018_06` 已确认 LC seed0/seed1；
- `018_08` 已补齐 SY seed1；
- `019_08` 已决定淋洗惩罚不加入正式 reward；
- `019_10` 已定义 `WP_ET/PFP_N/NUtE/NLCM` 并验证 DSSAT 原生指标。

因此，本任务不得重复训练、修改 IC、修改 reward 或重新测试 leaching cost。

## 目的

读取五站代表年份两个 seed 已保存的全部正式 checkpoint，重新提取：

- `HWAM`；
- `WP_ET = YPEM × 0.1`；
- `IWP_gross = YPIM × 0.1`；
- `PFP_N = YPNAM`（仅 NICM>0）；
- `NUtE = YPNUM`；
- `IRCM/NICM/NUCM/NLCM`；
- 现有训练 reward。

在不构造新的主观加权总分的前提下，筛选 checkpoint Pareto 前沿。

## 正式 checkpoint 来源

- HLA2010：`015_12` seed0/seed1 50K；
- YC2014：`015_04` seed0 与 `015_05` seed1；
- FQ2016：`015_14` seed0/seed1 50K；
- SY2014：`017_08` seed0 与 `018_08` seed1；
- LC2010：`017_12` seed0/seed1 5K smoke。

不得把其他历史训练支线混入本轮正式筛选。

## Pareto 判据

核心Pareto目标：

```text
maximize HWAM
maximize WP_ET
minimize irrigation
minimize applied N
minimize N leaching
```

`NUtE/PFP_N/IWP_gross`作为解释指标报告，不直接进入Pareto支配判定：

- `PFP_N/IWP_gross`在零投入时为NA；
- `NUtE`高值可能来自低吸氮和氮胁迫；
- 将它们直接纳入支配判定会造成指标缺失或误判。

## 基线比较

每个checkpoint分别与DSSAT auto和官方推广expert比较：

- 产量容差：1 kg/ha；
- `WP_ET`容差：0.01 kg/m3（对应DSSAT原生显示精度）；
- 水、氮、淋洗不得增加。

输出严格占优、产量-WP达标和近似高效三类标记，但不把NA指标强行判胜。

## 输出

- `DSSAT_auto_validation/five_site_checkpoint_pareto_020_01/020_01_all_checkpoint_native_metrics.csv`
- `DSSAT_auto_validation/five_site_checkpoint_pareto_020_01/020_01_pareto_front.csv`
- `DSSAT_auto_validation/five_site_checkpoint_pareto_020_01/020_01_checkpoint_baseline_flags.csv`
- `DSSAT_auto_validation/five_site_checkpoint_pareto_020_01/020_01_site_summary.csv`
- `DSSAT_auto_validation/five_site_checkpoint_pareto_020_01/020_01_source_match_audit.csv`
- `docs/2026-07-10_020_01_five_site_checkpoint_pareto_screen_record.md`
