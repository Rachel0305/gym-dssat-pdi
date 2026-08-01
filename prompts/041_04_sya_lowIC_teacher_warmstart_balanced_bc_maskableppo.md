# 041_04 SYA lowIC teacher warm-start：平衡非零动作 BC 后再 MaskablePPO

## 背景

041_03 已经证明 teacher warm-start 训练流程可以运行，但 2K smoke 暴露出一个关键问题：BC 数据集中 no-op 日样本过多，确定性策略在 `bc_init/1000/2000` 基本退化为 no-op，平均灌溉和施氮均接近 0。

因此，041_04 不进入新的奖励调参，也不改变 PPO 环境，只修正 warm-start 的行为克隆采样方式。

## 固定不变

- 数据源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 年份：2014–2023
- teacher：041_02 选出的 10 条 layered teacher 轨迹
- PPO 环境、奖励函数、动作安全层、离散动作档位：沿用 040_40
- PPO fine-tune：默认 100000 steps，checkpoint 为 25000/50000/75000/100000

## 本次唯一改动

BC warm-start 从原来的顺序 mini-batch 改为平衡采样：

- 每个 batch 中尽量包含 50% no-op 样本与 50% 非零动作样本；
- 不改 teacher 标签；
- 不额外筛选 teacher 年份；
- 不调 PPO 超参数。

这样做的理由是 041_03 smoke 的数据审计显示：

- 总样本 1398；
- 非零动作 73；
- no-op 1325；
- 即使非零样本单点权重更高，总权重仍是 no-op 占优，导致 BC accuracy 0.946 不能代表关键动作学会。

## Smoke 判据

先跑 2K smoke：

```bash
cd /workspace/src
/opt/gym_dssat_pdi/bin/python run_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_041_04.py --timesteps 2000 --checkpoint-steps 1000,2000 --suffix smoke2k
```

通过条件：

1. 脚本完整运行，无 DSSAT 失败；
2. BC 日志报告 `bc_nonzero_action_accuracy`；
3. `bc_init` 不应再是全季节 no-op；
4. 若 smoke 仍然 no-op，则停止，不跑 100K。

## 正式训练命令

仅在 smoke 通过后运行：

```bash
cd /workspace/src
/opt/gym_dssat_pdi/bin/python run_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_041_04.py
```

## 边界

这是 teacher-assisted same-year feasibility 训练，不声称严格跨年泛化。
