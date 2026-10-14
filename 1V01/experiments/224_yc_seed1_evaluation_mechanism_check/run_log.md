# YC seed1 evaluation mechanism check run log

日期：2026-09-11

## Scope

- 只读诊断；没有训练、没有修改 PPO/reward/action grid/wrapper/mask/data，也没有覆盖正式评估结果。
- 使用 Original seed1 @100K 的 2014–2023 daily trace 构造 shared observations，并在相同 observation 与 action mask 上比较 Original/Augmented seed1 @100K。
- 使用容器内项目解释器运行，避免污染宿主环境；没有启动长训练。

## Execution record

1. `py_compile`：通过。
2. 初次执行发现评估模块的实际导入名与预期别名不同，改为项目中存在的 `run_055_00_yca_lowIC_expanded_action_maskableppo`，未修改该模块。
3. 初次 replay 发现 `itertuples()` 结果不能按字典键访问，改为属性访问，未改变数据或评估逻辑。
4. 修正 raw DAP 与生产 wrapper `_dap()` 的语义对齐，并重新执行。
5. 最终状态：`completed_read_only`；shared states=1002；probability rows=32064；全部 integrity checks 通过。

## Reproducibility

运行脚本：

```text
docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python /workspace/experiments/224_yc_seed1_evaluation_mechanism_check/scripts/run_check.py
```

核心输出、模型/source SHA256 和完整性检查结果记录在：
`yc_seed1_evaluation_mechanism_check_manifest.json`。
