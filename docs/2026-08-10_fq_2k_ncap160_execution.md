# FQ originIC：N hard cap 160 kg/ha 单因素 2K smoke 执行记录

## 结论

2K smoke 已在用户启动的 `nifty_taussig` 容器中完成，且只将继承的 `action_safety.season_n_soft_limit` 从 250 改为 160 kg/ha。输入、渲染来源、年份、seed、raw observation、无天气预报、16 动作网格、reward、PPO 超参数和灌溉安全逻辑保持不变；输出为独立目录。

但该候选在 checkpoint 2000 的十个验证年只有 **1 个跨年动作签名**（全部为 `I0/N40; I15/N0; I30/N0; I45/N0`）。这违反了预注册的“不可退化为跨年单一签名”机制闸门。因此本 smoke **不批准后续 100K**。这不是性能失败的结论，而是动作适应性不足的停止结论。

## 环境、命令与隔离输出

- 容器：`nifty_taussig`；解释器：`/opt/gym_dssat_pdi/bin/python`，Python 3.10.12。
- 工作目录：`/workspaces/gym-dssat-pdi`。
- 线程限制：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- dry-run：

```bash
docker exec nifty_taussig bash -lc "cd /workspaces/gym-dssat-pdi; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/gym_dssat_pdi/bin/python src/062_fqa_originIC_ncap160_smoke/run_062_00_fqa_originIC_ncap160_smoke.py --dry-run"
```

- 2K 执行：

```bash
docker exec nifty_taussig bash -lc "cd /workspaces/gym-dssat-pdi; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/gym_dssat_pdi/bin/python src/062_fqa_originIC_ncap160_smoke/run_062_00_fqa_originIC_ncap160_smoke.py"
```

- 隔离输出：`benchmark_results/062_00_fqa_originIC_expanded_action_maskableppo_ncap160_smoke_smoke2k/`。
- manifest：`062_00_run_manifest.json`；有效运行时覆盖记录：`configs/062_00_effective_runtime_override.json`；机制闸门：`062_00_ncap160_gate.json`。
- 没有覆盖 `051_00`、`051_03` 或其他冻结结果；没有运行 100K。

## 预检与来源核验

dry-run 通过：FQA/FQ、`originIC`、输入根 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`、源 `FQ/CNFQ0801.MZX`、训练年份 2005--2013、验证年份 2014--2023 均与 051_00 一致。动作仍为 I `[0,15,30,45]` 和 N `[0,40,80,120]`，共 16 个组合；观测仍为 `046_02_raw_observation`、未归一化、无天气预报。

runner 沿用 051_00 的 `ppo_safe_rendering.MULTISITE_INPUT_ROOT` 设置，并在同一已核实 `originIC` 根下生成新的 `rendered_inputs/FQA/...`；manifest 同时记录唯一差异：`season_n_soft_limit: 250 -> 160`。日记录中的季节累计 N 最大值为 160，构成有效覆盖确已进入 safety/renderer/DSSAT 执行链路的行为证据。

## 2K checkpoint-2000 逐项闸门

| 闸门 | 结果 | 证据 |
|---|---|---|
| 单进程、最大 2K | 通过 | 只生成 checkpoint 1000 和 2000 |
| 请求--安全动作差异 | 通过 | 0 行 |
| 安全动作--DSSAT累计传输 mismatch | 通过 | 0 行 |
| 累计 N <=160 kg/ha | 通过 | 最大 160.0 kg/ha |
| 声明动作网格合法 | 通过 | 标准 action audit 10/10 年通过 |
| 非零动作对 >=3 | 通过 | 4 对：I0/N40、I15/N0、I30/N0、I45/N0 |
| DAP1 后正动作 | 通过 | 96 行 |
| 跨年不为单一签名 | **失败** | 10/10 年均为同一 4 对签名；distinct signatures=1 |
| 2018 状态 | 共同异常仍在 | 产量 0，N=160；不归因于 PPO |

最终机制 gate：`next_step_allowed=false`，唯一失败项为跨年单一动作签名。标准 051 action audit 的较宽松门槛会显示通过，但本任务采用更严格、在运行前指定的跨年适应性条件，故以本文件和 `062_00_ncap160_gate.json` 为准。

## 2K 描述性指标（非最终性能结论）

- 十年平均产量：7,249.4 kg/ha；平均灌溉：174.0 mm；平均 N：160.0 kg/ha；平均 PFP_N：45.31 kg/kg。
- 每年 N 均为 160.0 kg/ha，说明策略贴住了新上限；这进一步支持不进入 100K。
- `WP_ET` 未由该 2K checkpoint evaluator 导出，未进行额外 DSSAT replay，因此标为未提供，不能用此 smoke 宣称 WP_ET 改善。

## 停止条件与后续

停止 062 分支，不再把 reward、天气、观测、动作网格或 PPO 超参数叠加到该 N cap 实验上。若以后重新开启 FQ，必须先另行批准一个单因素、带预注册跨年动作适应性目标的设计；本次结果不允许作为 100K 训练依据。

