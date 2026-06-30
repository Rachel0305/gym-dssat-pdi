# 2026-06-28/29 011 official reward restart and action-channel diagnosis

## 目的

重新开启一条 HLA PPO 主线，使用：

- HLA 2010/2015；
- IC=1；
- 最新校准 HY0006；
- `references/rewards.py` 中的 gym-DSSAT 官方原始 reward；
- 先做 smoke，不长训练。

本轮核心不是判断 PPO 好坏，而是先确认：

> gym/PDI 的 `amir/anfer` 动作是否真的能进入 DSSAT 管理事件并改变作物生长。

## 011_01：官方 reward restart smoke

执行脚本：

`src/run_hla_official_reward_restart_smoke.py`

输出目录：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart`

2010 年 action-channel smoke 中，Python 层 forced action 为：

| DAP | amir | anfer |
| ---: | ---: | ---: |
| 1 | 0 | 165 |
| 49 | 10 | 0 |
| 70 | 10 | 0 |
| 95 | 10 | 0 |

结果：

| 指标 | 值 |
| --- | ---: |
| irrigation_events_mgmtevent | 0 |
| fertilizer_events_mgmtevent | 0 |
| irrigation_total_mgmtevent | 0 |
| fertilizer_total_mgmtevent | 0 |
| final_gwad | 6956 |
| final_cwad | 19344 |

初步判断：Python/gym 层动作被发出，reward 也使用了这些 action，但 DSSAT 管理事件和最终产量没有变化。

## 011_02：R/R 管理模式下 null vs forced action

执行脚本：

`src/run_hla_action_channel_diagnosis_011_02.py`

输出目录：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/action_channel_diagnosis`

同一套 2010 input 分别运行：

- `null_action`：全程 `amir=0, anfer=0`
- `forced_action`：DAP 1 `anfer=165`，DAP 49/70/95 `amir=10`

两者在原始 MZX 管理模式下都是：

```text
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 1 MA              R     R     R     R     M
```

结果：

| scenario | nonzero action rows | action irrigation | action N | MgmtEvent irrigation | MgmtEvent fertilizer | final GWAD | final CWAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| null_action | 0 | 0 | 0 | 0 | 0 | 6956 | 19344 |
| forced_action | 4 | 30 | 165 | 0 | 0 | 6956 | 19344 |

逐日比较 `PlantGro.OUT` 关键列：

| column | max abs diff |
| --- | ---: |
| DAP | 0 |
| CWAD | 0 |
| GWAD | 0 |
| LAID | 0 |
| WSPD | 0 |
| NSTD | 0 |

结论：在 `IRRIG=R, FERTI=R` 下，gym action 只影响 Python 层 action/reward 日志，不影响 DSSAT 管理事件和作物生长。

## 011_03：linked management 验证

检查 `references/dssat_pdi.py` 后发现：

```python
self.ferti = 'L' if mode in ['all', 'fertilization'] else 'R'
self.irrig = 'L' if mode in ['all', 'irrigation'] else 'R'
```

也就是说，`mode='all'` 设计上应该使用 DSSAT-PDI linked management：

- `IRRIG=L`
- `FERTI=L`

但当前 HLA MZX 是静态 DSSAT 文件，不含 `{{irrig}}` / `{{ferti}}` 占位符，所以 `_fill_template_from_string()` 没有实际改变 `fileX.MZX`。

011_03 将同一套 2010 input 扩展为四组：

| scenario | linked management | forced action | MgmtEvent irrigation | MgmtEvent fertilizer | final GWAD | final CWAD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| null_action | no, R/R | no | 0 | 0 | 6956 | 19344 |
| forced_action | no, R/R | yes | 0 | 0 | 6956 | 19344 |
| linked_null_action | yes, L/L | no | 0 | 0 | 6956 | 19344 |
| linked_forced_action | yes, L/L | yes | 30 | 165 | 7679 | 20665 |

`linked_forced_action` 的 `pdi_tmp_snapshot/fileX.MZX` 中管理模式为：

```text
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 1 MA              R     L     L     R     M
```

`linked_forced_action` 的 `MgmtEvent.OUT` 明确出现：

- DAP 1：`Fertilizer 165 kg[N]/ha`
- DAP 49：`Irrigation 10.0 mm`
- DAP 70：`Irrigation 10.0 mm`
- DAP 95：`Irrigation 10.0 mm`

逐日 `PlantGro.OUT` 对比：

| pair | GWAD max diff | CWAD max diff | WSPD max diff | NSTD max diff |
| --- | ---: | ---: | ---: | ---: |
| R/R null vs forced | 0 | 0 | 0 | 0 |
| L/L null vs forced | 723 | 1321 | 0.82 | 0.157 |

## 最终结论

gym/PDI/DSSAT 的 action 通信链路本身是有效的。

真正的问题是：HLA 静态 MZX 进入 gym-DSSAT 时，没有被正确切换为 `IRRIG=L, FERTI=L`。因此 action 在 `R/R` 模式下只影响 Python 层 reward/history，不影响 DSSAT 管理和生长。

## 修复方向

后续所有用于 PPO 训练/评估的 HLA MZX，在进入 gym 环境前必须显式设置：

- `IRRIG=L`
- `FERTI=L`
- `MI=1`
- `MF=1`

或者把静态 MZX 改成带 `{{irrig}}` / `{{ferti}}` 的 Jinja 模板，让 `DssatPdi` 源码自动渲染。

修复后必须重新跑 action-channel smoke，再进入 PPO 训练。

## 011_04：Jinja 占位符修复

根据官方 `my_data/UFGA8201-HL.jinja2`，在复制后的 HLA MZX 副本中插入占位符：

```text
@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL
 1 ME              {{ wther }}     M     E     R     S     L     R     1     G     R     2
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 1 MA              {{ plant }}     {{ irrig }}     {{ ferti }}     R     M
```

实现位置：

`src/run_hla_official_reward_restart_smoke.py`

函数：

`set_pdi_jinja_placeholders()`

注意：该修复只作用于复制后的实验 input，不修改原始 MZX。

修复后重新运行 2010 action-channel smoke，`pdi_tmp_snapshot/fileX.MZX` 被 `DssatPdi` 自动渲染为：

```text
@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL
 1 ME              M     M     E     R     S     L     R     1     G     R     2
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 1 MA              R     L     L     R     M
```

`MgmtEvent.OUT` 成功记录：

| DAP | Operation | Amount |
| ---: | --- | ---: |
| 1 | Fertilizer | 165 kg N/ha |
| 49 | Irrigation | 10 mm |
| 70 | Irrigation | 10 mm |
| 95 | Irrigation | 10 mm |

修复后 summary：

| 指标 | 值 |
| --- | ---: |
| irrigation_events_mgmtevent | 3 |
| fertilizer_events_mgmtevent | 1 |
| irrigation_total_mgmtevent | 30 |
| fertilizer_total_mgmtevent | 165 |
| final_gwad | 7679 |
| final_cwad | 20665 |

### 更新结论

Jinja 占位符修复后，action channel 已通过验证。

下一步可以进入极短 PPO training smoke，但仍需遵守：

- 先 smoke；
- 不直接长训练；
- 保存模型、日志、raw DSSAT 输出；
- 检查 PPO eval 的 `MgmtEvent.OUT` 是否真的有管理事件。

## 011_05：日交互耗时诊断

用户希望尽量保留日尺度交互，因此先诊断日交互本身是否太慢。

执行脚本：

`src/diagnose_hla_daily_linked_step_timing_011_05.py`

输出目录：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/daily_step_timing/2010`

结果：

| 指标 | 值 |
| --- | ---: |
| step rows | 10 |
| mean step elapsed | 0.009 s |
| max step elapsed | 0.014 s |
| min step elapsed | 0.006 s |

结论：日尺度 `env.step()` 本身并不慢。此前 PPO smoke 超时不是 DSSAT daily interaction 的天然成本，而是 SB3 wrapper/API 适配问题。

## 011_06：SB3 PPO smoke 跑通

发现原项目 `sb3_wrapper.GymDssatWrapper` 在 `__init__` 中会额外调用 `raw_env.reset()`，对 socket 型 DSSAT-PDI 环境风险较高。

同时，第一次自定义 lazy wrapper 没有继承 `gymnasium.Env`，导致 SB3 初始化报错。

修复：

- 在 `src/run_hla_official_reward_restart_smoke.py` 内新增 `LazyScalarGymDssatWrapper(gymnasium.Env)`；
- 不在 wrapper 初始化时额外 reset；
- 在 `step()` 内把官方 list reward 标量化；
- 不使用 `Monitor`；
- PPO smoke 参数保持极小：
  - `n_steps=5`
  - `batch_size=5`
  - `n_epochs=1`
  - `total_timesteps=5`

执行：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --child-train-smoke 2010 5 0"
```

结果：流程跑通。

调试日志：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/seed0_5steps/ppo_smoke_debug.log`

关键节点：

| phase | status |
| --- | --- |
| gym.make | done |
| wrapper | done |
| PPO init | done |
| learn | done |
| model save | done |
| eval reset | done |
| eval close | done |

eval 输出：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/seed0_5steps/ppo_smoke_eval_daily.csv`

`pdi_tmp_snapshot_eval/fileX.MZX` 正确为：

```text
1 MA              R     L     L     R     M
```

`MgmtEvent.OUT` 中出现 PPO eval action 管理事件，说明：

> SB3 PPO -> wrapper -> gym-DSSAT -> PDI -> DSSAT linked management 的完整链路已经跑通。

注意：5-step PPO 只是管道 smoke，不代表策略合理。eval 中未训练策略输出了近似每天 `~100 kg N/ha` 和 `~25 mm` 灌溉的动作，明显不具备农学意义。下一步如果继续日尺度 PPO，必须加入动作频率、预算或安全约束，否则会每天施肥灌溉。

## 011_06：带预算/频率约束的日尺度 PPO smoke

为了保留日尺度交互，同时避免 PPO 初始策略每天大量灌溉施肥，在 `src/run_hla_official_reward_restart_smoke.py` 中新增：

`BudgetedDailyActionWrapper`

约束：

| 约束 | 值 |
| --- | ---: |
| seasonal irrigation budget | 120 mm |
| seasonal nitrogen budget | 150 kg/ha |
| daily irrigation cap | 30 mm |
| daily nitrogen cap | 50 kg/ha |
| minimum operation interval | 7 days |

执行：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 50 --seed 0 --timeout 180"
```

结果：流程跑通。

状态：

| child | returncode | timed out |
| --- | ---: | --- |
| action smoke | 0 | false |
| budgeted PPO smoke, 50 steps | 0 | false |

eval CSV：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/seed0_50steps/ppo_smoke_eval_daily.csv`

CSV 中安全动作累计：

| 指标 | 值 |
| --- | ---: |
| safe irrigation total | 120 mm |
| safe nitrogen total | 150 kg/ha |

`MgmtEvent.OUT` 中实际事件：

| Operation | Count | Total |
| --- | ---: | ---: |
| Fertilizer | 3 | 150 kg N/ha |
| Irrigation | 5 | 120.1 mm |

注：`MgmtEvent.OUT` 中 irrigation 为 120.1 mm，是因为每日事件以一位小数输出后再求和；CSV 内部累计为 120 mm。

最终 `MgmtEvent.OUT` 中 harvest yield：

| 情景 | HWAM |
| --- | ---: |
| null/reference block | 6956 kg/ha |
| budgeted PPO eval block | 7854 kg/ha |

### 011_06 结论

日尺度 PPO + official reward + linked DSSAT-PDI + 预算/频率约束的完整流程已经跑通。

这仍然只是 smoke，不代表策略已经收敛；但它证明了现在可以在合理预算内开展日尺度 PPO 训练。

## 011_07：HLA 2010 budgeted daily PPO 5k

用户决定直接运行 5000 timesteps，以初步观察预算约束下 PPO 是否能形成可用策略。

执行：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 5000 --seed 0 --timeout 900"
```

结果：完成，无超时，无残留进程。

调试日志显示：

| phase | elapsed |
| --- | ---: |
| PPO learn | about 39.6 s |
| full child train + eval | about 44 s |

输出目录：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/seed0_5000steps`

模型：

`ppo_official_reward_smoke.zip`

预算检查：

| 指标 | CSV | MgmtEvent |
| --- | ---: | ---: |
| irrigation | 120 mm | 120.0 mm |
| nitrogen | 150 kg/ha | 150 kg/ha |

PPO eval 实际管理事件：

| DAP | Fertilizer | Irrigation |
| ---: | ---: | ---: |
| 1 | 50 | 30 |
| 8 | 50 | 30 |
| 15 | 50 | 30 |
| 22 | 0 | 30 |

最终产量：

| block | HWAM |
| --- | ---: |
| null/reference block | 6956 kg/ha |
| budgeted PPO eval block | 7854 kg/ha |

### 011_07 初步解释

5k 训练已经能在预算约束内完成日尺度 PPO 训练，并且相对 null 增产。

但策略仍然很粗暴：它倾向于在季节早期尽快用完全部水氮预算。当前结果说明流程可行，但还不能说明 PPO 学到了农学上合理的时序策略。

下一步候选：

1. 跑 seed1，确认“早期打满预算”是否稳定；
2. 加生育期/最早操作窗口，例如 emergence 后或指定 DAP 后才能施肥灌溉；
3. 或者保留日交互，但把 min interval 和 daily cap 设计成更贴近实际管理。

## 011_08：HLA 2010 budgeted daily PPO 5k seed1

为检查 seed 稳定性，保持 011_07 所有设置不变，只改 seed=1。

执行：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 5000 --seed 1 --timeout 900"
```

结果：完成，无超时。

预算检查：

| seed | irrigation | nitrogen | fertilizer events | irrigation events | HWAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 120 | 150 | 3 | 4 | 7854 |
| 1 | 120 | 150 | 3 | 7 | 7854 |

管理时序：

| seed | Fertilizer | Irrigation |
| ---: | --- | --- |
| 0 | DAP 2/9/16, 50 each | DAP 2/9/16/23, 30 each |
| 1 | DAP 2/9/16, 50 each | DAP 2 20.6, DAP 9 14.2, DAP 16 13.9, DAP 23 15.0, DAP 30 19.2, DAP 37 19.1, DAP 44 18.0 |

### 011_08 解释

两个 seed 的产量相同，预算均被完全使用。

稳定现象：

- 施氮都在早期 DAP 2/9/16 用完 N150；
- 灌溉也都在早期用完 I120，但 seed1 比 seed0 分配更分散。

这说明当前 official reward + budget wrapper 的主要倾向仍然是“尽早使用预算”，尤其是氮肥部分非常稳定。

下一步如果追求更合理农艺时序，需要引入生育期窗口或最早操作限制，而不只是继续增加训练步数。

## 011_09：单变量 PPO 诊断

用户指出此前只优化氮或只优化灌溉时，PPO 行为看起来较合理。因此进一步做单变量诊断。

为了避免 `mode='fertilization'` 触发 `PLANT=A` 的混杂，本轮仍使用 `mode=all` 和 linked management，但通过 `BudgetedDailyActionWrapper` 强制屏蔽另一个变量：

- irrigation-only：只允许 `amir`，`anfer=0`
- fertilization-only：只允许 `anfer`，`amir=0`

执行：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 5000 --seed 0 --variant irrigation_only --timeout 900"
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_official_reward_restart_smoke.py --year 2010 --timesteps 5000 --seed 0 --variant fertilization_only --timeout 900"
```

汇总表：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/single_variable_vs_joint_5k_summary.csv`

结果：

| run | HWAM | irrigation | nitrogen | events | management timing |
| --- | ---: | ---: | ---: | --- | --- |
| joint seed0 | 7854 | 120 | 150 | I=4, N=3 | DAP 2/9/16 N50+I30, DAP23 I30 |
| joint seed1 | 7854 | 120 | 150 | I=7, N=3 | DAP 2/9/16 N50, irrigation spread DAP2-44 |
| irrigation-only | 7854 | 120 | 0 | I=4, N=0 | DAP 2/9/16/23 I30 |
| fertilization-only | 6971 | 0 | 150 | I=0, N=3 | DAP 2/9/16 N50 |

### 011_09 解释

单变量结果也表现出“早期使用预算”的倾向：

- 只优化灌溉时，I120 在 DAP 2/9/16/23 用完；
- 只优化施氮时，N150 在 DAP 2/9/16 用完。

因此，“早期打满预算”不是水氮联合动作空间独有的问题，而是当前 reward + budget wrapper + 允许早期操作共同导致的稳定倾向。

同时，HLA 2010 的主要响应来自水分：

- irrigation-only 已达到 7854；
- joint PPO 也是 7854；
- fertilization-only 只有 6971，接近 null 6956。

下一步如果希望得到更合理农艺时序，应优先加入生育期/最早操作窗口，而不是继续单纯增加训练步数或只改成单变量。
## 011_10：旧 output_hl 模型在当前 corrected HLA2010 环境中的复评估

用户指出 `output_hl/irrigation` 和 `output_hl/fertilization/best_model.zip` 可能是以前单变量优化时看起来较合理的模型，但不确定是否就是当时使用的模型。因此本轮只做低成本复评估，不重新训练。

模型元信息：

| model | obs dim | action dim | note |
| --- | ---: | ---: | --- |
| `output_hl/irrigation/best_model.zip` | 24 | 1 | 可适配到当前 all-mode 的 `amir` |
| `output_hl/fertilization/best_model.zip` | 11 | 1 | 与当前 all-mode 24维 observation 不兼容 |

执行：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/evaluate_old_hla_model_in_current_env.py --year 2010 --model output_hl/irrigation/best_model.zip --label old_output_hl_irrigation_best_model"
```

输出：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/old_model_current_env_eval/2010/old_output_hl_irrigation_best_model`

结果：

| model | source | irrigation total | nitrogen total | nonzero actions |
| --- | --- | ---: | ---: | --- |
| old irrigation model adapted to current env | `MgmtEvent.OUT` / eval CSV | 1.9–1.99 mm | 0 kg/ha | DAP10≈0.95mm, DAP17≈1.04mm |

解释：

这个旧 `output_hl/irrigation/best_model.zip` 放到当前 corrected HLA2010 环境后，并没有表现为“正常灌溉”或“合理使用 I120 预算”，而是几乎不灌水。因此它很可能不是用户记忆中那个合理灌溉模型，或者它依赖旧 IC、旧品种参数、旧模板、旧年份、旧 observation/reward/action 定义。旧施肥模型由于 observation space 不兼容，且 `mode=fertilization` 试探会卡住，本轮不强行评估，以避免浪费算力。

结论：当前证据不支持“旧 output_hl 模型可以直接解决当前 corrected HLA2010 的早期打满/训练行为问题”。若之后找到其他候选旧模型 zip，可继续用同一复评估入口检查，但必须记录模型 observation/action space 与当前评估环境是否一致。

## 011_11：HLA 2004 新品种参数下 DSSAT 原生自动施肥抢救

目的：用户希望继续抢救 DSSAT 原生 `FERTI=A`，确认更新品种参数和 IC=1 后是否能触发自动施肥。

输入基底：

- `DSSAT_auto_validation/HLA_2004/run_CNHL0408_DSSAT480_2004/CNHL0408.MZX`
- `DSSAT_auto_validation/HLA_2004/CNHL0401.WTH`
- `DSSAT_auto_validation/HLA_2004/run_CNHL0404/SOIL.SOL`
- `DSSAT_auto_validation/HLA_2004/cultivar_calibration_HLA2004_480/input_corrected_package/MZCER048.CUL`

执行：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2004_auto_fertilizer_rescue_011_11.py"
```

输出：

`DSSAT_auto_validation/HLA_2004/auto_fertilizer_rescue_2004_newcul_011_11`

扫描变体：

- `FERTI=R` 自动灌溉对照；
- `FERTI=A` + `FE001/FE005`；
- `NMTHR=50/99`；
- `NAMNT=25/50`；
- `NAOFF=GS000/GS999`；
- `IRRIG=A` 与 `IRRIG=N`。
- 初始条件土壤无机氮强制降低：所有层 `SNH4/SNO3=0.0` 或 `0.1`。

结果：所有 `FERTI=A` 变体的 `fertilizer_events_mgmtevent=0`、`fertilizer_total_mgmtevent=0`。自动灌溉可以触发并达到 1132.8 mm，但自动施肥仍为 0；关闭自动灌溉后，作物水分严重胁迫且籽粒为 0，自动施肥仍为 0。进一步把初始土壤 `SNH4/SNO3` 降到 0 或 0.1 后，最大 `NSTD=0.799`，产量降到 374/495 kg/ha，说明氮胁迫已经非常强，但自动施肥仍然 0 次。

解释：更新品种参数和 IC=1 没有修复 DSSAT 原生自动施肥。此次低成本扫描进一步排除了材料代码、阈值、单次用量、截止阶段、是否自动灌溉、初始土壤氮过高这些显性输入解释。当前证据支持继续把 `FERTI=A` 视为未验证/不可用路径；正式规则基线应使用外部显式施肥规则或 PPO linked action，而不是 DSSAT 原生自动氮。
