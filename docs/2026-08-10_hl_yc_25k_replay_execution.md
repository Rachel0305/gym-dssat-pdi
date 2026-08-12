# HL/YC 25K 同口径五情景回放：执行 gate 记录

> 更新（2026-08-10）：实际容器回放已完成。本更新覆盖下文容器启动前“运行时不可用”的状态。用户启动 Docker 容器 `nifty_taussig` 后，使用 `/opt/gym_dssat_pdi/bin/python` 成功导入 `sb3_contrib`，并在 `/workspaces/gym-dssat-pdi` 串行完成 HL、YC 的既有 25K checkpoint 十年五情景回放；没有 Windows Python、训练或对冻结 100K 目录的覆盖。

实际命令均为完整十年（单年 `--years 2014 --dry-run` 会因 helper 只接受完整十年 validation summary 而在 DSSAT 前停止）。完整十年 dry-run 均为 `missing_inputs: []` 后才运行：

```text
docker exec nifty_taussig /usr/bin/bash -lc "cd /workspaces/gym-dssat-pdi; /opt/gym_dssat_pdi/bin/python src/054_hla_lowIC_site_transfer/run_054_03_hla_lowIC_five_scenario_figures.py --checkpoint 25000 --years 2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 --label prereg25k_hl_full --skip-daily --skip-bars"
docker exec nifty_taussig /usr/bin/bash -lc "cd /workspaces/gym-dssat-pdi; /opt/gym_dssat_pdi/bin/python src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py --checkpoint 25000 --years 2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 --label prereg25k_yc_full --skip-daily --skip-bars"
```

HL 用时 51.8 s，YC 用时 43.5 s；无 OOM 或资源异常，且仅在 HL 完成后启动 YC。输出根：

- `benchmark_results/054_03_hla_lowIC_054_00_hla_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt25000_prereg25k_hl_full/`（约 14 MB）
- `benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt25000_prereg25k_yc_full/`（约 10 MB）

|站点，25K PPO 十年均值|产量 kg/ha|WP_ET kg/m3|PFP_N kg/kg|灌溉 mm|施氮 kg/ha|胜场（产量/WP_ET/PFP_N）|
|---|---:|---:|---:|---:|---:|---:|
|HL|6844.3|1.514|28.53|225.0|240.0|6/10，4/10，0/10|
|YC|8201.7|2.219|34.16|228.0|240.0|3/10，0/10，10/10|

HL-25K 比冻结 100K 高约 599.8 kg/ha 产量和 0.067 `WP_ET`，但低 22.18 `PFP_N`；YC-25K 则高约 2129.1 kg/ha 和 0.340 `WP_ET`，低 3.79 `PFP_N`。逐年终点产量均由入口脚本以 1 kg/ha 容差与既有 checkpoint validation summary 核验通过。management-event 累计灌溉/N 与季节汇总最大绝对差均为 0；HL 平均正事件/DAP1后事件为 16.4/15.4，YC 为 10.2/9.2。

**更新后的 gate：技术回放通过，批准进入独立年份折叠的 checkpoint 选择验证设计；不批准直接冻结 25K、不批准新 100K 训练或多站点调参。** 比较仍是同一 seed0、同一十年上的 post-hoc 候选证据，且 HL-25K 的 PFP-N 明显低于 100K。

日期：2026-08-10。范围仅为 HL `054_00/054_03` 和 YC `055_00/055_03` 的既有 25K MaskablePPO checkpoint；不新训、不改 reward/observation/action grid/wrapper，不覆盖冻结 100K 结果，不使用 Windows Python。

## 预注册式执行规则（在看 25K 五情景结果前固定）

本轮不是事后把“看起来更好”的 25K 直接写成论文模型。若容器 gate 通过，先做 HL-2014 单年 smoke；只有以下项同时通过，才按一次一个进程、先 HL 后 YC 的顺序完成十年回放：

1. 必须由 `/opt/gym_dssat_pdi/bin/python` 加载 `sb3_contrib`，CPU 单进程运行；不得替换为 Windows Python。
2. 25K 模型、低 IC 输入根、四基线/外部自动氮根均存在；PPO 回放的终点产量必须与 `054_00/055_00` 既有 checkpoint 验证 CSV 在 1 kg/ha 容差内一致。
3. action transmission mismatch 必须为 0；输出只能写入 checkpoint=25000 且含唯一 label/run-id 的新目录，不能写入冻结 `054_03`/`055_03`。
4. checkpoint 的正式选择不得在同一十年上挑选。建议以后采用年份折叠：在预先指定的训练折叠选择 25K 或 100K，以平均产量为主目标，并设定 `WP_ET`、PFP-N 和 DAP1 后动作数的非劣/下限 gate；在未参与选择的年份折叠报告五情景结果。当前同一 seed0、同十年审计仅能产生候选证据。

## 静态 provenance 核验（通过）

|检查项|HL|YC|结论|
|---|---|---|---|
|25K checkpoint|`054_00.../models/HLA/HLA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip`，181,880 B|`055_00.../models/YCA/YCA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip`，178,734 B|存在|
|训练/验证年份|2004--2013 / 2014--2023|2004--2013 / 2014--2023|一致|
|动作网格|I=[0,15,30,45] mm；N=[0,40,80,120] kg/ha（16 组合）|相同|一致，禁止修改|
|输入 profile/root|`lowIC`；`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`；源 MZX `HL/CNHL0701_corrected_IC123.MZX`|`lowIC`；同一 root；源 MZX `YC/CNYC0801.MZX`|与 054_00/055_00 manifest 一致|
|冻结基线 root|`054_02_hla_lowIC_four_baselines_static_level1` 和 `054_01_hla_lowIC_external_auto_n_rule_nstd050_minimal`|`055_02_yca_lowIC_four_baselines_static_level1` 和 `055_01_yca_lowIC_external_auto_n_rule_nstd050_minimal`|与 054_03/055_03 result JSON 一致|
|已有 checkpoint 动作审计|25K transmission mismatch=0|25K transmission mismatch=0|动作到 DSSAT 的既有证据通过|

正式回放入口已核对为：

- `src/054_hla_lowIC_site_transfer/run_054_03_hla_lowIC_five_scenario_figures.py`
- `src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py`

两脚本均有 `--checkpoint`、`--years`、`--label`、`--dry-run`，并在 PPO 回放中同时设置 engine 的 `LOWIC_INPUT_ROOT` 和 `ppo_safe_rendering.MULTISITE_INPUT_ROOT`，且将回放终点与既有 checkpoint 验证表进行 1 kg/ha 容差核验。若环境可用，拟执行的单年 smoke 命令为（只允许在容器解释器下执行）：

```text
/opt/gym_dssat_pdi/bin/python src/054_hla_lowIC_site_transfer/run_054_03_hla_lowIC_five_scenario_figures.py --checkpoint 25000 --years 2014 --label prereg25k_smoke --skip-daily --skip-bars
```

该命令因 label 形成新的 `...ckpt25000_prereg25k_smoke` 输出根；脚本拒绝写入非空目录，因而不会覆盖冻结 100K 输出。注意：脚本也会生成一个 054/055 record 文件，执行前应确认该文件不存在或改由隔离的执行包装入口处理，以满足“只新增本执行记录”的文档边界。

## 容器运行时 gate（失败，停止）

执行的只读解释器探测命令：

```text
wsl.exe --exec /opt/gym_dssat_pdi/bin/python -c "import sys, sb3_contrib; print(sys.executable); print(sb3_contrib.__file__)"
```

返回：`WSL ... execvpe(/opt/gym_dssat_pdi/bin/python) failed: No such file or directory`。

这表示项目指定的 WSL/container gym 解释器当前不可用；没有执行 Python 脚本、没有启动 DSSAT、没有产生 snapshot/CSV/图件，亦没有使用 Windows Python 作为替代。根据任务约束，此处停止，不尝试长回放或新训练。

## 当前可报告的既有 25K 数值（非新回放）

来自原始 checkpoint validation summary，而非五情景重放：HL 平均产量 6844.4 kg/ha、灌溉 225 mm、N 240 kg/ha、PFP-N 28.52，平均产量差 vs 四基线最优 +351.4 kg/ha；YC 分别为 8201.6、228、240、34.17、-1.93。这些数值支持 25K 是候选 checkpoint，但没有同口径 `WP_ET`、五情景胜场或重新生成的 snapshot，不能作为最终论文结论。

## 结论

**不批准进入 25K replay 的下一阶段。** 解除条件是恢复或明确提供项目指定的 `/opt/gym_dssat_pdi/bin/python` WSL/container runtime，并能用其导入 `sb3_contrib`；之后仍须从 HL-2014 单年 smoke 开始，单进程、低内存执行。
