# 021_00 DQN Benchmark Framework 项目结构审计

## 1. 审计状态

- 日期：2026-07-11
- 状态：`completed`
- 审计方式：只读扫描代码、五站点输入、历史实验目录、checkpoint、summary、图和文档；审计阶段未修改旧脚本和旧结果。
- 当前分支：`codex-reward-sweep-backup`
- 审计开始时 HEAD：`b304fdf`（`Add five-site status and benchmark refactor prompt`）
- 审计开始时未跟踪但不纳入本任务提交的用户资料：`references/` 下 4 个 PPTX/DOCX/PDF 文件。

## 2. 结论摘要

当前项目已经有可用的冻结 n-step DQN 证据，但还没有统一的训练、评估、复用、续跑和报告入口。`src/` 中有 35 个实际调用 `model.learn()` 的 DQN 或 DQN 变体脚本，实验迭代主要通过复制脚本完成。

本次重构应采用“新增框架 + adapter”的方式：

1. 不改动旧训练脚本、reward、IC 和已有结果。
2. 把冻结配置、站点差异、结果 schema 和旧目录映射移入 YAML、registry 和 adapter。
3. 第一版仅把已经验证的 9 动作、`n_steps=5`、本地 null 相对终端奖励定义为正式冻结主线。
4. 5/16/25 动作、淋洗 reward、观测消融等只预留配置接口，不写成已经验证。
5. 历史 checkpoint 未保存 replay buffer，因此只能用于确定性评估或 warm-start；严格 resume 必须从新框架生成的 checkpoint 开始验证。

## 3. 当前训练入口

### 3.1 冻结配置

冻结主线常量位于：

- `src/frozen_nstep_dqn_config_020_11.py`

当前配置为：

| 项目 | 冻结值 |
|---|---|
| 算法 | Stable-Baselines3 DQN，MlpPolicy |
| 动作 | 9 个离散水氮组合，I={0,15,30} mm，N={0,50,100} kg/ha |
| 季节预算 | I≤120 mm，N≤300 kg/ha |
| 单次上限 | I≤30 mm，N≤100 kg/ha |
| 操作间隔 | 水氮共享最小 7 DAP |
| 决策窗口 | DAP 1–120 |
| reward | 每步 `-1×I-5×N`；终止时 `+max(0, GWAD_final-GWAD_local_null)` |
| DQN | lr=1e-4，buffer=10000，learning_starts=50，batch=32，train_freq=1，gradient_steps=1，gamma=0.99，target_update_interval=10000，n_steps=5 |
| 正式训练量 | 50,000 steps，5,000 steps/checkpoint |
| checkpoint 选择 | deterministic total reward 最大；并列时取更早 checkpoint |

### 3.2 HLA

HLA 当前不是一个统一入口，而是多个脚本组成的流水线：

- seed0：`src/run_hla2010_nstep_dqn_seed0_020_08.py`
- seed1：`src/run_hla2010_nstep_dqn_seed1_020_06.py`
- seed2：`src/run_hla2010_nstep_dqn_seed2_020_07.py`
- 2015 跨年评估：`src/run_hla2010_nstep_to_hla2015_transfer_020_09.py`
- 2007/2016/2022 跨年评估：`src/run_hla2010_nstep_transfer_2007_2016_2022_020_10.py`
- 五情景和三 seed 汇总：`src/run_hla_five_scenario_completion_020_11.py`

旧 HLA 训练入口存在删除同名 run directory 的逻辑，不能由新框架直接无保护调用。

### 3.3 YC/FQ

目前最接近统一入口的是：

- 训练与 checkpoint 评估：`src/run_yc_fq_frozen_nstep_cross_site_020_12.py`
- 跨 seed 汇总：`src/summarize_yc_fq_frozen_nstep_seed1_020_13.py`

该入口会拒绝覆盖已存在结果，包含 local same-input null、运行时 action/MgmtEvent 对账和 checkpoint 选择，但尚未保存 replay buffer，也没有真正的 resume。

### 3.4 LC/SY

- LC：`src/run_lc2010_baseline_relative_dqn_smoke_017_12.py`
- SY：`src/run_sy_local_dqn_train_cross_year_transfer_017_08.py`
- SY seed1 最小复现：`src/run_sy2014_seed1_minimal_reproduction_018_08.py`

这些结果不属于冻结 `n_steps=5` 正式主线；LC 仅 5K smoke，LC/SY 历史训练没有显式传入 `n_steps=5`。

## 4. 当前评估入口

| 范围 | 入口/数据源 | 说明 |
|---|---|---|
| HLA checkpoint | `run_hla_unified_dqn_long_train_015_09.py::evaluate_model` | 每个 checkpoint deterministic 评估 |
| HLA 跨年 | `020_09`、`020_10` | 真实加载 2010 模型迁移评估 |
| HLA 五情景 | `020_11::load_existing` | DQN 部分复用历史 daily CSV；固定情景另外运行/复用 |
| YC/FQ | `020_12::evaluate_checkpoint` | 保存 daily、summary、MgmtEvent 和 runtime audit |
| YC2014 DAP 修正 | `src/reaudit_yc2014_operation_dap_020_12.py` | 只加载模型再评估，不训练 |
| LC/SY | 各站点旧脚本 | schema、n-step 和 provenance 不统一 |

## 5. reward、action 和约束实现

### 5.1 冻结 reward

正式公式为：

```text
R_t = -1.0 × I_t - 5.0 × N_t
      + 1_terminal × max(0, GWAD_final - GWAD_local_same_input_null)
```

实现分散于：

- HLA：`src/run_hla_baseline_relative_dqn_checkpoint_015_12.py`
- YC/FQ：`src/run_yc_fq_frozen_nstep_cross_site_020_12.py`
- 早期同公式复制：`run_yc2014_baseline_relative_dqn_015_10.py`、`run_fq2016_baseline_relative_dqn_checkpoint_015_14.py` 等。

旧 `delta_grnwt` reward、Tao RF1、淋洗惩罚 smoke 均属于历史/敏感性试验，不是本次重构要修改或推广的正式 reward。

### 5.2 action/constraint

实际离散动作与约束主要由：

- `src/run_yc2014_linked_dqn_5k_multiseed_013_07.py::YCDiscreteBudgetedWrapper`
- `src/frozen_nstep_dqn_config_020_11.py::apply_environment_constants`

共同实现。

高风险点：

1. 冻结入口通过修改历史模块全局变量，把原 4 动作表替换为 9 动作表。
2. 水、氮共用一个 `last_operation_dap`；任一非零操作都会触发两个通道共同等待 7 DAP。
3. 同一 Python 进程连续运行多配置可能发生全局常量串扰。

新框架第一版必须保持上述已验证语义，并优先让每个 site-year-seed 在隔离的子进程中运行。

## 6. 五站点输入配置审计

### 6.1 Canonical code

新框架统一采用 `HLA / YC / FQ / LC / SY`，同时接受旧别名 `HL / YCA / FQA / LCA / SYA`。旧 `dssat_site_config.py` 指向早期 `my_data/UFGA8201-*.jinja2`，不能作为当前论文主线真值。

### 6.2 HLA

HLA 020_11 的实际 prepared input 映射：

| 年份 | prepared input |
|---|---|
| 2007 | `DSSAT_auto_validation/HLA_2004/hla2010_nstep_transfer_2007_2016_2022_020_10/test_cases/2007/input` |
| 2010 | `DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed0_020_08/nstep5_seed0_50000steps/input` |
| 2015 | `DSSAT_auto_validation/HLA_2004/hla2010_nstep_to_hla2015_transfer_020_09/test_case_hla2015/input` |
| 2016 | `DSSAT_auto_validation/HLA_2004/hla2010_nstep_transfer_2007_2016_2022_020_10/test_cases/2016/input` |
| 2022 | `DSSAT_auto_validation/HLA_2004/hla2010_nstep_transfer_2007_2016_2022_020_10/test_cases/2022/input` |

这些目录均包含 MZX、WTH、`MZCER048.CUL` 和 `SOIL.SOL`；正式 HLA 输入为 treatment=1、IC=1。

### 6.3 YC/FQ

- YC2014：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0801.MZX`，treatment=2，WTH=`CNYC1401.WTH`，soil=`YC99001200`，cultivar=`ZD0985`，IC=1。
- FQ2016：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ0801.MZX` 经年份 adapter 平移到 2016，treatment=2，WTH=`CNFQ1601.WTH`，soil=`FQ99001200`，cultivar=`FQ0985`，IC=1。

两者已有 `020_12 --static-check` 逻辑，可作为环境 adapter 的第一批接入对象。

### 6.4 LC

原始 `CNLC0801.MZX` 的 soil reference 为 `LC99001200`，实际 profile 为 `LC990012007`，且逐年 IC/SDATE 需要克隆对齐。必须使用：

- `src/run_lc_fixed_input_year_screening_017_11.py`

提供的修复 adapter。已验证的 2010 prepared input 为：

- `DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2010/null/input/CNLC1001.MZX`

LC 不能直接把原始 MZX 写成“正式可运行输入”。

### 6.5 SY

SY 存在未解决的 provenance 冲突：

- 文档 `019_09`、`020_14` 写 SY2014 使用 IC=2；
- 当前源码、017_08/018_08 落盘训练输入实际为 IC=0；
- 当前项目中未找到包含第二 IC level 的权威 MZX。

因此 SY 配置只能标记为 `input_status: ambiguous`。用户确认权威 IC=2 文件前，框架不得把 SY 写成已完成或自动启动正式训练。

## 7. 已有结果与 checkpoint 审计

### 7.1 HLA 020_11

目录：`DSSAT_auto_validation/HLA_2004/hla_five_scenario_nstep_020_11`

- 922 files，约 33.51 MB。
- 汇总：`020_11_hla_five_scenario_summary.csv`，35 行。
- 日值：`020_11_hla_five_scenario_daily.csv`，4893 行。
- 冻结 manifest：`020_11_hla_nstep5_frozen_config.json`。
- 5 年 PNG/SVG/PDF 图均存在。
- 模型位于外部 seed0/1/2 训练根；正式选择点分别为 30K/10K/20K。
- 适合 report/evaluation reuse；无 replay buffer，不支持 exact resume。

### 7.2 YC/FQ 020_12–020_13

目录：`DSSAT_auto_validation/frozen_nstep_cross_site_020_12`

- 4229 files，约 91.18 MB。
- 汇总：`020_13_yc_fq_seed0_seed1_summary.csv`。
- 跨 seed：`020_13_yc_fq_cross_seed_dqn_comparison.csv`。
- YC2014、FQ2016 seed0/seed1 正式 selected model、summary 和 daily 均存在。
- YC seed0 应优先使用修正后的 `reaudit_operation_dap_020_12/checkpoint_35000/eval_daily.csv`。
- 020_13 本身无统一图/PPT，适合 plot/report-only 补建。
- 容器绝对路径需要 adapter 映射到项目相对路径。

### 7.3 LC

- `lc_pdi_initialization_rescue_017_10`：仅 baseline 诊断；可 report-only，不能自动映射正式 config_hash。
- `lc2010_baseline_relative_dqn_smoke_017_12`：seed0/1、5K、n_steps=1 旧 smoke；可登记 legacy 证据，不能复用为冻结正式 benchmark。

### 7.4 SY

- `sy_local_dqn_train_cross_year_transfer_017_08`：2014 seed0 50K 与 2012 transfer，旧 n_steps=1。
- `sy2014_dqn_resource_space_017_09`：四情景派生汇总/图，reward 为后处理 proxy。
- 可用于历史报告，不可自动视为冻结 n-step 配置。

### 7.5 020_14

- `docs/2026-07-11_020_14_five_site_current_status_and_priority.md`
- 仅作为五站点证据上下文，不是训练运行。

## 8. 结果复用等级

单一 `can_reuse` 不足以表达复用含义。registry 将同时记录 `reuse_scope`：

| scope | 含义 |
|---|---|
| `report` | 可直接用于汇总、表格、图和报告 |
| `evaluation` | 模型与输入身份可确认，可重新评估 |
| `warm_start` | 可加载模型权重继续训练，但不保证轨迹完全连续 |
| `exact_resume` | 必须同时恢复模型、replay buffer、训练步数和状态；历史结果目前均不满足 |
| `diagnostic` | 只作为历史诊断证据，不参与正式同配置比较 |

旧结果没有 `experiment_id/config_hash`。自动发现器只能建立 provenance index；只有当输入 hash、算法、reward、action、constraint、IC 和 Git/代码来源足够一致时，才允许自动跳过训练。

## 9. 可复用模块

- 冻结配置：`frozen_nstep_dqn_config_020_11.py`
- 状态抽取：`src/ppo_evaluate.py::latest_observation_dict/scalar`
- 动作归一化：`src/ppo_action_safety.py::normalize_action`
- raw gym/PDI 环境：`run_yc2014_linked_dqn_5k_multiseed_013_07.py`
- 站点输入审计/hash：`run_yc_fq_frozen_nstep_cross_site_020_12.py`
- local null：同文件 `run_local_null`
- checkpoint evaluation/MgmtEvent 审计：同文件 `evaluate_checkpoint`
- HLA 固定情景和图：`run_hla_five_scenario_completion_020_11.py`
- WUE/NUE 已核验口径：`src/calculate_five_site_wue_nue_from_summary_019_10.py`

## 10. 重复代码和潜在破坏风险

1. 35 个 DQN/变体训练脚本，reward/action/env/checkpoint 循环多次复制。
2. HLA 多个旧入口会 `shutil.rmtree(run_dir)`。
3. 模块级常量和 monkeypatch 可能跨实验串扰。
4. 旧结果列名不统一：`GWAD/HWAM/grnwt/final_gwad/final_grain_kg_ha` 等。
5. 旧路径混用 Windows、`/workspace` 和 `/workspaces/gym-dssat-pdi`。
6. 环境 seed 与 DQN seed 混用，不能仅从 env_args 推断 model seed。
7. 历史 checkpoint 没有 replay buffer，不能伪称 exact resume。
8. YC seed0 旧 daily 缺 `operation_dap`，必须使用已修正版。
9. LC/SY 旧结果不是冻结 n-step=5，不能按站点年份直接复用。
10. SY2014 IC 来源冲突尚未解决。
11. `requirements.txt` 未显式锁定 openpyxl、python-pptx、pytest 等报告/测试依赖。
12. 当前没有正式 `tests/` 测试体系。

## 11. 推荐重构方案

1. 新建 `benchmark/`，旧代码保持只读。
2. YAML 只保存项目相对路径；运行时映射 Windows/容器根。
3. 通过 `environment_adapter` 分别处理 HLA、YC/FQ、LC、SY。
4. 用 factory 固化 action、reward、constraint 元数据；第一版不改变数学逻辑。
5. 每个 site-year-seed 默认子进程隔离，避免全局常量污染。
6. 先实现 config、schema、hash、registry、manifest、dry-run 和 report-only。
7. 对新框架生成的 checkpoint 同时保存模型和 replay buffer；若状态不完整，标记 `partial_resume`。
8. 统一输出 schema，并对缺失字段填 NA，同时记录 source column mapping。
9. 旧结果仅建立可追溯索引；不为无法证明相同的结果生成虚假 config_hash 等价声明。
10. 报告能力在主机 Python 生成；DSSAT 训练/评估在指定容器 `/opt/gym_dssat_pdi/bin/python` 执行。

## 12. 本次实际修改范围

本任务将新增：

- `benchmark/` 配置、registry、runner、adapter、汇总、绘图和报告模块；
- `configs/` 默认配置、五站点配置、schema 和 021_01–021_08 模板；
- `tests/` 最小非 DSSAT 单元测试；
- `benchmark_results/` registry 与 smoke test 输出；
- `docs/` 审计、重构记录、PPT 和项目进度。

## 13. 本次明确不修改

- 不修改任何已有训练、评估、绘图脚本。
- 不修改冻结 reward。
- 不修改 IC 或站点输入数据。
- 不改写底层 gym-DSSAT / DSSAT-PDI。
- 不删除/覆盖已有模型、CSV、图、PPT、MD。
- 不启动五站点完整敏感性实验。
- 不把氮淋洗惩罚接入正式 reward。
- 不声称 5/16/25 动作已经验证。
- 不把 warm-start 写成 exact resume。
- 不把 SY2014 模糊输入写成已确认。

## 14. 审计后的实施判定

审计结论支持继续进行兼容式 Benchmark Framework 重构。首个真实 DSSAT smoke test采用 HLA2007、seed0、冻结 9 动作与冻结 reward，并使用低于正式 50K 的安全 timesteps；若容器或旧接口阻塞，则报告 `partial`，不进入完整训练。
