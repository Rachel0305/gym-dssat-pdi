# 021_00 Benchmark Framework Refactor

## 任务背景

当前项目已经完成五个站点的多轮 DQN 实验，并形成了 HLA、YC、FQ、LC、SY 的阶段性证据。现有实验脚本、评估脚本、绘图脚本和结果目录较多，后续还计划开展动作空间、奖励成本、预算、决策间隔、观测变量消融、跨年份泛化和跨站点迁移等论文级实验。

本任务的目标不是立即开展新的大规模敏感性训练，而是先把现有项目重构为一个统一、可配置、可断点续跑、可复用已有结果、可自动生成论文材料的 Benchmark 实验框架。后续所有敏感性实验应尽量通过新增或修改配置文件完成，而不是复制大量训练、评估和绘图代码。

本任务必须优先保护已有可运行代码和已有实验结果，不允许为了重构破坏现有主线。

---

## 总体目标

构建一个统一的 DQN Benchmark 实验平台，使其至少支持以下能力：

1. 通过 YAML 或 JSON 配置实验。
2. 自动训练、评估、汇总、绘图和生成报告。
3. 自动发现并复用已有实验结果。
4. 自动避免重复训练同一配置。
5. 支持中断后继续运行。
6. 支持五个站点和多个年份、多个 seed。
7. 支持动作空间、reward、budget、decision interval、observation ablation 等实验类型。
8. 统一输出 CSV、Excel、PNG、SVG、Markdown 和 PPT。
9. 自动记录软件环境、Git commit、配置文件和运行命令。
10. 自动备份关键代码、配置和文档到 GitHub。

---

# 1. 开始前审计

在修改任何代码前，必须先完成项目审计。

## 1.1 扫描内容

扫描并整理以下内容：

1. 当前项目根目录结构。
2. 所有 DQN 训练脚本。
3. 所有评估脚本。
4. 所有绘图脚本。
5. 所有 reward 实现。
6. 所有动作空间定义。
7. 所有预算和窗口约束实现。
8. HLA、YC、FQ、LC、SY 的输入配置。
9. 已有 checkpoint。
10. 已有 summary CSV、history、模型文件、PPT 和 Markdown。
11. 当前冻结的 n-step DQN 设置。
12. 现有 Git 状态和当前分支。

## 1.2 审计输出

生成：

```text
docs/2026-07-11_021_00_benchmark_framework_audit.md
```

审计报告至少包含：

1. 当前训练入口。
2. 当前评估入口。
3. 当前结果目录结构。
4. 可复用模块。
5. 重复代码位置。
6. 潜在破坏风险。
7. 已有结果复用策略。
8. 推荐重构方案。
9. 本次实际修改范围。
10. 明确列出本次不会修改的部分。

必须先完成审计，再开始重构。

---

# 2. 安全原则

必须遵守以下要求：

1. 不覆盖已有实验结果。
2. 不删除已有训练脚本。
3. 不删除已有模型、CSV、PPT、图片或 Markdown。
4. 不直接改坏当前能够运行的五站点主线。
5. 优先新增模块，再逐步接入旧代码。
6. 对必须修改的关键文件，先备份。
7. 所有备份放入带日期和任务编号的目录。
8. 任何训练开始前先进行 smoke test。
9. 若现有代码接口不统一，使用 adapter 或 wrapper 兼容，不要大规模重写底层 gym-DSSAT 环境。
10. 若发现现有结果已经满足某个配置，不重复训练。
11. 若某个步骤失败，记录错误并继续处理其他可执行部分。
12. 不允许因为单个站点失败导致整个任务完全终止。
13. 所有异常必须记录到最终实验报告。

建议备份目录：

```text
backups/2026-07-11_021_00/
```

---

# 3. 目标目录结构

优先建立以下结构；若项目已有类似结构，可在保持兼容的前提下调整，但必须在文档中说明。

```text
benchmark/
├── __init__.py
├── benchmark_runner.py
├── config_loader.py
├── experiment_registry.py
├── result_registry.py
├── train_runner.py
├── evaluation_runner.py
├── summary_builder.py
├── plot_builder.py
├── statistics.py
├── report_builder.py
├── ppt_builder.py
├── environment_adapter.py
├── action_space_factory.py
├── observation_factory.py
├── reward_factory.py
├── constraint_factory.py
├── checkpoint_manager.py
├── provenance.py
└── validators.py

configs/
├── benchmark_defaults.yaml
├── sites/
│   ├── hla.yaml
│   ├── yca.yaml
│   ├── fqa.yaml
│   ├── lca.yaml
│   └── sya.yaml
├── experiments/
│   ├── action_space_template.yaml
│   ├── reward_cost_template.yaml
│   ├── budget_template.yaml
│   ├── decision_interval_template.yaml
│   ├── observation_ablation_template.yaml
│   ├── cross_year_template.yaml
│   └── cross_site_template.yaml
└── schemas/
    └── benchmark_config_schema.yaml

benchmark_results/
└── <experiment_id>/
    ├── configs/
    ├── logs/
    ├── checkpoints/
    ├── evaluations/
    ├── summaries/
    ├── figures/
    ├── tables/
    ├── reports/
    └── manifests/
```

如果不能完全采用以上结构，至少必须实现同等功能。

---

# 4. 配置系统

## 4.1 配置格式

使用 YAML 作为主要配置格式。

至少支持以下字段：

```yaml
experiment:
  experiment_id: "021_00_smoke"
  experiment_type: "framework_validation"
  description: "Benchmark framework validation"
  output_root: "benchmark_results"
  reuse_existing_results: true
  resume: true
  dry_run: false

site:
  station_code: "HLA"
  station_name: "Hailun"
  train_years: [2007]
  eval_years: [2007]
  fileX_template_path: ""
  weather_path: ""
  soil_path: ""
  cultivar_path: ""
  initial_condition_mode: 1

algorithm:
  name: "DQN"
  total_timesteps: 50000
  learning_rate: 0.0001
  buffer_size: 100000
  learning_starts: 1000
  batch_size: 64
  gamma: 0.99
  target_update_interval: 1000
  train_freq: 4
  gradient_steps: 1
  n_steps: 5
  seed: 0
  checkpoint_interval: 5000

action_space:
  irrigation_levels: [0, 15, 30]
  nitrogen_levels: [0, 50, 100]

constraints:
  irrigation_budget: 120.0
  nitrogen_budget: 300.0
  daily_irrigation_cap: 30.0
  daily_nitrogen_cap: 100.0
  min_interval_days: 7
  decision_window_start_dap: 1
  decision_window_end_dap: 120

reward:
  type: "local_null_terminal"
  irrigation_cost: 1.0
  nitrogen_cost: 5.0
  yield_gain_coefficient: 1.0
  leaching_cost: 0.0

observations:
  include:
    - dap
    - dtt
    - ep
    - grnwt
    - istage
    - nstres
    - rtdep
    - srad
    - sw
    - swfac
    - tmax
    - topwt
    - totir
    - vstage
    - wtdep
    - xlai

evaluation:
  seeds: [0, 1, 2]
  baselines:
    - null
    - recorded
    - dssat_auto
    - official_extension_expert
    - dqn
  save_daily_trajectory: true
  save_action_history: true

reporting:
  generate_markdown: true
  generate_pptx: true
  generate_excel: true
  generate_png: true
  generate_svg: true
  dpi: 300
```

以上只是结构示例。实际值必须以项目当前冻结配置和现有站点文件为准，不允许凭空编造路径。

## 4.2 配置校验

必须实现配置校验，至少检查：

1. 文件路径是否存在。
2. station code 是否有效。
3. train_years 和 eval_years 是否存在。
4. 动作值是否非负。
5. 最大动作是否超过 daily cap。
6. budget 是否小于单次动作。
7. `min_interval_days` 是否为正整数。
8. seed 是否为整数。
9. 输出目录是否会覆盖旧结果。
10. reward 参数是否完整。
11. observation 名称是否存在于环境输出。
12. n-step 设置是否与当前实现兼容。

配置错误时应给出清晰错误信息，不能静默失败。

---

# 5. 实验唯一标识和结果复用

## 5.1 配置哈希

为每个实验生成稳定的 `config_hash`。

哈希内容至少包括：

1. site。
2. year。
3. seed。
4. algorithm 参数。
5. action space。
6. constraints。
7. reward。
8. observations。
9. 初始条件模式。
10. 关键输入文件路径及文件摘要。
11. 当前 Git commit。

## 5.2 复用规则

开始训练前必须检查：

1. 是否已有相同 `config_hash` 的完整结果。
2. 是否已有相同配置的 checkpoint。
3. 是否只有评估结果缺失。
4. 是否只有报告或图片缺失。
5. 是否已有旧目录中可复用的结果。

处理逻辑：

- 完整结果存在：直接复用。
- 模型存在但评估缺失：只评估。
- checkpoint 存在：断点续跑。
- CSV 存在但图片缺失：只生成图片。
- 结果不完整且无法确认一致：重新训练，但保留旧结果。
- 任何复用必须写入 manifest。

---

# 6. 断点续跑

必须支持：

1. 训练 checkpoint 恢复。
2. 单站点失败后重新执行。
3. 单 seed 失败后重新执行。
4. 单年份失败后重新执行。
5. 报告生成失败后单独重跑。
6. 图片生成失败后单独重跑。
7. 使用 manifest 标记状态：

```text
pending
running
completed
failed
reused
partial
```

程序再次启动时，应自动跳过 `completed` 和 `reused`，继续处理 `pending`、`failed` 和 `partial`。

---

# 7. Benchmark Runner

实现统一入口，例如：

```bash
python -m benchmark.benchmark_runner \
  --config configs/experiments/action_space_template.yaml
```

至少支持：

```bash
--config
--dry-run
--resume
--force
--train-only
--evaluate-only
--report-only
--plot-only
--site
--year
--seed
```

## 7.1 dry-run

`--dry-run` 必须输出：

1. 将运行多少实验。
2. 哪些实验会训练。
3. 哪些实验会复用。
4. 预计涉及哪些站点、年份和 seed。
5. 输出目录。
6. 配置错误。
7. 不得真正开始训练。

## 7.2 force

`--force` 可以重新运行，但不能删除原结果。应新建带时间戳的输出目录，并记录它是强制重跑。

---

# 8. 训练模块

训练模块必须：

1. 兼容当前 Stable-Baselines3 DQN。
2. 保留当前冻结 n-step DQN 主线。
3. 保留现有 9 动作设置。
4. 支持后续 16、25 个动作。
5. 支持多个 seed。
6. 支持多个站点和年份。
7. 自动保存 checkpoint。
8. 保存训练日志。
9. 保存最终模型。
10. 保存 replay buffer，若当前实现允许且文件大小可接受。
11. 保存完整配置。
12. 保存训练时间。
13. 保存异常栈。
14. 不能把某个站点配置硬编码到公共模块。
15. 不能假设所有站点使用相同 IC。

---

# 9. 评估模块

评估必须统一输出以下核心字段，若环境不存在某字段则记录为缺失，不允许伪造：

```text
experiment_id
config_hash
station_code
year
seed
scenario
checkpoint
dap
date
istage
vstage
topwt
grnwt
xlai
swfac
nstres
totir
cumulative_irrigation
cumulative_nitrogen
action_irrigation
action_nitrogen
raw_action
executed_action
reward
terminal_reward
final_yield
```

季节汇总至少包含：

```text
yield_kg_ha
irrigation_mm
nitrogen_kg_ha
reward_total
number_of_irrigation_events
number_of_nitrogen_events
water_budget_use_ratio
nitrogen_budget_use_ratio
WUE
NUE
yield_per_irrigation
yield_per_nitrogen
```

注意：

1. WUE 和 NUE 必须在文档中给出明确公式。
2. 若某个分母为 0，应使用合理的 NA 规则，不允许产生误导性无穷值。
3. recorded、DSSAT auto、official expert 与 DQN 的指标定义必须一致。
4. baseline 必须优先复用 same-input 结果。
5. local null 必须对应同一站点、同一年份和同一输入条件。

---

# 10. 汇总和统计

统一生成：

```text
season_summary.csv
daily_trajectory.csv
action_summary.csv
cross_seed_summary.csv
baseline_comparison.csv
failure_summary.csv
result_manifest.csv
```

并生成：

```text
benchmark_summary.xlsx
```

Excel 至少包含以下 sheet：

```text
season_summary
daily_trajectory_index
action_summary
cross_seed_summary
baseline_comparison
config_index
failure_summary
reused_results
```

统计至少包括：

1. Mean。
2. Standard deviation。
3. Minimum。
4. Maximum。
5. Median。
6. 跨 seed 稳定性。
7. 相对 null 的产量增益。
8. 相对 recorded 的产量差。
9. 相对 DSSAT auto 的产量差。
10. 相对 official expert 的产量差。
11. 水投入差。
12. 氮投入差。
13. 预算使用率。
14. 动作频次。
15. 非零动作比例。

不要强行进行不适合小样本的显著性检验。若统计检验前提不满足，应在报告中明确说明。

---

# 11. 绘图模块

所有图同时输出：

```text
PNG
SVG
```

PNG 要求：

```text
300 dpi
```

至少支持：

1. 训练 reward 曲线。
2. 跨 seed 产量均值和标准差。
3. 灌溉量比较。
4. 施氮量比较。
5. WUE 比较。
6. NUE 比较。
7. 产量-灌溉 Pareto 图。
8. 产量-施氮 Pareto 图。
9. 水氮二维动作热图。
10. 动作频次柱状图。
11. DAP-动作时间线。
12. SWFAC/NSTRES 与动作关系图。
13. 不同 baseline 对比图。
14. budget 使用率图。
15. 失败与缺失结果概览图。

绘图要求：

1. 图例名称统一。
2. 单位完整。
3. 不混淆 `GWAD`、`grnwt` 和最终产量。
4. 不同站点可以分图，避免信息过密。
5. 图标题和轴标签使用英文，便于论文使用。
6. Markdown 和 PPT 中对图进行中文解释。
7. 不因缺少某个 baseline 而让整个绘图流程崩溃。
8. 所有图对应的数据必须可追溯到 CSV。

---

# 12. 报告生成

本任务必须生成两个正式实验记录文件：

```text
docs/2026-07-11_021_00_benchmark_framework_refactor_record.md
docs/2026-07-11_021_00_benchmark_framework_refactor_record.pptx
```

## 12.1 Markdown 内容

至少包含：

1. 任务背景。
2. 当前问题。
3. 原项目结构审计。
4. 重构目标。
5. 新目录结构。
6. 配置系统。
7. 配置哈希和结果复用。
8. 断点续跑逻辑。
9. 训练模块。
10. 评估模块。
11. 统计模块。
12. 绘图模块。
13. 报告模块。
14. smoke test。
15. 实际运行命令。
16. 实际生成文件。
17. 成功项。
18. 失败项。
19. 已知限制。
20. 对已有结果是否产生影响。
21. 后续如何开展 `021_01` 至 `021_08`。
22. Methods Source。
23. References。
24. Git commit 和 push 状态。
25. 最终结论。

报告必须明确区分：

- 已实现。
- 已测试。
- 仅预留接口。
- 尚未实现。
- 因环境或数据限制无法验证。

不允许把“计划支持”写成“已经支持”。

## 12.2 PPT 内容

PPT 至少包含：

1. Title。
2. Why refactor。
3. Current workflow problems。
4. Target architecture。
5. Configuration-driven experiment design。
6. Result reuse and checkpoint resume。
7. Unified output structure。
8. Smoke test。
9. Example command。
10. Example manifest。
11. Risks and safeguards。
12. What is ready for the next experiment。
13. Next steps。
14. References。

PPT 中必须有至少一张流程图，展示：

```text
config
→ validation
→ existing-result lookup
→ train/resume
→ evaluate
→ summarize
→ plot
→ Markdown/PPT
→ Git backup
```

---

# 13. Methods Source 和参考文献

在实验记录中增加：

```text
Methods Source
References
```

至少说明以下方法来源：

1. DQN。
2. Experience replay。
3. Target network。
4. n-step return。
5. 可复现实验中的 seed 和 checkpoint 设计。
6. 消融实验与敏感性实验的一般方法。
7. 农业强化学习或作物管理强化学习中的相关设计。
8. Stable-Baselines3 官方文档。
9. gym-DSSAT 或 DSSAT-PDI 相关来源。

要求：

1. 优先引用原始论文和官方文档。
2. 不允许编造参考文献。
3. 引用条目必须包含作者、年份、题目、来源。
4. 对暂时无法确认的参考文献标记为待核验，不得伪造 DOI。
5. 报告中说明哪些设计来自文献，哪些是本项目工程选择。

---

# 14. Smoke Test

重构完成后，必须先执行 smoke test，不得直接启动大规模实验。

建议 smoke test：

```text
station: HLA
year: 2007
seed: 0
total_timesteps: 5000 或更低的安全值
action space: 当前冻结 9 动作
reward: 当前冻结 reward
```

smoke test 必须验证：

1. 配置能读取。
2. 环境能创建。
3. 模型能训练。
4. checkpoint 能保存。
5. 评估能执行。
6. CSV 能输出。
7. PNG/SVG 能输出。
8. Markdown 能生成。
9. PPT 能生成。
10. manifest 状态正确。
11. 第二次运行能识别并复用已有结果。
12. 人为中断后可以 resume。
13. `--dry-run` 不启动训练。
14. 错误配置能被拦截。

如果 smoke test 失败，优先修复框架，不进入大规模训练。

---

# 15. 与已有实验结果的连接

必须尽可能建立已有结果索引，至少覆盖：

1. HLA 020_11。
2. YC/FQ 020_13。
3. LC 017_10/017_12。
4. SY 017_08/017_09。
5. 当前五站点状态总结 020_14。

生成已有结果索引：

```text
benchmark_results/existing_results_registry.csv
```

至少包含：

```text
source_experiment
station_code
year
seed
scenario
model_path
summary_path
trajectory_path
figure_path
status
can_reuse
reuse_reason
notes
```

若旧结果无法自动对应新配置，记录原因，不要强行认定相同。

---

# 16. 后续实验模板

完成框架后，必须生成以下可编辑模板，但不要在本任务中自动跑完整大实验：

```text
configs/experiments/021_01_action_space_sensitivity.yaml
configs/experiments/021_02_reward_cost_sensitivity.yaml
configs/experiments/021_03_budget_sensitivity.yaml
configs/experiments/021_04_decision_interval_sensitivity.yaml
configs/experiments/021_05_observation_ablation.yaml
configs/experiments/021_06_cross_year_benchmark.yaml
configs/experiments/021_07_cross_site_transfer.yaml
configs/experiments/021_08_paper_report_generator.yaml
```

每个模板必须：

1. 包含说明。
2. 默认 `dry_run: true`。
3. 不自动覆盖结果。
4. 标明需要用户确认的参数。
5. 给出示例运行命令。
6. 明确哪些配置继承 `benchmark_defaults.yaml`。

---

# 17. 代码质量

要求：

1. Python 文件包含必要 docstring。
2. 关键函数包含类型注解。
3. 路径使用 `pathlib.Path`。
4. 避免在代码中硬编码站点路径。
5. 使用 logging，不要只依赖 print。
6. 错误信息必须包含 experiment_id、station、year、seed。
7. 公共模块尽量避免循环依赖。
8. 对配置解析、哈希、结果注册和状态恢复编写最小测试。
9. 测试不应依赖完整 DSSAT 长训练。
10. 保留兼容旧代码的入口说明。
11. 不进行与本任务无关的大规模格式化。
12. 不批量修改所有旧脚本。

---

# 18. 项目进度记录

更新或创建：

```text
docs/project_progress.md
```

新增 `021_00` 记录，至少包含：

1. 日期。
2. 任务名称。
3. 状态。
4. 主要输出。
5. smoke test 状态。
6. Git commit。
7. 下一任务建议。
8. 未解决问题。

若项目已有其他进度索引，也应同步更新，但不能删除旧记录。

---

# 19. GitHub 备份

完成后执行：

```bash
git status
git diff --stat
```

确认不包含以下内容：

1. 大型模型文件。
2. replay buffer 大文件。
3. 临时缓存。
4. DSSAT 临时运行目录。
5. 原始私有数据。
6. 无关的系统文件。

必要时更新 `.gitignore`。

然后执行：

```bash
git add <本任务相关文件>
git commit -m "Add configurable DQN benchmark framework"
git push
```

要求：

1. 不使用 `git add .`，除非确认没有无关改动。
2. 不覆盖用户已有未提交修改。
3. 如果 push 失败，保留 commit，并把失败原因和后续命令写入实验记录。
4. 在 Markdown 和 PPT 中记录 commit hash。
5. 备份关键配置、脚本、Markdown 和 PPT。
6. 不把大型训练结果强制推送到 GitHub。

---

# 20. 完成标准

只有同时满足以下条件，才能将本任务标记为完成：

1. 已完成项目审计。
2. 已备份关键文件。
3. 已建立统一 Benchmark 目录。
4. 已实现配置读取和校验。
5. 已实现配置哈希。
6. 已实现结果注册和复用。
7. 已实现断点续跑。
8. 已实现统一训练入口。
9. 已实现统一评估入口。
10. 已实现统一汇总输出。
11. 已实现 PNG 和 SVG 绘图。
12. 已实现 Markdown 生成。
13. 已实现 PPT 生成。
14. 已通过至少一个 smoke test。
15. 已验证第二次运行不会重复训练。
16. 已生成后续实验 YAML 模板。
17. 已生成正式实验记录 MD。
18. 已生成正式实验记录 PPT。
19. 已更新项目进度文档。
20. 已完成 Git commit。
21. 已尝试 Git push。
22. 已清楚记录尚未完成或未验证的功能。

---

# 21. 最终终端输出

任务结束时，在终端打印简洁摘要：

```text
021_00 Benchmark Framework Refactor Completed

Audit:
- ...

Framework:
- ...

Smoke test:
- ...

Existing results reused:
- ...

Generated files:
- ...

Failed or partial items:
- ...

Git commit:
- ...

Git push:
- ...

Recommended next prompt:
prompts/021_01_action_space_sensitivity_benchmark.md
```

不得只打印“完成”，必须列出实际状态。

---

# 22. 本任务禁止事项

1. 不直接启动全部五站点、多年份、多 seed 的完整敏感性训练。
2. 不修改初始条件来制造优化空间。
3. 不修改冻结 reward 作为本任务主要内容。
4. 不引入氮淋洗惩罚作为正式主线。
5. 不删除旧脚本。
6. 不覆盖旧结果。
7. 不把未验证功能写成已完成。
8. 不伪造 WUE、NUE、产量或 baseline 数据。
9. 不伪造参考文献。
10. 不因追求统一而破坏站点特定配置。
11. 不重复运行已有完整实验。
12. 不忽略失败记录。
13. 不提交大型临时文件到 GitHub。

---

## 执行优先级

按以下顺序执行：

1. 审计。
2. 备份。
3. 设计目录和接口。
4. 配置系统。
5. 哈希、注册和复用。
6. 断点续跑。
7. 训练和评估适配。
8. 汇总、绘图和报告。
9. smoke test。
10. 二次运行复用验证。
11. 后续 YAML 模板。
12. 实验记录 MD。
13. 实验记录 PPT。
14. 更新进度。
15. Git commit 和 push。

若时间或资源不足，必须优先保证：

```text
审计
→ 配置系统
→ 结果注册
→ 断点续跑
→ smoke test
→ MD 记录
→ Git 备份
```

并把未完成部分明确记录为 `partial`，不要用临时代码伪装完成。
