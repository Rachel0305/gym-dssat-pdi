# 021_00 DQN Benchmark Framework 重构记录

## 1. 状态

- 日期：2026-07-11
- 总体状态：`completed`
- 范围：论文实验前的配置、复用、续跑、评估和报告基础工程。
- 未做事项：未修改冻结 DQN、reward、IC；未启动五站点完整敏感性训练。

## 2. 背景与当前问题

项目已有 HLA、YC、FQ、LC、SY 多轮结果，但训练入口、评估入口、路径、列名和 checkpoint 规则分散在大量脚本中。继续复制脚本会增加输入错配、结果覆盖、配置漂移和无法复核的风险。

本任务先审计，再用新增 `benchmark/` 模块和 adapter 兼容旧主线。旧脚本、旧 reward、旧输入和旧结果均未删除或覆盖。

## 3. 审计结果

完整证据见 `docs/2026-07-11_021_00_benchmark_framework_audit.md`。核心结论：

1. `src/` 中有 35 个实际调用 DQN 学习的脚本，没有统一入口。
2. 冻结主线为 9 个离散水氮动作、I120/N300、共享 7 DAP 间隔、`n_steps=5` 和本地 null 相对终端奖励。
3. HLA、YC/FQ 可由 adapter 接入；LC 必须使用已验证输入修复；SY 的 IC 来源仍存在冲突。
4. 历史 checkpoint 没有 replay buffer，只能做评估或 warm-start，不能称为精确续跑。
5. 已索引 85 条历史证据，但只有来源和配置足够匹配时才允许自动复用。

## 4. 架构设计

```text
YAML -> merge/validate -> config_hash -> result registry
     -> dry-run / existing-result lookup
     -> train/resume -> deterministic evaluation
     -> CSV/Excel -> PNG/SVG -> Markdown/PPTX -> Git evidence
```

站点差异放入 `configs/sites/` 与 `environment_adapter.py`；冻结算法、奖励和约束由 factory 读取，旧代码只通过 adapter 调用。

## 5. 实现内容

| 功能 | 状态 | 证据 |
|---|---|---|
| YAML 配置、默认值、站点覆盖 | 已实现并测试 | `benchmark/config_loader.py`、`configs/` |
| experiment_id / config_hash | 已实现并测试 | host/container hash 完全一致 |
| 结果 registry 和历史发现 | 已实现并测试 | 85 条历史记录 + smoke registry |
| dry-run | 已实现并测试 | 容器内未启动训练 |
| train-only/evaluate-only/report-only | 已实现；train/report 已实测 | CLI 与 smoke manifest |
| checkpoint resume | 已实现并测试 | checkpoint 100 中断后续至 200 |
| 结果复用 | 已实现并测试 | 第二次运行状态 `reused`，未训练 |
| CSV、Excel、PNG、SVG | 已实现并测试 | smoke 输出目录 |
| Markdown、PPTX、QA | 已实现并测试 | 14 页 PPT，结构 QA 0 个高/中风险缺陷 |
| 5/16/25 动作敏感性 | 仅配置接口 | 非冻结动作训练尚未实现，不得写成已完成 |
| 五站点完整敏感性实验 | 未执行 | 本任务明确禁止启动 |

## 6. 配置与科学身份

`config_hash` 对合并后的科学配置、关键 MZX/WTH/CUL/SOL 文件哈希和 Git commit 建立身份。运行模式、输出位置等控制参数不冒充科学变量。站点统一使用 HLA/YC/FQ/LC/SY，并保留旧别名。

## 7. reward、action 与 IC

本任务没有修改以下冻结定义：

```text
R_t = -1.0 * irrigation_t - 5.0 * nitrogen_t
      + terminal * max(0, GWAD_final - GWAD_local_same_input_null)
```

- 动作：I={0,15,30} mm 与 N={0,50,100} kg/ha 的 9 个组合。
- 预算：I≤120 mm，N≤300 kg/ha；共享最小间隔 7 DAP；决策窗 DAP 1–120。
- IC：读取各站点现有确认输入；未在本任务中修改。

## 8. checkpoint 与 resume 边界

新框架保存模型、replay buffer、训练步数和 Python/NumPy/Torch 随机状态。本次恢复成功加载 replay buffer 和 RNG，从 100 步继续到 200 步。

状态必须写作 `partial_reproducible_checkpoint_resume`：DSSAT 进程在 episode 边界重建，不能声称是逐位完全相同的进程级恢复。旧模型因缺 replay buffer 也不能声称 exact resume。

## 9. smoke test

- 场景：HLA2007、seed0、200 timesteps。
- 容器：`b2fd6726c8c1`。
- Python：`/opt/gym_dssat_pdi/bin/python`。
- config_hash：`5fa1364a8add26121e19fd781c6dd5fff1a40d3229a6d5405ec95936fda4274a`。
- 结果：13 项检查全部通过。

验证链：dry-run → 100 步模拟中断 → 保存 model/replay/RNG → resume 到 200 → deterministic evaluation → host report-only → 再运行并复用 → 错误动作配置返回 exit 2。

该 smoke test 只验证工程管线，不作为 DQN 农艺性能证据。

## 10. 主要运行命令

```powershell
docker exec -w /workspace b2fd6726c8c1 /opt/gym_dssat_pdi/bin/python -m benchmark.benchmark_runner --config configs/experiments/021_00_framework_smoke.yaml --dry-run
docker exec -w /workspace b2fd6726c8c1 /opt/gym_dssat_pdi/bin/python -m benchmark.benchmark_runner --config configs/experiments/021_00_framework_smoke.yaml --train-only
docker exec -w /workspace b2fd6726c8c1 /opt/gym_dssat_pdi/bin/python -m benchmark.benchmark_runner --config configs/experiments/021_00_framework_smoke.yaml --train-only --resume
python -m benchmark.benchmark_runner --config configs/experiments/021_00_framework_smoke.yaml --report-only
python -m pytest tests/test_benchmark_core.py tests/test_benchmark_reporting.py -q
```

## 11. 输出文件

- 审计：`docs/2026-07-11_021_00_benchmark_framework_audit.md`
- 框架：`benchmark/`
- 配置：`configs/`
- 测试：`tests/`
- 历史索引：`benchmark_results/existing_results_registry.csv`
- smoke：`benchmark_results/021_00_hla2007_framework_smoke__hla_2007_seed0/`
- 本记录：`docs/2026-07-11_021_00_benchmark_framework_refactor_record.md`
- 汇报：`docs/2026-07-11_021_00_benchmark_framework_refactor_record.pptx`

模型、replay buffer、RNG 和 DSSAT 临时输出保留在本地但不提交 Git。

## 12. 已解决问题

1. 一个 YAML 对应一个可追溯实验身份。
2. 统一 dry-run、训练、评估、报告与复用入口。
3. 新实验具备可验证的 checkpoint 续跑能力。
4. 旧结果可发现、可分级复用，不再一律重跑。
5. CSV/图/报告由同一 manifest 追溯，减少图表与数据错配。

## 13. 未解决问题与限制

1. SY 权威 IC=2 输入尚未确认，正式配置保持 blocked/ambiguous。
2. LC/SY 尚未完成冻结 n-step=5、50K 正式训练。
3. 5/16/25 动作、淋洗 reward、观测消融和预算敏感性只有模板，尚未运行。
4. ET 和 crop N uptake 尚未统一接入当前 DQN evaluation；`WP_ET`、`NUtE` 可能为 NA，不能伪造。
5. `evaluate-only` 已实现但本次未单独执行；`--force` 接口也未做破坏性覆盖测试。
6. 容器缺少 `python-pptx`，因此训练/评估在容器，报告在主机生成；这是有意的依赖隔离。
7. 未做五站点完整敏感性或新长训练。

## 14. 后续实验计划

1. 先用 `021_08_paper_report_generator.yaml` 对已有冻结结果做 report-only 统一汇总。
2. 在不改 reward/IC 的前提下，依次运行 021_01–021_05 单因素敏感性；每次先 dry-run 和 smoke。
3. 再执行 021_06 跨年份、021_07 跨站点验证。
4. 在正式跨站点训练前解决 SY IC provenance，并为 LC/SY建立冻结输入。
5. 增加 Summary.OUT 的 ET 和 N uptake adapter 后再正式报告 WP_ET 与 NUtE。

## 15. Methods Source

- DQN 与 target network：Mnih et al. (2015)。
- experience replay：Lin (1992)。
- multi-step Q-learning：Peng and Williams (1996)；本项目 `n_steps=5` 是冻结工程配置，不声称来自该论文的特定推荐值。
- 实现库：Stable-Baselines3（Raffin et al., 2021）。
- 农田环境：DSSAT（Jones et al., 2003）与 gym-DSSAT（Gautron et al., 2022）。
- 多 seed 和可复现性纪律：Henderson et al. (2018)。
- 作物管理 RL 参照：Tao et al. (2023)。

## 16. References

1. Mnih V, et al. Human-level control through deep reinforcement learning. *Nature* 518, 529–533 (2015). https://doi.org/10.1038/nature14236
2. Lin L-J. Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning* 8, 293–321 (1992). https://doi.org/10.1007/BF00992699
3. Peng J, Williams RJ. Incremental Multi-Step Q-Learning. *Machine Learning* 22, 283–290 (1996). https://doi.org/10.1007/BF00114731
4. Raffin A, et al. Stable-Baselines3. *JMLR* 22(268), 1–8 (2021).
5. Henderson P, et al. Deep Reinforcement Learning that Matters. *AAAI* 32 (2018). https://doi.org/10.1609/aaai.v32i1.11694
6. Jones JW, et al. The DSSAT cropping system model. *European Journal of Agronomy* 18, 235–265 (2003). https://doi.org/10.1016/S1161-0301(02)00107-7
7. Gautron R, et al. gym-DSSAT: a crop model turned into a Reinforcement Learning environment. HAL `hal-03711132` (2022).
8. Tao R, et al. Optimizing Crop Management with Reinforcement Learning and Imitation Learning. *IJCAI-23*, 6228–6236 (2023). https://doi.org/10.24963/IJCAI.2023/691

## 17. Git 信息

- 重构前备份提交：`b304fdf`，已推送。
- 框架实现提交：`3bec052`（`Add configurable DQN benchmark framework`）。
- 正式记录与 PPT 首次提交：`ad8a3dc`（`Document benchmark framework refactor`）。
- push 状态：框架提交已成功推送到 `codex-reward-sweep-backup`。
