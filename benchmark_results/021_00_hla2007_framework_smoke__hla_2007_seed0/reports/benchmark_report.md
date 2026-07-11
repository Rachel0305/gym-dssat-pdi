# 021_00 Benchmark Framework Refactor 实验记录

- experiment_id: `021_00_hla2007_framework_smoke`
- config_hash: `5fa1364a8add26121e19fd781c6dd5fff1a40d3229a6d5405ec95936fda4274a`
- 总体状态: **reused**

## 背景

- 五站点 DQN 证据已形成，但训练、评估、绘图和报告入口分散。
- 本框架在不改 reward、IC 和旧结果的前提下统一控制面。

## 当前问题

- 历史脚本存在复制、全局常量耦合和输出 schema 不统一。
- 历史模型没有 replay buffer，不能声称 exact resume。
- SY2014 IC 来源仍冲突，正式训练被配置层阻塞。

## 审计结果

- 审计证据未注入；不得据此声称审计完成。

## 架构设计

- YAML config → validation → existing-result lookup → train/resume
- train/resume → evaluate → summarize → plot → Markdown/PPT → Git backup
- 旧脚本通过 adapter 接入；底层 gym-DSSAT 与已验证 reward/IC 不在本任务中改动。

## 实现内容与状态

| 功能 | 状态 | 证据或说明 |
|---|---|---|
| 未提供 | 未验证 | 未提供实现状态 |

## 配置系统

- 冻结主线：9 动作、I120/N300、共享 7 DAP、n_steps=5。
- reward：local-null terminal yield gain - water cost - nitrogen cost。
- 站点路径使用项目相对路径；逐年 prepared input 参与 hash。

## 配置哈希与结果复用

- config_hash 完全匹配且 manifest completed/reused 时跳过训练。
- 旧结果按 report/evaluation/warm_start/exact_resume 分级，不强行认定等价。
- 020_11、020_13、LC/SY legacy 证据已写入 existing-results registry。

## Checkpoint resume 与运行模式

- 新 checkpoint 保存 model、replay buffer、RNG state 和 training_state。
- 恢复在 episode/checkpoint 边界重建 DSSAT 环境，标记 partial reproducible resume，而非 bitwise exact continuation。

## 训练与评估适配

- 运行模式：report-only
- 训练结果：{'daily_path': 'benchmark_results/021_00_hla2007_framework_smoke__hla_2007_seed0/evaluations/daily_trajectory.csv', 'model_path': 'benchmark_results/021_00_hla2007_framework_smoke__hla_2007_seed0/checkpoints/checkpoint_200/model.zip', 'replay_buffer_loaded': True, 'resume_semantics': 'partial_reproducible_checkpoint_resume', 'resumed_from': 100, 'rng_state_loaded': True, 'season_path': 'benchmark_results/021_00_hla2007_framework_smoke__hla_2007_seed0/evaluations/season_summary.csv', 'selected_checkpoint': 200, 'status': 'completed'}
- 所有 daily/season 字段由统一 schema adapter 生成；缺失值保持 NA。

## 统计模块

- WP_ET = yield / ET / 10；IWP_gross = yield / irrigation / 10。
- PFP_N = yield / applied N；NUtE = yield / crop N uptake。
- 分母为 0 或缺失时记录为 NA，不生成无穷值。
- 小样本仅报告 mean、standard deviation、minimum、maximum、median 与跨 seed CV，不强行进行显著性检验。

### 已生成汇总表

| 表 | 行数 | 列数 |
|---|---:|---:|
| `action_summary` | 1 | 13 |
| `baseline_comparison` | 4 | 17 |
| `cross_seed_summary` | 1 | 46 |
| `daily_trajectory` | 144 | 26 |
| `failure_summary` | 0 | 10 |
| `result_manifest` | 6 | 6 |
| `season_summary` | 1 | 24 |

## 绘图模块

- Python/matplotlib 后端；PNG 300 dpi 与可编辑 SVG 成对输出。
- 图中英文标题与单位；缺失 baseline 或缺失指标时显示不可用，不伪造数值。

## 报告模块

- Markdown 与 PPTX 从同一 config、manifest、表格和图形清单生成。
- PPTX 至少包含 config → validation → reuse → train/resume → evaluate → report → Git 的流程图。

## Smoke test

- 状态: **passed**
- 配置读取与校验通过
- HLA2007 gym-DSSAT 环境创建通过
- DQN 在 checkpoint 100 后按设计中断
- model、replay buffer、RNG 与 checkpoint 保存通过
- 从 checkpoint 100 恢复并训练到 200
- deterministic checkpoint 评估通过
- 统一 CSV 与 Excel 输出通过
- PNG 300 dpi 与 SVG 成对输出通过
- Markdown 与 14 页 PPTX 生成通过
- manifest 状态转换通过
- 第二次 train-only/report-only 复用且未训练
- dry-run 未启动训练
- 错误动作配置被拦截并返回 exit 2

## 实际运行命令

- python -m benchmark.benchmark_runner --config configs/experiments/021_00_framework_smoke.yaml --report-only

## 输出文件

- `figures.action_timeline`: `C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi\benchmark_results\021_00_hla2007_framework_smoke__hla_2007_seed0\figures\action_timeline.png`
- `figures.baseline_comparison`: `C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi\benchmark_results\021_00_hla2007_framework_smoke__hla_2007_seed0\figures\baseline_comparison.png`
- `figures.efficiency_comparison`: `C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi\benchmark_results\021_00_hla2007_framework_smoke__hla_2007_seed0\figures\efficiency_comparison.png`
- `figures.resource_comparison`: `C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi\benchmark_results\021_00_hla2007_framework_smoke__hla_2007_seed0\figures\resource_comparison.png`
- `figures.reward_curve`: `C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi\benchmark_results\021_00_hla2007_framework_smoke__hla_2007_seed0\figures\reward_curve.png`
- `figures.yield_resource_pareto`: `C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi\benchmark_results\021_00_hla2007_framework_smoke__hla_2007_seed0\figures\yield_resource_pareto.png`

## 已解决问题

- 未提供已验证解决项。

## 未解决问题与已知限制

- 未提供；不等于不存在限制。

## 对已有结果的影响

- 未提供影响评估。

## 后续实验计划

- 先完成并复核 021_00 smoke、resume、reuse 和 invalid-config tests。
- 随后只在用户确认后 dry-run 021_01–021_08。

## Methods Source

- Experience replay 可追溯至 Lin (1992)；DQN 与独立 target network 采用 Mnih et al. (2015)。
- n-step/multi-step Q-learning 依据 Peng and Williams (1996)；本项目冻结 n_steps=5 是工程配置。
- 工程实现采用 Stable-Baselines3 DQN (Raffin et al., 2021)，并由 adapter 保留旧 gym-DSSAT 主线。
- gym-DSSAT / DSSAT-PDI 依据 Gautron et al. (2022) 与官方技术文档；YAML、hash、registry 和自动报告是本项目工程选择。

## References

1. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529–533. https://doi.org/10.1038/nature14236
2. Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine Learning, 8, 293–321. https://doi.org/10.1007/BF00992699
3. Peng, J., & Williams, R. J. (1996). Incremental Multi-Step Q-Learning. Machine Learning, 22, 283–290. https://doi.org/10.1007/BF00114731
4. Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research, 22(268), 1–8.
5. Henderson, P., Islam, R., Bachman, P., et al. (2018). Deep Reinforcement Learning that Matters. AAAI, 32, 3207–3214. https://doi.org/10.1609/aaai.v32i1.11694
6. Jones, J. W., Hoogenboom, G., Porter, C. H., et al. (2003). The DSSAT cropping system model. European Journal of Agronomy, 18, 235–265. https://doi.org/10.1016/S1161-0301(02)00107-7
7. Gautron, R., Padrón, E. J., Preux, P., Bigot, J., Maillard, O.-A., & Emukpere, D. (2022). gym-DSSAT: a crop model turned into a Reinforcement Learning environment. Inria Research Report RR-9460, HAL hal-03711132.
8. Tao, R., Zhao, P., Wu, J., et al. (2023). Optimizing Crop Management with Reinforcement Learning and Imitation Learning. IJCAI-23, 6228–6236. https://doi.org/10.24963/IJCAI.2023/691

## Git 状态

- commit: `未提交或未提供`
- push: `未尝试或未提供`

## 最终结论

- 当前任务状态为 reused；只把有测试证据的功能视为已完成。
