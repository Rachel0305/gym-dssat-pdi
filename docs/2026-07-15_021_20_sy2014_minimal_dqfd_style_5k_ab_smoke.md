# 021_20 SY2014 最小 DQfD-style 5K seed0 A/B smoke 记录

## 背景与目标

021_18 已证明在冻结 SY2014 环境内存在 HWAM=11205 kg/ha、I75/N200 的确定性 oracle；021_19 又证明简单逆频率行为克隆虽然识别了5个非零动作，却把大量 no-op 状态误判为操作，62.5%准确率低于“永远预测 no-op”的96.875%基线。本轮因此测试“标准TD学习 + 永久示范样本 + 1-step/n-step TD + large-margin约束”的项目本地最小实现。

本实现只能称为 **minimal DQfD-style**：没有实现原始 DQfD 的优先经验回放、示范优先级加成和统一混合 replay sampler，不能写成完整 DQfD 复现。

## 冻结条件与唯一变量

- 两组均为 SY2014 IC=2、seed0、相同DSSAT输入、奖励、9动作、I120/N300预算、7d共享间隔和DQN超参数；
- 两组均使用50K全局探索日程、callback在5K停止，5K epsilon均为0.728626；
- control为标准SB3 DQN；treatment只增加示范机制；
- treatment预注册参数：margin=0.8、lambda_n=1、lambda_margin=1、lambda_L2=1e-5、n=5、100次示范预训练、每个在线gradient step追加1次示范更新；
- reward没有缩放；没有修改IC、观测、动作、预算或DSSAT输入。

## 示范与单元测试

示范再次精确复现 HWAM=11205、I75/N200；共160条transition、5个非零事件、无裁剪，终止边界与5-step数据均通过。

large-margin零损失案例为0，违规案例损失为0.30；toy batch一次更新后组合损失由1.30降到1.105，expert margin由0升到0.10。示范永久结构保持160条，普通 replay 在在线训练前为0条。准备阶段全部通过。

## 5K A/B结果

| arm | checkpoint | yield_kg_ha | irrigation_mm | nitrogen_kg_ha | reward_total | late_n_after_dap90_kg_ha |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1000 | 11175.0 | 120.0 | 300.0 | 4147.0 | 0.0 |
| control | 2000 | 11143.0 | 120.0 | 300.0 | 4115.1 | 0.0 |
| control | 3000 | 11177.0 | 120.0 | 300.0 | 4148.8 | 0.0 |
| control | 4000 | 11167.0 | 120.0 | 300.0 | 4139.3 | 0.0 |
| control | 5000 | 11093.0 | 120.0 | 300.0 | 4064.5 | 0.0 |
| minimal_dqfd_style | 1000 | 11228.0 | 120.0 | 300.0 | 4200.3 | 50.0 |
| minimal_dqfd_style | 2000 | 5741.0 | 30.0 | 100.0 | -196.7 | 100.0 |
| minimal_dqfd_style | 3000 | 7719.0 | 120.0 | 300.0 | 691.4 | 300.0 |
| minimal_dqfd_style | 4000 | 5741.0 | 90.0 | 150.0 | -506.7 | 150.0 |
| minimal_dqfd_style | 5000 | 5408.0 | 0.0 | 0.0 | 0.1 | 0.0 |

Treatment 在1K取得11228 kg/ha，但仍打满I120/N300，且DAP90后仍施氮50 kg/ha，不满足节水节氮门槛。随后其确定性策略发生剧烈漂移：

- 2K：仅在DAP110执行I30/N100，产量5741；
- 3K：在DAP98/105/112连续晚施氮，I120/N300，产量7719；
- 4K：在DAP100/107/114晚施氮，I90/N150，产量5741；
- 5K：全季no-op，回到null产量5408。

Control 在1K–5K始终为I120/N300，产量11093–11177 kg/ha；它同样没有达到资源目标，但没有出现Treatment这种从高产到null的剧烈退化。

## 预注册判定

5个Treatment checkpoint均未同时达到“HWAM≥11077、I≤90、N≤250、DAP90后不施氮、相对Control无明显权衡恶化”。通过checkpoint：`[]`。

结论：**本轮预注册门槛失败**。最小 DQfD-style 示范机制没有把oracle的稀疏时序稳定传递给DQN，反而在短训练内出现明显动作时序漂移和最终no-op退化。因此不能进入多seed或长训练，也不能把1K高产点挑出来称为成功。

## 可解释范围

1. 失败说明这一组预注册的最小示范实现不充分，不等于完整DQfD无效；
2. 本实现缺少原论文的优先回放、示范优先级加成等组件，不能把阴性结果外推为对DQfD方法的否定；
3. 由于只有seed0和5K，本轮只负责筛掉当前最小实现，不提供跨seed稳定结论；
4. 按停止规则，不事后扫描margin、lambda、epoch或追加训练步数。

## 执行异常与处理

第一次A/B启动命令设置了1秒shell超时，进程被终止；检查确认尚未创建训练目录或结果文件。随后使用同一预注册代码重新启动，没有复用或覆盖半成品。该异常不涉及模型、输入或实验变量变化。

## 输出

- `prompts/021_20_sy2014_minimal_dqfd_style_5k_ab_smoke.md`
- `src/minimal_dqfd_style.py`
- `src/run_sy2014_minimal_dqfd_style_5k_ab_021_20.py`
- `src/finalize_sy2014_minimal_dqfd_style_5k_ab_021_20.py`
- `benchmark_results/021_20/021_20_demonstration_transitions.csv/.npz`
- `benchmark_results/021_20/021_20_unit_tests.json`
- `benchmark_results/021_20/021_20_ab_checkpoint_trajectory.csv`
- `benchmark_results/021_20/021_20_preregistered_candidate_checks.csv`
- `benchmark_results/021_20/021_20_ab_checkpoint_trajectory.png/.svg`
- 每组训练audit、1K–5K模型checkpoint、逐checkpoint日值和summary。

## Methods Source

Hester, T. et al. Deep Q-learning from Demonstrations. *Proceedings of the AAAI Conference on Artificial Intelligence* 32 (2018). DOI: 10.1609/aaai.v32i1.11757. 本任务只采用其示范TD、n-step和large-margin思想，并明确记录了未实现的组件。

## 状态

`completed`：预注册5K A/B已完成；结果为阴性，已触发停止规则。
