# 021_21 SY2014 文献对齐 DQfD 离线组件与单元审计记录

## 目的与边界

本轮只实现并审计 Hester et al. (2018) DQfD 的关键 replay/loss 组件，没有调用DSSAT、没有训练网络、没有修改reward、IC、动作、预算或旧结果。参数在运行前写入prompt，未根据结果调整。

## 冻结参数

- alpha=0.4，beta=0.6；
- epsilon_demo=1.0，epsilon_agent=0.001；
- margin=0.8；
- lambda_n=1，lambda_margin=1，lambda_L2=1e-5；
- 项目适配gamma=0.99、n-step=5。

## 单元测试结果

| 审计项 | 结果 |
| --- | --- |
| 示范永久保留 | True |
| 示范优先级加成 | True |
| 采样概率公式 | True |
| IS权重 | True |
| 固定seed复现 | True |
| priority更新不改身份/内容 | True |
| margin仅作用于示范 | True |
| 损失分解恒等式 | True |

全部核心测试通过。160条021_20 oracle示范在50次agent插入（agent容量20、发生两轮以上覆盖）后内容哈希不变；相同TD error=2时，示范优先级为3.0，agent优先级为2.001；统一采样概率、alpha幂次和beta重要性权重与手算一致。

## 合成混合batch损失量级

| component | raw_value | weighted_value | weighted_to_td1_ratio |
| --- | --- | --- | --- |
| td_1 | 0.0455 | 0.0455 | 1.0 |
| td_n | 0.11 | 0.11 | 2.41758226 |
| margin | 0.28000003 | 0.28000003 | 6.15384646 |
| l2 | 2.37000012 | 2.37e-05 | 0.00052088 |

本表只是代码恒等式和日志链路测试，不是SY真实训练损失，不能据此调整lambda或推断5K结果。四项分量均有限，加权分项之和与总损失误差为1.460e-08；agent-only batch的margin loss严格为0。

## 判定

`completed`：优先回放、示范永久保留、示范优先级加成、IS权重、priority更新和四项损失分解具备进入下一轮真实短诊断设计的工程条件。

这不等于已经证明DQfD有效，也不授权自动运行5K。下一轮若继续，应另写prompt：先在真实网络上运行极短的损失量级诊断并保存逐update原始/加权损失；若审计未通过则停止，不能在同一任务现场调lambda。

## 方法来源

Hester, T. et al. Deep Q-learning from Demonstrations. *Proceedings of the AAAI Conference on Artificial Intelligence* 32 (2018). DOI: 10.1609/aaai.v32i1.11757。

## 输出

- `prompts/021_21_sy2014_literature_aligned_dqfd_offline_unit_audit.md`
- `src/literature_aligned_dqfd.py`
- `src/test_literature_aligned_dqfd_offline_021_21.py`
- `benchmark_results/021_21/021_21_unit_test_results.json`
- `benchmark_results/021_21/021_21_priority_table.csv`
- `benchmark_results/021_21/021_21_loss_component_audit.csv`
