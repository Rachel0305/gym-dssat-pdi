# 021_35 SY2014 event-balanced 冻结网络在线保持性 1K smoke

## 目的

021_34 已证明：500 次 event-balanced 离线示范更新形成的冻结 DQN 网络，在单季确定性前向中达到 11175 kg/ha、I90/N300，超过 official expert 产量并显著节水。本任务只回答：在该网络上继续在线 DQN/DQfD-style 更新后，优质策略能否保持。

## 单变量与边界

- 起点必须逐位复现 021_34 的 online-Q 参数哈希；
- reward、IC、DSSAT 输入、动作空间、预算、网络结构、学习率均不变；
- 只增加在线环境交互和梯度更新；
- 仅 seed0、1000 environment steps；不自动扩展到 5K 或其他 seed；
- deterministic checkpoint evaluation，无 epsilon；训练交互使用冻结 50K 计划下的 epsilon 日程；
- 在线开始前把 target Q 同步到 021_34 online Q；同步不改变 deterministic 起点策略；
- 不覆盖 021_34，不删除失败结果。

## 在线 batch（预注册）

batch_size=32：

- 16 条永久示范：8 no-op + 8 nonzero；
- 8 条 nonzero 覆盖五个 oracle 事件，每次各至少 1 条，额外 3 条按 update 轮转；
- 16 条 agent transition，按 agent 内部 PER 抽样；
- demo no-op、demo nonzero、agent 三组 mixture mass 分别为 0.25/0.25/0.50；
- beta=0.6、alpha=0.4、epsilon_demo=1.0、epsilon_agent=0.001；
- demonstration 1-step TD 屏蔽，agent 1-step TD 保留；n-step TD 对全部样本生效；large-margin 只对 demonstration 生效；
- agent transition 使用冻结 n_step=5；demonstration 保持 021_34 已验证的 full return-to-go，不在本任务改定义。

## 训练与检查点

- learning_starts=50；每个环境步至多一次梯度更新；
- checkpoint：0（复用021_34）、250、500、750、1000；
- target_update_interval=10000，因此本次1K中除起点同步外不更新 target；
- 每个检查点单独做一个确定性 SY2014 前向季节。

## 预注册判据

每个在线检查点通过条件：yield>=11077 kg/ha、I<=120 mm、N<=300 kg/ha、DAP>90 的 N=0。

- A：4个在线检查点中至少3个通过，1000步通过，且最低产量>=recorded 9613；允许另立5K/多seed任务；
- B：至少1个在线检查点通过，或1000步产量>=9613，但不满足A；说明部分保持，不允许自动放大；
- C：没有在线检查点通过且1000步产量<9613；在线更新破坏优质起点，停止该配置。

## 强制验证

- 起点 online-Q 哈希与021_34一致；
- checkpoint0指标直接复用021_34并保留来源；
- replay demonstration哈希训练前后不变；
- agent replay确实增长、sample严格16 demo/16 agent；
- 全部Q/loss/参数有限；
- 每次评估正常终止；
- 输出逐步训练日志、更新日志、checkpoint评估日值、汇总、PNG/SVG和中文记录；
- 不Git push。

