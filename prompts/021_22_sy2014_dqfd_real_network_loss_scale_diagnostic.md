# 021_22 SY2014 文献对齐 DQfD 真实网络损失与梯度诊断

## 1. 目标

021_21 已通过纯离线组件测试，但合成batch不能代表真实SY训练动力学。本轮使用真实SY2014 IC=2 observation/reward和真实SB3 DQN网络，只运行：

1. 100次示范预训练更新；
2. 100个真实环境交互步，其中按冻结主线`learning_starts=50`执行在线式更新。

逐更新记录1-step TD、5-step TD、large-margin、L2的原始值、加权值和梯度范数，判断示范预训练是否已造成大幅参数偏移，以及在线阶段各信号是否持续失衡。本轮不评价产量、不运行完整生长季、不做5K训练。

## 2. 冻结参数

- SY2014 IC=2，复用021_20/021_14同源输入；
- seed=0；SB3默认MlpPolicy网络；学习率1e-4；gamma=0.99；max_grad_norm=10；
- alpha=0.4，beta=0.6；
- epsilon_demo=1.0，epsilon_agent=0.001；
- margin=0.8；lambda_n=1，lambda_margin=1，lambda_L2=1e-5；
- n-step=5；batch_size=32；
- 100次示范预训练；预训练结束后同步online→target一次；
- 在线阶段计划探索日程仍按50K计算，前100步epsilon约从1.0降到0.9946；
- agent transition以当前网络计算的一步TD误差初始化优先级；
- 每个在线交互步最多一次梯度更新；target在100步窗口内不再次更新；
- 不缩放reward，不修改IC、动作、预算、观测或DSSAT输入。

## 3. 实际数据与更新逻辑

- 示范：复用021_20的160条I75/N200 oracle transitions；
- agent：在真实SY环境中按epsilon-greedy收集100步；
- 原始1-step transition进入5步pending队列，形成完整5-step return后才加入统一replay；
- 示范永久区和agent环形区按021_21公式统一优先采样；
- Double-Q target：online选择next argmax，target网络评估该动作；
- 每次更新后用采样transition的一步TD error更新priority；
- 重复采样索引使用该batch内最大绝对TD error更新，避免顺序依赖。

## 4. 必须记录

每次预训练/在线更新保存：

- phase、update、env_step、epsilon、demo/agent样本数；
- 四项raw loss和weighted loss；
- 四项加权损失各自的未裁剪梯度范数；
- total gradient norm（裁剪前）和是否超过10；
- optimizer step前后参数相对L2变化；
- Q绝对均值、Q绝对最大值；
- replay中的示范/agent数量及采样比例；
- 所有数值是否有限。

分别绘制示范预训练和在线阶段的损失、梯度范数随update变化图，不得只报告终点或平均值。

## 5. 预注册诊断标志

这些是工程警报，不是调参依据：

- 任一loss、Q、gradient或参数出现非有限值：核心失败；
- 任一加权分量相对weighted 1-step TD的阶段中位数比值>100：标记`component_dominance_alert`；
- 超过50%的更新在裁剪前total grad norm>10：标记`frequent_gradient_clipping_alert`；
- 预训练阶段参数累计相对L2变化>在线阶段100步变化的10倍：标记`pretraining_shift_alert`。

无论是否触发警报，本轮都不现场调整lambda、优先参数或更新频率，也不自动启动5K。若核心失败，保留现场并停止；若只是触发警报，将其作为下一任务预注册依据。

## 6. 输出

- `src/run_sy2014_dqfd_real_network_loss_diagnostic_021_22.py`
- `benchmark_results/021_22/021_22_pretrain_update_log.csv`
- `benchmark_results/021_22/021_22_online_update_log.csv`
- `benchmark_results/021_22/021_22_agent_interactions.csv`
- `benchmark_results/021_22/021_22_stage_summary.csv`
- `benchmark_results/021_22/021_22_loss_gradient_diagnostic.png/.svg`
- `benchmark_results/021_22/021_22_summary.json`
- `docs/2026-07-15_021_22_sy2014_dqfd_real_network_loss_scale_diagnostic.md`

## 7. 资源与停止规则

- 使用指定容器和虚拟环境；
- 仅100个真实交互步，单进程，不并行；
- 不完成整季、不做checkpoint选择、不评价策略优越性；
- 不安装新包，不覆盖旧结果；
- 本轮完成后停止，等待结果解释，不自动进入5K。

