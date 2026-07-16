# 021_47 SY2014 terminal bonus 的 replay 采样与传播：1K 精确重放插桩审计

## 研究问题

021_45 已证明 terminal feasibility bonus 在环境奖励中正确触发，但在 online seed1 的 1K 短期诊断中没有改变最终 checkpoint。021_46 排除了 epsilon 过早归零。代码审计进一步发现：021_35–45 的自定义 1K online loop 只在开始时同步一次 target，随后 1K 内不再刷新；agent n-step 为 5。

本任务回答：bonus-bearing agent transitions 在 PER 中是否被充分采样；其 TD target/Q 如何变化；标准 bootstrap 通道在 target 固定条件下覆盖多远。

## 范围与单变量纪律

- 精确重放 021_45 SY2014、online action seed=1、sample seed=21036、1K。
- 仅增加样本级日志；不修改 reward、IC、DSSAT 输入、动作、预算、网络、优化器、采样公式、epsilon、loss、target 同步行为或 RNG 调用。
- 不运行 seed2，不扩到 5K，不改 target interval。
- Control 为原 021_45 Treatment；本任务不是新的性能比较。

## 复现门槛

在解释新增日志前必须全部满足：

1. 021_47 与 021_45 的 1000 行 training interactions 逐列精确一致；
2. checkpoint 0/250/500/750/1000 的 online Q 参数哈希逐位一致；
3. 四个确定性 checkpoint 的产量、I、N、late-N、reward 和 gate 逐项一致；
4. 插桩函数每次更新仍只调用一次原 `mixed_sample`；不得增加任何 RNG 调用；
5. replay agent index 与 origin env step 的映射通过 action/reward/done 交叉核对；
6. target 参数哈希在每次 online update 后记录；若不止一个唯一值，停止使用“target全程冻结”解释。

若任一复现门槛失败，保留失败产物并停止科学解释。

## 预注册关键时点与传播定义

- oracle 的施氮关键 DAP 固定为 29、42、56，来源为 021_18 `R_I75_no_early_mid_N200`，禁止事后挑选。
- 021_45 终止 action DAP 预期约 139；实际值由重放日志核对。
- `direct bonus transition`：1-step reward 本身含 bonus 的终止 transition。
- `n-step bonus transition`：其冻结 5-step return 内含 terminal bonus 的 agent transition。
- `标准bootstrap直接覆盖`：无训练后 target refresh 时仅指终止前最多5个transition；这不是神经网络参数共享造成的“总影响”硬上限。

## 样本级日志

每次 online update 对16个 agent draws保存：

- replay global index、推导的 origin env step/episode/DAP/action；
- direct/n-step bonus 标志及折扣bonus；
- 即时水氮成本及5-step折扣成本；
- PER采样概率、importance weight、采样前priority；
- chosen Q、1-step target、n-step target、TD1/TDn error。

同时对每个active agent transition按当次PER条件概率累计理论期望采样次数和方差。

## “采样不足”的预注册判据

对每条bonus-bearing transition：

- `expected = Σ 16 p_t`；
- `variance = Σ 16 p_t(1-p_t)`；
- 若 `observed < expected - 1.96×sqrt(variance)`，记为显著低于PER期望。

类别层面同时用每次更新的类别总概率计算精确的期望与方差。若多数bonus-bearing transitions显著低于期望，且类别总观测次数也低于95%下界，判为“采样不足候选”；否则不允许用“很少抽中”解释。

## 预注册解释分支

- A：bonus transitions 显著低采样，支持采样暴露不足候选。
- B：采样不低于PER期望，但target全程冻结且DAP29/42/56远超直接5-step覆盖，支持“标准bootstrap传播受限”候选；不排除网络泛化。
- C：采样充分且target发生刷新/传播条件不受限，但早期Q仍无响应，需要另查机制。
- D：精确重放或数据映射失败，不作科学解释。

## 输出

- 复现验证JSON和checkpoint哈希CSV；
- sample-level agent draw CSV；
- per-transition exposure CSV；
- category exposure CSV；
- target hash trace CSV；
- DAP传播距离表；
- 采样/TD-Q图PNG+SVG；
- 中文实验记录。

