# 027_02 HLA2010 阶段型 MaskablePPO seed0 240 步预注册训练

## 1. 唯一任务

在 027_01 已通过的 HLA2010 独立 24 维阶段环境中，只运行 seed0 的一次 240 阶段步 MaskablePPO 训练，判断该固定配置是否出现相对 DSSAT auto 和 official extension expert 的高质量候选。

本任务不是多站点联合训练，不使用 SY 权重，不运行 seed1/2，不做跨年验证，也不根据结果调参。

## 2. 必须复用的冻结证据

- 环境与 scaler：`benchmark_results/027_01_attempt2/` 的权威 attempt2。
- `local_null_yield = 6956.453857421875 kg/ha`。
- `local_feasibility_yield = 7853.6651611328125 kg/ha`。
- 决策点：DAP 1/30/50/65/85/110。
- 原始观测 24 维，使用 027_01 HLA 专属 scaler。
- 9 动作：I∈{0,15,30} mm × N∈{0,50,100} kg/ha。
- 季节预算：I≤120 mm、N≤300 kg/ha；超预算动作必须 mask，禁止裁剪形成动作别名。
- DAP110 禁止施氮；其他晚期规则与 026/027_01 已验证实现一致。

若以上任一文件、数值、维度、哈希或 smoke 状态不一致，立即停止，不训练。

## 3. 冻结 PPO 配置

- 算法：SB3-Contrib `MaskablePPO`。
- policy 网络：`[32,32]`。
- `learning_rate=3e-4`。
- `gamma=1.0`。
- `gae_lambda=1.0`。
- `n_steps=60`。
- `batch_size=30`。
- `n_epochs=5`。
- `seed=0`。
- 总训练步数：240 个阶段步。
- checkpoint：0、60、120、180、240。

禁止扫描或修改网络、学习率、GAE、batch、epoch、步数、seed、奖励或阶段点。

## 4. 冻结奖励

每个决策步的资源成本：

```text
-(irrigation + 5*nitrogen) / 1000
```

收获终止时额外加入：

```text
max(0, yield - 6956.453857421875) / 1000
+ 1620 / 1000, if yield >= 7853.6651611328125
```

recorded 不进入奖励或选模。必须逐季保存产量增益、资源成本和 feasibility bonus 分项，并验证分项之和等于环境总奖励。

## 5. 训练前 smoke

正式训练前只允许以下低成本检查：

1. Python 编译与依赖导入。
2. 027_01 result 为 `A_ready_for_027_02_seed0`。
3. 24 维 scaler 可加载且数值有限。
4. 环境 reset、action mask、一次 no-op 季节继续复现 null，奖励为 0。
5. 输出目录不存在正式 checkpoint；不得覆盖旧结果。

smoke 失败则判工程失败，不得进入训练。

## 6. checkpoint 评估与选择

每个 checkpoint 都用冻结权重、`deterministic=True`、无在线更新完成一个完整 HLA2010 季节评估，保存：

- 六阶段请求动作与实际动作；
- 每阶段 mask、累计/剩余水氮预算；
- 产量、生物量、总灌溉、总施氮；
- ETCP、WP_ET、PFP_N；
- reward 分项和总和；
- 模型 SHA-256；
- 非法动作数和 `learn()` 调用统计。

预注册选模规则：

1. checkpoint0 只作随机初始化参照，不得入选；
2. 在 60/120/180/240 中选择确定性季节 reward 最大者；
3. reward 严格并列时选择更早 checkpoint；
4. 不得根据 primary、recorded 或图形观感重新选择。

## 7. 科学判据

### Primary：相对 auto + official expert

选中 checkpoint 必须同时满足：

- yield ≥ 7853.665161 kg/ha；
- WP_ET ≥ max(auto 1.64, expert 1.63) = 1.64 kg/m3；
- PFP_N ≥ expert 26.2 kg/kg；auto 因 N=0 的 PFP_N 不可定义，不伪造比较值。

若 PPO 的 N=0，则 PFP_N 标记不可定义，不能自动判为超过 expert。

### Recorded 独立比较

另行比较 recorded 的 yield=7679 kg/ha、WP_ET=1.70 kg/m3、PFP_N=46.5 kg/kg。recorded 结果不参与 reward 或 checkpoint 选择。只有所有可定义指标均不低于 recorded、auto、expert 的最大值时，才允许称为 full three-baseline pass。

## 8. 预注册分支

### A_seed0_primary_signal

工程检查全部通过，且预注册选中 checkpoint 通过 primary。只允许下一步另写 seed1/2 三 seed 复核 prompt；不得直接跨年。

### B_seed0_no_primary

工程通过，但选中 checkpoint 未通过 primary。照实记录并停止 HLA 训练线；不加步数、不换 checkpoint 规则、不改 reward、不现场追加 seed。

### C_execution_failed

输入、scaler、环境、mask、数值、输出或依赖检查失败。停止并只诊断工程原因。

## 9. 算力与输出纪律

- 仅 seed0、240 阶段步；串行运行，避免 OOM。
- 不启动 HLA 其他年份、YC、FQ、LC 或联合训练。
- 不覆盖 027_01 或任何既有结果。
- 模型 zip 保存在本地结果目录，不默认加入 Git。
- 保存 prompt、训练曲线、checkpoint 汇总、动作表、JSON 结论、失败记录和中文实验记录。
- 本任务完成后停止，等待用户确认是否进入 seed1/2。

