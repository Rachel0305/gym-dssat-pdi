# 027_07 YC/FQ/LC 站点专属阶段型 MaskablePPO 训练锚点验证

## 1. 唯一任务

在不修改 DSSAT 输入、IC、奖励结构、动作档位或 PPO 核心超参数的前提下，完成 YC2014、FQ2016、LC2010 三个训练锚点的：

1. 输入与四基线 provenance 复核；
2. 站点专属原始观测 scaler；
3. 完整季节 no-op smoke；
4. seed0 240 阶段步训练；
5. seed0 通过冻结的原主判据后，才运行 seed1/2；
6. 保存每个 checkpoint、动作序列、指标和分支结论。

本任务只判断三个训练锚点，不启动同站跨年迁移，不做多站点联合训练，也不搬运 SY/HLA 权重或 scaler。

用户已明确授权本任务按硬门槛连续执行，因此不需要在 readiness、seed0、seed1/2 之间再次等待确认；但任何科学停止条件仍必须执行。

## 2. 既有证据与适用边界

- 总协议：`prompts/027_00_five_station_site_specific_stage_maskable_ppo_protocol.md`。
- YC readiness 草案：`prompts/027_04_yc2014_stage_maskable_ppo_readiness_scaler_and_smoke.md`。
- 026_10 已确认原始观测维度：YC=22、FQ=23、LC=26；不得复用 HLA 24 维或 SY 25 维 scaler。
- 026_12 已证明固定 DAP 1/30/50/65/85/110 不具备五站点兼容性，且 YC/FQ/LC 的最大 DAP 分别约为 104/96/93；不得继续把 HLA/SY 阶段硬编码套到三站点。
- 华北黄淮夏玉米官方名义六阶段为 DAP 7/30/45/60/80/100。

本任务沿用 027_00 “官方区域方案阶段属于站点环境事实”的原则，并预先冻结如下可执行阶段：

| 站点 | 官方名义阶段 | 本训练年可执行阶段 | 依据 |
|---|---|---|---|
| YC2014 | 7/30/45/60/80/100 | 7/30/45/60/80/100 | 最大 DAP≈104，六点均可达 |
| FQ2016 | 7/30/45/60/80/100 | 7/30/45/60/80 | 最大 DAP≈96，DAP100 收获后不可执行 |
| LC2010 | 7/30/45/60/80/100 | 7/30/45/60/80 | 最大 DAP≈93，DAP100 收获后不可执行 |

FQ/LC 不得把 DAP100 移到 DAP93、DAP96 或季末；最后一个真实可执行阶段 DAP80 后直接推进到收获并结算终季奖励。官方 expert 在这两站的既有确定性回放同样因收获而未执行 DAP100 事件。该差异是物候/季长适配，不是为结果调整 PPO 参数。

## 3. 冻结输入

### YC2014

- `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0801.MZX`
- treatment=2；weather=`CNYC1401.WTH`；soil=`YC99001200`；cultivar=`ZD0985`；IC pointer=1。
- ICDAT=08153、SDATE=14152、PDATE=14168 作为既有 provenance caveat 记录，本任务不修改。

### FQ2016

- `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ0801.MZX`
- treatment=2；weather=`CNFQ1601.WTH`；soil=`FQ99001200`；cultivar=`FQ0985`；IC pointer=1。
- ICDAT=07152、SDATE=16153、PDATE=16162 作为既有 provenance caveat 记录，本任务不修改。

### LC2010

- 原始根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/LC`。
- 必须复用 `src/run_lc_fixed_input_year_screening_017_11.py` 的已审计输入适配，不能直接把原始错误 MZX 当训练输入。
- treatment=3；weather=`CNLC1001.WTH`；soil=`LC990012007`；cultivar=`XY0004`；IC pointer=3；ICDAT/SDATE=10121；PDATE=10171。

所有源文件只读。每个运行只在 `benchmark_results/027_07_site_specific_stage_maskable_ppo/` 下建立新副本；执行前后核对 SHA-256。

## 4. Phase A：四基线、scaler 与 no-op smoke

每个站点重新确定性前向运行：

- null；
- recorded/farmer practice；
- DSSAT auto；
- official extension expert。

旧 027_05 五情景表只用于端点交叉核验，不能用其整数值代替本次新鲜完整精度结果。必须保存 Summary.OUT 指标：yield、biomass、IRCM、NICM、ETCP、WP_ET、PFP_N。

scaler 仅使用该训练年四基线在本站可执行阶段上的原始观测：

- YC：4×6=24 个状态，22 维；
- FQ：4×5=20 个状态，23 维；
- LC：4×5=20 个状态，26 维。

展开顺序按 live runtime vector 核验：9 个前置字段 + 实际 `sw` 层数 + 7 个后置字段。population std<1e-6 的维度 scale=1 并标记 near_constant。

no-op smoke 使用 linked input、所有阶段 action0：必须命中全部本站可执行阶段、零非法动作、I=N=0、终值在 2 kg/ha 内复现 fresh null、总 reward 绝对值≤1e-10。

任一输入、基线、阶段、维度、scaler 或 smoke 失败，该站点停止，不训练。

## 5. 冻结 MaskablePPO

三个站点完全相同：

- `MaskablePPO("MlpPolicy")`；
- policy net `[32,32]`；
- learning_rate=3e-4；
- gamma=1；gae_lambda=1；
- n_steps=60；batch_size=30；n_epochs=5；ent_coef=0；
- 9 动作：I∈{0,15,30} mm × N∈{0,50,100} kg/ha；
- 季节预算 I≤120 mm、N≤300 kg/ha；DAP≥90 禁氮；
- seed=0/1/2；总计 240 阶段步；checkpoint=0/60/120/180/240；
- checkpoint 只按预注册的确定性完整季节总 reward 最大选择，精确并列取更早者；不得按结果重选。

FQ/LC 每季 5 个阶段，因此 240 阶段步对应 48 个训练季；YC 每季 6 个阶段，对应 40 个训练季。这是相同 `n_steps/total_timesteps` 下由真实可执行阶段数产生的事实差异，不改变 PPO 参数。

## 6. 冻结 reward

每站只替换 fresh 本地数值：

```text
step resource cost = -(I + 5*N) / 1000
terminal yield gain = max(0, yield - local_null_yield) / 1000
terminal feasibility bonus = 1.620,
  if yield >= max(local_auto_yield, local_official_expert_yield)
```

recorded 不进入 reward 或 checkpoint 选择。

## 7. 训练顺序和停止线

严格串行执行 YC→FQ→LC，禁止并行 DSSAT/PPO 以防 OOM。

每站：

1. Phase A 全通过；
2. 仅运行 seed0；
3. seed0 工程检查通过且其预注册选中 checkpoint 通过原 027_00 primary，才运行 seed1/2；
4. 三 seed 中至少 2/3 primary 才称训练年初步复现成功；否则停止该站跨年线；
5. 不追加 seed、不延长步数、不扫描 reward/超参数、不改 IC。

原 027_00 primary 保持不变：yield、WP_ET、可比 PFP_N 均不低于 auto/expert 对应最大值。导师最新“至少一个指标超过其余四情景、另两项接近”同时作为报告视图：

- 对 yield、WP_ET、正氮情景下可比的 PFP_N，逐项报告相对四基线最大值的绝对/百分比差；
- 标记是否至少一个指标严格高于四基线；
- “接近”尚无数值阈值，不擅自判定，也不得用该视图修改 reward、选模或 seed 扩展门槛。

## 8. 输出

至少生成：

- `benchmark_results/027_07_site_specific_stage_maskable_ppo/` 下的输入哈希、baseline、scaler、smoke、checkpoint、训练 episode、阶段动作、模型哈希、结果 JSON；
- `docs/2026-07-17_027_07_yc_fq_lc_site_specific_stage_maskable_ppo.md`；
- 本 prompt 与执行脚本；
- 所有失败尝试和硬停止分支。

模型 zip 和 DSSAT runtime 不自动提交 Git。本任务未得到新的 push 授权，不执行 Git push。
