# 021_03 SY2014 IC=2 统一 DQN smoke 与正式 seed0 验证

## 目的

在 021_02 已确认 SY2014 treatment 2 / IC=2 输入和三情景前向链路的基础上，验证冻结统一 DQN 框架能否在该输入上正常训练，并判断是否出现值得继续做 seed1/2 的优化信号。

## 冻结配置

- 输入：`configs/sites/sya.yaml` 当前确认的 SY2014 IC=2。
- 算法：Stable-Baselines3 DQN，`n_steps=5`。
- 动作：灌溉 `[0,15,30]` mm × 施氮 `[0,50,100]` kg N/ha，共 9 个离散动作。
- 季节预算：灌溉不超过 120 mm，施氮不超过 300 kg N/ha。
- 共享操作间隔：7 DAP；决策窗口 DAP 1–120。
- reward：`max(0, DQN最终产量 - 本站点年份null产量) - 1×灌溉量 - 5×施氮量`。
- 不加入氮淋洗惩罚；不改 reward 系数、IC、DSSAT 输入、动作空间和预算。

## 分级执行

### 第一阶段：审计与 dry-run

1. 确认 021_02 输入哈希和当前 MZX 一致。
2. 对 smoke YAML 进行 dry-run，只展开 SY2014 seed0 一个 case。
3. 检查输出目录不存在，禁止覆盖旧结果。

### 第二阶段：5K smoke

运行 `021_03_sy2014_ic2_dqn_smoke.yaml`，只训练 seed0 5000 steps。

通过标准：

- 训练和 checkpoint 评估完整结束；
- PDI runtime audit 通过，实际运行输入仍为 IC=2；
- reward、产量和动作总量均为有限值；
- 灌溉与施氮不超过 I120/N300；
- 没有模板指针、端口、NaN、OOM 或提前异常终止。

smoke 不用于宣称 DQN 优越性。若上述任一项失败，停止，不运行 50K。

### 第三阶段：正式 50K seed0

仅在 smoke 通过后运行 `021_03_sy2014_ic2_dqn_seed0_50k.yaml`。每 5K 保存 checkpoint 与确定性评估，最终从已有 checkpoint 中按冻结协议选择最佳 reward checkpoint。

## 结果判定

比较 DQN 与 021_02 的 true null、recorded 和 DSSAT auto：

- true null：5408 kg/ha，I0/N0；
- recorded：9613 kg/ha，I0/N293；
- DSSAT auto：5498 kg/ha，I66/N0。
- 官方推广 expert：11077 kg/ha，I266.1/N300（复用 018_03 已验证结果）。

seed0 进入后续多 seed 验证的最低条件：

1. 明显高于 true null；
2. 动作满足预算且不是运行异常产生的伪结果；
3. 相对 recorded/auto 至少形成可解释的产量—资源权衡；
4. 最佳 checkpoint 不是仅凭单一最终 50K 数值臆选，而有完整 checkpoint 轨迹支持。

即使 seed0 表现良好，也只能标记为“候选”，不能宣称跨 seed 稳定成功；seed1/2 必须另行确认后才能运行。

## 输出

- `benchmark_results/021_03/` 下独立 smoke 与 50K 目录；
- 配置、manifest、checkpoint 评估、日值、动作、日志；
- 中文实验记录；
- checkpoint 产量/水/氮/reward 汇总与图；
- 不提交模型、replay buffer、PDI 临时文件到 Git。
