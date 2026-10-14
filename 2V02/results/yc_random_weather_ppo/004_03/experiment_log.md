# 004_03 YC random-weather PPO controlled pilot 实验记录

## 设计冻结

- 纠正依据：`prompt_02/004_03_01_correct_canonical_reward_config_conflict.md` 优先于原任务中旧 reward 口径。
- 配置来源核验：`canonical_source_config` 指向指定的 055_00/032_00 YAML；它声明 5,000 timesteps，属 smoke/base 模板值。正式 055_00 任务配置 JSON 与 completed formal result 均为 100,000 timesteps，因此以正式任务配置/结果为 formal budget authority，pilot 两组均请求 100,000；来源分层已写入机器可读配置和报告。
- 旧公式 `0.06 * final_grnwt - 0.04 * cumfert` 标记为 `SUPERSEDED / NOT_APPLICABLE`；本轮沿用 `055_00` 有效 reward wrapper、reward、安全约束和 PPO 实现，不修改 reward。
- 正式训练预算：每模型请求 100,000 timesteps；checkpoint 25,000、50,000、75,000、100,000。PPO 参数、网络、动作、观测、约束和 lowIC YC 输入两组一致。
- 训练 schedule 预先冻结为 100,000 条 episode 记录：历史年份 schedule seed `64003`，WGEN seed-order schedule seed `64004`。两组共用完全相同的 2004–2013 年份上下文 sequence；WGEN schedule 每个 80-seed block 覆盖 `1001–1080` 全池一次。
- Evaluation 冻结为 held-out WGEN `1081–1100`，每个 seed 使用 2008 YC crop-year context；observed evaluation 为 2014–2023 independent comparison period，不称 pristine final test。
- `004_02` WGEN episode QC 为 `PASS_WITH_NOTES`；synthetic Tmin 均值约比 fitting observed 高 `0.58 C`，此 note 保留，不作为本轮阻断条件。全年 WGEN validation 仍为 deferred/optional。

## 配置与输入门禁

- `--self-test`：PASS；schedule 可复现、年份每 10 条平衡、WGEN 每 80 条覆盖完整 seed pool、训练/评估 WGEN seed 无交集。
- `--freeze-config`：PASS；保存 canonical config、experiment design、100,000 行历史 schedule、100,000 行 WGEN schedule 和 30 行 evaluation manifest。
- 本轮配置来源复核的首次轻量 preflight 将 `055_00_formal_result.json` 误按 `config.training` 结构读取，触发 `KeyError`；检查真实 schema 后改用 `action_gate.final_checkpoint` 作为正式完成预算证据。修正后 preflight PASS；未启动 DSSAT、未重跑训练或评估。
- Frozen CLI SHA-256：`65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`；fitting weather SHA-256 与 `004_02` provenance 一致。

## Seed Smoke 过程

### 首次尝试：`smoke/smoke`

- 432 timesteps 尚未开始 PPO rollout；首次 reset 校验发现 schedule seed `1070` 与 runtime `_rseed1` `55411` 不一致，按门禁立即停止。
- 失败日志和 run manifest 保留；未生成 checkpoint，未纳入性能分析。

### 诊断尝试：`smoke/smoke_retry1`

- 仍在 rollout 前停止。诊断确认 WGEN seed adapter 调用数为 0。
- 根因：gym-DSSAT wrapper 通过 `__getattr__` 代理底层字段；原查找函数命中了外层代理，seed adapter 没有设置在真正的 `DssatPdi` 实例上。另有 `GymDssatWrapper` 构造期 reset，会在 PPO 首次 reset 前消费一次 WGEN seed。
- 修复限于本任务脚本中的 seed plumbing：按实际 runtime 类定位底层 `DssatPdi`；构造 wrapper 前及每次 episode reset 前设置显式 sequence seed。未改安装包文件、DSSAT binary、CLI、reward 或 PPO 参数。

### 修复后尝试：`smoke/smoke_retry2`

- 状态：PASS；请求/实际 `432/432` timesteps，4 个完整 episode。
- PPO seed：0；实际注入 PDI 的 `rseed1_` sequence：`1070, 1065, 1053, 1009`，全部属于训练 pool `1001–1080`，且至少两个不同 seed 到达 PDI。
- checkpoint `checkpoint_432.zip` 存在且可加载；reward component sum 与 episode return 一致；schedule seed 与 PPO seed 分离。
- 实测耗时约 `10.7 s`，峰值进程树 RSS 约 `628 MB`。6×100K 串行预算粗略外推约 `4.1 h`；这是按 smoke 线性外推的资源估计，不是正式运行时间承诺。

## 正式阶段

- 首轮六模型训练和 evaluation 虽完成，但事后核验发现当时 FileX 实际 `WTHER=M`，并未启用 WGEN；该批模型、评估、汇总和图保留为首轮尝试，不能作为受控随机天气比较，也不参与最终结论。

## WGEN 执行链纠正与复核

- 根因：`random_weather=True` 和 seed 注入本身不会把 YC FileX 的 `WTHER=M` 改为 WGEN。此前每年天气字段/seed 日志不足以证明实际生成天气已切换。
- 修复仅作用于任务生成的 FileX 模板副本：所有 `METHODS.WTHER` 明确设为 `W`；原模板、canonical 输入、安装 runtime、DSSAT binary、CLI、PPO 和 reward 均未修改。
- 天气差异证据改为复用 `003_06_05_02` 已验证的 daily runtime-state 提取，按每日 `RAIN/SRAD/TMAX/TMIN` 序列生成 SHA-256；不再把输入 `.WTH` 文件哈希当成 WGEN 输出。
- 同年 smoke `wther_w_cached_process_probe_01`：PASS，PPO seed 0，432/432 timesteps，4 个完整 episode；年份固定 2008，PDI 实际 seed 为 `1070,1065,1053,1009`，FileX `WTHER=W`，4 个天气哈希互异，checkpoint 可加载，reward 分解一致。运行约 5.46 秒，峰值进程树 RSS 433.64 MB。
- 另一 smoke `wther_w_daily_capture_smoke_01` 的训练/checkpoint 成功；首次门禁落盘遇 Pandas `bool_` JSON 序列化错误，手工复核 episode CSV 后确认相同四个天气 seed/hash 条件成立，并修正门禁类型转换。该 attempt 与 `weather_refresh_*` 诊断均保留，不纳入正式统计。
- 有效正式模型将写入 `training_verified/`，有效评估 episode 将写入 `evaluation/verified_weather_refresh/`。首轮 `WTHER=M` 汇总已备份至 `backups/004_03_initial_unverified_outputs/`；最终汇总只读取修复后的目录。
- 正式训练策略仍为单容器、单模型串行，进程树 RSS 达到 `6000 MB` 即停止并保留现场，不自动重试；Git push 禁止。

## 修复后正式训练

- 六个模型全部完成；每个请求 `100000`、实际 `100080` timesteps，均保存 25K/50K/75K/100K 四个 checkpoint。历史组完整 episode 数为 `940/942/940`；WGEN 组均为 `962`。
- 历史组峰值 RSS 为 `927.79/985.84/994.89 MB`，耗时 `556.9/563.8/564.6 s`；WGEN 组峰值 RSS 为 `1030.81/1046.07/1055.90 MB`，耗时 `573.7/573.7/570.9 s`。未触发 6000 MB 资源停止线。
- `--validate-training`：PASS；6/6 run 完成、六组配置签名相同、实际步数相同、checkpoint 齐全，训练/held-out WGEN seed 交集为空。三个 WGEN PPO seeds 的 year/seed/daily-weather-hash 序列按 962 个 episode 完全一致；每次 run 有 170 个不同 daily-weather hashes。
- 观察 note：同一年中有 7 个重复的 weather-seed/year pair 在跨 schedule block 重现时得到不止一种日天气序列；这些实际序列在三个 PPO seed 间仍逐 episode 完全相同，不造成组间天气暴露不匹配。报告会保留该复现边界，不宣称同一 seed/year 重置必定生成唯一固定序列。
- 对外正式评估只读取 `evaluation/verified_weather_refresh/`；旧 `WTHER=M` 首轮评估留档但不合并。

## 修复后正式评估与结论

- 六个模型均完成 held-out WGEN `1081–1100`（每模型 20 episodes）和 observed `2014–2023`（每模型 10 episodes），deterministic actions=true。有效评估汇总分别为 120 条与 60 条。
- `evaluation_qc_verified.json`：PASS；每个 held-out seed 的实际 `rseed1_` 与请求相等，WGEN runtime daily-weather hash 在 20 个 seed 上互异，且同一 seed 对应的天气序列在六个模型间一致；observed evaluation 不误填 WGEN seed。Reward decomposition 最大绝对误差为 0。
- Held-out pooled：reward 历史组 `0.830541`、WGEN 组 `0.907329`（`+9.25%`）；yield `6985.45` vs `7006.34 kg/ha`（`+0.30%`）；fertilizer `120.0` vs `118.67 kg N/ha`（`-1.11%`）；irrigation `123.5` vs `55.0 mm`（`-55.47%`）；P10 reward `0.510195` vs `0.685132`（`+34.29%`）。
- Per-seed held-out reward 方向：seed 0 positive、seed 1 negative、seed 2 negative；Observed comparison 方向分别 worse/better/worse。预设判定为 `MIXED_SIGNAL`，更大实验建议 `NEEDS_DIAGNOSIS`；不宣称统计显著或天气增强已稳定有效。
- 中文报告：`docs/yc_random_weather_ppo_controlled_pilot.md`；六张图、paired/per-regime 汇总和机器可读决策均已生成。NUE/WUE 未报告，无 canonical 计算口径时不另造定义。
