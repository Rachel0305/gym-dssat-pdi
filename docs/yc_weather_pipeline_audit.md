# YC weather pipeline audit

日期：2026-09-22

## 一句话结论

当前 `055_00` 的 YC/YCA PPO 训练没有启用 gym-DSSAT 的 `random_weather` / WGEN 随机天气；episode 间天气变化来自 `RandomYearEnv` 在 2004-2013 训练年份中按 seed 抽取不同的真实历史 `.WTH` 年份。

## 审计范围

- 站点：`YCA` / DSSAT site `YC`。
- 当前 PPO 入口：`src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py`。
- 当前配置：`configs/055_00_yca_lowIC_expanded_action_maskableppo.json`。
- 继承方法：`046_10_sya_originIC_expanded_action_maskableppo`。
- 输入 profile：`lowIC`。
- 未修改 `LC`、`SY`、`HL`、`FQ` 配置、脚本或结果。

## 当前训练入口和配置

`055_00` runner 读取 `configs/055_00_yca_lowIC_expanded_action_maskableppo.json`，校验：

- `station_code = YCA`，`site = YC`。
- 训练年份：2004-2013。
- 验证年份：2014-2023。
- seed：0。
- 16 维动作网格：灌溉 `[0, 15, 30, 45]` mm，施氮 `[0, 40, 80, 120]` kg/ha。
- observation contract：`046_02_raw_observation`，不启用 normalization，不启用 weather forecast。

训练主链路为：

1. `run_055_00_yca_lowIC_expanded_action_maskableppo.py`
2. `run_sya_lowIC_binary_timing_maskableppo_042_10.py`
3. `run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py`
4. `run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py`
5. `run_all_year_direct_action_safe_ppo.py`
6. `ppo_safe_rendering.py`
7. `gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args)`

## 实际天气来源

当前 `YCA` 使用的输入根目录为：

`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC`

该目录包含 `CNYC0001.WTH` 到 `CNYC2301.WTH` 等逐年历史天气文件，不包含 `.CLI` 文件。渲染诊断以训练首年 2004 为例，实际 `env_args` 为：

- `fileX_template_path`: `results/yc_weather_audit/rendered_probe/rendered_inputs/YCA/2004/yc_weather_audit_probe/YCA_2004_yc_weather_audit_probe.jinja2`
- `auxiliary_file_paths[0]`: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/MZCER048.CUL`
- `auxiliary_file_paths[1]`: `results/yc_weather_audit/rendered_probe/rendered_inputs/YCA/2004/yc_weather_audit_probe/CNYC0401.WTH`
- `auxiliary_file_paths[2]`: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/SOIL.SOL`

`CNYC0401.WTH` 运行验证摘要：

- 366 行逐日天气。
- 日期码范围：2004001 到 2004366。
- 年降雨和：67.4。
- SHA256：`333e976a52ed6135a37774adb7925b0189f2d6b30137692e792fb8ec2283a700`。

## random_weather / WGEN 状态

实际安装的 `gym_dssat_pdi` 版本为 `0.0.5`，环境类签名包含：

`random_weather=True`, `seed=None`, `auxiliary_file_paths=None`, `fileX_template_path=None`, `evaluation=False`

安装源码里 `random_weather` 的含义是：

- `random_weather=True` 时，DSSAT fileX 的 weather mode 使用 `W`，reset 时重新抽 `_rseed1`。
- `random_weather=False` 时，fileX 的 weather mode 使用 `M`，读取传入的 `.WTH` 文件。

当前 YC/YCA 训练链路显式传入：

```json
"random_weather": false
```

因此当前没有启用 WGEN 随机天气，也没有读取 YC `.CLI` 气候参数。

## episode 间天气是否变化

已运行 `scripts/diagnose_yc_weather.py --episodes 12 --gym-reset --gym-resets 3`。

固定 `ppo_seed = 0` 且当前未分离 weather seed 时，年份抽样序列为：

`2012, 2010, 2009, 2006, 2007, 2004, 2004, 2004, 2005, 2012, 2010, 2013`

重复同一 seed 得到完全相同序列。将 seed 改为 1 后，序列改变为：

`2008, 2009, 2011, 2013, 2004, 2005, 2012, 2013, 2006, 2007, 2012, 2008`

真实 gym reset 轻量验证的前三次 `active_year` 为：

`2012, 2010, 2009`

结论：episode 间天气确实会变，但变化单位是训练年历史天气文件，不是同一年内由 WGEN 生成的新随机天气。

## PPO seed 与 weather seed

当前没有独立的 `weather_seed` 字段。`RandomYearEnv` 使用传入的 `SEED = 0` 初始化年份抽样 RNG；MaskablePPO 同样使用 `seed = 0`。

因此当前：

- `ppo_seed = 0`
- `weather_seed = 0`
- 二者在接口上未分离

这会使“模型初始化随机性”和“训练 episode 天气年顺序随机性”耦合。建议下一阶段只在 YC 专用 runner 中新增可选 `weather_seed`，默认等于 `ppo_seed` 以保持旧结果可复现；正式对照时显式记录两者。

## 文件引用

当前 YC/YCA 链路涉及：

- 源 DSSAT experiment template：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYC0801.MZX`
- 渲染后的 2004 工作 experiment：`results/yc_weather_audit/rendered_probe/rendered_inputs/YCA/2004/yc_weather_audit_probe/YCA_2004_yc_weather_audit_probe.jinja2`
- 2004 天气文件：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYC0401.WTH`
- 渲染工作目录天气副本：`results/yc_weather_audit/rendered_probe/rendered_inputs/YCA/2004/yc_weather_audit_probe/CNYC0401.WTH`
- cultivar：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/MZCER048.CUL`
- soil：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/SOIL.SOL`
- `.CLI`：当前 YC lowIC 输入目录未发现 `.CLI`
- WSTA：2004 渲染为 `CNYC0401`

## 是否存在默认 Florida 气候参数污染

当前 `055_00` 链路未启用 `random_weather`，也未向 `auxiliary_file_paths` 传入 `.CLI`，因此没有证据显示当前 YC PPO 训练正在使用默认 Florida / UF 气候参数生成随机天气。

需要注意：仓库历史结果目录中存在其他任务的 `.CLI` 和 WGEN 痕迹，但它们不在当前 `055_00` 运行链路内，不能据此判断当前 YC 训练已启用随机天气。

## 当前发现的问题

1. `random_weather` 参数在 gym-DSSAT 默认值为 `True`，但项目当前 YC/YCA 训练链路靠 `ppo_safe_rendering.build_env_args()` 显式设为 `False`。后续新入口必须继续显式记录，不能依赖默认值。
2. 当前 episode 天气变化来自历史年份抽样，不是生成式增强天气。报告时必须区分“多历史年训练”和“WGEN 增强天气训练”。
3. `ppo_seed` 与训练年抽样 seed 当前未分离，后续多 seed 结果可能同时混入模型随机性和天气顺序随机性。
4. YC lowIC 输入目录没有 `.CLI`，不能直接声称已具备 YC WGEN 参数。
5. 若下一阶段要启用 WGEN，不能手写 `.CLI`；必须先用训练期历史天气估计参数，并保持 validation/test 真实天气独立。

## 下一阶段最小改动建议

建议建立两个严格公平的 YC-only 分支：

### baseline_weather

- 使用当前历史天气机制。
- `random_weather = False`。
- episode 在训练年历史 `.WTH` 中抽样。
- 保持 PPO 算法、reward、action space、observation space、训练步数、超参数、评估年份不变。

### augmented_weather

- 仅改变训练天气来源。
- 基于 YC 训练期完整生长季历史天气估计增强天气参数。
- 每个 episode 使用记录在 manifest 中的完整年天气情景。
- 不改 PPO 算法、reward、action grid、observation、约束。
- 评估仍使用真实 validation/test 年份，不把 validation/test 天气用于拟合天气生成器。

### 数据划分

- train：2004-2013，用于训练 PPO，也只能用这些年份估计增强天气参数。
- validation：2014-2023，用于 checkpoint / seed 选择和报告。
- test：如果后续要做最终独立检验，应另行冻结完整生长季年份或跨站点外部检验，且不得参与天气生成器参数拟合。

### 记录结构

后续每颗 seed 至少记录：

- yield
- irrigation
- fertilizer
- WUE / `WP_ET`，仅在 `Summary.OUT` / ETCP replay 证据齐全时报告
- NUE / `PFP_N`
- constraint violations
- reward
- 管理时序摘要
- 成功 seed 数 / 总 seed 数
- `ppo_seed`
- `weather_seed`
- episode-level weather manifest

## 已运行验证与静态判断边界

已通过运行验证：

- `gym_dssat_pdi` 实际安装版本、环境类签名和源码位置。
- 当前渲染 `env_args` 的 `random_weather=false`。
- 当前 `auxiliary_file_paths` 为 cultivar、`.WTH`、soil，没有 `.CLI`。
- 固定 seed 下训练年序列可复现。
- 不同 seed 下训练年序列改变。
- 3 次真实 gym reset 返回的 `active_year` 与固定 seed 序列一致。

代码静态判断：

- 当前未启用 WGEN，因为 `random_weather=false` 且没有 `.CLI` 进入当前 env_args。
- 默认 Florida / UF 气候参数未污染当前 `055_00` 链路。该判断基于当前运行链路，不代表仓库所有历史任务。

## Git / GitHub 状态

开始前记录：

- 当前分支：`codex/sya-forecast-freeze-2026-08-16`
- remote：`origin git@github.com:Rachel0305/gym-dssat-pdi.git`
- 工作区已有大量未跟踪文件和若干用户既有修改；本任务只显式新增/修改指定审计文件。

根据项目 `AGENTS.md`，未经用户确认不得 push 到 GitHub。因此本次不执行 GitHub push；这不是认证失败，而是项目级安全规则阻止。

## 产物

- `scripts/diagnose_yc_weather.py`
- `results/yc_weather_audit/yc_weather_config_snapshot.json`
- `results/yc_weather_audit/yc_weather_reset_diagnostic.json`
- `results/yc_weather_audit/rendered_probe/`
- `docs/yc_weather_pipeline_audit.md`
