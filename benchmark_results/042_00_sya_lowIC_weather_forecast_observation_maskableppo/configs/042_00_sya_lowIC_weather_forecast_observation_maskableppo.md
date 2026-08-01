# 042_00 SYA lowIC 天气预报增强 observation MaskablePPO

## 背景

041_06 审计发现：

- 当前 25 维 observation 中没有直接的 `RAIN` 和 `TMIN`；
- PPO 对 `SWFAC/NSTRES/soil water` 的动作敏感性很弱；
- PPO 主要响应 `grnwt/topwt/totir/cumsumfert/dap`，容易学成固定管理模板。

导师允许使用“历史天气作为完美天气预报”。因此本任务新增天气/预报信息进入 PPO observation，测试策略是否能获得更合理的天气响应。

## 本次改动

只改 PPO observation 信息结构，在原 25 维 DSSAT 状态后追加 5 个原始物理量：

1. `rain_today_mm`
2. `tmin_today_c`
3. `rain_past7_mm`
4. `rain_future7_mm`
5. `tmean_future7_c`

其中未来 7 天天气使用历史观测天气构造，为 perfect forecast 场景。

## 固定不变

- 数据源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 训练年：2005–2013
- 验证年：2014–2023
- 算法：MaskablePPO
- 动作档位：灌溉 `[0,30,45]` mm；施氮 `[0,80,120]` kg/ha
- 约束：
  - 单季灌溉软上限 240 mm；
  - 单季施氮软上限 250 kg/ha；
  - 灌溉最小间隔 7 天；
  - 施氮最小间隔 7 天；
  - DAP90 后禁氮；
  - DAP90 前累计灌溉不得超过 195 mm，给后期保留至少 45 mm 灌溉机会。
- reward：沿用 040_40，不新增 reward 项。
- total timesteps：正式 100000，先 2000 smoke。

## 前置安全核查

训练前必须输出并检查：

1. lowIC input root 是否为 `multisite_new_cultivar_inputs_013_lowIC_manual`；
2. 039_00 authoritative lowIC audit 是否通过；
3. observation 维度是否从 25 增加到 30；
4. 新增 5 个天气特征是否非全空、非全 0；
5. dry-run 和 2K smoke 通过后，才允许正式 100K。

## Smoke 命令

```bash
cd /workspace/src
/opt/gym_dssat_pdi/bin/python run_sya_lowIC_weather_forecast_observation_maskableppo_042_00.py --timesteps 2000 --checkpoint-steps 1000,2000 --suffix smoke2k
```

## 正式训练命令

```bash
cd /workspace/src
/opt/gym_dssat_pdi/bin/python run_sya_lowIC_weather_forecast_observation_maskableppo_042_00.py
```

## 判读边界

本任务改变了 agent 的信息结构，因此不能和 040/041 直接混为同一种 PPO。它回答的是：

> 加入当天天气、近期降雨和完美天气预报后，自由时序 MaskablePPO 是否比旧 observation 更有机会学习天气响应。
