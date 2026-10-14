# YC isolated SRAD lower-bound weather-only pilot（004_22）

## 执行结论

```text
source_build_gate = FAIL
pilot_status = BLOCKED_SOURCE_GATE
weather_generation_started = false
dssat_crop_simulation_started = false
ppo_started = false
```

004_22 已按顺序执行只读 source/build preflight。停止是任务预设安全门槛的结果，不是 pilot 失败或 WGEN 科学结论：当前容器没有可核对的安装版 Fortran 源码，因此不能安全修改/重建 SRAD 下限，也不能验证 `DailyWeatherCheck` 是否重新处理 SRAD。

## 已核实证据

| 项目 | 结果 |
|---|---|
| Container | `nifty_taussig`；项目 bind mount 指向 `/workspace` 与 `/workspaces/gym-dssat-pdi` |
| 已安装包 | `dssat-pdi 4.8.0.24+2~a4ae650-1`，与目标 revision 标签一致 |
| Binary | `/opt/dssat_pdi/dscsm048`；SHA256 `a0714e0fe7cdc6dc62114647142d202286ed37fa2e4dfcded6e0a8b1c08637d5` |
| Wrapper | `/opt/dssat_pdi/run_dssat`；SHA256 `4f32d85a2d5df405785328c6c56209d3abffc784a0890f30380ca50bf881eb07` |
| Debug source root | `/home/gymdev/gym_dssat_pdi/dssat-pdi-4.8.0.24+2~a4ae650`，路径在二进制调试字符串中出现，但目录不存在 |
| Expected `Weather/WGEN.for` | 未找到 |
| `DailyWeatherCheck` implementation | 在限定只读搜索根(`/home/gymdev`、`/opt/dssat_pdi`、`/usr/src`)内没有源码匹配 |
| Binary diagnostic strings | 可作旁证，不足以恢复或绑定具体实现源码 |

因此，包版本与调试路径提供了 revision 线索，binary SHA 提供了当前安装件身份，但**源码内容与实际 executable 的构建绑定仍未闭合**。004_17只保留了 runtime wrapper hash；它不能替代 DSSAT crop executable hash，也不能证明当年二进制与当前二进制完全一致。

## Pilot为何没有启动

004_21的下限候选要求先确认准确补丁版 `WGEN.for` 和 `DailyWeatherCheck`。尤其，0.1 MJ m⁻² d⁻¹只是待核验的候选：binary string显示相关诊断文本，但没有对应源码就不能证明该条件当前如何执行、是否会再次覆盖WGEN的 SRAD 修改，或是否同步影响 PAR。故没有制作candidate binary，也没有启动天气生成。

之前对 `run_dssat` 做版本探测时，命令因缺少初始化所需的 paraconf 文件立即报错退出；该调用没有进入天气生成或作物模拟。本轮 runner 不再调用 `run_dssat`，只用 `docker inspect`、`dpkg-query`、文件查找、`sha256sum` 和二进制字符串/节区读取完成 preflight。

## 产物

- Prompt: `prompt_02/004_22_yc_isolated_srad_floor_weather_only_pilot.md`
- Read-only gate runner: `src/run_004_22_yc_isolated_srad_floor_weather_only_pilot.py`
- Machine result: `results/yc_random_weather_ppo/004_22_yc_isolated_srad_floor_weather_only_pilot/decision.json`
- Detailed gate evidence: `results/yc_random_weather_ppo/004_22_yc_isolated_srad_floor_weather_only_pilot/source_gate.json`
- Raw command output: `results/yc_random_weather_ppo/004_22_yc_isolated_srad_floor_weather_only_pilot/source_gate_raw_output.txt`

## 下一步所需输入

要继续，需取得并放入项目内可审计的位置：与包版本 `4.8.0.24+2~a4ae650-1` 对应的源码包/构建源码，至少包含 `Weather/WGEN.for` 和 `DailyWeatherCheck` 的实现，以及能将该源码 revision 与当前 `dscsm048` SHA256绑定的 package/build provenance。恢复这些证据后，应重新运行 Phase 1 gate；只有 gate 通过，才可按 prompt 做隔离、单因素 weather-only smoke。

当前结果不授权：扩大天气池、运行DSSAT crop experiment、用Observed调参、修改canonical runtime、PPO训练、commit或push。
