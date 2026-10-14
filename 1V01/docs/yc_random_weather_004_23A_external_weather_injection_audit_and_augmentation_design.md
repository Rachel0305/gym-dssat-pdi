# YC 外部天气接入路径审计与增强设计（004_23A）

## 执行结论

```text
external_weather_injection_status = STATICALLY_SUPPORTED_UNSMOKED
augmentation_design_status = DESIGN_ONLY
weather_generation_started = false
dssat_started = false
ppo_started = false
next_step = WEATHER_ONLY_INJECTION_SMOKE
```

本轮是源码/provenance 静态审计和设计，不是天气生成或 crop experiment。没有导入 Gym/DSSAT，没有加载模型，没有读写旧天气文件；唯一写入是 004_23A 新产物。

## 接入链审计

| 来源 | 位置 | 观察到的证据 | 证据状态 |
|---|---|---|---|
| `004_03_runner` | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py:453` | args["random_weather"] = regime == "RANDOM_WEATHER_WGEN" | VERIFIED_IN_PROJECT_SOURCE |
| `004_03_runner` | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py:461` | args["auxiliary_file_paths"] = auxiliary | VERIFIED_IN_PROJECT_SOURCE |
| `004_03_runner` | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py:471` | instance._rseed1 = int(bootstrap_weather_seed) | VERIFIED_IN_PROJECT_SOURCE |
| `004_03_runner` | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py:398` | wgen_text, replacements = re.subn( | VERIFIED_IN_PROJECT_SOURCE |
| `004_03_runner` | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py:400` | r"\g<1>W", | VERIFIED_IN_PROJECT_SOURCE |
| `004_03_runner` | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py:404` | if not replacements or not modes or any(mode != "W" for mode in modes): | VERIFIED_IN_PROJECT_SOURCE |
| `004_03_runner` | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py:415` | args["fileX_template_path"] = str(target) | VERIFIED_IN_PROJECT_SOURCE |
| `yc_renderer` | `src/ppo_safe_rendering.py:79` | standard_name = f"{info['weather_prefix']}{int(year) % 100:02d}01.WTH" | VERIFIED_IN_PROJECT_SOURCE |
| `yc_renderer` | `src/ppo_safe_rendering.py:91` | return f"{info['weather_prefix']}{int(year) % 100:02d}01" | VERIFIED_IN_PROJECT_SOURCE |
| `yc_renderer` | `src/ppo_safe_rendering.py:303` | shutil.copyfile(source_weather, rendered_weather) | VERIFIED_IN_PROJECT_SOURCE |
| `yc_renderer` | `src/ppo_safe_rendering.py:336` | "random_weather": False, | VERIFIED_IN_PROJECT_SOURCE |
| `yc_renderer` | `src/ppo_safe_rendering.py:340` | "auxiliary_file_paths": [str(cultivar), str(weather), str(soil)], | VERIFIED_IN_PROJECT_SOURCE |
| `gym_dssat_source_snapshot` | `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/dssat_pdi.py:92` | self.wther = 'W' if random_weather else 'M' | VERIFIED_IN_PROJECT_SOURCE |
| `gym_dssat_source_snapshot` | `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/dssat_pdi.py:339` | def _copy_auxiliary_files(self): | VERIFIED_IN_PROJECT_SOURCE |
| `gym_dssat_source_snapshot` | `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/dssat_pdi.py:342` | shutil.copyfile(path, f'{self._tmp_folder}/{file_name}') | VERIFIED_IN_PROJECT_SOURCE |
| `gym_dssat_source_snapshot` | `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/dssat_pdi.py:275` | cwd=self._tmp_folder, | VERIFIED_IN_PROJECT_SOURCE |
| `yc_filex_template_snapshot` | `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/configs/maize/UFGA8201.jinja2:27` | @L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME | VERIFIED_IN_PROJECT_SOURCE |
| `yc_filex_template_snapshot` | `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/configs/maize/UFGA8201.jinja2:116` | @N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL | VERIFIED_IN_PROJECT_SOURCE |
| `weather_archive_contract` | `docs/random_weather_archival_contract.md:7` | - Preserve the actual runtime WTH file byte-for-byte when the runtime emits one; copy it before cleanup or reuse. | VERIFIED_IN_PROJECT_SOURCE |
| `weather_archive_contract` | `docs/random_weather_archival_contract.md:8` | - Preserve the canonical daily series with `DATE,DAP,SRAD,TMAX,TMIN,RAIN` and document how dates/DAP were derived. | VERIFIED_IN_PROJECT_SOURCE |
| `weather_archive_contract` | `docs/random_weather_archival_contract.md:20` | Call `src/archive_runtime_weather.py::archive_runtime_weather` immediately after the episode weather states have been captured and before closing the environment. Pass an actual runtime WTH path only when one exists. Keep archives under the experiment's own `weather_archive/` directory; never write them into old experiment outputs. | VERIFIED_IN_PROJECT_SOURCE |
| `004_22_report` | `docs/yc_random_weather_004_22_isolated_srad_floor_weather_only_pilot.md:6` | source_build_gate = FAIL | VERIFIED_IN_PROJECT_SOURCE |
| `004_22_report` | `docs/yc_random_weather_004_22_isolated_srad_floor_weather_only_pilot.md:8` | weather_generation_started = false | VERIFIED_IN_PROJECT_SOURCE |
| `004_22_report` | `docs/yc_random_weather_004_22_isolated_srad_floor_weather_only_pilot.md:13` | 004_22 已按顺序执行只读 source/build preflight。停止是任务预设安全门槛的结果，不是 pilot 失败或 WGEN 科学结论：当前容器没有可核对的安装版 Fortran 源码，因此不能安全修改/重建 SRAD 下限，也不能验证 `DailyWeatherCheck` 是否重新处理 SRAD。 | VERIFIED_IN_PROJECT_SOURCE |

项目代码显示 YC renderer 会按年构造站点 weather stem，并将 `.WTH` 加入 `auxiliary_file_paths`；环境 source snapshot 显示 auxiliary 文件复制到 DSSAT 临时目录，DSSAT 子进程以该目录为 working directory。004_03 的实验代码区分 `WTHER=W` 的 WGEN 方案与 measured-weather 路径；该项目历史也记录了 `WTHER=M` 未启动 WGEN、改用任务级 `WTHER=W` 后才得到随机 WGEN 天气。

**静态可行路径**：在任务隔离输入目录中，把 fitting-only candidate WTH 放到 FileX `WSTA` 所指向的 basename，并将该文件路径作为 weather auxiliary input；FileX 使用 measured-weather 模式。这样复用现有文件输入链，不需要改 canonical runtime 或重编 DSSAT。

**尚未证明**：本轮没有启动 runtime，因而没有证明特定候选文件被安装版 DSSAT 正确解析并逐日消费；项目 source snapshot 与当前安装二进制的完整 build identity 也没有在本任务重新绑定。故结果只达到 `STATICALLY_SUPPORTED_UNSMOKED`，不能声称 injection 已通过运行验证。

## 输入与归档状态

| 资产 | 路径 | SHA256 | 状态 |
|---|---|---|---|
| `004_03_runner` | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py` | `042B36B84B1BFE2C2085068CAE987B4EFBC8E575F6820ECE2D277EDB88CA9136` | READ_ONLY_INPUT |
| `yc_renderer` | `src/ppo_safe_rendering.py` | `0714DEEA114C3FDB029BAF1210A1A09E650F4418EF6AA52DFC9E90DE2A4E15CD` | READ_ONLY_INPUT |
| `gym_dssat_source_snapshot` | `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/dssat_pdi.py` | `960784A95FBF3D68027B659D441CC91F744713417D5D1DEDA58C989EF78A2058` | READ_ONLY_INPUT |
| `yc_filex_template_snapshot` | `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/configs/maize/UFGA8201.jinja2` | `867C8FB5F66960D06597A3C645F34371AA1D33EF75BACDE152E20B606567543C` | READ_ONLY_INPUT |
| `weather_archive_contract` | `docs/random_weather_archival_contract.md` | `E1438A44D0614A60211F2830967A2C73865EBC080E80B58AAB69F5DEC52059E4` | READ_ONLY_INPUT |
| `004_17_decision` | `results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/decision.json` | `47590F63E70DAC404851214D17E6BAD5C9AE173EFC9030AB010D622F1769BC0E` | READ_ONLY_INPUT |
| `004_17_verified_manifest` | `results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/weather_archive_manifest_verified.csv` | `1B70A4B337041425E319C728EFDA2D0F71DDDBB2DE96BD02A390CEAE097B6411` | READ_ONLY_INPUT |
| `004_21_decision` | `results/yc_random_weather_ppo/004_21_yc_wgen_srad_generation_mechanism_and_correction_design/decision.json` | `5EB0D26CB785F138BFA9DF6FC44EDB197C384B889A8C48C966FC247A82FA4F96` | READ_ONLY_INPUT |
| `004_22_decision` | `results/yc_random_weather_ppo/004_22_yc_isolated_srad_floor_weather_only_pilot/decision.json` | `FA55F7D07CD257524076869EC8013D7716B14BD914A9D51B8F94A95309E1AB90` | READ_ONLY_INPUT |
| `004_22_report` | `docs/yc_random_weather_004_22_isolated_srad_floor_weather_only_pilot.md` | `92B6FDBE39C7291D937FA45A90B5411DEA1436A262F46D8AA5334F2E3E9C64DE` | READ_ONLY_INPUT |
| `fitting_weather` | `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv` | `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34` | READ_ONLY_INPUT |
| `frozen_cli` | `results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI` | `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929` | READ_ONLY_INPUT |

004_17 verified manifest 共登记 100 行；按 set 统计为 `{"training": 80, "heldout": 20}`，manifest 指向且当前可见的 canonical CSV 为 100 个。该核对仅确认归档清单及文件存在，不在本任务重算气候覆盖结论。004_17 历史 raw WTH 状态是 `NOT_EMITTED_OR_NOT_CHANGED_FROM_INPUT`；canonical state-derived CSV 与 raw WTH 必须继续区分。

## 增强候选比较

| 候选方法 | 保留结构 | 可增加的支持 | 主要风险 | 本轮定位 |
|---|---|---|---|---|
| Whole-season donor resampling | 整季RAIN/SRAD/TMAX/TMIN联合轨迹、全部原始日序 | 重抽样不会超出2004-2013供体支持；唯一季节数仍有限 | 年份级样本仅10个，重复率高；标签年份与实际气候供体必须分开记录 | 低复杂度基线，不作为最终扩增首选 |
| Season/date-context-aligned multivariate moving-block bootstrap | 块内四变量联合关系、短期自相关与局部wet/dry和热/辐射事件；DAP仅在fitting context可核验时使用 | 可组合fitting供体块形成新季节序列，但不保证超出单变量或联合供体支持 | 拼接处可能产生不连续；block长度与DAP/月分层须fitting-only预先冻结 | 推荐的最小 weather-only pilot 候选 |
| Fitting-only conditional multivariate residual synthesis | 取决于模型；可显式保留阶段/湿干条件及变量协方差 | 可能探索供体未出现的变量组合；必须证明物理与统计可行 | 10个年份支撑参数有限；模型错设、尾部放大、RAIN离散分布与物理约束伪影 | 较复杂备选；只有块重采样不足且QA失败机制明确后再设计 |

### 推荐的最小起步设计

先核验 2004–2013 fitting weather 日期与 YC planting/season context 的映射；只有起点/日期对齐可追溯时才按 DAP 分段，否则按固定月份匹配。随后做多变量连续 block resampling，整体搬运每日 `RAIN/SRAD/TMAX/TMIN` 向量，不做逐变量独立扰动。block 长度和分段规则须仅用 fitting 数据的连续性诊断预先冻结；随后做 fitting-only leave-one-year-out 检查，评估分布保持、时间结构与块边界伪影。整季 donor resampling作为低复杂度基线；条件多变量 residual synthesis暂列备选，只有前者无法满足已预先定义的天气 QA 时才考虑。

这类基于历史供体/块的方案可以把 fitting期已有的低辐射、降雨和温度组合保留到候选 weather 中，但不会凭空保证超出 2004–2013 支持；也不能预先宣称一定覆盖 2014–2023 中已暴露的年份。Observed 2014–2023仅可在方法冻结后作描述性外部 QA，不能参与参数拟合或方法优选。该约束与用户已明确的 YC 目标一致：关注各验证年的产量、水氮投入、效率与 reward 的尽可能好的权衡，而非要求每个单项指标都成为五情景冠军或改写成单一跨年均值目标。 

## 下一阶段 gate

建议先做 004_23B 的单情景 weather injection loop smoke：选一份现有、已知 provenance 的 YC WTH，在任务独立目录中按 WSTA basename 接入，确认 FileX mode、传入文件 hash、实际 runtime daily-state 的日期和 `SRAD/TMAX/TMIN/RAIN` 逐日一致，并确认临时目录/输入隔离。此 smoke 只验证路径闭环，不调 PPO、不更改配置、不形成作物性能结果。通过后再冻结 fitting-only block construction 和小批候选 weather-only QA；初始 20–40 套只是建议规模，非本轮执行。

### QA gate

- date/DAP 对齐、日序连续、目标 season 长度及 leap-year 规则均可追溯；
- 缺测/哨兵、单位、`RAIN >= 0`、`SRAD >= 0`、`TMAX >= TMIN` 检查；
- SRAD 与太阳几何 plausibility、降雨 occurrence/amount、热/干事件持续性和多变量联合结构；
- 与 fitting donors 的来源映射、block seam诊断、候选间重复率；
- raw WTH 与 canonical daily series 分开归档、各自 SHA256、FileX/WSTA/CLI/runtime provenance、episode到 archive ID映射；
- 2014–2023只作已暴露的外部描述性 QA，不用于生成参数或调参；held-out WGEN维持隔离。

## 当前边界与决策

004_22 的 source/build gate 阻止的是修改 DSSAT 内部 WGEN，不阻止未来复用外部 measured-weather 输入路径；也不自动证明外部接入成功。当前不建议直接做 `80→160` 同分布 WGEN扩容或 PPO retraining。下一步先通过一份既有天气文件的注入回环，之后才考虑 fitting-only weather candidate pilot。

```text
external_weather_injection_status = STATICALLY_SUPPORTED_UNSMOKED
augmentation_design_status = DESIGN_ONLY
weather_generation_started = false
dssat_started = false
ppo_started = false
next_step = WEATHER_ONLY_INJECTION_SMOKE
```

## 产物

- Prompt：`prompt_02/004_23A_yc_external_weather_injection_audit_and_augmentation_design.md`
- Runner：`src/run_004_23A_yc_external_weather_injection_audit_and_augmentation_design.py`
- 机器审计：`results/yc_random_weather_ppo/004_23A_yc_external_weather_injection_audit_and_augmentation_design/`
