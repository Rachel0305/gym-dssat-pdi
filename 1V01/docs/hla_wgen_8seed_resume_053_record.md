# 053 HLA WGEN 八种子恢复：训练前核对

日期：2026-10-09。状态：`PREFLIGHT_COMPLETE_WEATHER_POOL_AND_RUNTIME_GATES_PENDING`。

本轮根据用户要求启动 HLA 天气增强与训练的准备工作，参照已冻结 FQA 051–052 的逐 episode 天气归档、100K checkpoint 和顺序八种子方案。新流程见 `prompt_02/053_hla_wgen_8seed_resume.md`。没有改动历史 HL/FQ/YC 结果，也没有启动新的 PPO 训练。

## 已核对的输入

- 历史 HL 八种子对照：每个 seed 使用精确 100K checkpoint，训练期 2004–2013，lowIC 输入；见 `results/hl_fq_8seed_cross_site/analysis_v1/checkpoint_inventory.csv` 与 `run_metadata.json`。
- 029 已冻结五条拟合/CLI 情景：`GPCC_RAW`、`CPC_RAW`、`GPCC_BIASCORR`、`CPC_BIASCORR`、`ENSEMBLE_BIASCORR`。各 `CNHL.CLI` 当前均存在，大小 3598 字节；038 已核对拟合输入与清单哈希。
- 029 `candidate_count_generated=0`，没有正式通过的 HLA WGEN 候选池。
- 032 仅证明 GPCC_RAW、2007 年、seed101 的单季 WGEN 状态可被捕获且同 seed 可复现；其 `coordinate_gate` 为 `FAIL_NATIVE_FIELD_COORDINATES_REMAIN_PLACEHOLDERS`，候选 QA 未通过。
- 033/034/037 未验证 native FIELD 坐标修复或其物理影响；035/036 缺精确 PDI 源码/构建绑定，不能安全替换运行时。
- 039 的事件源抄录审计通过，但 2013 大雨量级未独立裁决。用户已明确选择保留权威 D222 的原值用于研究；本流程不把未独立裁决写成已验证，也不擅自缩放这些雨值。
- 111 天降水缺口仍由五种格点参考方案分别补入；此前用户选择五方案分别生成限量候选。正式八种子训练输入方案待确定。

## 门禁结论

当前可以推进五方案的有限候选生成和天气质量检查；**尚不满足直接启动八个 100K PPO 的条件**。下一步在独立 053 结果目录进行天气候选与 runtime smoke，并记录坐标问题对候选天气及基线可比性的影响。只有训练输入、天气 QA 和 smoke 明确后，才排队执行长训练。内存控制沿用 FQA 单进程顺序模式。
