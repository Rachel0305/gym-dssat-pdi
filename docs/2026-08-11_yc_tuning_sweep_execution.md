# 2026-08-11 YC 单因素 PPO 调参 2K sweep

## 预注册

固定 lowIC、YCA/YC、2004--2013 train、2014--2023 validation、seed0、16 动作网格、原始 observation、reward/safety 与 `weather_forecast_enabled=false`。基线有效 kwargs 为 `lr=3e-4, gamma=1, n_steps=144, batch=144, n_epochs=5, ent_coef=0.01, net=[64,64]`。

| 候选 | 唯一改变 | 状态 |
| --- | --- | --- |
| A | `net_arch=[32,32]` | 已完成 2K。 |
| B | `learning_rate=1e-4` | 已完成 2K。 |
| C | `ent_coef=0.02` | 已完成 2K。 |

## A：net32 2K

容器 `nifty_taussig` 内 dry-run 通过，随后单进程运行 2K（1K/2K checkpoint）。隔离输出为 `benchmark_results/065_a_yca_lowIC_tuning_net32_maskableppo_smoke2k`，未修改 055 原始结果。

checkpoint 2K 验证均值：yield `8201.46 kg/ha`，灌溉 `214.5 mm`，施氮 `240 kg/ha`，`PFP_N=34.17`；此 run 未产出同口径 `WP_ET`，不推断。

机制审计：10/10 daily 存在、off-grid 水/氮为 0、transmission mismatch 为 0、每年均有 DAP1 后动作且使用 3--4 个非零动作对（总体不少于 3），跨年并非单一签名。因此 A 通过机制 smoke gate，但不代表性能成功，也不授权 25K/100K。

## B/C 结果（checkpoint 2K）

| 候选 | 有效唯一改变 | yield kg/ha | 灌溉 mm | N kg/ha | `PFP_N` | DAP1 后正动作 | transmission mismatch | 机制 gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | net `[32,32]` | 8201.46 | 214.5 | 240 | 34.17 | 107 | 0 | 通过 |
| B | lr `1e-4` | 8202.52 | 214.5 | 200 | 41.01 | 113 | 0 | 通过 |
| C | ent_coef `0.02` | 8188.98 | 189.0 | 240 | 34.12 | 87 | 0 | 通过 |

B/C 均先通过 dry-run，effective kwargs 与预注册一致，且各自约 81--83 秒完成、RSS 增量约 124 MB。三个候选在 10 个验证年均有 daily 输出、off-grid 为 0、request→safe 的累计传输 mismatch 为 0，均使用至少 3 个非零动作对并存在 DAP1 后动作，未退化为跨年单一签名。三者都没有同口径 `WP_ET` 输出，故不作推断。

唯一可申请下一阶段的候选是 **B（learning_rate=1e-4）**：与 A/C 相比，其 yield 近似，但在较低总 N（200 vs 240 kg/ha）下 `PFP_N=41.01`，高于 A 的 34.17 和 C 的 34.12。这个判断仅是 2K 机制/资源效率筛选，不是论文性能结论，也不自动授权训练。

## 执行限制

三个候选已保持串行完成；没有主代理另行授权时，B 也不得进入 25K/100K。

## B 的已授权 25K rescue

B 随后在独立 25K rescue 中完成 5K/10K/25K checkpoint，但 25K 的 DAP1 后动作降为 0、yield 降至 5992.08 kg/ha，机制 gate 失败。因此不进入 100K；详见 `docs/2026-08-11_yc_lr1e4_rescue25k.md`。
