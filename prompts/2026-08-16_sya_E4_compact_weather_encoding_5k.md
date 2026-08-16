# E4：压缩天气特征表达（2K smoke → 5K seed0）

## 目的

E3 已表明：天气信号对 E2 有帮助，但双分支结构本身没有带来优势。E4 只改变天气分支中的信息表达，检验 12 个天气指标是否存在冗余或噪声。

## 唯一改变

保留 E2 的双分支网络和 12 维天气输入宽度，但只保留 6 个核心天气指标的实际值，其余 6 个位置固定为 0：

- 保留：`rain_next1_norm`、`rain_next3_norm`、`rain_next7_norm`、`first_rain_lead_norm`、`tmax_max_next7_norm`、`dry_spell_next7_flag`；
- 置零：`rain_max_next7_norm`、`rain_days_next7_norm`、`tmax_mean_next7_norm`、`tmin_mean_next7_norm`、`srad_mean_next7_norm`、`valid_days_next7_norm`。

原始天气值和归一化天气值的列仍全部写入审计文件；置零列必须在每个决策时刻严格为 0。

## 必须保持不变

- 奖励函数、奖励权重、动作评价、安全约束和 action mask；
- 16 格动作网格：灌溉 `[0,15,30,45]` mm × 氮肥 `[0,40,80,120]` kg/ha；
- 046_02 raw observation 基础分支、E2 双分支 extractor、25+12 观测宽度、64+32 latent、96 维 combined features、下游 `[64,64]`；
- SYA / originIC、训练年 2005–2013、验证年 2014–2023、seed0；
- 5K 训练计划 `[2000,5000]`，不运行 10K 或更长训练。

## 2K smoke 通过条件

1. 观测维度为 37（25 个基础值 + 12 个天气槽位）；
2. 6 个保留天气列来自未来 1–7 天窗口，并且在季节内有变化；
3. 6 个置零列在观测和 daily CSV 中全部为 0；
4. 10 个验证年份完整，动作均在 16 格内；
5. 存在 DAP1 之后的正灌溉或正施氮动作；
6. 训练、奖励、动作审计没有异常。

## 结果判定

- 若 E4 在产量、WP_ET、PFP-N 中至少一项相对 E2/no-forecast 有一致改善，再做独立 seed 验证；
- 若 E4 仍不优于 raw no-forecast，则停止继续堆天气特征，转向动作时序或 forecast 不确定性建模；
- WP_ET 必须对最终 5K checkpoint 做同口径 DSSAT ETCP replay 后再下结论。
