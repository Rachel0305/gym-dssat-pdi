# 219YCA 归一化 PPO 结果包

## 结论

观测归一化后的 PPO 改善了管理动作的时间分布和过程合理性：动作不再只集中在生长季初期，且不同年份的动作日期出现差异。但它没有证明能够根据年份天气自适应调整全年水氮总量；20K 末期两个 seed 仍基本固定为 `I=150 mm`、`N=240 kg/ha`。

与旧 218YCA-5K 相比，产量、利润和 WP_ET 只是在部分年份或部分 seed 有小幅变化，PFP_N 没有整体改善，因此当前结果仍不达标，不能作为最终 PPO 候选。

## 实验边界

- 站点：YC/YCA lowIC。
- 唯一训练改动：输入观测归一化 `VecNormalize(norm_obs=True)`。
- 动作组：原 16 个灌溉—施氮组合。
- 奖励：产量奖励减灌溉和施氮成本；没有增加胁迫惩罚。
- 训练：两个 seed，检查点至 20,160 步；未继续 500K。
- 2014–2023 是开发验证，不作为未见最终测试集。

## 图形说明

本目录中的逐年五情景日过程图和两张逐年柱状图，复用了已保存的 218YCA-5K 五情景 replay 数据和历史颜色，不把 219 的真实天气验证误标为五情景 replay。

若要得到 219YCA 归一化 checkpoint 的同格式五情景图，需要另做一次五情景 replay；这属于评估，不是重新训练。

## 主要文件

- `219YCA_result.json`：正式 pilot 结果。
- `tables/219YCA_checkpoint_summary.csv`：检查点汇总。
- `tables/219YCA_seed1_ck20160_vs_218YCA5k_yearly.csv`：逐年对照数据。
- `figures/219YCA_ycaYYYY_five_scenario_daily.png`：逐年五情景过程图。
- `figures/219YCA_five_scenario_management_bars.png`：逐年水氮投入柱状图。
- `figures/219YCA_five_scenario_metrics_bars.png`：逐年产量、WP_ET、PFP_N 柱状图。
