# 012_05 HLA2015 economic reward DQN seed0 跨年验证记录

## 目的

在 012_03/012_04 中，HLA2010 economic reward DQN 已经出现了一个重要现象：

- seed0：高产、I120、N50；
- seed1：高产、I120、N100；
- 两个 seed 都没有打满 N150。

因此本实验把同一套 DQN 离散动作、同一套 economic reward、同一套窗口和预算设置迁移到 HLA2015，先跑 seed0，检查它是否也能做到“高产 + 少氮”。

## 实验设置

- 站点年份：HLA2015
- 算法：DQN
- 训练步数：5000
- seed：0
- 动作空间：
  - 0：不操作
  - 1：灌溉 30 mm
  - 2：施氮 50 kg/ha
  - 3：灌溉 30 mm + 施氮 50 kg/ha
- 灌溉预算：120 mm
- 追加氮预算：150 kg/ha
- 灌溉窗口：DAP 20–35、45–65、70–95
- 施氮窗口：DAP 25–40、55–70
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

这里的水氮成本仍是诊断用的“籽粒等价成本”，不是最终论文中的经济价格参数。

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_dqn_economic_reward_probe_012_03.py --year 2015 --timesteps 5000 --seed 0 --water-cost 1.0 --nitrogen-cost 5.0 --label medium_N_cost"
```

## 输出文件

- 训练/评估输出目录：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/2015/medium_N_cost_seed0_5000steps/
```

- 四情景过程图：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_05_hla2015/hla2015_four_scenario_with_economic_dqn_seed0_process.png
```

- 四情景日值数据：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_05_hla2015/hla2015_four_scenario_with_economic_dqn_seed0_daily.csv
```

- 四情景汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_05_hla2015/hla2015_economic_dqn_four_scenario_summary.csv
```

- 绘图脚本：

```text
src/plot_hla2015_dqn_economic_012_05.py
```

## 结果汇总

| 情景 | 产量 kg/ha | 生物量 kg/ha | 灌溉 mm | 施氮 kg/ha | 最大水分胁迫 | 最大氮胁迫 |
|---|---:|---:|---:|---:|---:|---:|
| null zero | 6486 | 17168 | 0.0 | 0.0 | 1.000 | 0.093 |
| 2007专家策略平移 | 7296 | 18639 | 30.0 | 165.0 | 0.779 | 0.015 |
| DSSAT auto irrigation + auto-N attempt | 7648 | 19021 | 141.5 | 0.0 | 0.000 | 0.047 |
| Economic DQN seed0 | 6485.8 | 17167.8 | 0.0 | 0.0 | 1.000 | 0.093 |

## 关键观察

1. HLA2015 economic DQN seed0 没有产生任何有效灌溉或施肥事件。
2. 其产量、胁迫和生物量几乎与 null zero 完全一致。
3. 日值评估文件显示模型在窗口外曾输出 raw action，但这些动作被安全窗口拦截；在真正允许操作的窗口内，没有形成有效管理动作。
4. 因此，HLA2010 中出现的“高产 + 少氮”行为没有直接跨年复现。

## 当前结论

这一结果是负结果，但非常重要：

- HLA2010 economic reward DQN 的 seed 稳定性较好；
- 但同一设置迁移到 HLA2015 后，seed0 直接退化为 null；
- 因此目前不能说 DQN economic reward 已经是稳健的跨年水氮联合优化方案。

更准确的表述是：

> DQN + economic reward 在 HLA2010 上表现出有希望的少氮高产行为，但该行为尚未跨年份稳定复现。当前限制可能来自年份差异、训练样本不足、窗口设计、奖励稀疏性或 DQN 探索不足，需要进一步诊断。

## 下一步建议

不要马上继续跑 HLA2015 seed1。因为 seed0 已经等同 null，直接加 seed 可能只是再生成一个不清楚原因的失败样本。

建议下一步先做低成本诊断：

1. 用 HLA2015 的固定动作反事实测试，检查在 2015 年同样的窗口下，I120/N50、I120/N100、I120/N150 是否确实能明显增产。
2. 如果固定动作能增产，说明问题主要在 DQN 没学到动作；
3. 如果固定动作也不增产，说明 2015 这个年份下经济奖励选择不操作可能是合理的，需要重新解释目标函数。

