# 021_10 SY2014 延长探索衰减 seed1 复现

## 状态

- dry-run：**passed**
- seed1 25K：**completed**
- 训练失败：**0**
- OOM：**未发生**
- 严格跨seed稳定：**未通过**
- 25K永久N0锁死：**两个seed均未发生**

## 目的与控制变量

021_09 seed0显示，将`exploration_fraction`从0.35延长至0.70后，原配置20K出现的N300→N0永久坍缩在25K前消失。本轮只将seed0改为seed1，其他全部不变，判断是否能跨seed复现。

- 站点年份：SY2014；
- IC：确认的IC=2；
- reward、动作空间、I120/N300预算、7日间隔、DAP窗口不变；
- DQN学习率、buffer、batch、n-step、gamma、target interval不变；
- exploration fraction=0.70；
- 全局计划50K，本次停止25K，每5K保存和确定性评估。

021_09已对相同代码、输入和参数链路完成5K smoke，本轮复用该工程证据以节省算力，只先做dry-run核对seed差异。

## 可追溯性

- 配置：`configs/experiments/021_10_sy2014_extended_exploration_seed1_25k.yaml`
- config hash：`4e74e5479503a45a69bcf6962004313fbaf55535a4cca0db444dbb351d4996b6`
- dry-run：1个SY2014 seed1案例，无结果复用；
- runner：`status=completed`、`failures=[]`；
- 运行约325秒，无OOM；
- 5个checkpoint均保存model、replay buffer、RNG state及确定性评估。

## seed1结果

| checkpoint | epsilon | 产量kg/ha | 生物量kg/ha | 灌溉mm | 施氮kg/ha | reward |
|---:|---:|---:|---:|---:|---:|---:|
| 5K | 0.864 | 11074 | 19522 | 120 | 300 | 4046.46 |
| 10K | 0.729 | 11046 | 19752 | 120 | 300 | 4018.16 |
| 15K | 0.593 | 9620 | 17723 | 120 | 100 | 3591.74 |
| 20K | 0.457 | 7352 | 13978 | 120 | 50 | 1573.75 |
| 25K | 0.321 | 10794 | 19245 | 120 | 300 | 3765.72 |

seed1在15K和20K出现明显但非永久的氮投入/产量下降；25K又恢复到N300和10794 kg/ha。整个过程中灌溉始终为I120。

## seed0/seed1对比

| checkpoint | seed0产量 | seed0氮 | seed1产量 | seed1氮 |
|---:|---:|---:|---:|---:|
| 5K | 11199 | 300 | 11074 | 300 |
| 10K | 11145 | 300 | 11046 | 300 |
| 15K | 11070 | 300 | 9620 | 100 |
| 20K | 11068 | 300 | 7352 | 50 |
| 25K | 11007 | 300 | 10794 | 300 |

## 判断

### 得到支持的结论

1. 与旧0.35配置20K后永久停在N0不同，0.70配置的两个seed在25K时都保持或恢复到N300和约10.8–11.0 t/ha；
2. 延长探索可能提高了策略从错误低氮区域重新恢复的能力；
3. 探索调度确实是训练动力学的重要因素，不是无效参数。

### 未得到支持的结论

1. “延长探索已经消除训练坍缩”：**不成立**。seed1在15K–20K仍明显退化；
2. “只有epsilon降到0.05才会坍缩”：**不成立**。seed1在epsilon=0.593/0.457时已经退化；
3. “两个seed所有checkpoint稳定一致”：**不成立**；
4. “已经形成节水节氮策略”：**不成立**。两个seed最终仍使用I120/N300；
5. “可以恢复五站点正式结论”：**不成立**。

## 科学解释边界

本轮更准确的解释是：**延长探索降低了策略永久锁死在N0的风险，但没有解决online Q策略在训练中期的大幅波动。** 探索不足是参与机制之一，而不是唯一根因。seed1在20K低氮、25K恢复，也说明只挑最佳checkpoint会掩盖策略轨迹的不稳定性。

## 下一步建议

暂不直接增加seed2。优先对seed1的10K、15K、20K、25K做与021_06/021_07相同的纯离线Q排序、target更新、TD残差和replay动作分布审计，重点判断：

1. 15K–20K低氮阶段是否出现online氮排序反转；
2. 20K→25K恢复是否与target网络更新或replay中高氮样本恢复同步；
3. 这种“下降后恢复”是否进一步支持部分可观测/动作混叠造成的价值波动。

确认机制后再决定seed2或环境观测修复，避免继续堆训练。

## 输出

- `prompts/021_10_sy2014_extended_exploration_seed1_replication.md`
- `configs/experiments/021_10_sy2014_extended_exploration_seed1_25k.yaml`
- `src/finalize_sy2014_extended_exploration_seed1_021_10.py`
- `benchmark_results/021_10/021_10_seed0_seed1_checkpoint_comparison.csv`
- `benchmark_results/021_10/021_10_seed0_seed1_checkpoint_comparison.png`
- `benchmark_results/021_10/021_10_seed_replication_summary.json`
- `benchmark_results/021_10/021_10_sy2014_extended_exploration_seed1_25k__sy_2014_seed1/`
- 本记录。

## Git

- 用户计划稍后手动提交；本轮未执行commit或push。
