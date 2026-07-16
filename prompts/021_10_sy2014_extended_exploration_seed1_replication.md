# 021_10 SY2014 延长探索衰减 seed1 复现

## 目标

复核021_09 seed0结果能否跨seed复现：在SY2014 IC=2上，将`exploration_fraction=0.70`的单变量配置训练至25K，检查15K、20K、25K是否持续保持高产和有意义施氮，不发生N300→N0坍缩。

## 严格控制变量

与021_09 seed0正式25K完全一致，仅将随机种子从0改为1：

- reward及系数不变；
- IC=2及输入哈希不变；
- 9动作空间、I120/N300预算、单次上限、共享7日间隔、DAP窗口不变；
- DQN学习率、buffer、batch、n-step、gamma、target interval不变；
- `exploration_fraction=0.70`、initial epsilon=1.0、final epsilon=0.05；
- 全局计划50K，本次停止25K，checkpoint间隔5K。

## 算力纪律

021_09 seed0已完成同一代码路径和配置的5K smoke，epsilon、IC=2、runtime audit、replay/RNG保存及内存均通过。因此本次不重复相同工程smoke，先dry-run核对差异仅为seed，再执行一个seed的25K；不得并行其他训练。

## 判定

1. 若seed1在20K和25K仍保持产量>10000 kg/ha且施氮>0，则支持延长探索可跨两个seed避免25K前坍缩；
2. 若seed1仍坍缩，则021_09只能视为seed0特例；
3. 无论结果如何，不以单个最佳checkpoint代替完整轨迹；
4. 即使复现，也不代表节水节氮目标完成，因为当前策略可能继续打满I120/N300；
5. 不修改其他参数，不扩展站点或seed2。

## 输出

- 独立YAML与输出目录；
- 5K–25K checkpoint、replay buffer、RNG state；
- 日值和季节汇总CSV；
- seed0/seed1同checkpoint比较CSV和图；
- 中文实验记录；
- 不执行Git push。
