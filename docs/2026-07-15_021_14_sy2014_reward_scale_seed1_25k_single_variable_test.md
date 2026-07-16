# 021_14 SY2014 seed1 reward 缩放 25K 单变量对照

## 1. 目的与严格对照

本轮以 021_10 为对照，两组均为 SY2014、IC=2、seed1、`exploration_fraction=0.70`、50K全局探索日程、运行至25K。唯一差异是：

```text
021_10: training_reward_scale = 1.0
021_14: training_reward_scale = 0.1
```

因此，021_10→021_14 的差异可作为相同延长探索配置下 reward 缩放的单变量证据；但021_14的绝对配置仍是“延长探索+缩放”，不能外推到原始探索0.35。

## 2. 执行与边界

- 指定 Docker 容器和虚拟环境；
- 每5K保存模型、replay、RNG、评价、逐步reward与缩放审计；
- 未修改IC、动作、预算、reward相对权重、target interval、观测空间或DSSAT输入；
- 未运行其他seed或站点；
- 正式评价和checkpoint选择继续使用未缩放原始经济reward；
- 训练结束后强制执行固定状态Q值、参数L2和完整氮排序离线审计。

## 3. 10K第一次运行的安全失败

第一次正式运行在10K缩放审计处停止，没有继续至25K。原因不是reward公式错误，而是审计器最初只处理一个callback边界：

- 5K停止时有1条环境已返回、尚未写入replay的transition；
- 10K停止时又产生第2条；
- wrapper记录10000条，replay实际提交9998条；
- 原实现仅排除最后一个边界，导致序列错位。

修复后审计器显式登记每个未提交边界step，并保留足够长的记录覆盖完整环形replay。使用失败现场离线回归测试：

| 项目 | 结果 |
|---|---:|
| wrapper记录 | 10000 |
| 未提交边界 | 5000、10000 |
| 已提交记录 | 9998 |
| replay记录 | 9998 |
| 最大对齐误差 | 0.000012207 |
| 回归测试 | 通过 |

第一次失败的日志、checkpoint、replay和审计全部保留。随后使用新实验ID从头运行，不复用缺失10K评价的失败checkpoint。

## 4. 25K训练结果

| checkpoint | 未缩放产量 | 缩放产量 | 未缩放I/N | 缩放I/N |
|---:|---:|---:|---:|---:|
| 5K | 11074 | 10929 | 120/300 | 120/300 |
| 10K | 11046 | **8120** | 120/300 | 120/300 |
| 15K | 9620 | 10548 | 120/100 | 120/300 |
| 20K | **7352** | **8165** | 120/50 | 120/250 |
| 25K | 10794 | 11199 | 120/300 | 120/300 |

单位：产量 kg/ha，灌溉 mm，施氮 kg/ha。

缩放模型在10K和20K仍出现明显掉产，产量范围8120–11199 kg/ha。虽然比未缩放的7352–11074范围略窄，仍不能称为稳定。25K恢复至11199，只说明该训练轨迹后来再次经过高产策略，不证明训练已收敛。

特别是10K：缩放模型同样用满I120/N300，却只有8120 kg/ha。这说明季节资源总量不能解释全部差异，操作时机和状态依赖决策仍是重要候选因素。

## 5. reward缩放链路审计

5K、10K、15K、20K、25K全部通过：

- 整体缩放恒等式误差为0；
- 分项缩放恒等式误差为0；
- 所有reward有限；
- replay与已提交training reward最大误差始终为`1.2207×10^-5`；
- 未提交callback边界依次明确登记为5000、10000、15000、20000、25000。

因此本轮结果确实来自reward×0.1，不是配置未生效或replay错位。

## 6. Q值与参数过程审计

### 6.1 online平均绝对Q

| checkpoint | 未缩放 | 缩放0.1 |
|---:|---:|---:|
| 5K | 46.21 | 45.54 |
| 10K | 52.68 | 45.82 |
| 15K | 66.29 | 51.00 |
| 20K | 66.69 | 50.56 |
| 25K | 74.70 | 54.70 |

缩放抑制了Q值随训练增长的幅度，但Q并没有简单变成未缩放的0.1倍。神经网络初始化、bootstrap和优化轨迹使这一关系不是线性的。

### 6.2 online参数相对L2变化

| 窗口 | 未缩放 | 缩放0.1 |
|---|---:|---:|
| 5K–10K | 0.317 | 0.217 |
| 10K–15K | 0.292 | 0.221 |
| 15K–20K | 0.253 | 0.186 |
| 20K–25K | 0.226 | 0.205 |

缩放后四个窗口的online参数变化均较小。两次target更新的相对L2也从0.547/0.531降至0.375/0.374。reward尺度确实改变并缓和了数值更新幅度。

### 6.3 online完整氮排序改变数（总计54）

| 窗口 | 未缩放 | 缩放0.1 |
|---|---:|---:|
| 5K–10K | 38 | 39 |
| 10K–15K | 25 | 22 |
| 15K–20K | 17 | 21 |
| 20K–25K | 39 | 28 |

排序重组没有消失。前3个窗口缩放前后相近或互有高低，只有20K–25K明显减少。target网络第一次更新仍改变50/54组氮排序，与未缩放完全相同；第二次从29降至17。不能说缩放系统性消除了动作价值重排。

## 7. 结论

1. reward×0.1是真实生效的单变量，训练和replay链路均通过严格审计。
2. 缩放降低了参数变化和多数窗口的Q值变化幅度，因此reward数值尺度是训练动力学的参与因素。
3. 但缩放模型仍在10K和20K发生严重掉产，完整氮排序仍频繁重组；reward缩放不是充分修复，也不能称为已确认根因。
4. 25K高产结果不能覆盖中间不稳定性；单seed也不能支持跨seed稳定结论。
5. 同样I120/N300却产生10929和8120 kg/ha，提示下一步应关注动作时机、预算状态不可见和操作间隔信息缺失，而不是继续只扫reward尺度。

## 8. 下一步建议

不建议立即扩大reward scale或增加seed。更便宜且更有针对性的下一步是纯离线比较缩放模型5K、10K、15K、20K、25K的实际灌溉/施氮时点和固定状态动作选择，解释“相同总量、不同产量”来自哪里。

完成该时机审计后，再把以下环境定义问题作为单独决策交给用户/导师：是否在观测中增加剩余水预算、剩余氮预算和距上次操作天数，使状态更接近Markov。该改动不能与reward缩放同时进行。

## 9. 输出

- `prompts/021_14_sy2014_reward_scale_seed1_25k_single_variable_test.md`
- `configs/experiments/021_14_sy2014_reward_scale_seed1_25k.yaml`
- `configs/experiments/021_14_sy2014_reward_scale_seed1_25k_retry.yaml`
- `src/test_reward_scale_multiboundary_021_14.py`
- `src/analyze_sy2014_reward_scale_25k_021_14.py`
- `benchmark_results/021_14/021_14_multiboundary_regression_test.json`
- `benchmark_results/021_14/021_14_scaled_vs_unscaled_trajectory.csv`
- `benchmark_results/021_14/021_14_scaled_vs_unscaled_parameter_changes.csv`
- `benchmark_results/021_14/021_14_scaled_vs_unscaled_q_magnitude.csv`
- `benchmark_results/021_14/021_14_scaled_vs_unscaled_q_changes.csv`
- `benchmark_results/021_14/021_14_scaled_vs_unscaled_fixed_state_q_long.csv`
- `benchmark_results/021_14/021_14_scaled_vs_unscaled_process_comparison.png`
- `benchmark_results/021_14/021_14_summary.json`
- 两次正式运行的完整配置、日志、checkpoint、replay和审计文件。

## 10. 状态

`completed`：25K单变量对照与强制过程审计已完成；结论为reward缩放改变数值动力学，但未消除策略振荡。

