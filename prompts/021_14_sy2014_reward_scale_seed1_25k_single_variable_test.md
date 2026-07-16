# 021_14 SY2014 seed1 reward 缩放 25K 单变量对照

## 目标

在 `021_10` SY2014 IC=2 seed1、`exploration_fraction=0.70` 的基础上，只把训练 reward 的整体尺度从 1.0 改为 0.1，运行至 25K，判断数值缩放是否改变中期策略振荡、Q 值尺度和动作排序重组。

## 严格对照

| 项目 | 对照 021_10 | 本轮 021_14 |
|---|---:|---:|
| station/year | SY2014 | SY2014 |
| IC | 2 | 2 |
| seed | 1 | 1 |
| exploration_fraction | 0.70 | 0.70 |
| total schedule | 50K | 50K |
| run until | 25K | 25K |
| checkpoint interval | 5K | 5K |
| reward相对权重 | yield:water:N = 1:1:5 | 相同 |
| training reward scale | 1.0 | **0.1** |

因此，`021_10` 与 `021_14` 的差异是 reward 缩放的单变量证据；但本轮绝对配置仍属于“延长探索 + 缩放”，不能外推为原始探索0.35下的结论。

## 执行纪律

1. 先 dry-run，确认 config hash、SY2014 IC=2、seed1 和唯一变量；
2. 021_13 smoke 已通过，不重复 smoke；
3. 使用指定 Docker 容器和虚拟环境；
4. 保存5K、10K、15K、20K、25K checkpoint、replay、RNG、评价和缩放审计；
5. 不因中间结果好看提前停止过程审计；
6. 无论最终产量如何，训练完成后都执行：
   - 参数绝对/相对 L2；
   - 18个固定状态的online/target Q变化；
   - 54组N0/N50/N100完整排序改变数；
   - 18状态全局argmax改变数；
   - 与021_10同checkpoint产量、水、氮、原始reward轨迹对照；
7. 不改观测空间、target interval、成本相对权重、动作、预算、IC或DSSAT输入；
8. 不启动seed0/seed2或其他站点实验。

## 判定边界

- 单个seed平稳不能证明跨seed稳定；
- 没有坍缩不能证明reward尺度是唯一或普遍根因；
- 若过程重组减弱，只能说在相同0.70探索配置下，缩放与更温和的训练动力学一致；
- 若最终策略改善但中间仍剧烈振荡，不能称为训练稳定；
- 若没有改善，reward尺度假设在该单变量设置下不受支持。

## 输出

- `configs/experiments/021_14_sy2014_reward_scale_seed1_25k.yaml`
- `benchmark_results/021_14/...`
- 过程对照CSV/JSON/PNG；
- `docs/2026-07-15_021_14_sy2014_reward_scale_seed1_25k_single_variable_test.md`

