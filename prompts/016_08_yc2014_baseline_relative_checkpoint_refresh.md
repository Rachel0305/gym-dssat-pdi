# 016_08 YC2014 baseline-relative checkpoint refresh

## 目标

在不覆盖旧结果、不额外改奖励结构的前提下，复跑一条 YC2014 baseline-relative DQN 主线，确认：

1. 禹城站是否确实有稳定优化空间；
2. 5K 冒烟是否正常；
3. 如果正常，再跑 50K checkpoint 版本，筛选最佳 checkpoint，而不是盲目使用最终步模型。

## 已知背景

- 旧结果显示 YC2014 存在明显优化空间；
- deterministic scan 表明 `I60/N150` 附近已接近高产平台；
- 旧 DQN 结果说明 seed0/seed1 都有机会达到高产，但后期训练会漂移，因此 checkpoint 选择比“最后一步模型”更关键。

## 本轮约束

- 不修改旧脚本核心逻辑；
- 不覆盖 `015_10` 历史结果；
- 使用新包装脚本 `src/run_yc2014_baseline_relative_checkpoint_refresh_016_08.py`；
- 先 5K smoke，再决定是否继续 50K；
- 结果全部写入：
  - `DSSAT_auto_validation/yc2014_baseline_relative_checkpoint_refresh_016_08/`
  - `docs/2026-07-05_016_08_yc2014_baseline_relative_checkpoint_refresh_record.md`

## 执行命令

### 1. 5K smoke

```bash
docker exec b2fd6726c8c1 bash -lc 'cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_yc2014_baseline_relative_checkpoint_refresh_016_08.py --seed 0 --timesteps 5000 --checkpoint --checkpoint-interval 5000'
```

### 2. 若 smoke 正常，再跑 50K checkpoint

```bash
docker exec b2fd6726c8c1 bash -lc 'cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_yc2014_baseline_relative_checkpoint_refresh_016_08.py --seed 0 --timesteps 50000 --checkpoint --checkpoint-interval 5000'
```

## 输出检查

至少检查：

- `.../015_10_yc2014_baseline_relative_checkpoint_summary.csv` 是否生成；
- 最优 checkpoint 的 `final_grain_kg_ha / total_reward / irrigation / fertilizer`；
- 是否出现明显漂移（例如 10K 前后最好，后面下降）。

## 预期解释

- 若 5K 就接近平台：说明 YC2014 仍是强候选主站点；
- 若 50K 中前段最佳、后段退化：说明这条线适合“checkpoint 选优”，不适合“盲信最终模型”；
- 若全程都退化：再回头诊断 reward/动作约束，不直接堆更多步数。
