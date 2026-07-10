# 018_08 SY2014 seed1 最小复现实验

## 目的

在不覆盖 `017_08` 旧结果的前提下，补做 `SY2014` 的 `seed1` 最小复现实验，判断它是否能在与 `seed0` 相同的 DQN 训练框架下复现“高产 + 合理资源配置”的结果。

## 约束

- 必须使用指定 Docker 容器：`b2fd6726c8c1`
- 必须使用指定解释器：`/opt/gym_dssat_pdi/bin/python`
- 严格先做 5K smoke，再决定是否继续 50K
- 不覆盖 `017_08` 任何旧 CSV / 图 / 记录
- 输出全部写入：
  - `DSSAT_auto_validation/sy2014_seed1_minimal_reproduction_018_08/`
  - `docs/2026-07-09_018_08_sy2014_seed1_minimal_reproduction_record.md`

## 执行脚本

- `src/run_sy2014_seed1_minimal_reproduction_018_08.py`

## 需要产出

1. baseline daily / events / summary
2. 5K smoke checkpoint daily / events / summary
3. 50K formal checkpoint daily / events / summary（仅在 smoke 成功后）
4. `seed0` vs `seed1` best checkpoint 对比表
5. 中文实验记录

## 判定

- 若 5K smoke 本身失败，则停止，不继续 50K
- 若 50K 成功，则重点比较：
  - best checkpoint 的 `final_gwad`
  - `irrigation_total`
  - `fertilizer_total`
  - `total_reward`
  - 相对 `null` / `dssat_auto` 的差值
