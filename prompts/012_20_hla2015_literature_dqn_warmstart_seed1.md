# 012_20 HLA2015 文献式 DQN warm-start seed1 稳定性验证

## 背景

012_19 在 HLA2015 seed0 上加入 I60/N0 启发式 warm-start 后，得到：

- 产量约 7653 kg/ha；
- 灌溉 66 mm；
- 施氮 0 kg/ha；
- reward 明显高于 012_17 的 N150 策略。

这说明合理初始 Q 排序可以阻止 DQN 漂回高氮动作。但目前只验证了 seed0，不能说明稳定。

## 目的

做 seed1 稳定性验证：

> 在完全相同设置下，只把 seed0 改为 seed1，检查 warm-start 是否仍能得到少氮、适量灌溉、高产策略。

## 固定设置

与 012_19 保持一致：

- HLA2015；
- IC=1；
- 文献式 terminal reward；
- 25 个水氮离散动作；
- I120/N150 总预算；
- 操作窗口和 7 天间隔；
- `n_steps=5`；
- warm-start epochs=8；
- timesteps=5000。

唯一变化：

```text
seed: 0 -> 1
```

## 执行要求

使用已有脚本：

```text
src/run_hla2015_literature_dqn_warmstart_012_19.py
```

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_literature_dqn_warmstart_012_19.py --year 2015 --timesteps 5000 --seed 1 --n-steps 5 --warmstart-epochs 8 --label literature_warmstart"
```

保存 daily CSV、event_summary.json、debug log、model、warm-start 数据和中文记录。

## 判断

- 如果 seed1 也接近 I60/N0 或少氮适量灌溉，则 warm-start 线值得继续；
- 如果 seed1 又回到 N150，则 warm-start 仍有 seed 敏感性，需要更系统的 imitation/replay 方案。
