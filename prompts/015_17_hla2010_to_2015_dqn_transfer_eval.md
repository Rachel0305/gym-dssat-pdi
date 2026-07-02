# 015_17 HLA2010-trained DQN transfer evaluation on HLA2015

## 目的

验证 DQN 是否学到可迁移的水氮管理规律，而不是只在单一年份上拟合。

本轮不重新训练，只加载 HLA2010 已训练好的 baseline-relative DQN checkpoint，并在 HLA2015 环境中直接评估。

## 迁移设置

训练年：

```text
HLA2010
```

测试年：

```text
HLA2015
```

待测试模型：

| train year | train seed | checkpoint | 选择理由 |
|---|---:|---:|---|
| HLA2010 | 0 | 35000 | HLA2010 seed0 best reward checkpoint 之一，I120/N0/GWAD≈7854 |
| HLA2010 | 1 | 25000 | HLA2010 seed1 best reward checkpoint，I60/N0/GWAD≈7573 |

## 统一框架

保持与 015_12/015_13 一致：

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

注意：

- 训练时模型来自 HLA2010；
- 评估时环境是 HLA2015；
- 评估 reward 使用 HLA2015 自己的 null baseline；
- 不进行任何继续训练。

## 对照对象

HLA2015 baseline：

| 情景 | GWAD kg/ha | I mm | N kg/ha |
|---|---:|---:|---:|
| null | 6486 | 0 | 0 |
| recorded/expert | 7296 | 30 | 165 |
| DSSAT auto | 7648 | 141.5 | 0 |
| local DQN seed0 | 7653 | 75 | 0 |
| local DQN seed1 | 7653 | 75 | 0 |

## 执行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2010_to_2015_dqn_transfer_eval_015_17.py"
```

## 输出

```text
DSSAT_auto_validation/HLA_2004/hla2010_to_2015_dqn_transfer_eval_015_17/
```

需要输出：

- transfer daily CSV
- transfer summary CSV
- 与 2015 baseline/local DQN 的对比图
- docs 实验记录

## 判定标准

如果 2010 训练模型在 2015 上也能达到：

```text
GWAD > null
I <= auto
N <= recorded
策略操作有明确水氮管理意义
```

则说明 DQN 存在初步跨年份迁移能力。

如果能接近或超过 2015 local DQN / DSSAT auto，则迁移能力很强。

如果明显退化到 null 或资源浪费策略，则说明当前 DQN 更偏年份特异性，需要按年份训练或做多年份训练。

