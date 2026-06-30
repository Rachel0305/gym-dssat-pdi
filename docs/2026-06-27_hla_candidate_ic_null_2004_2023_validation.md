# HLA 候选初始条件 2004-2023 null 多年验证

## 1. 目的

在前一轮小扫描中，候选初始条件被暂定为：

```text
water_fraction = 0.55
N_scale = 0.25
```

其含义为：

- 水分：沿用已有项目文档中的 `SH2O = SLLL + 0.55 × (SDUL - SLLL)`；
- 氮素：将 HLA 2004 record-based 初始矿质氮剖面缩放到 25%；
- 管理：null，无灌溉、无施肥。

本轮目标是把该候选条件扩展到 HLA 2004-2023 全部年份，和 IC=0、原 IC=1 多年 null 并排比较，判断它是否适合作为下一步小规模四情景对照的候选基线。

本轮不训练 PPO，不切换主实验。

## 2. 输入和脚本

新增脚本：

- `src/run_hla_candidate_ic_null_2004_2023.py`

输出目录：

- `DSSAT_auto_validation/HLA_2004/candidate_ic055_n025_null_2004_2023/`

对比三套情景：

| 情景 | 说明 |
|---|---|
| `IC0_null` | HLA 2004-2023，IC=0，null，无灌溉、无施肥 |
| `Original_IC1_null` | 原 IC=1，直接平移 HLA 2004 record-based 初始剖面 |
| `Candidate_IC055_N025_null` | 候选 IC：0.55 水分公式 + 25% 初始矿质氮 |

主要输出：

- `hla_candidate_ic_null_run_status.csv`
- `hla_ic0_origic1_candidate_null_summary_long_2004_2023.csv`
- `hla_ic0_origic1_candidate_null_summary_wide_2004_2023.csv`
- `hla_ic0_origic1_candidate_null_daily_2004_2023.csv`
- `hla_ic0_origic1_candidate_null_yield_2004_2023.png`
- `hla_2004_ic0_origic1_candidate_null_daily.png`
- `hla_2009_ic0_origic1_candidate_null_daily.png`
- `hla_2012_ic0_origic1_candidate_null_daily.png`

运行结果：

- 20 年候选 null 前向模拟全部完成；
- `returncode=0`，无超时；
- 未进行 PPO 训练。

## 3. 年度产量结果

多年籽粒产量范围：

| 情景 | 年数 | 最小值 | 中位附近 | 最大值 |
|---|---:|---:|---:|---:|
| IC=0 null | 20 | 0 kg/ha | 约 260 kg/ha | 421 kg/ha |
| 原 IC=1 null | 20 | 0 kg/ha | 约 6124 kg/ha | 8027 kg/ha |
| 候选 IC null | 20 | 0 kg/ha | 约 2287 kg/ha | 2869 kg/ha |

年度图：

- `DSSAT_auto_validation/HLA_2004/candidate_ic055_n025_null_2004_2023/hla_ic0_origic1_candidate_null_yield_2004_2023.png`

关键结果：

- 候选 IC 明显高于 IC=0，说明它避免了“null 几乎完全不能生长”的极低基线；
- 候选 IC 明显低于原 IC=1，说明它避免了“无管理也达到 5-8 t/ha”的过强基线；
- 2004 和 2012 仍然是零籽粒异常年份，说明候选 IC 没有把异常年份强行抹平成正常高产；
- 2009 这类正常响应年份从原 IC=1 的约 8027 kg/ha 降到候选 IC 的约 2869 kg/ha，保留了施肥管理空间。

## 4. 关键年份过程解释

### 4.1 HLA 2004

候选 IC 下：

- 籽粒产量仍为 0；
- 生育后期 WSPD 升高到接近 1，说明仍存在强水分胁迫；
- NSTD 有中等幅度变化，但不再像 IC=0 那样长期高氮胁迫；
- 说明候选 IC 没有把 2004 异常年直接救成高产，同时保留了水分管理响应空间。

### 4.2 HLA 2009

候选 IC 下：

- 籽粒产量约 2869 kg/ha；
- WSPD 基本为 0，说明 2009 在该设置下不是明显水分限制；
- NSTD 后期逐步升高，说明产量受氮素限制；
- 这说明候选 IC 对正常年份形成了“中等 null + 明显施肥空间”的状态。

### 4.3 HLA 2012

候选 IC 下：

- 籽粒产量仍为 0；
- WSPD 基本为 0；
- NSTD 仅有小幅变化；
- 说明 2012 的零籽粒不能主要归因于初始水氮不足，可能还涉及年份气象、物候或生殖期过程。

## 5. 当前判断

候选 `water_fraction=0.55, N_scale=0.25` 通过了第一轮多年 null 合理性检查：

1. 它比 IC=0 更合理，不再让多数年份 null 停留在几百 kg/ha；
2. 它比原 IC=1 更保守，不再让多数年份无管理也达到 5-8 t/ha；
3. 它保留了异常年份 2004、2012；
4. 它在正常年份保留了氮素管理空间；
5. 水分设定有已有文档支撑，氮素缩放可解释为“统一多年初始剖面的保守缩放”。

但它仍不是最终主实验 IC。当前只能说：

> 候选 IC 适合进入 HLA 2004 小规模四情景对照验证。

## 6. 对下一步四情景的限制说明

如果下一步做 HLA 2004 四情景小规模对照，需要注意：

1. 候选 IC 改变了环境初始状态；
2. 旧 PPO 模型是在旧 IC/旧奖励/旧输入条件下训练得到的；
3. 如果直接把旧 PPO 用到候选 IC 上，只能称为“旧策略迁移/回放测试”，不能称为“候选 IC 下重新训练得到的 PPO 最优策略”；
4. 如果要严格比较 PPO 优越性，需要在候选 IC 下重新训练或至少做小步数 smoke/短训；
5. 为节省算力，下一步建议先做 HLA 2004 的非训练版四情景过程检查：
   - null；
   - 固定/专家管理；
   - 规则管理；
   - 旧 PPO 策略回放或短 smoke PPO；
   并明确标注 PPO 的性质。

## 7. 下一步建议

建议下一步做 HLA 2004 候选 IC 小规模四情景对照，但不要直接宣称 PPO 已经在候选 IC 下最优。

优先顺序：

1. 先复用候选 IC 生成 HLA 2004 null、固定管理、规则管理；
2. 再决定 PPO 是旧策略回放，还是候选 IC 下短训 smoke；
3. 如果短训 smoke 行为合理，再考虑正式训练。

