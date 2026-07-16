# 022_11 SY2014 DAP65 action1/action7 三前缀一致性验证

## 1. 目的

022_10在W60 critical固定状态下证明DAP65 action1在联合目标上明显优于action7。本任务按预注册硬停止线，只补W75 uniform-pre90和W120 critical两个水分背景；三前缀必须3/3一致，否则停止且不得补第4个。

## 2. 固定门槛

材料性action1优势定义为：

1. Control和Treatment产量均≥11077 kg/ha；
2. `Treatment−Control G0_raw ≤ −250`。

−250来自额外N100对应reward成本500的一半，不依据结果选取。

## 3. 实现检查

两个新增前缀的全部检查通过：

- 每组完整6阶段；
- 唯一选定动作差异为DAP65 action1/action7；
- DAP65前氮均为N200，剩余N100；
- action7完整执行N100，无预算裁剪或动作别名；
- 两arm灌溉总量相同；
- 最终氮为N200 vs N300；
- Summary匹配误差均≤1。

本任务只运行4个DSSAT季，DQN梯度更新为0。

## 4. 三前缀结果

| 前缀 | Control/Treatment灌溉 | Δ产量 kg/ha | Δ生物量 kg/ha | ΔG0 | PFP Control→Treatment | 达到材料性门槛 |
|---|---:|---:|---:|---:|---:|---|
| W60 critical（022_10） | 60/60 | +0.106 | +79.814 | −499.894 | 56.0→37.3 | 是 |
| W75 uniform-pre90 | 75/75 | +0.106 | +79.022 | −499.894 | 56.0→37.3 | 是 |
| W120 critical context | 105/105 | +0.273 | +109.591 | −499.727 | 55.9→37.3 | 是 |

三个前缀的Control和Treatment均达到产量门槛。额外N100的籽粒增益均不足0.3 kg/ha，而G0均下降约500。

## 5. 判定

结果为 **A：3/3前缀均显示action1材料性占优**。

可以支持：

- 在三个预先固定的不同水分背景、DAP65前均为N200的状态下，额外施N100几乎不增加籽粒产量；
- action1相对action7的联合回报优势不是W60单一前缀偶然现象；
- 当前offline checkpoint在DAP65一致偏好action7，是需要处理的稳定排序病灶；
- 允许下一步进行一次纯离线pairwise-loss与MC-loss梯度冲突审计。

不能支持：

- 不能声称所有DAP65状态都应选action1；
- 不能把“库容饱和”写成已证实机制；生物量增、籽粒几乎不增只支持该解释；
- 不能直接启动在线warm-start或扫描pairwise权重。

## 6. 下一步硬边界

下一步只允许一次离线审计：在三个同状态action1/action7配对上，计算pairwise ranking loss与现有MC loss的梯度方向、余弦相似度、范数和是否冲突。

- 权重必须在训练前固定；
- 若pairwise与MC目标在受控配对上方向冲突，停止，不进入在线训练；
- 若方向兼容，才允许另立一次固定权重的离线训练任务；
- 不补第4个前缀。

## 7. 输出

- `prompts/022_11_sy2014_dap65_three_prefix_consistency_test.md`
- `src/run_sy2014_dap65_three_prefix_consistency_022_11.py`
- `benchmark_results/022_11/022_11_new_prefix_summary.csv`
- `benchmark_results/022_11/022_11_new_prefix_stage_actions.csv`
- `benchmark_results/022_11/022_11_new_prefix_daily_values.csv`
- `benchmark_results/022_11/022_11_three_prefix_consistency.csv`
- `benchmark_results/022_11/022_11_three_prefix_consistency.png`
- `benchmark_results/022_11/022_11_three_prefix_consistency.svg`
- `benchmark_results/022_11/022_11_result.json`
