# 021_47 SY2014 terminal bonus replay采样与传播精确重放审计

## 复现结果

- 训练interactions逐列精确一致：True，最大数值误差=0.0。
- checkpoint online Q哈希逐位一致：True；target哈希逐位一致：True。
- 确定性评估最大误差：0.0；gate逐项一致：True。
- replay index→origin step映射：action全部一致=True，done全部一致=True，reward最大误差=0.000244（float32存储）。
- 插桩没有新增RNG调用；每次update只调用一次原mixed sampler。

| checkpoint | online_hash_exact | target_hash_exact |
| --- | --- | --- |
| 0 | True | True |
| 250 | True | True |
| 500 | True | True |
| 750 | True | True |
| 1000 | True | True |

## PER采样暴露

| category | expected_draws | observed_draws | lower_95 | upper_95 | significantly_below_per_expectation |
| --- | --- | --- | --- | --- | --- |
| direct_bonus | 1684.333 | 1703 | 1609.348 | 1759.317 | False |
| nstep_bonus | 1874.636 | 1890 | 1796.123 | 1953.149 | False |
| immediate_resource_cost | 4089.248 | 4166 | 3982.482 | 4196.015 | False |

n-step含bonus的transition共20条；其中显著低于各自PER期望的比例为0.000。类别总采样是否低于95%下界：False。

## 已采样样本的value拟合

| category | draws | chosen_q_mean | target_1_mean | target_n_mean | abs_td1_mean | abs_tdn_mean | td1_huber_linear_fraction | tdn_huber_linear_fraction | importance_weight_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_bonus | 1703 | 12.646 | 7390.028 | 7390.028 | 7377.383 | 7377.383 | 1.0 | 1.0 | 0.08 |
| nstep_bonus | 1890 | 12.498 | 6660.054 | 7372.333 | 6647.668 | 7359.835 | 0.957 | 1.0 | 0.137 |
| immediate_resource_cost | 4166 | 4.717 | -304.156 | -304.747 | 308.873 | 309.464 | 1.0 | 1.0 | 0.204 |

bonus样本不是“没有被看到”：direct bonus被抽中1703次，n-step含bonus样本被抽中1890次。但其平均chosen Q仍远低于target，且TDn全部位于Huber线性区；PER importance correction还使其平均权重低于普通资源成本样本。这表明除bootstrap覆盖外，还存在局部value拟合不足，不能把全部失败单独归因于target冻结。

## target与传播距离

951次online update后记录到的target唯一哈希数：1。因此target在本次1K自定义loop中全程冻结：True。

| critical_n_dap | representative_terminal_dap | dap_distance_to_terminal | direct_nstep_transition_count | maximum_direct_lookback_gap_steps | within_direct_nstep_coverage |
| --- | --- | --- | --- | --- | --- |
| 29 | 139 | 110 | 5 | 4 | False |
| 42 | 139 | 97 | 5 | 4 | False |
| 56 | 139 | 83 | 5 | 4 | False |

这里的5-step只表示标准n-step target对终止奖励的直接覆盖（终止transition及最多前4个transition），不是神经网络参数共享造成的总影响硬上限。

## 预注册判定

分支：**B**。bonus-bearing transitions未显示系统性低采样；target在1K内全程冻结且关键施氮DAP远超直接5-step覆盖，同时已采样bonus transitions仍存在巨大target-Q残差。支持传播与局部拟合受限候选，但尚未确认单一根因。

## 适用范围

本任务精确重放的是021_45的1K固定target短期诊断，不是完整的长期DQN训练。结果不修改021_36–40的离线梯度/Q/loss证据，但要求021_35/41/42/45的在线“保持/坍缩”结论带上固定target限定。没有启动seed2或5K，也没有改reward、target同步或其他超参数。
