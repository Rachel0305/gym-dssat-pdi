# 022_15 SY2014 固定 pairwise+MC 唯一一次离线训练

## 1. 依据

022_12 已确认三个 seed 的 pairwise 与 MC 梯度兼容；022_14 在 SmoothL1 二次区局部外推下得到最慢排序翻转约 2884 步，并按预注册规则固定统一预算 3000 updates。

## 2. 唯一配置

- 起点：022_08 seed0/1/2 的 2000-update checkpoint；
- 所有 seed 使用同一 `lambda=0.0060048738917845`；
- loss：`L_MC + lambda * L_pair`；
- `L_pair=mean SmoothL1(Q(a1)-Q(a7), (G1-G7)/1000)`；
- Adam `lr=1e-4`，因022_08未保存optimizer state，本任务明确从加载的网络权重重新初始化Adam；
- 全批次、固定3000 updates；不得延长、缩短或扫描lambda；
- 保存并评估 update 0/500/1500/3000；中间checkpoint只用于观察轨迹，不用于事后挑选成功模型。

## 3. 数据边界

- MC训练集仍为022_08的36场景/216条；
- pairwise为022_10/11三个受控DAP65状态；
- W120 pair状态来自原022_08测试场景，因此原12场景指标仅作描述；
- 主要guardrail使用未被pairwise监督的其余11个测试场景，不再把W120场景称为严格独立留出。

## 4. 预注册判据

单seed在最终3000步通过，必须同时满足：

1. 三个受控DAP65状态的 `Q(a1)-Q(a7)>0`；
2. 原MC训练loss相对update0增加不超过1%；
3. 11个未污染测试场景中，DAP1与DAP110支持集argmax相对update0改变数均为0；
4. 数值有限。

总体：

- A：至少2/3 seed最终通过；
- B：仅1/3最终通过；
- C：0/3最终通过；
- D：实现或数值失败。

若中间checkpoint通过但3000步回退，该seed仍判失败并标记`intermediate_pass_then_regressed`，不得挑中间checkpoint充当成功结果。

## 5. 输出与停止线

- 保存四个观察点的checkpoint、pair margin、MC loss、各阶段argmax扰动和曲线；
- DQN离线更新共9000次（3×3000），DSSAT调用0次；
- 即使A分支，也只证明一次离线排序修复有效，不能直接宣称在线DQN成功；
- 不自动启动022_13 warm-start或任何在线训练。

