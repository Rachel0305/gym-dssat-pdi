# HLA 2010/2015 四情景 forward 对比 prompt

## 目的

在不重新训练 PPO 的前提下，检查 HLA 2010 和 2015 在新 HY0006 品种参数与 IC=1 输入口径下，四类管理策略的结果：

1. null：无灌溉、无施肥；
2. expert_2007_shifted：把 HLA 2007 真实记录管理作为专家策略模板，按 DAP 平移到目标年份；
3. dssat_auto：DSSAT 原生自动管理，不设置 PPO 预算；
4. ppo_00906_seed0：加载既有 HLA 2004 联合水氮 PPO 模型，在目标年份环境中实时 `model.predict()` 决策。

## 关键约束

- 不训练新模型；
- 不构造固定预算规则情景；
- DSSAT auto 不设水氮预算；
- PPO 使用历史训练模型和原训练约束口径，所以保留 I120/N150 安全裁剪，用于观察其在新年份里会做什么；
- 2010/2015 没有真实专家记录，因此 expert 情景明确标注为“2007 expert shifted”，不是目标年份实测管理。

## 专家策略搬移定义

以 HLA 2007 TRNO1 记录管理为模板：

- 灌溉：DAP 50、71、96，各 10 mm，总 30 mm；
- 施肥：DAP 1，两条记录合计 N 165 kg/ha，P 30 kg/ha；
- 将这些 DAP 平移到目标年份种植日。

## 输出

- 每个 scenario/year 的 input 与 PDI snapshot；
- daily CSV；
- events/action CSV；
- summary CSV；
- 每年一张过程图：降雨、灌溉、施肥、WSPD、NSTD；
- 一张产量/生物量柱状图；
- 实验记录 MD。

## 判读注意

这一步只用于比较管理响应与 PPO 旧模型跨年行为，不等同于证明 PPO 最优。
