# 021_29 SY2014 示范采样与梯度平衡离线审计

## 背景

021_28 在两组相同的 full return-to-go 示范训练中，仅对 Treatment 屏蔽示范 1-step TD。结果表明 Treatment 的 5 个非零动作召回率始终为 0，并在 1000 次更新时退化为全部预测 no-op，非零动作平均 Q-margin 恶化至 -27.376。

021_27 已证明 1-step/full-return 梯度冲突存在；021_28 又证明单独移除该冲突仍不够。本任务不继续改算法，而是审计剩余的两个候选机制：

1. PER 是否仍被 155 个 no-op 或终止样本主导，导致 5 个非零动作实际采样不足；
2. 在同一个真实训练 batch 中，no-op 样本与非零示范样本的实际梯度贡献是否量级失衡或方向冲突。

## 任务边界

- 完全离线，不调用 DSSAT/PDI；
- 不做在线环境交互；
- 只复现 021_28 Treatment：固定标准化、full return-to-go、示范 1-step TD 不进入总损失；
- 使用同一份示范、seed、网络、PER 和 1000 次更新；
- 不调整 reward、margin、lambda、batch size、PER 参数、网络或里程碑；
- 不根据本轮结果现场修改权重；
- 不自动进入 5K 在线训练。

## 审计内容

在更新 `1, 10, 25, 50, 100, 250, 500, 1000` 前后记录：

1. 当前 batch 中 no-op、非零动作、终止样本数量；
2. replay 中三类样本的 priority 中位数、最大值和总采样概率质量；
3. 实际总损失中 no-op 样本和非零样本的 full-return TD、margin 贡献；
4. no-op 子目标与非零子目标分别产生的梯度范数；
5. 两个梯度向量的 cosine similarity；
6. 该里程碑的 no-op 准确率、非零召回率和 expert Q-margin；
7. 复现结果是否与 021_28 Treatment 一致。

分组损失必须按原始 batch 总大小归一化，确保分组梯度相加能还原实际总梯度中的数据项；不能使用各组独立均值制造虚假的“等权”。L2 单独记录，不并入 no-op/非零分组归因。

## 预注册解释

- A：非零样本的采样概率质量明显低于其学习需要，且 no-op 梯度范数持续主导——优先考虑示范分层/平衡采样，但另立任务预注册；
- B：非零样本已被充分采样，但 no-op 与非零梯度持续强烈反向——说明共享网络上的分类/信用目标冲突，需要另立表示或损失结构实验；
- C：采样和梯度均无明显失衡或冲突——现有两个候选不足以解释失败，停止凭猜测修改。

本轮只做描述性机制审计，不用单一固定阈值强行把连续证据二元化。所有解释必须同时报告样本量和时间变化。

## 输出

- `benchmark_results/021_29/021_29_sampling_gradient_audit.csv`
- `benchmark_results/021_29/021_29_learning_curve.csv`
- `benchmark_results/021_29/021_29_replication_validation.json`
- `benchmark_results/021_29/021_29_summary.json`
- PNG/SVG 图
- `docs/2026-07-15_021_29_sy2014_demo_sampling_gradient_balance_audit.md`

不自动 Git push。
