# 022_09 SY2014 DAP65 离线checkpoint专项排序审计

## 1. 目的

022_08 的 recorded-action Spearman约0.80，但阶段聚合Spearman仅0.471–0.557。022_06曾定位DAP65为N300形成前的关键错位阶段。本任务在任何warm-start在线训练前，专项检查022_08三个离线checkpoint是否修复DAP65的动作排序。

## 2. 边界

- 只读取022_08已保存CSV，不训练、不调用DSSAT、不重新计算checkpoint；
- 对seed0/1/2分别计算DAP65六个可估计动作的Q-target Spearman；
- 固定比较action7（I15/N100）与action1（I15/N0）的Q margin和经验target margin；
- 与022_06 season30/45/60的DAP65 Spearman约−0.60作描述性比较；
- DAP65不同动作样本来自不同历史状态，经验target排序不得写成严格因果真值。

## 3. 判定

- **A 病灶已修复**：至少2/3 seed的DAP65 Spearman≥0.50，且至少2/3 seed满足Q(action1)>Q(action7)；允许执行warm-start草案。
- **B 部分改善但关键对仍反向**：Spearman相对022_06改善，但不足A；暂停seed1 warm-start，先做固定历史的DAP65受控动作对照。
- **C 未改善**：多数seed Spearman≤0或关键排序无任何改善；停止当前warm-start初始化。
- **D 数据失败**：动作、seed或数值缺失。

不得根据结果更换seed后直接在线训练。

## 4. 输出

- DAP65 seed×action表；
- seed摘要和JSON；
- PNG/SVG；
- 中文记录。
