# 027_03 HLA2010 阶段型 MaskablePPO seed1/2 复核

## 1. 目的

在不改变027_02任何算法、环境、奖励或选模规则的条件下，串行运行HLA2010 seed1和seed2，判断seed0的primary正向信号能否跨seed复现。

本任务不调参、不追加seed、不做跨年、不启动YC/FQ/LC或联合训练。

## 2. 冻结配置

除seed分别为1和2外，逐项复用027_02：

- HLA2010、24维HLA专属scaler；
- DAP 1/30/50/65/85/110；
- 9动作，I≤120、N≤300，预算与晚期禁氮mask；
- MaskablePPO `[32,32]`；
- learning_rate=3e-4；gamma=1；gae_lambda=1；
- n_steps=60；batch_size=30；n_epochs=5；
- 每个seed精确240阶段步；checkpoint 0/60/120/180/240；
- 精确null=6956.453857421875；产量gate=7853.6651611328125；
- 相同资源成本、产量增益和1620 feasibility bonus。

先核对seed0选中checkpoint180的SHA-256：

`6715fffb4bbe251cf0cdad13121331d9e6f875a3e6629c26423500e502e042da`

不一致则停止。

## 3. 选模规则

每个seed独立执行同一预注册规则：

1. checkpoint0不入选；
2. 在60/120/180/240中选择确定性季节reward最大者；
3. 精确并列选择更早checkpoint；
4. 禁止按primary、recorded或图形观感重选。

## 4. Primary与recorded

Primary要求选中checkpoint同时满足：

- yield≥7853.665161 kg/ha；
- WP_ET≥1.64 kg/m3；
- PFP_N可定义且≥26.2 kg/kg。

recorded独立比较：yield≥7679、WP_ET≥1.70、PFP_N≥46.5。recorded不进入reward或选模，也不改变primary定义。

## 5. 三seed判据

### A_three_seed_primary_replicated

- seed1/2全部工程检查通过；
- 将seed0/1/2按相同规则汇总；
- 至少2/3选中checkpoint通过primary。

只有该分支允许下一步冻结三个选中模型及哈希，并另立HLA站内跨年迁移任务书。不得声称3/3稳定，除非数字确实为3/3。

### B_three_seed_primary_not_replicated

工程通过但少于2/3 seed的选中checkpoint通过primary。停止HLA跨年迁移；不增加seed、不延长训练、不改reward。

### C_execution_failed

任一输入、scaler、mask、数值、版本、步数、checkpoint或输出检查失败。停止并仅诊断工程原因。

## 6. 输出与算力

- seed1和seed2必须串行，避免OOM；每个seed仅240步。
- 每个checkpoint保存模型哈希、确定性完整季节指标、动作/mask、奖励分项。
- 保存两seed训练季日志、三seed汇总CSV/JSON、PNG/SVG和中文实验记录。
- 不覆盖027_02；模型zip不默认提交Git。
- 完成本任务后停止，等待用户确认下一步。

