# 025_00 确定性水氮优化正式协议与 SY 跨年准备度审计记录

## 1. 背景

021–024 已确认学习式方法目前没有形成稳定、覆盖完整动作空间的成功策略。025 路线不再继续调 DQN 或监督网络，而是把已成功找到候选的 DSSAT 约束确定性搜索发展为正式方法。

本任务首先防止一种重要的过度表述：SY2014 的18个严格成功方案使用了2014完整天气，只能称为 retrospective oracle / site-year upper bound，不能直接称为跨年策略。

## 2. 冻结的方法协议

### 2.1 两层证据

- 单年 oracle：允许使用目标年完整天气进行搜索，只证明可达上界；
- 跨年固定策略：只在训练年选择方案，验证年不得重新优化、移动阶段或修改总量。

### 2.2 指标

- `WP_ET = yield_kg_ha / ET_mm / 10`；
- `PFP_N = yield_kg_ha / applied_N_kg_ha`；
- 产量和WP_ET必须不低于同输入 official expert 与 DSSAT auto 中更严格者；
- PFP_N不低于official expert；
- 同时检查资源预算和晚期施氮合理性；
- recorded/farmer只作描述比较，不替代official expert。

采用可行性优先的词典序，不引入可调加权和。可行方案并列时，依次优先低氮、低灌溉和高产。

## 3. 第一组跨年验证

- 选择年：SY2014，IC=2；
- 候选：022_01 的18个严格成功固定阶段方案；
- 验证年：SY2012，IC=1；
- SY2012不得参与候选选择；
- 方案只允许按2012播种日保持相同DAP偏移，不得重新选择动作或总量。

## 4. 准备度审计

### 4.1 已通过

- SY2014严格候选数量为18且场景唯一；
- 当前`CNSY1201.MZX`同时包含2012 treatment/IC=1和2014 treatment/IC=2；
- 2012天气、SOIL、当前品种文件均存在；
- 2012 IC=1历史前向运行正常终止，可作为输入准备度证据；
- 当前MZX SHA-256：`20b071bc49549cbf561be3aae81caa355aa0582564db524e17d68ab6a274418a`；
- 2012天气SHA-256：`4e2489653b7a72f75ce255602b2a58953e40e6424b4f27e8cf64d900361e7c68`。

### 4.2 尚未完成

- `configs/sites/sya.yaml`目前只正式登记2014；
- SY2012尚未在当前输入链下完成null、recorded、DSSAT auto、official extension expert四基线统一前向验证；
- 因此18个候选现在还不能直接迁移运行。

## 5. 判定

**A_protocol_ready_input_validation_required。**

协议和阻塞输入均可用，但候选迁移尚未获准。下一步必须先执行：

`025_01_sy2012_input_provenance_and_four_baseline_forward_validation`

只有025_01确认2012处理行、IC=1、天气、品种、土壤与四基线都一致后，才允许另立025_02，把18个SY2014候选原样迁移到SY2012。

## 6. 执行边界

- DSSAT调用0次；
- 正式优化0次；
- 学习训练0步；
- 没有修改MZX、WTH、SOL、CUL、IC或站点配置；
- 第一次运行因历史诊断表字段名误写为`label`而失败，失败目录保留为`benchmark_results/025_00_failed_attempt_1_diagnosis_column_name/`；修正为实际字段`case`后原样重跑。

## 7. 输出

- `prompts/025_00_deterministic_optimization_protocol_and_sy_crossyear_readiness.md`
- `configs/deterministic_optimization_protocol_025.yaml`
- `src/audit_deterministic_protocol_sy_crossyear_readiness_025_00.py`
- `benchmark_results/025_00/025_00_sy_crossyear_readiness.csv`
- `benchmark_results/025_00/025_00_result.json`
- `docs/2026-07-16_025_00_deterministic_optimization_protocol_and_sy_crossyear_readiness.md`

本记录生成时尚未执行本任务的Git commit或push。
