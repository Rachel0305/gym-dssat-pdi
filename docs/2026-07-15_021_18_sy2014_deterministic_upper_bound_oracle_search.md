# 021_18 SY2014 确定性上界 / oracle 调度搜索记录

## 背景与问题

导师的目标是让 DQN 尽量同时超过 expert 与 DSSAT auto 的产量和水氮利用效率。021_18 不训练 DQN，而是先验证当前 SY2014 IC=2、离散动作、共享 7 d 间隔、I≤120/N≤300 约束内是否存在客观可达的高质量策略。

## 冻结条件与输入

- 输入：复用 `021_14` 的 SY2014 IC=2 PDI/DSSAT 4.8.0 输入。
- 动作：I=0/15/30 mm，N=0/50/100 kg ha-1；DAP 1–120；共享间隔≥7 d。
- 无 DQN 训练；无 reward、IC、DSSAT 输入修改。
- 所有候选保存原始 `Summary.OUT`、`PlantGro.OUT`、日值、动作和输入快照。

## 同输入基准

| scenario | yield_kg_ha | irrigation_mm_summary | nitrogen_kg_ha_summary | WP_ET_kg_m3 | IWP_gross_kg_m3 | PFP_N_kg_kg | NUtE_kg_kg |
|---|---|---|---|---|---|---|---|
| null | 5408.000 | 0.000 | 0.000 | 1.220 | NA | NA | 55.800 |
| recorded | 9613.000 | 0.000 | 247.000 | 2.160 | NA | 38.900 | 35.100 |
| dssat_auto | 5498.000 | 66.000 | 0.000 | 1.200 | 8.330 | NA | 53.400 |
| official_extension_expert | 11077.000 | 266.000 | 300.000 | 2.260 | 4.160 | 36.900 | 37.200 |

注意：recorded 的管理事件请求总氮是 293 kg ha-1，但 DSSAT `Summary.OUT` 的实际 `NICM` 为 247 kg ha-1；本表和利用效率统一采用 Summary.OUT 实际值。官方 expert 的事件汇总为266.1 mm，Summary.OUT 为266 mm；利用效率采用 Summary.OUT。

## 预注册成功条件

- 产量≥11077 kg ha-1；
- WP_ET≥2.26 kg m-3；
- 对 N>0 候选，PFP_N≥36.9 kg kg-1；
- I≤120 mm、N≤300 kg ha-1。

IWP、NUtE、PNB 同时报告，但 DSSAT auto 的 N=0，PFP_N 不可定义，不能解释为无穷大。

## Smoke

scaled 25K 的 I120/N300 高产时序被确定性回放为11199 kg ha-1，与历史结果一致；IRCM/NICM=120/300，未漏动作。因此搜索链路通过。

## 搜索过程

1. 固定 I120，扫描 early/mid/late × N150/200/250/300，共12组。
2. 选取前三个候选，测试 I90/I60/I30，共9组。
3. 粗扫描发现 I60–I90 跨越预注册阈值，因此追加登记两种 I75 × N200/250/300，共6组 adaptive refinement；没有修改成功阈值。

汇总脚本第一次执行时因容器缺少 pandas 可选依赖 `tabulate` 而停止；没有重新模拟，也没有修改结果。随后改为脚本内置 Markdown 表格生成，未安装或升级环境包。

## 严格通过候选

| scenario | phase | final_gwad | summary_irrigation_total | summary_nitrogen_total | WP_ET_kg_m3 | IWP_gross_kg_m3 | PFP_N_kg_kg | NUtE_kg_kg |
|---|---|---|---|---|---|---|---|---|
| R_I75_no_early_mid_N200 | adaptive_refinement | 11205.000 | 75.000 | 200.000 | 2.300 | 14.940 | 56.000 | 40.000 |
| R_I75_no_early_mid_N250 | adaptive_refinement | 11205.000 | 75.000 | 250.000 | 2.300 | 14.940 | 44.800 | 37.100 |
| R_I75_no_early_mid_N300 | adaptive_refinement | 11205.000 | 75.000 | 300.000 | 2.300 | 14.940 | 37.300 | 37.100 |

## 最省氮的严格通过候选

- `R_I75_no_early_mid_N200`：产量 11205 kg ha-1，I=75 mm，N=200 kg ha-1。
- 相对官方 expert：增产 128 kg ha-1，少灌 191 mm，少施氮 100 kg ha-1。
- WP_ET=2.30 kg m-3，IWP=14.94 kg m-3，PFP_N=56.0 kg kg-1，NUtE=40.0 kg kg-1。
- 实际时序：灌溉 DAP22/29/42/56/79，各15 mm；施氮 DAP29=50、42=100、56=50 kg ha-1。

## 结论

当前约束内确实存在同时超过官方 expert 和 DSSAT auto 产量、并提高主要水肥投入效率的确定性策略。因此 SY2014 的困难不是“动作空间里没有好策略”，而是 DQN 尚未稳定学到这种时序。

该结论不能写成“DQN 已经成功”。021_18 的策略是人工剪枝搜索得到的 oracle，只能作为：

1. 可达上界证据；
2. demonstration-guided DQN 的示范轨迹；
3. 后续 DQN 是否真正学到合理时序的独立评估标准。

## 限制

- 只验证 SY2014；不能直接外推到其他站点或年份。
- 搜索不是全局穷举；“oracle”指当前剪枝候选集中的确定性上界，不是数学全局最优。
- NUtE=40.0 低于 N=0 的 DSSAT auto（53.4），但 auto 的低产且无外施氮情景不能用 PFP_N 比较；因此“全面超过每一个效率指标”仍不成立。

## 下一步

以 `R_I75_no_early_mid_N200` 及其他严格通过候选构建小型示范集，先做纯离线数据校验和行为克隆预训练 smoke，再在保持 DQN、reward、IC、动作约束不变的条件下进行短步数微调。必须与无示范 DQN 做单变量、多 seed 对照；不直接启动大规模训练。
