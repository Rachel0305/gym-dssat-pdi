# 019_03 leaching 变量链路审计记录

## 结论先行

本轮没有训练，也没有修改模板。审计结果显示：PDI 通信模板中已经存在 `CLeach -> cleach` 和 `TLeachD -> tleachd` 的状态映射；DSSAT 输出中也能看到 `Summary.OUT` 的 `NLCM` 与 `SoilNi.OUT` 的 `NLCC`。因此当前证据不支持“必须先手动改模板才能获得 leaching”。

真正缺的是：当前主线 DQN reward wrapper 和评估 CSV 没有稳定保存/使用 `cleach` 或 `tleachd`，所以如果要把氮淋洗加入奖励，下一步应优先改 reward wrapper 与日值记录，而不是马上重训。

## 关键数量

- 含 `cleach` 状态映射的 PDI 快照数：423
- `Summary.OUT` 含 `NLCM` 的快照数：423
- `SoilNi.OUT` 含 `NLCC` 的快照数：423
- 已有 CSV 表头包含 leaching 字段的文件数：13
- 被审计 reward 脚本中提到 leaching/no_leaching 的文件数：1

## 示例数值

| station   |   year |   irrigation_label |   nitrogen_label | timing_label   |   summary_final_NLCM |   soilni_final_NLCC |   soilni_max_NLCC |
|:----------|-------:|-------------------:|-----------------:|:---------------|---------------------:|--------------------:|------------------:|
| FQ        |   2007 |                  0 |                0 | critical       |                  176 |                3.3  |              3.3  |
| FQ        |   2007 |                  0 |                0 | early          |                  176 |                3.3  |              3.3  |
| FQ        |   2007 |                  0 |              150 | critical       |                  177 |                6.17 |              6.17 |
| FQ        |   2007 |                  0 |              150 | early          |                  177 |                6.22 |              6.22 |
| FQ        |   2007 |                  0 |              300 | critical       |                  177 |                6.28 |              6.28 |
| FQ        |   2007 |                  0 |              300 | early          |                  177 |                6.57 |              6.57 |
| FQ        |   2007 |                120 |                0 | critical       |                  167 |               10.53 |             10.53 |
| FQ        |   2007 |                120 |                0 | early          |                  151 |               21.31 |             21.31 |
| FQ        |   2007 |                120 |              150 | critical       |                  174 |               17.7  |             17.7  |
| FQ        |   2007 |                120 |              150 | early          |                  169 |               32.34 |             32.34 |
| FQ        |   2007 |                120 |              300 | critical       |                  174 |               18.77 |             18.77 |
| FQ        |   2007 |                120 |              300 | early          |                  169 |               36.9  |             36.9  |

## 文件

- 快照字段审计：`DSSAT_auto_validation\leaching_chain_audit_019_03\019_03_snapshot_leaching_field_audit.csv`
- 已有 CSV 字段审计：`DSSAT_auto_validation\leaching_chain_audit_019_03\019_03_existing_csv_leaching_header_hits.csv`
- reward 脚本审计：`DSSAT_auto_validation\leaching_chain_audit_019_03\019_03_reward_file_leaching_mentions.csv`

## 下一步建议

1. 在 DQN 评估日值表中加入 `cleach`、`tleachd`、`cnox` 字段，先不改变奖励。
2. 选择一个高氮案例和一个低氮案例，对比 `cleach` 是否随施氮和灌水合理变化。
3. 若变量稳定，再做一个 leaching-aware reward 的 smoke test。
4. 只有当 PDI 快照缺失这些字段时，才需要回头改模板。
