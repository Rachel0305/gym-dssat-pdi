# SYA / LCA 当前天气增强试验临时冻结、收尾归档与 GitHub 备份任务

## 任务定位

本任务用于对当前 SYA / LCA 随机天气增强实验线进行**阶段性收尾**。

注意：

- 这是**临时冻结（temporary freeze）**，不是永久终止；
- 仅表示：在本轮实验中，不再继续为 SYA / LCA 推进 WGEN 随机天气增强；
- 未来如果获得更高质量天气数据、外部补测数据或采用新的天气生成方法，可以重新开启 SYA / LCA 天气增强实验；
- 现有 historical-weather PPO 冻结结果继续保留并用于当前研究；
- 不重新训练 SYA / LCA PPO；
- 不运行 WGEN、DSSAT、PPO。

本任务同时负责：

1. 整理 SYA / LCA 015–019 相关实验记录；
2. 生成当前阶段统一收尾说明；
3. 生成实验备份清单；
4. 将关键实验文件、代码、报告和机器可读结果提交并 push 到 GitHub；
5. 为当前阶段生成 `.md` 和 `.pptx` 两种任务记录，保存到 `docs/`。

## 一、当前阶段冻结结论

### 1. SYA

当前 historical-weather PPO：

`TEMPORARILY_FROZEN_ACCEPTED_FOR_CURRENT_EXPERIMENT`

当前天气增强状态：

`EXCLUDED_FROM_WGEN_FOR_CURRENT_EXPERIMENT`

当前阻塞原因：

`BLOCKED_BY_WEATHER_STATISTICS`

已确认：

- 2005–2013 historical weather WTH 可继续用于当前已冻结的历史天气 PPO 结果；
- 当前问题不表示 SYA 历史天气模拟无效；
- 当前问题只表示：SYA 现有天气序列不适合作为本轮 WGEN 随机天气增强的正式拟合输入；
- 2005 年存在长时间连续均值填补；
- 2005 年 1–4 月 SRAD、TMAX、TMIN 月内日际标准差为 0；
- 当前 train-only fill 虽解决了验证期信息泄漏，但无法恢复 2005 年缺段的真实日际变率；
- 因此本轮不继续推进 SYA WGEN。

未来可重新开启的条件包括但不限于：

- 获得 2005 年相关 SRAD/TMAX/TMIN 的可靠外部补测；
- 获得可验证的独立天气重建方案；
- 使用新的、经过单独验证的天气生成方法。

### 2. LCA

当前 historical-weather PPO：

`TEMPORARILY_FROZEN_ACCEPTED_FOR_CURRENT_EXPERIMENT`

当前天气增强状态：

`EXCLUDED_FROM_WGEN_FOR_CURRENT_EXPERIMENT`

当前阻塞原因：

`BLOCKED_BY_JANUARY_GAMMA_SAMPLE`

已确认：

- LCA historical weather WTH 可继续用于当前已冻结的历史天气 PPO 结果；
- 当前问题不表示 LCA 历史天气模拟无效；
- 当前问题只表示：当前可用历史窗口无法稳定支持本轮 WGEN 冬季降水 Gamma 参数拟合；
- 2005–2013、2001–2013、2000–2013 候选窗口均未解除 1 月 Gamma 阻塞；
- 2000–2013 虽增加湿日数量，但 2000 年湿日贡献过度集中，原始 ALPHA 仍超过 CLI 截断范围；
- 1995–2013 无连续合格历史天气序列；
- 因此本轮不继续推进 LCA WGEN。

未来可重新开启的条件包括但不限于：

- 获得更长且连续、质量合格的历史降水记录；
- 获得更合理的冬季降水参数化方法；
- 使用新的、经过单独验证的天气生成方法。

## 二、本任务核心目标

请将 SYA / LCA 当前实验线正式标记为：

`TEMPORARILY_FROZEN_FOR_CURRENT_WEATHER_AUGMENTATION_EXPERIMENT`

不要使用：

- `PERMANENTLY_CLOSED`
- `ABANDONED`
- `FAILED_SITE`
- `INVALID_BASELINE`

不要把 historical-weather PPO 结果标记为失败或无效。

需要明确区分：

1. historical-weather PPO：
   - 当前研究中继续保留；
   - 不需要因 WGEN 阻塞而重新训练。

2. stochastic weather augmentation：
   - 本轮实验暂不继续；
   - 未来可重新开启。

## 三、收尾归档范围

请系统整理与 SYA / LCA 当前天气增强线直接相关的实验文件。

至少检查并纳入归档清单：

- `docs/sy_lc_random_weather_source_audit_015.md`
- `docs/provenance_resolution_report.md`
- `docs/sy_lc_wgen_readiness_review.md`
- `docs/lca_cli_parameter_audit_016.md`
- `docs/lca_wgen_window_review.md`
- `docs/sya_train_only_fill_sensitivity_review.md`
- `docs/sya_cli_parameter_audit_018.md`
- `docs/sya_wgen_readiness_review_019.md`

以及对应的：

- `results/sy_lc_random_weather_015/`
- `results/lca_cli_parameter_audit_016/`
- `results/lca_wgen_window_review/`
- `results/sya_train_only_fill_sensitivity/`
- `results/sya_cli_parameter_audit_018/`
- `results/sya_wgen_readiness_review_019/`

同时检查并纳入本轮实际使用的：

- provenance 审计脚本；
- readiness review 脚本；
- CLI 参数审计脚本；
- train-only sensitivity 脚本；
- LCA window review 脚本；
- SYA 019 statistics review 脚本；
- 与上述任务直接相关的验证脚本。

不要为了“备份更多”而提交大型原始天气数据、模型 checkpoint 或无关中间文件。

## 四、生成统一阶段性收尾记录

生成：

`docs/sy_lc_weather_augmentation_temporary_freeze.md`

至少包括：

1. 实验目的；
2. SYA 当前最终状态；
3. LCA 当前最终状态；
4. 为什么 historical-weather PPO 仍保留；
5. 为什么 SYA 当前不进入 WGEN；
6. 为什么 LCA 当前不进入 WGEN；
7. 哪些问题已经排除；
8. 当前真正剩余的 WGEN 方法限制；
9. 本轮实验到此收尾；
10. 未来重新开启 SYA / LCA 的条件；
11. 论文中推荐如何描述这两个站点；
12. 不得将两站描述为“天气无效”或“PPO结果无效”。

## 五、生成机器可读冻结状态

生成：

`results/sy_lc_weather_augmentation_closeout/freeze_status.json`

建议字段：

```json
{
  "scope": "current_weather_augmentation_experiment",
  "freeze_type": "temporary",
  "historical_weather_ppo_results": {
    "SYA": "FROZEN_ACCEPTED",
    "LCA": "FROZEN_ACCEPTED"
  },
  "weather_augmentation": {
    "SYA": {
      "status": "EXCLUDED_FOR_CURRENT_EXPERIMENT",
      "reason": "BLOCKED_BY_WEATHER_STATISTICS",
      "reopen_allowed": true
    },
    "LCA": {
      "status": "EXCLUDED_FOR_CURRENT_EXPERIMENT",
      "reason": "BLOCKED_BY_JANUARY_GAMMA_SAMPLE",
      "reopen_allowed": true
    }
  }
}
```

可以根据项目现有 schema 调整字段名，但语义必须保持一致。

## 六、生成实验备份清单

生成：

- `docs/sy_lc_weather_augmentation_backup_manifest.md`
- `results/sy_lc_weather_augmentation_closeout/backup_manifest.json`

清单至少记录：

- 文件路径；
- 文件类别；
- 是否纳入 Git；
- SHA256；
- 生成任务编号；
- 是否为关键证据；
- 是否属于大型/不应提交文件；
- 当前 Git tracked / untracked 状态。

重点确保以下类别有备份：

1. 关键报告；
2. gate JSON；
3. 参数 QC JSON；
4. 关键 CSV；
5. CLI 候选文件；
6. 审计与复核脚本；
7. 本轮最终冻结说明；
8. 当前 prompt（若项目规范要求保留）。

## 七、任务记录：MD + PPTX

按照当前项目要求，本任务必须生成两种记录并保存到 `docs/`：

1. `docs/sy_lc_weather_augmentation_closeout_020.md`
2. `docs/sy_lc_weather_augmentation_closeout_020.pptx`

PPT 建议 5–7 页，简洁记录：

1. 本轮天气增强实验目标；
2. SYA 审计链与最终阻塞原因；
3. LCA 审计链与最终阻塞原因；
4. 为什么 historical-weather PPO 结果仍保留；
5. 两站 temporary freeze 状态；
6. GitHub 归档内容；
7. 未来 reopen 条件。

PPT 不需要复杂美化，以清晰、可汇报、可追溯为主。

## 八、GitHub 备份

本任务需要将 SYA / LCA 当前天气增强实验线的关键文件备份到 GitHub。

执行要求：

1. 先检查：

```bash
git status
git branch --show-current
git remote -v
```

2. 不覆盖或删除用户已有未提交工作。
3. 只提交与本轮 SYA / LCA 天气增强实验直接相关的文件。
4. 避免提交：
   - 大型模型 checkpoint；
   - 临时缓存；
   - 无关结果；
   - 原始下载天气大文件；
   - `.gitignore` 明确排除的运行时文件。
5. 如果关键机器可读结果被 `.gitignore` 排除：
   - 不要擅自全局修改 `.gitignore`；
   - 先判断是否属于应备份的小型关键证据；
   - 如需 `git add -f`，仅对明确列入 backup manifest 的小型关键文件使用；
   - 在报告中记录。
6. 推荐 commit message：

```text
archive SYA LCA weather augmentation closeout 020
```

7. 完成 commit 后 push 到当前工作分支对应 remote。
8. push 前后记录：

```bash
git rev-parse HEAD
git status
```

9. 在最终报告中记录：
   - branch；
   - commit hash；
   - remote；
   - push 是否成功；
   - 未提交文件是否仍存在。

## 九、输入保护与禁止事项

禁止：

1. 修改原始 WTH；
2. 修改 frozen historical-weather PPO checkpoint；
3. 重新运行 PPO；
4. 运行 WGEN；
5. 运行 DSSAT；
6. 修改已完成的历史实验结果；
7. 删除 015–019 现有报告或产物；
8. 将 temporary freeze 改写为永久关闭；
9. 因归档需要重写或覆盖已有关键结果。

如需生成新的 summary / manifest / closeout 文件，应写入新的路径。

## 十、最终输出

完成后请给出简洁总结：

1. SYA temporary freeze 状态；
2. LCA temporary freeze 状态；
3. historical-weather PPO 是否保留；
4. 是否修改任何 WTH / PPO 结果；
5. 新生成的 closeout 文档；
6. 新生成的 PPT；
7. backup manifest；
8. Git commit hash；
9. Git push 状态；
10. 当前是否存在未提交变更。

本任务完成后停止。

不要继续推进 SYA / LCA WGEN，不要启动新的训练。
