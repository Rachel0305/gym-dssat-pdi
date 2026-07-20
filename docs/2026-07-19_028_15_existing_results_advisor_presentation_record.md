# 028_15 现有强化学习结果导师汇报PPT记录

## 状态

completed

## 输出

- `docs/2026-07-19_028_15_existing_results_advisor_presentation.pptx`
- 幻灯片：43页
- 站点年份：17
- 五情景终值记录：85行

## 内容

1. 总体结论与三指标差距热图；
2. 统一判定口径和证据边界；
3. 两套官方推广expert阶段水肥模板；
4. 五站点结果总览；
5. 每个站点年份两页：
   - 五情景终值对照图；
   - 五情景精确数据表、PPO阶段措施、指标差距和seed证据；
6. 综合分级；
7. 历史DQN provisional对照；
8. 结论和下一步建议。

## 执行边界

- 新训练：0；
- 新DSSAT运行：0；
- reward、IC、输入和模型修改：0；
- 仅复用028_12、028_13和028_14冻结证据。

## 数据质量与QA

- 17/17站点年份均包含五情景终值图；
- 17/17站点年份均包含5行五情景精确数据表；
- N=0时PFP_N显示为NA；
- 产量、WP_ET、PFP_N胜出计数保持9/17、7/17、12/17；
- PowerPoint由`@oai/artifact-tool`生成；
- 最终PPTX重新渲染43/43页；
- `slides_test.py`通过：No overflow detected；
- 已检查全套缩略图总览，并抽查标题页、expert表、五情景图和逐年数据页的原始尺寸渲染。

## 失败与修复记录

1. 第一次构建调用等待窗口过短，命令超时；未产生科学结果，延长等待后重跑。
2. Windows环境未设置`HOME`导致Artifact Tool依赖路径解析到项目目录；显式设置`HOME=C:\Users\DELL`后恢复。
3. CSV首列带UTF-8 BOM导致`site`字段首次读取失败；解析器去除首列BOM后恢复。
4. 首版逐年表按旧情景别名匹配，只显示3行情景；核对源CSV后改为`recorded_farmer`和`rl_candidate`，最终每页完整5行。
5. `slides_test.py`首次同样受`HOME`路径影响；设置环境变量后通过。

## Git

本任务未收到Git提交或推送授权，因此未执行commit或push。
