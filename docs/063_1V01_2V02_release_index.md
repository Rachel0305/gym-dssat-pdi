# 五站双版本项目归档

日期：2026-10-14。用户确认版本边界并授权GitHub备份。

两版各固定SY、LC、HL、FQ、YC两套八种子，共80个100K模型、2014–2023的800个年度端点及400张统一主图。80个模型SHA256逐一一致，没有重新训练。

1V01保留原FQ2018输入与端点，以及YC当时30/160条主年度精确WP_ET记录。2V02采用FQ修正TMIN的2018验证端点，并补齐YC两套八种子的160/160条主年度精确WP_ET。仅16条FQ2018产量端点改变，其他站点与年份主产量一致。YC heldout-WGEN辅助记录按原样保留，不声称补了其精确WP_ET。

## 本地入口

- [1V01项目](../1V01/README.md)
- [2V02项目](../2V02/README.md)
- 主图表位置：版本/deliverables/站点/historical或augmented/seed_00–seed_07/figures或tables。
- 此前逐日过程图与事件表保留在每版本results原目录，README和MODEL_REGISTRY.csv提供索引。
- 完整性证据：各版本FINAL_RELEASE_QA.json、ARCHIVE_MANIFEST.csv、MODEL_REGISTRY.csv和METRIC_DEFINED_YEAR_COUNTS.csv。

## GitHub定位

仓库：Rachel0305/gym-dssat-pdi；分支：codex/five-site-1V01-2V02-20261014。

原版标签：five-site-1V01-20261014；修订版标签：five-site-2V02-20261014。

上传只包含两个版本及本次整理脚本/说明，不混入其他工作区修改。大CSV按字节分块，运行各版本restore_large_files.py恢复并核验。系统级DSSAT和Python环境不复制，REPLAY_ENVIRONMENT.json记录当前核验通过的环境。Windows检出长路径归档时使用Git的core.longpaths临时配置或较短根目录。

主报告统一四情景，不把存档旧FQ2018或替代自动灌溉图混入2V02比较。旧筛选表只作来源，不将修订后的端点默认为已重新筛选；原25/50/75K证据保留，但没有声明其FQ输入和YC效率也做了相同修订。
