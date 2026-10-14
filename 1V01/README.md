# 1V01 五站双域八种子项目

本版本固定80个100K策略与2014–2023共800个年度端点。代码在src/scripts，配置在configs，源输入在DSSAT_auto_validation/data，模型与验证在原相对路径benchmark_results/results。主报告统一四情景：Null、Recorded、Expert、PPO。

原版：FQ保留原2018天气及零产量结果；YC保留当时精确WP_ET的缺项，缺失不填0。

## 主结果入口

- [SY 未增强](deliverables/SY/historical/) / [SY 增强](deliverables/SY/augmented/)；seed_00–seed_07各有figures与tables。
- [LC 未增强](deliverables/LC/historical/) / [LC 增强](deliverables/LC/augmented/)；seed_00–seed_07各有figures与tables。
- [HL 未增强](deliverables/HL/historical/) / [HL 增强](deliverables/HL/augmented/)；seed_00–seed_07各有figures与tables。
- [FQ 未增强](deliverables/FQ/historical/) / [FQ 增强](deliverables/FQ/augmented/)；seed_00–seed_07各有figures与tables。
- [YC 未增强](deliverables/YC/historical/) / [YC 增强](deliverables/YC/augmented/)；seed_00–seed_07各有figures与tables。

每策略5张独立PNG，共400张统一主图；此前逐日过程图及事件表保留在results中的原目录。MODEL_REGISTRY.csv索引80个真实模型；METRIC_COVERAGE.csv记录指标可用性；deliverables/all_80_policy_800_year_metrics.csv为主端点表。五情景旧图属于历史存档，2V02比较时优先用deliverables，避免误读旧FQ2018图。

## 复现

在本文件夹作为项目根目录运行，先用Python执行restore_large_files.py恢复字节分块的大表；文件恢复后hash自动核验。项目代码需要已有gym-DSSAT/PPO环境与DSSAT4.8运行时（例如原容器/opt/gym_dssat_pdi/bin/python和/opt/dssat_pdi/run_dssat）；本归档不复制系统级运行时或大缓存。代码保留原路径，模型保持原sha256。SOURCE_MANIFEST.csv记录复制来源，RELEASE_MANIFEST.csv为最终归档逐文件hash。

主对照固定100K，不做逐年择优。原25/50/75K训练验证保留；V02的FQ输入修订与YC效率补齐仅针对100K主端点，不把其他checkpoint描述成已做相同修订。各站增强算法和天气库不同，不做混站排名，探索性筛选不替代八seed总体证据。
