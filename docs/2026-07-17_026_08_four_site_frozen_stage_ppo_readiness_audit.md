# 026_08 其余四站点冻结阶段型 PPO 验证就绪审计

任务为零训练、零 DSSAT 的静态审计。第一次运行因 Pandas 把字符串 `null` 解析为 NA、且 LC expert 标签使用了缩写而错误判为 0/8；失败保存在 `benchmark_results/026_08/`。修正标签读取后，`benchmark_results/026_08_attempt2/` 得到：

- 分支：`A_all_cases_ready_for_serial_smoke`；
- 候选站点年份：8；
- 就绪：8；
- 阻塞：0；
- SY 026_07：`A_SY_all_years_transfer_after_icdat_alignment`；
- DSSAT 调用：0；训练步数：0。

后续验证集固定为 HLA 2007/2010/2015/2016/2022、YC2014、FQ2016、LC2010。它们来自已有正式 adapter/证据链，不等同于“目录中所有 WTH 年份”。执行顺序固定为 HLA、YC、FQ、LC，每个站点必须先 smoke，模型固定为 026_02/03/04 三个 SY2014 checkpoint。
