# 021_07 SY2014 target更新、离线TD残差与三档氮排序审计

## 目标

基于021_05/021_06已有checkpoint、replay buffer和固定状态Q值，完成纯离线诊断：

1. 核对target network实际更新计数是否符合`target_update_interval=10000`；
2. 排除checkpoint保存/加载复用错误；
3. 计算各checkpoint当前模型在其replay buffer上的离线Bellman/TD残差；
4. 汇总N0/N50/N100完整排序变化，包括N50与N100之间的变化；
5. 明确哪些证据可支持结论，哪些因历史日志缺失只能标记为不可判定。

## 边界

- 不训练、不调用DSSAT；
- 不修改reward、IC、动作空间、预算、target interval或观测空间；
- 不把离线重算的Bellman残差冒充训练过程中逐步记录的loss；
- 不覆盖021_05/021_06结果；
- 观测空间Markov修复另行立项，本任务只诊断。

## target更新审计

对5K、10K、15K、20K、25K checkpoint提取：

- model.zip SHA256；
- `num_timesteps`、`_n_calls`、`_n_updates`；
- `target_update_interval`；
- online和target网络参数指纹；
- 相邻checkpoint网络参数最大绝对差。

判定：若model.zip各不相同、online参数变化、target参数只在`_n_calls`跨越10000整数倍后变化，则支持“正常target冻结窗口”；否则记录异常，不得强行解释。

## 离线TD残差

对每个checkpoint加载其完整replay buffer，按当前保存的online/target网络重算：

`TD target = n-step reward + (1-done) × discount × max_a Q_target(s',a)`

`TD error = TD target - Q_online(s,a_executed)`

统计总体和按请求动作索引分组的：均值、绝对均值、中位数、P95、最大值和Huber loss。该结果是“checkpoint时刻当前网络对当前buffer的Bellman残差”，不是历史训练loss曲线。

## 三档氮完整排序

读取021_06固定状态Q值，对15K与20K分别输出每个灌溉档位下N0/N50/N100的完整排名字符串，并统计：

- N0与任一高氮的相对顺序变化；
- N50与N100的相对顺序变化；
- 全局argmax变化；
- 无动作混叠状态中的变化。

## 输出

- `benchmark_results/021_07/021_07_target_update_audit.csv`
- `benchmark_results/021_07/021_07_network_parameter_differences.csv`
- `benchmark_results/021_07/021_07_td_residual_summary.csv`
- `benchmark_results/021_07/021_07_td_residual_by_action.csv`
- `benchmark_results/021_07/021_07_full_nitrogen_ranking.csv`
- `benchmark_results/021_07/021_07_full_nitrogen_ranking_changes.csv`
- 汇总图和JSON；
- 中文实验记录。

## 判定纪律

- target冻结与坍缩时间重合不等于target interval导致坍缩；
- Bellman残差变大只能证明数值拟合恶化，不能单独证明reward或某个超参数是根因；
- 不根据结果立即改target interval或reward；
- 若需要单变量训练，必须另写prompt并先做smoke。
