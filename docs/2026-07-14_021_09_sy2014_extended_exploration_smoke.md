# 021_09 SY2014 延长探索衰减单变量 smoke

## 状态

- dry-run：**passed**
- 5K smoke：**passed**
- 正式25K对照：**用户已确认，使用独立配置启动**
- 跨seed/跨站点：**未授权、未进行**

## 依据

021_08发现，SY2014 seed0 的15K–20K坍缩窗口中：

- 请求正灌溉比例基本不变（65.5%→65.2%）；
- 请求正氮比例由61.6%降至45.5%；
- 水氮同时操作由37.7%降至21.9%；
- epsilon由0.186降至0.05，动作熵同步下降。

因此本任务只改变`exploration_fraction`，检验探索率到达下限是否参与氮策略坍缩。

## 单变量设计

| 参数 | 021_05基准 | 021_09 smoke |
|---|---:|---:|
| exploration_fraction | 0.35 | 0.70 |
| exploration_initial_eps | 1.0 | 1.0 |
| exploration_final_eps | 0.05 | 0.05 |
| 全局计划步数 | 50K | 50K |
| smoke停止步数 | 5K | 5K |

其他内容全部冻结：SY2014、IC=2、seed0、reward、动作空间、I120/N300预算、7日间隔、DAP窗口、学习率、buffer、n-step、target interval及DSSAT输入。

## dry-run

- 实验数：1
- 站点年份：SY2014
- seed：0
- config hash：`a762b0e1d9dba553e80cd1a7a82bbc4b4abbc502448733420e604fae507a5751`
- 输出目录独立于021_05，不覆盖旧结果。

## smoke结果

| 检查项 | 结果 | 判定 |
|---|---:|---|
| completed_steps | 5,000 | passed |
| exploration schedule total | 50,000 | passed |
| SB3 `_total_timesteps` | 50,000 | passed |
| 5K epsilon | 0.864313 | 符合0.70衰减计划 |
| runtime audit | true | passed |
| replay buffer | saved | passed |
| RNG state | saved | passed |
| IC mode | 2 | passed |
| IC源SHA256 | `20b071bc...a274418a` | 与确认输入一致 |
| 耗时 | 45.6秒 | 无OOM |

5K确定性评估输出为：产量11199 kg/ha、生物量20055 kg/ha、I120/N300、reward 4171.24。该数字只说明链路和动作均能运行，**不能用于判断延长探索是否解决15K–20K坍缩**。

## 运行异常记录

第一次启动命令的外层shell超时设置为1秒，返回超时；随后改用可持续等待的执行方式，正式任务正常完成，runner报告`failures=[]`。没有发现重复checkpoint或旧结果覆盖。

## 判定

smoke满足进入最多25K单变量正式对照的工程条件：

1. 新epsilon曲线准确生效；
2. 50K全局时间轴没有被smoke的5K停止点破坏；
3. IC=2和其他冻结配置未漂移；
4. 无OOM或运行链路异常。

但目前只完成5K，尚未产生关于坍缩机制的新科学结论。正式对照必须覆盖15K、20K、25K，检查高产和氮投入是否持续，而不能只选择单个最佳尖峰。

## 输出

- `prompts/021_09_sy2014_extended_exploration_single_variable_test.md`
- `configs/experiments/021_09_sy2014_extended_exploration_smoke.yaml`
- `benchmark_results/021_09/021_09_sy2014_extended_exploration_smoke__sy_2014_seed0/`
- 本记录。

## Git

- 用户计划稍后手动提交；本轮未执行commit或push。
