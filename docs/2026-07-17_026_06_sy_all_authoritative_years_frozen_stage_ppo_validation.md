# 026_06 SY 全部权威年份冻结阶段型 PPO 验证记录

## 当前状态

`partial / C_input_or_execution_blocked`

本任务尚未进入 SY2014/SY2015 批量前向验证，也没有训练 PPO。第一次只执行了 SY2015 null 单季 smoke。

## 年份审计

当前 SY MZX 的正式 treatment 为2012、2014、2015。虽然目录还包含2001–2023多个WTH，但其他年份没有权威 treatment/IC，因此不纳入本轮。

## 第一次 smoke 失败

SY2015 null smoke 在外层等待300秒后仍未返回。检查容器进程和 `pdi_gym.log` 后确认，DSSAT已经输出：

```text
STOP 99
Input date is greather than 99 years after start simulation date.
File: fileX.MZX Line: 33 Error key: Y4KDOY
```

对应输入日期为：

| year | treatment | IC pointer | SDATE | ICDAT | PDATE | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 2012 | 1 | 1 | 12092 | 14099 | 12119 | 年份不一致 |
| 2014 | 2 | 2 | 14091 | 14099 | 14111 | 合法 |
| 2015 | 3 | 1 | 15091 | 14099 | 15108 | 年份不一致且导致Y4KDOY |

失败进程已经终止；没有并行残留进程，没有产生PPO结果。

## 对026_05的影响

026_05 的 SY2012 运行在工程上正常终止，模型哈希、动作、Summary等检查均通过；但本次新增审计发现其IC=1的 `ICDAT=14099` 不属于2012季节。因此026_05目前只能保留为“当前原始MZX可运行结果”，不能在日期溯源问题解决前升级成完全可信的正式跨年结论。

## 为什么没有自动修复

一种最小修复候选是在**运行副本**中保持IC profile和treatment IC指针不变，只把ICDAT按年份对齐，例如2012改为12099、2015改为15099。早期 `ppo_safe_rendering.py` 曾采用同类跨年日期对齐方法，但这仍属于输入日期处理规则的科学选择，不能在本任务中静默实施。

本轮没有：

- 把2015改成IC=0；
- 修改原始MZX；
- 修改IC水氮剖面；
- 启动14个正式前向季节；
- 继续进入其他站点。

## 下一步所需确认

需明确批准以下方案后才能继续：只在每个运行副本中按目标年份对齐ICDAT，保留IC profile、IC指针、SDATE、PDATE及全部水氮初值不变；先重新验证2012和2015的null/recorded/auto，再恢复冻结PPO迁移。所有对齐前结果与对齐后结果分目录保存，不覆盖。
