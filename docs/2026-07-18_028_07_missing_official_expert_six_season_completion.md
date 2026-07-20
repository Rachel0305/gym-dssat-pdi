# 028_07 缺失 official expert 六季补齐记录

状态：`completed`

## 范围与纪律

- 只包含 YC2008 与 FQ2013/2014/2019/2020/2023 的 official expert。
- null、recorded、DSSAT auto 复用 028_06 已验证快照，没有重跑。
- RL 训练 0 次。
- 使用 018_03 已冻结华北黄淮/汾渭夏玉米固定 DAP 调度。

## Input smoke

|site|year|experiment|weather token|weather check|status|
|---|---:|---:|---|---|---|
|YC|2008|1|CNYC0801|True|pass|
|FQ|2013|2|CNFQ1301|True|pass|
|FQ|2014|2|CNFQ1401|True|pass|
|FQ|2019|2|CNFQ1901|True|pass|
|FQ|2020|2|CNFQ2001|True|pass|
|FQ|2023|2|CNFQ2301|True|pass|

## DSSAT 结果

|site|year|status|yield kg/ha|irrigation mm|N kg/ha|error|
|---|---:|---|---:|---:|---:|---|
|YC|2008|ok|8157.0|228.8|247.0|nan|
|FQ|2013|ok|7381.0|198.8|247.0|nan|
|FQ|2014|ok|8643.0|228.8|247.0|nan|
|FQ|2019|ok|8821.0|228.8|247.0|nan|
|FQ|2020|ok|9541.0|228.8|247.0|nan|
|FQ|2023|ok|9284.0|198.8|247.0|nan|

## 失败与边界

- 输入 smoke 不通过时不得启动 DSSAT。
- 第一次结果检查错误地把全季计划总量与实际执行总量作严格浮点相等比较；FQ2013/2023 在 DAP100 前终止，且 MgmtEvent.OUT 有输出舍入。该错误判定已保留，修正只重算检查结果，没有重跑 DSSAT。
- 该任务只补齐比较基线，不构成 RL 成功或失败判定。
