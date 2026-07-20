# 026_09 HLA 五年份冻结 SY 阶段型 PPO 跨站点验证

## 1. 目的

将 SY2014 训练并在 SY 三个权威年份完成验证的三个冻结阶段型 MaskablePPO，直接迁移到 HLA 2007/2010/2015/2016/2022。目标年份不训练、不重新选 checkpoint。

## 2. 输入与基线

- 使用 020_11 已验证的五个 prepared input 目录；
- 四基线与 Summary.OUT 复用 020_11 的 null、recorded farmer、DSSAT auto、official extension expert；
- 重新从 Summary.OUT 统一计算 ETCP、WP_ET、PFP_N；
- recorded 单独报告，主判据仍为每年 auto 与 official expert 的较高产量、WP_ET、正施氮 PFP_N。

## 3. 执行

1. HLA2010 seed0 单季 smoke；
2. smoke 通过后串行运行五年三模型，共 15 季；
3. CPU-only，不并行；
4. 模型与 scaler 哈希运行前后不变；
5. 阶段、动作、预算和 mask 与 SY 完全冻结；
6. 冻结评估中的环境 reward 只作工程日志，不作跨站点科学比较。

## 4. 判据与边界

每年至少 2/3 模型同时通过当地 yield/WP_ET/PFP_N 门槛，记该年通过。五年全部通过为 A；部分年份通过为 B；输入、DAP110、mask、Summary 或哈希失败为 C。

本任务检验的是 SY 模型参数的跨站点迁移，不是 HLA 本地 PPO 训练，也不能由单站点成功推出五站点普遍成功。
