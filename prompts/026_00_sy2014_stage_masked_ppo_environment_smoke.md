# 026_00 SY2014 阶段型 Masked PPO 环境与奖励闭合 smoke

## 1. 背景

当前不能概括为五站点DQN都已在修复后失败；只有SY2014完成了深度阶段型诊断。为继续满足“必须使用强化学习”的研究要求，新路线先在SY2014测试阶段型PPO，而不是继续修改DQN。

本任务只验证RL环境，不训练PPO。

## 2. 固定环境

- 输入：SY2014、IC=2、当前已验证MZX/WTH/SOL/CUL；
- 决策阶段：DAP1/30/50/65/85/110；
- 动作：9个I×N离散组合；
- DAP≥90禁止施氮；
- 超过剩余I120/N300预算的请求动作直接mask，不允许wrapper裁剪形成动作别名；
- 非决策日强制no-op；
- 观测复用021_24固定25维scaler。

## 3. 奖励

每个阶段即时奖励：

`r_t = -(irrigation + 5*nitrogen)/1000`

收获终止步额外加入：

`(max(0, yield-null_yield) + feasibility_bonus)/1000`

因此整季奖励和必须严格等于既有`terminal_complete_returns`的G0/1000。未来PPO固定`gamma=1, gae_lambda=1`，但本任务不训练。

## 4. Smoke

使用022_02已验证动作序列`[3,4,7,1,1,0]`跑一个确定性季节，要求：

- 六个阶段精确出现；
- 每步选中动作均在mask内；
- 其余日期全部no-op；
- 产量11202±2 kg/ha；
- I60/N200；
- 总奖励6.354004±0.002；
- WP_ET=2.31附近、PFP_N=56附近；
- 原生Summary资源总量匹配；
- 所有观测和奖励有限。

## 5. 依赖边界

- 审计SB3与sb3-contrib是否存在；
- 不安装、升级或卸载任何包；
- 若环境smoke通过但sb3-contrib缺失，判`A_environment_ready_dependency_missing`；
- 只有用户明确批准项目Docker虚拟环境安装匹配版本后，才允许另立026_01 MaskablePPO短smoke；
- 不使用无mask普通PPO冒充MaskablePPO。

