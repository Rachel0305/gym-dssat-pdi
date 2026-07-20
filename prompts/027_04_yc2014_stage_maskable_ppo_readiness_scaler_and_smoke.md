# 027_04 YC2014 阶段型 MaskablePPO 输入、scaler 与环境smoke

## 1. 唯一任务

按照027_00站点顺序，为YC2014建立站点专属阶段型MaskablePPO训练前入口。本任务只完成输入provenance、四基线、YC专属观测scaler、本地reward数值和完整季节no-op smoke；训练步数必须为0。

不得复用SY/HLA权重或scaler，不得联合训练，不得启动seed0正式训练。

## 2. 先审计再实现

必须从既有YC实验记录和代码中逐项确认，不凭记忆填写：

- YC2014权威MZX/WTH/SOL/CUL及treatment、IC、SDATE/PDATE/ICDAT；
- 当前原始观测维度及每个字段标签；
- null、recorded、DSSAT auto、official extension expert四基线的权威结果目录；
- 官方华北区域方案对应的六个决策阶段及其DSSAT/DAP映射依据；
- YC2014可定义的ETCP、WP_ET和PFP_N。

若四基线或输入provenance不完整，判C并停止，不得用旧的非同口径结果拼表。

## 3. 数据保护

- 原始输入只读；运行只使用新目录副本；执行前后核对SHA-256。
- 不修改IC、DSSAT输入、reward结构、动作空间或PPO超参数。
- 不覆盖已有YC结果。
- 若发现旧结果和当前权威输入不一致，保留旧结果并新建attempt记录。

## 4. YC专属scaler

只使用YC2014训练年的null、recorded、auto、official expert四条轨迹，在预注册六决策点采集完整原始观测；禁止使用验证年份。

- 记录真实观测维度和字段顺序；
- population mean/std；std<1e-6时scale=1并标记near_constant；
- 非常量维标准化均值绝对值<1e-4，std偏离1<1e-4；
- 逆变换最大误差<2e-3；
- 保存来源情景、DAP、原始状态及scaler表。

## 5. 本地reward数值

从本次确定性、完整精度的四基线前向结果生成：

```text
local_null_yield = YC2014 null yield
local_feasibility_yield = max(YC2014 auto yield, YC2014 official expert yield)
step reward = -(I + 5*N)/1000
terminal yield gain = max(0, yield-local_null_yield)/1000
terminal feasibility bonus = 1620/1000 if yield >= local_feasibility_yield
```

recorded不进入reward或选模，只用于最终比较。不得使用整数汇总值替代完整精度前向值。

## 6. 完整季节no-op smoke

- 精确到达全部预注册决策点；跳过任一点即停止，不能现场移动DAP；
- 全部动作action0；每步no-op有效，非法动作0；
- 观测维度正确且有限；
- 最终yield在2 kg/ha内复现精确null；I=0、N=0；
- 总reward严格为0（容差1e-10）；
- Summary.OUT与环境记账一致；
- 训练步数0，PPO模型0。

## 7. 分支

### A_ready_for_YC_seed0

输入、四基线、阶段点、scaler、reward闭合与no-op smoke全部通过。只允许另写YC2014 seed0 240步预注册训练prompt。

### B_environment_or_scaler_failed

输入完整，但阶段/scaler/smoke失败。停止并只诊断工程原因。

### C_input_or_baseline_provenance_incomplete

权威输入或四基线证据不完整。停止，不训练。

## 8. 输出

保存prompt、审计表、四基线CSV、四轨迹重放表、scaler来源状态、scaler表、reward JSON、六阶段动作表、result JSON、失败记录、代码和中文实验记录。完成后停止，等待用户确认。

