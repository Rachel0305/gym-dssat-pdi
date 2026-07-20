# 027_00 五站点站内训练—同站跨年迁移 MaskablePPO 总协议

## 1. 唯一主任务

验证阶段型 MaskablePPO 能否在五个站点分别学到高质量水氮策略，并把训练年冻结模型直接迁移到同一站点其他年份，判断产量、WP_ET 和 PFP_N 能否尽量同时超过 official extension expert、DSSAT auto 和 recorded/farmer practice。

本系列**不是多站点联合训练**，也**不是把 SY 权重搬到其他站点**。026_09–026_12 的跨站点接口审计保留作未来支线，现在暂停。

## 2. 严格串行顺序

| 顺序 | 站点 | 训练锚点 | 已确认首批验证年 |
|---:|---|---:|---|
| 1 | SY | 2014 | 2012、2015（026 已完成） |
| 2 | HLA | 2010 | 2007、2015、2016、2022 |
| 3 | YC | 2014 | 训练前从既有审计列出全部权威可用年 |
| 4 | FQ | 2016 | 训练前从既有审计列出全部权威可用年 |
| 5 | LC | 2010 | 训练前从既有审计列出全部权威可用年 |

一个站点完成判定后才进入下一个。不得因结果不理想删除权威年份；YC/FQ/LC 不得凭记忆写年份。

## 3. 框架冻结项

- SB3-Contrib MaskablePPO，网络 `[32,32]`；
- `learning_rate=3e-4`，`gamma=1`，`gae_lambda=1`；
- `n_steps=60`，`batch_size=30`，`n_epochs=5`；
- 9动作：`I∈{0,15,30} mm × N∈{0,50,100} kg/ha`；
- 季节预算 I≤120 mm、N≤300 kg/ha；预算mask、晚期禁氮mask和无效动作检查；
- seed=0/1/2；先固定240阶段步，checkpoint=0/60/120/180/240；
- checkpoint只按训练年前注册的确定性季节reward最大选择，并列取更早者；
- 不根据primary、recorded或full-pass结果重选checkpoint。

## 4. 只允许本地化的事实参数

- 本站点原始观测维度和固定scaler；
- 训练年null yield；
- 训练年auto/expert产量阈值；
- 官方区域方案的六个决策点；
- 输入文件、treatment、IC pointer和年份；
- 各年份本地四基线。

这些属于环境事实，不属于调参。不得为救单站点修改reward水氮相对成本、预算、动作档位或PPO超参数。

## 5. Reward结构

每站点结构相同，仅替换本地数值：

```text
step resource cost = -(I + 5*N) / 1000
terminal yield gain = max(0, yield - local_null_yield) / 1000
terminal feasibility bonus = 1620/1000,
  if yield >= max(local_auto_yield, local_official_expert_yield)
```

1620来自固定预算最大资源成本 `120 + 5×300`。recorded不进入reward或选模，避免验证基准泄漏，但必须进入最终比较。

## 6. 每站点执行链

### Phase A：输入、四基线、scaler、环境smoke

1. 审计权威年份、MZX/WTH/SOL/CUL、treatment、IC、SDATE/PDATE/ICDAT；
2. 备份并记录重要输入SHA-256，所有运行只用新目录副本；
3. 复用或重跑训练年null、recorded、auto、official expert；
4. 保存yield、I、N、ETCP、WP_ET、PFP_N；
5. 阶段点来自对应官方区域方案，不从成功策略反推；
6. 只用训练年预注册轨迹拟合站点专属scaler；
7. 完整全no-op阶段环境smoke必须复现null；
8. 本阶段0 PPO训练。

### Phase B：训练年seed0 smoke

只运行seed0、240阶段步，保存五个checkpoint的确定性评估、动作、mask、reward分项和模型哈希。科学失败照实记录，不现场调参。

### Phase C：三seed复核

seed0工程通过后，以完全相同配置运行seed1/2。训练年初步成功要求至少2/3预注册选中checkpoint通过auto+expert primary。不得增加seed凑通过率。

### Phase D：同站跨年冻结迁移

- 固定三个训练年checkpoint和SHA-256；
- 验证年 `learn()` 调用次数严格为0；
- 不在验证年重选训练年checkpoint；
- 每模型每年份一次确定性完整季节评估；
- 每年先确认四基线和输入provenance；
- 至少2/3固定模型通过才称该年份迁移成功。

### Phase E：表、图与汇总

每年份生成五情景结果：null、recorded、DSSAT auto、official expert、PPO。日值表和图必须含降雨、土壤水分/水氮胁迫、灌溉、施氮、产量、生物量和reward。

## 7. 三层成功定义

### Primary：超过auto + expert

- yield ≥ `max(auto, expert)`；
- WP_ET ≥ `max(auto, expert)`；
- PFP_N ≥ 所有正氮auto/expert中的最大值。

### Recorded扩展比较

yield、WP_ET逐项比较；recorded的PFP_N仅在NICM>0且可定义时参与。recorded为零氮时不得伪造成无穷或自动判失败。

### Full three-baseline pass

在所有可定义指标上，PPO同时不低于recorded、auto、expert三者最大值。不可定义项标记`not_comparable`；不得把primary通过包装成全面超过三基线。

## 8. 停止规则

- 训练年少于2/3 seed通过primary：停止该站点跨年迁移；
- 输入、baseline、mask或数值异常：停止，不修改IC/DSSAT输入追求结果；
- 不扫描reward、不延长训练、不加bonus、不改动作空间，除非另立prompt并经用户批准；
- 先smoke、后3 seed、再跨年，禁止并行启动五站点大训练。

## 9. 记录与Git

每一步必须有prompt、代码、CSV/JSON、失败记录和中文docs。模型zip、运行时临时目录不提交；summary、动作表、图、prompt、代码和docs可提交。未得到用户明确确认时不push。

