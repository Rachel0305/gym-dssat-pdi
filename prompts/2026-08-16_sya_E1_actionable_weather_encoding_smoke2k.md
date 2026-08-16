# 141E1 SY 天气特征编码 2K smoke

## 目的

上一轮 140 的 28 维逐日 rolling forecast 在 5K 出现策略收缩；139 rolling oracle 又表明未来降雨的数量和时序确实会改变更优灌溉动作。E1 只改变天气输入表达，检查 PPO 是否更容易利用这类信息。

## 唯一变化

把原来 `7 天 × rain/srad/tmax/tmin = 28` 个逐日值，替换成 12 个有界、面向动作时序的编码：

- 降雨前缀：未来 1、3、7 天累计降雨；
- 降雨时序：未来 7 天最大单日雨、雨日比例、首次降雨提前量、干旱标志；
- 热量和辐射：未来 7 天平均/最高温、平均最低温、平均太阳辐射；
- 覆盖率：未来窗口有效天数比例，避免季末缺测被误当作零温度。

窗口每天滚动，仅使用 date+1 至 date+7，不含当天。历史 WTH 只作为 perfect-hindcast 诊断输入，不代表业务预报精度。

## 严格固定

- SYA/SY、originIC、seed=0；训练 2005--2013、验证 2014--2023；
- 与 140F 相同的 16 动作格、046_02 reward、安全 mask、PPO 超参数和 DSSAT 输入；
- 不改变奖励函数，不改变动作合法性、传输或多样性评价机制；
- 不启用 native DSSAT automatic irrigation，不使用 `dssat_auto_external_n`，不修改源 WTH。

## 本轮执行边界

只允许 dry-run 和串行 2K smoke，checkpoint 为 1K/2K。2K 结果出来后再决定是否授权 5K；本轮禁止 5K、25K、50K、75K、100K。

## 2K 闸门

沿用既有 forecast smoke gate：天气列存在且季内变化；动作全部在 16 格内；存在正动作和 DAP1 后正动作；request/safe 无传输 mismatch；至少两个非零动作组合并使用新增档位。闸门只判断链路与早期动作非塌缩，不把 2K 指标写成稳定优势。

