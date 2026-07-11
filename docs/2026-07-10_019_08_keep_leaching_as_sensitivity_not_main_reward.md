# 019_08 决策记录：淋洗惩罚暂不并入正式 reward

## 结论先行

本轮决定：**氮淋洗惩罚暂不并入正式 DQN reward**。

正式主线仍然保持当前统一经济型奖励：

```text
reward = max(0, final_grnwt - local_null_yield)
         - water_cost * irrigation
         - nitrogen_cost * nitrogen
```

氮淋洗相关变量和惩罚项作为敏感性分析、环境效益扩展和后续可选方向保留。

## 为什么这样决定

019_03–019_05 已经确认链路没问题：

- DSSAT/PDI 能输出 `CLeach`、`TLeachD`、`NLCC` 等淋洗变量；
- gym-DSSAT full state 可以读取 `cleach`；
- reward 中加入 `leaching_cost * delta_cleach` 技术上可行。

019_06–019_07 的 FQ2016 测试说明：

- `leaching_cost=20` 在 500-step smoke 和 5K 的 2000-step checkpoint 上有过较好表现；
- 但 5K 后期退化明显，4000/5000-step checkpoint 变成不灌溉、只施氮，产量降到 7106 kg/ha；
- 无淋洗惩罚 `leaching_cost=0` 在 5K 下反而更稳定，4000/5000-step 都达到 8012 kg/ha。

因此，淋洗惩罚目前可以证明“能接入、能影响策略”，但还不能证明“适合直接作为正式 reward 的一部分”。

## 对论文/汇报的表达

可以这样向导师解释：

> 我们已经检查了氮淋洗变量是否能从 DSSAT/PDI 接入 DQN reward，技术链路是通的。初步敏感性测试显示，加入淋洗惩罚会显著影响策略学习，但目前会带来训练不稳定和 checkpoint 选择问题。因此正式主线暂时聚焦产量和水氮投入效率，淋洗项作为环境效益扩展保留，不在当前阶段强行并入主 reward。

## 后续安排

1. 正式 DQN 主线不改 reward。
2. 不继续对 `leaching_cost=20` 做长训练。
3. 淋洗惩罚相关结果作为敏感性分析保留。
4. 下一步继续五站点优化空间审计和 DQN 成功策略筛选，核心目标仍是：产量、水分利用效率、氮肥利用效率尽量同时超过 expert 和 DSSAT auto。
