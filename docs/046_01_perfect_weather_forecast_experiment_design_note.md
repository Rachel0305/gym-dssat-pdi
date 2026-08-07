# 046_01 perfect weather forecast 实验设计说明

## 参考文章

文章：Li Kexin 等，*An integrated meteorological adaptive simulation-optimization framework for real-time irrigation scheduling considering perfect weather forecasts*，Agricultural Systems，2026，DOI: 10.1016/j.agsy.2025.104567。

公开摘要显示，该文构建了一个实时灌溉模拟-优化框架，把“短期 5 天”和“中期 15 天”的完美天气预报接入 AquaCrop-OSPy + NSGA-III 优化，并与多年优化得到的固定灌溉策略作基准对比。核心结论是：完美天气预报能帮助推迟或减少不必要灌溉，在保持产量的同时减少灌溉用水、提高灌溉水生产力。

## 对我们实验的启发

这篇文章支持的不是“只加一个天气变量就算创新”，而是一个成对比较设计：

1. 固定同一站点、同一训练/验证年份、同一动作约束、同一奖励函数；
2. 只改变 observation 是否能看到未来天气；
3. 比较最终产量、灌溉量、施氮量、WP_ET、PFP_N；
4. 同时比较动作是否更像天气响应：例如未来有雨时是否减少/推迟灌溉，未来连续无雨时是否提前灌溉。

## 建议我们的主设计

### A. no-forecast PPO

PPO 只能看到当前作物/土壤状态、DAP、累计水氮等，不给未来天气。

### B. perfect-forecast PPO

PPO 额外看到来自真实历史天气的“完美预报”特征，例如：

- future_7d_rain_sum
- future_7d_tmax_mean
- future_7d_tmin_mean
- future_7d_srad_mean
- past_7d_rain_sum

这里的“perfect”意思是：在历史回放实验中，把后 7 天真实天气当作理想预报输入。它不声称现实预报没有误差，只是在量化“如果 agent 有未来天气信息，理论上能改善多少”。

## 为什么主窗口建议先用 7 天

- 我们的管理安全层已有最小操作间隔 7 天，未来 7 天窗口和管理节奏匹配。
- 参考文章使用 5 天和 15 天预报；我们用 7 天属于同类“短期预报”设计。
- 后续如果 7 天有效，再做 3/7/15 天 horizon sensitivity；现在不建议一开始就扫多个窗口，避免任务膨胀。

## 成功判据建议

forecast 版本不应只看最终 reward。更合适的判据是：

- 在产量不下降超过预设容差的前提下，灌溉量下降或 WP_ET 提高；
- 干旱/未来无雨年份比湿润/未来有雨年份更愿意灌溉；
- 未来 7 天有明显降雨时，PPO 灌溉概率或灌溉事件数下降；
- 逐年动作序列不能重新坍缩成完全模板化。

## 当前建议

可以把 042_15 作为 no-forecast 当前最好线，后续再跑一条严格配对的 perfect-forecast PPO：

- 相同 lowIC；
- 相同训练/验证年份；
- 相同 binary 或 9-action 动作约束；
- 相同 checkpoint 选择规则；
- 唯一差异是 observation 是否包含 future weather。

这样才能回答导师关心的核心问题：天气预报信息是否真的让 RL 决策更合理、更节水，而不是只是换了一次算法配置。

