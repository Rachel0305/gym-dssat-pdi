# E2 跨 seed 5K 稳定性与 WP_ET 核验

## 目的

冻结 E2 的 5K checkpoint，使用已经完成的 seed0/seed1/seed2 验证和 WP_ET replay，判断 E2 相对各自 no-forecast 是否具有跨 seed 稳定优势。

## 数据范围

- 站点：SYA / originIC；验证年：2014–2023；checkpoint：5000。
- E2：`e2_s0`、`e2_s1`、`e2_s2`；no-forecast：`nof_s0`、`nof_s1`、`nof_s2`。
- 读取既有 5K daily validation summary、147E2 跨 seed 动作/天气响应审计、151E2N WP_ET replay；不重新训练、不修改奖励函数或动作评价机制。

## 必须报告

1. 每个 seed 的平均产量、PFP-N、WP_ET、加权 WP_ET、ETCP；
2. E2−no-forecast 的逐 seed 均值差、胜出年份数和 pooled/weighted 汇总；
3. 5K checkpoint 的动作多样性门禁和天气响应门禁；
4. WP_ET replay 可复现性核验（产量、灌溉、施氮、动作序列闭合）。

## 判定规则

- E2 只有在主要指标方向跨 seed 一致时，才可称为稳定优势；
- 若产量/PFP-N 有优势但 WP_ET 方向不一致或为负，应表述为“有条件的 forecast 优势”，不能称为全面优于 no-forecast；
- 5K 若动作多样性通过但天气响应少于 3/3 seed，通过“可运行”但不通过“完全跨 seed 稳定”。
