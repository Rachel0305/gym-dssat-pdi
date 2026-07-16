# 021_40 SY2014 agent/demo n-step梯度来源拆分记录

## 复现验证

五个checkpoint online-Q哈希全部与021_39相同：True。因此新增autograd日志没有改变训练轨迹。本轮没有重复DSSAT评估。

## 坍缩与恢复期

| metric | collapse_median | recovery_median | absolute_stage_difference |
| --- | --- | --- | --- |
| grad_tdn_demo | 5.01336 | 2.98955 | 2.02381 |
| grad_tdn_agent | 2.38287 | 2.41591 | 0.03304 |
| grad_margin_demo | 0.54692 | 0.32047 | 0.22645 |
| cos_tdn_demo_margin | -0.70633 | -0.70444 | 0.00188 |
| cos_tdn_agent_margin | 0.18443 | 0.24131 | 0.05687 |
| cos_tdn_demo_agent | -0.32967 | -0.46821 | 0.13855 |

预注册来源判定：**no_clear_source**。demo cosine阶段差=0.0019，agent cosine阶段差=0.0569。

## 边界

本结果只定位候选来源，不证明因果，也未现场修改任何loss。
