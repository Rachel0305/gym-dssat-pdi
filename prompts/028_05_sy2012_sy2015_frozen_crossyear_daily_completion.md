# 028_05 SY2012/SY2015 固定权重跨年日值证据补齐

## 目标

复用 SY2014 已训练的三个阶段型 MaskablePPO 模型及 026_07 已发布的零训练跨年
结果，为 SY2012、SY2015 的每个 seed 补齐冻结评估 DSSAT 快照、五情景日值表和
027_05 样式 PNG/SVG。不得重训、不得按测试年重选 checkpoint。

## 固定对象

- seed0：SY2014 checkpoint120；
- seed1：SY2014 checkpoint60；
- seed2：SY2014 checkpoint240；
- 测试年份：2012、2015；
- 共6次确定性冻结季节评估。

## 约束与验收

1. 复用026_07的四基线快照，不重跑；
2. 继续使用已批准的运行时ICDAT对齐，原始CNSY1201.MZX哈希必须不变；
3. 模型哈希、动作序列、产量、水氮投入、WP_ET、PFP_N与026_07逐项一致；
4. `learn_calls=0`且模型`num_timesteps`不变；
5. 保存每个seed的完整快照、动作CSV、五情景日值CSV、终值CSV、PNG和SVG；
6. 逐seed计算至少一项严格超过四基线的导师汇报标签，其余指标只报告差距；
7. 串行、CPU单线程运行，防止OOM。

