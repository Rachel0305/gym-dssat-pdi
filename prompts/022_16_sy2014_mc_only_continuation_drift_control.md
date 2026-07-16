# 022_16 SY2014 MC-only续训跨阶段漂移对照

## 1. 唯一问题

022_15在MC+pairwise继续训练后成功纠正DAP65局部排序，但DAP110支持集argmax发生跨seed漂移。本任务只判断：**没有pairwise时，仅继续优化MC回归，DAP110漂移是否仍会发生。**

本任务不能解释漂移的完整机制，也不用于调整lambda、训练步数或guardrail。

## 2. 严格A/B对应

Control（本任务）：

- 起点仍为022_08 seed0/1/2 checkpoint；
- 重新初始化Adam，lr=1e-4；
- 只优化原216条训练转移的`L_MC`；
- 每seed固定3000 updates；
- 保存0/500/1500/3000。

Treatment（既有022_15）：配置完全相同，但loss为`L_MC+0.0060048738917845*L_pair`。

除pairwise项外不得改变任何设置。

## 3. 预注册判定

- A（pairwise非必要）：MC-only最终至少2/3 seed的DAP110未污染测试状态发生至少1个argmax改变；
- B（pairwise在当前设置下是必要条件）：MC-only最终0/3 seed发生DAP110改变；
- C（混合）：仅1/3 seed发生改变；
- D：实现/数值失败。

“非必要”不等于“pairwise完全无影响”；“必要”也不等于已经解释lambda或结构耦合机制。

## 4. DAP110支持说明

DAP110合法支持仅`{0,1}`，但原固定网格数据仍有48条DAP110转移（action0=40、action1=8），不能预先称为样本量最小。额外保存每个未污染测试状态的`Q(a0)-Q(a1)`、argmax与变化标记，用于描述是否存在近似平局，但本任务不因此放宽“argmax零变化”guardrail。

## 5. 输出与停止线

- MC loss、pair margin、逐阶段argmax变化、DAP110逐状态Q差距；
- 与022_15同seed最终状态变化重叠表；
- checkpoint、CSV、PNG/SVG、JSON和中文记录；
- DQN离线更新9000次，DSSAT 0次，在线交互0次；
- 不自动启动warm-start、不调整pairwise、不追加训练。

