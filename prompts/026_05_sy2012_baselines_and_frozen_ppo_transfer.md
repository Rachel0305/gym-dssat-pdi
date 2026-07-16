# 026_05 SY2012 四基线与三个冻结 PPO 模型零训练迁移

## 1. 问题

检验在SY2014训练并选出的三个具体MaskablePPO checkpoint，能否在不重新训练、不重新选模的条件下直接迁移到SY2012。

本任务验证的是固定模型参数的跨年份泛化，不是“在SY2012重新训练同一算法”。

## 2. 冻结模型

- seed0：`benchmark_results/026_03/checkpoint_000120.zip`；
- seed1：`benchmark_results/026_02/checkpoint_000060.zip`；
- seed2：`benchmark_results/026_04/checkpoint_000240.zip`。

运行前记录SHA256；运行后再次核对，禁止覆盖或修改模型。

## 3. SY2012输入

- 当前权威输入包：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/`；
- treatment=1，year=2012，IC=1；
- weather=`CNSY1201.WTH`；
- soil=`SOIL.SOL`，cultivar=`MZCER048.CUL`；
- 不修改原始MZX/WTH/SOL/CUL；所有场景只在新结果目录复制输入并运行。

## 4. 第一阶段：四基线

在当前输入链下重新前向运行：

1. null；
2. recorded/farmer practice；
3. DSSAT auto；
4. official extension expert fixed DAP（东北春玉米官方推广阶段方案）。

记录产量、灌溉、施氮、ETCP、WP_ET和PFP_N。若任一场景未终止、输入处理行不一致、Summary无法匹配或official expert未形成有效正氮基准，停止，不执行PPO迁移。

施氮采用双口径审计：MgmtEvent/管理表记录实际管理事件总量，Summary.OUT 的 NICM/YPNAM记录DSSAT季节统计口径。PFP_N使用DSSAT原生NICM/YPNAM；两者差值必须保留，不能为了强行一致而覆盖。首次执行因recorded事件量293与NICM247被严格匹配器中止，保留为attempt1；本次修正只解除“两个不同口径必须数值相等”的错误假设，产量和灌溉仍须与Summary匹配。

第二次执行发现2012 treatment原始指针为MI=0/MF=1；只把模拟控制改为IRRIG=L/FERTI=L并不能启用外部灌溉，expert请求266.25 mm而Summary仍为0。attempt2保留。expert/PPO运行副本必须显式保持IC=1，同时设置MI=1/MF=1、清零副本中原记录事件并启用L/L；这只修复agent外部动作输入链，不修改权威原始MZX。null/recorded/auto仍按各自原定义运行。

## 5. 第二阶段：冻结模型迁移

- 每个模型在SY2012只做一次确定性完整季节评估；
- 继续使用六阶段DAP1/30/50/65/85/110、原9动作、预算mask和DAP>=90禁氮；
- 使用SY2014训练时冻结的25维scaler；
- 不调用`learn()`，训练步数严格为0；
- 不根据SY2012结果重新挑SY2014 checkpoint。

## 6. 本地跨年判据

以SY2012的DSSAT auto和official extension expert为主要对照：

- yield至少不低于两者的较高值；
- WP_ET至少不低于两者的较高值；
- PFP_N仅在基线施氮量>0时定义，并至少不低于所有可定义主要基线的较高值；
- 水氮投入单独报告，不把零施氮基线的PFP_N错误设为无穷大。

### A_fixed_models_transfer

四基线审计通过，且至少2/3冻结模型满足全部本地判据。

### B_models_do_not_transfer

基线和工程审计通过，但少于2/3模型满足。

### C_baseline_or_execution_blocked

输入、四基线、模型哈希、环境、mask或数值检查失败。

## 7. 边界

- 不训练；
- 不修改reward、IC、scaler、动作空间或阶段点；
- 不增加模型或年份；
- 不用SY2012选择checkpoint；
- 不覆盖任何历史结果。
