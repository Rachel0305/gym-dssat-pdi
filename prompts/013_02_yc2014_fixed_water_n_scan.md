# 013_02 YC2014 固定水氮扫描

## 目标

在不做任何 RL 训练的前提下，对 `YC 2014` 新品种/新输入包做低成本前向扫描，判断这个年份在当前输入条件下的水氮响应空间，回答：

1. 记录管理 `120 mm / 374 kg N/ha` 是否明显过高；
2. 是否存在更低氮、更低水但接近记录产量的组合；
3. 后续是否值得把 `YC 2014` 作为 RL 主试验候选年份。

## 输入

- 输入包目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC`
- MZX：`CNYC0801.MZX`
- 目标年份：`2014`
- 目标 treatment：`TRNO=2`
- 运行环境：Docker 容器 `b2fd6726c8c1`
- Python：`/opt/gym_dssat_pdi/bin/python`

## 运行原则

- 只做前向模拟，不训练；
- 基础场景使用 `null`（关闭原记录灌溉施肥），然后由脚本在环境 step 中注入固定管理动作；
- 控制算力，优先跑一个 `3 × 4 = 12` 的水氮组合网格；
- 每个组合保存日值、事件、最终汇总；
- 额外生成 4 张热图：
  - `GWAD`
  - `CWAD`
  - `max WSPD`
  - `max NSTD`

## 扫描组合

- 灌溉总量：`0, 60, 120 mm`
- 施氮总量：`0, 150, 300, 375 kg N/ha`

## 固定施用时机

参考 `YC 2014 recorded` 原记录时机：

- 氮肥两次：
  - `DAP 0`
  - `DAP 43`
- 灌溉一次：
  - `DAP 43`

其中氮肥分配比例按照原记录 `96 : 278` 拆分。

## 输出

输出目录：

`DSSAT_auto_validation/multisite_new_cultivar_yc2014_fixed_water_n_scan_013_02/`

至少包含：

- `013_02_yc2014_fixed_scan_summary.csv`
- `013_02_yc2014_fixed_scan_daily.csv`
- `figures/`
- 中文实验记录 `docs/2026-06-30_013_02_yc2014_fixed_water_n_scan_record.md`

## 判读重点

- 若 `I60/I120 + N150/N300` 已接近 `I120 + N374`，说明有较强节水节氮优化空间；
- 若只有接近高水高氮时才有明显增产，则后续 PPO 的“节本增效”空间较小；
- 若热图显示 `GWAD` 对水或氮的边际响应很弱，也要明确记录，不要强推 RL。
