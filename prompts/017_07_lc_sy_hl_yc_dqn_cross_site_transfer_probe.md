# 017_07 LC/SY 跨站点 DQN 策略迁移 probe

## 目的

在不重新训练、不额外堆算力的前提下，检查已经在 HLA/YC 得到的 baseline-relative DQN 策略是否可以直接迁移到另外两个站点：

- LC：栾城站，输入包位于 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/LC`
- SY：沈阳站，输入包位于 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY`

核心问题：

1. LC/SY 的本地 null、recorded、DSSAT auto 基准是否能正常运行？
2. HLA2010 训练得到的 DQN checkpoint 能否直接用于 LC/SY？
3. YC2014 训练得到的 DQN checkpoint 能否直接用于 LC/SY？
4. 如果不能，失败原因是输入/观测空间不兼容、模型迁移无效，还是 LC/SY 本身缺少可优化空间？

## 严格约束

- 不训练新模型。
- 不覆盖已有 HLA/YC/FQ 结果。
- 使用指定 Docker 容器 `b2fd6726c8c1`。
- 使用指定虚拟环境 `/opt/gym_dssat_pdi/bin/python`。
- 保持统一 DQN 框架：
  - 9-action 离散动作表；
  - I in `{0, 15, 30}` mm；
  - N in `{0, 50, 100}` kg/ha；
  - 总灌溉预算 `I <= 120 mm`；
  - 总施氮预算 `N <= 300 kg/ha`；
  - 最小操作间隔 7 天；
  - 奖励仅用于评估：`max(0, GWAD_final - local_null_GWAD) - 1*I - 5*N`。

## 输入

LC/SY 文件夹中已有 MZX 及对应 WTH/SOL/CUL/MZA/MZT 文件。优先使用原 MZX 中已经存在的 treatment 年份：

- LC：2008、2009、2010、2011
- SY：2012、2014、2015

## 迁移模型

HLA2010：

- seed0 checkpoint 35000
- seed1 checkpoint 25000

YC2014：

- seed0 checkpoint 5000，高产 checkpoint
- seed0 checkpoint 25000，高 reward checkpoint
- seed1 checkpoint 30000，少水高产 checkpoint
- seed1 checkpoint 50000，高 reward checkpoint

## 输出

保存到：

`DSSAT_auto_validation/lc_sy_hl_yc_dqn_cross_site_transfer_probe_017_07`

至少包括：

- `017_07_lc_sy_cross_site_summary.csv`
- `017_07_lc_sy_cross_site_daily.csv`
- `017_07_lc_sy_cross_site_events.csv`
- `017_07_lc_sy_cross_site_status.csv`
- `figures/017_07_lc_sy_cross_site_summary.png`
- `docs/2026-07-06_017_07_lc_sy_hl_yc_dqn_cross_site_transfer_probe_record.md`

## 判定口径

本轮不要求模型“成功”，只做诊断：

- 如果模型无法加载/评估：记录具体错误，说明跨站点直接迁移存在结构不兼容。
- 如果模型能运行但产量不优或资源浪费：说明 direct policy transfer 不成立，下一步应改为“同一训练框架在 LC/SY 本地重新训练”。
- 如果模型能在 LC/SY 同时优于 null，并接近或超过 recorded/auto，同时资源投入更少：才认为存在直接跨站点策略迁移迹象。

