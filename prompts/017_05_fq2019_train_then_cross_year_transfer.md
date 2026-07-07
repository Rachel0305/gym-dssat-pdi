# 017_05 FQ2019 单年训练与跨年份迁移

## 目的

017_04 确定性扫描表明，FQ2019 在当前 I≤120/N≤300 约束内存在明显上界空间：人工调度 `I90_mid_N0` 可达到约 8829 kg/ha，高于 DSSAT auto 和 FQ2016 模型迁移结果。  
因此本实验单独训练 FQ2019 DQN，并把 FQ2019 的 best checkpoint 迁移到封丘站其他可用年份。

## 执行原则

- 先小 smoke，再正式 50K；
- 使用指定 Docker 容器和指定虚拟环境；
- 不修改奖励函数、不修改 wrapper；
- 不覆盖旧结果；
- 保存模型、checkpoint evaluation、跨年迁移日值、管理事件、summary 和实验记录。

## 奖励函数与约束

奖励函数仍为 baseline-relative 版本：

`reward = max(0, final_GWAD - local_null_GWAD) - 1.0 * irrigation - 5.0 * nitrogen`

只在终止时给产量增益，过程中扣水氮成本。

约束：

- I≤120 mm
- N≤300 kg/ha
- 单日 I≤30 mm
- 单日 N≤100 kg/ha
- 最小操作间隔 7 天
- 动作窗口：DAP 1–120

## 训练年份

- FQ2019

## 迁移年份

使用 017_03 中可用的 FQ 年份集合，继续排除 FQ2007/FQ2008。

## 输出目录

`DSSAT_auto_validation/fq2019_baseline_relative_dqn_train_transfer_017_05/`

## 判读重点

1. FQ2019 训练后是否能接近 017_04 人工上界；
2. FQ2019 best checkpoint 是否优于 null / recorded / DSSAT auto；
3. FQ2019 模型迁移到其他年份后，是否比 FQ2016 模型更稳；
4. 是否出现类似 FQ2010 那样白用资源但不增产的年份。

