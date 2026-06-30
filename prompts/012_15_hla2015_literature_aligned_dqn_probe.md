# 012_15 HLA2015 文献对齐版 DQN 奖励/动作空间低成本探针

## 背景

前一轮 HLA2015 economic reward DQN 发现：

- 固定反事实扫描显示当前情景下合理方向是少施氮、适量灌溉；
- DQN seed1 能学到接近固定扫描最优的 I60/N0；
- seed0、seed2 不稳定；
- Double DQN 和 Dueling DQN 单独替换并没有解决问题。

因此本轮不继续盲目堆算法模块，而是参考文献 **A comparative study of deep reinforcement learning for crop production management** 中 DQN 对 mixed fertilization and irrigation task 的基本设定，做一次小规模探针。

## 目的

检验“文献式终端产量奖励 + 投入成本惩罚 + 更细离散动作空间”是否能比当前 `ΔGRNWT - input cost` 奖励更稳定地引导 DQN 学到灌溉/施氮策略。

## 设计

### 固定不变

- 站点年份：HLA 2015
- 使用当前已经确认的 IC=1 与更新品种参数输入链路
- 使用 PDI/gym-DSSAT 4.8.0
- 使用指定 Docker 容器 `b2fd6726c8c1`
- 使用指定虚拟环境 `/opt/gym_dssat_pdi/bin/python`
- 保持现有窗口和预算，用于避免与前面实验完全不可比：
  - 灌溉窗口：DAP 20–35, 45–65, 70–95
  - 施氮窗口：DAP 25–40, 55–70
  - I ≤ 120 mm
  - N ≤ 150 kg/ha
  - 最小操作间隔 7 天

### 新变化

参考文献 mixed task 的思想：

1. 奖励改为终端产量奖励 + 投入成本惩罚：

   ```text
   非终止步：R_t = - w2 * N_t - w3 * W_t
   终止步：R_T = w1 * GRNWT_final - w2 * N_T - w3 * W_T
   ```

   初始采用文献报告量级：

   ```text
   w1 = 0.158
   w2 = 0.79
   w3 = 1.10
   w4 = 0
   ```

   本轮暂不加入 nitrate leaching 项，因为当前 wrapper 中未稳定整理该变量，而且文献示例中 `w4=0`。

2. 动作空间从 4 个动作改为 5×5 离散水氮组合：

   ```text
   灌溉：0, 6, 12, 18, 24 mm
   施氮：0, 40, 80, 120, 160 kg/ha
   ```

   其中施氮 160 kg/ha 会被当前 N150 总预算裁剪到剩余预算内。

## 执行要求

1. 新建脚本，不覆盖已有 012_03–012_14 脚本和结果。
2. 先运行 200 timesteps smoke test，检查：
   - 容器和虚拟环境是否正常；
   - 动作映射是否正常；
   - 终端奖励是否能写入日志；
   - 结果文件、CSV、event summary 是否生成。
3. smoke test 通过后，再运行 HLA2015 seed0 5000 timesteps。
4. 保存：
   - daily CSV；
   - event_summary.json；
   - debug log；
   - pdi_tmp_snapshot_eval；
   - 简要中文实验记录。

## 判断标准

本轮不要求立刻证明 DQN 最优，只回答：

1. 文献式奖励是否能正常驱动训练和评估；
2. seed0 是否仍然退化成 null / 纯施肥 / 不灌溉；
3. 相比 012_03 的 economic reward，动作是否更接近合理灌溉；
4. 是否值得继续 seed1 或 HLA2010/2015 双年验证。
