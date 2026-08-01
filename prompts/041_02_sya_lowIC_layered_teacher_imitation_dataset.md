# 041_02 SYA lowIC 分层 teacher imitation 数据集

## 背景

041_00 和 041_01 已经在 lowIC 条件下重新生成了 teacher 候选轨迹。结果显示：

- 2014、2016、2018、2020、2021 有三项全超 teacher；
- 2015、2017、2019、2022、2023 暂时只有 near-miss teacher，即两项超过或一项明显接近。

当前目标不是继续搜索，也不是训练 PPO，而是把已有 teacher 候选整理成后续 imitation warm-start 能直接使用的数据集。

## 目的

生成一个分层 teacher 数据集：

- strong teacher：三项全超四情景最高值；
- near-miss teacher：没有三项全超，但在当前候选中最接近三项全超。

后续 041_03 可先用该数据集做 policy warm-start，再进入自由时序 PPO fine-tune。

## 选择规则

每个年份只选 1 条 teacher 轨迹。

### strong teacher 选择

若该年份存在三项全超候选，则只在三项全超候选中选择。排序规则：

1. 三项全部超过；
2. 产量 gap 更大；
3. WP_ET gap 更大；
4. PFP_N gap 更大；
5. 灌溉量更低；
6. 施氮量更低。

### near-miss teacher 选择

若该年份没有三项全超候选，则选择最接近三项全超的候选。排序规则：

1. 超过指标数更多；
2. 归一化缺口更小：
   - 产量缺口按 1000 kg/ha 缩放；
   - WP_ET 缺口按 0.1 kg/m3 缩放；
   - PFP_N 缺口按 10 kg/kg 缩放；
3. 产量 gap 更大；
4. WP_ET gap 更大；
5. PFP_N gap 更大；
6. 灌溉量更低；
7. 施氮量更低。

near-miss 不是“三项全优真值”，只能作为较弱 teacher。

## 样本权重

为避免 near-miss 轨迹被当作强真值，同时避免 no-op 天数淹没动作天数，预注册样本权重：

- strong teacher 非零动作日：1.0
- strong teacher no-op 日：0.2
- near-miss teacher 非零动作日：0.5
- near-miss teacher no-op 日：0.1

## 输出

输出目录：

`benchmark_results/041_02_sya_lowIC_layered_teacher_imitation_dataset/`

主要文件：

- `tables/041_02_selected_teacher_trajectories.csv`
- `tables/041_02_imitation_daily_dataset.csv`
- `tables/041_02_imitation_action_days.csv`
- `tables/041_02_action_distribution.csv`
- `041_02_result.json`
- `docs/041_02_sya_lowIC_layered_teacher_imitation_dataset_record.md`

## 停止线

- 如果 041_00 结果表缺失，停止；
- 如果某个年份选不出任何 teacher/near-miss，停止；
- 如果选中的轨迹存在 mask 强制 no-op，停止；
- 如果选中的轨迹缺少 daily_values.csv，停止；
- 不训练 PPO；
- 不根据输出结果现场改权重。

