# 037_08 FQA 验证十年 PPO 五情景图与指标汇总

## 目的

为组会汇报快速整理 FQ/FQA 验证年份 2014–2023 的现有 PPO 结果：

1. 每个验证年份输出一张五情景日过程对照图；
2. 输出产量、WP_ET、PFP_N 三个指标相对四基线最高值的汇总图；
3. 明确记录数据来源、checkpoint 选择和当前口径限制。

## 固定口径

- 站点：FQ/FQA。
- 年份：2014–2023，共 10 年。
- PPO 候选：使用 036_04 预先选定的 FQA checkpoint，不重新挑选结果。
  - 当前选定 checkpoint：036_01 run 的 seed0, checkpoint_step=25000。
- 四基线：使用 037_07 修正后的静态 level=1 四基线结果。
  - null
  - recorded_farmer_template
  - official_extension_expert
  - dssat_auto

## 图表输出

### 日过程图

每年一张，包含：

- 天气：降雨、Tmax、Tmin；
- 土壤水：SWTD。注意 PPO daily CSV 未保存 SWTD，因此该面板仅展示四基线；
- 水分胁迫指数：WSPD/SWFAC；
- 氮胁迫指数：NSTD/NSTRES；
- 灌溉事件；
- 施氮事件；
- 籽粒/生物量轨迹；
- 统一累计奖励：`ΔGRNWT - 1 × irrigation - 5 × nitrogen`。

### 汇总图

输出三张：

- 产量：PPO vs 四基线最高值；
- WP_ET：PPO vs 四基线最高值；
- PFP_N：PPO vs 四基线最高值；

同时导出 year-level CSV，用于复核。

## 注意事项

- FQ2018 四基线和 PPO 均为零产异常年，不作为正常农学解释，但保留在图表中并在记录中标注。
- 本任务不训练、不重放 DSSAT，只读取已有结果并绘图。
- 若缺少某列，不猜测、不补造，直接在记录中声明。
