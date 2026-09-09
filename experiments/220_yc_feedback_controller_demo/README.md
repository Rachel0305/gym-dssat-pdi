# 220 YC daily state-feedback controller demo

这是一个单站点、单年份 proof-of-concept demo，不是正式论文实验。编号 `220` 是当前仓库中未占用的实验编号；本目录不覆盖 04X 或更早实验。

## 固定范围

- site-year: `YC / YCA / 2019`，lowIC 输入副本。
- controller: 外部逐日 state-feedback；水、氮两个 controller 独立运行，不耦合。
- 固定 basal: DAP1 外部固定施氮 `96 kg N/ha`；不计入 controller N budget，但计入最终 `total_nitrogen`。
- fixed safety window: `DAP 7--90`，只用于避免播种当天和成熟后触发，不是专家日历节点。
- objectives: maximize grain yield; minimize total N; minimize total irrigation。
- constraint: `grain_yield >= 0.97 * official_extension_expert_simulated_yield`，参照值来自同一 DSSAT/gym-DSSAT 环境，不是田间实测产量。
- optimizer: explicit type-aware mixed-variable NSGA-II，population `32`、generations `20`、optimizer seed `1`；不安装 pymoo，不做多 seed，不做 optional 40/30 confirmation。
- irrigation dose set includes `50 mm`: it is anchored to the audited expert maximum single irrigation of about `48.75 mm` and modestly expanded; the DSSAT action upper bound is `50 mm`, so a 60 mm diagnostic was rejected by action-clipping closure.

## 运行顺序

1. `python experiments/220_yc_feedback_controller_demo/src/run_demo.py --mode parameter-audit`
2. `python experiments/220_yc_feedback_controller_demo/src/run_demo.py --mode unit`
3. 在 `nifty_taussig` 容器内运行 `--mode smoke`，确认外部动作和 DSSAT 事件闭合。
4. smoke 通过后，在同一容器内运行 `--mode nsga`。

完整结果、候选日志、DSSAT 快照和图表写入本目录下的 `results/` 与 `logs/`；原始 `.MZX/.WTH/.SOL/.CUL` 不修改。
