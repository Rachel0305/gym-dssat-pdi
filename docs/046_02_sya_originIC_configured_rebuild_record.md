# 046_02 SYA originIC 配置化重建记录

## 状态

尚未运行。

## 为什么新开这一轮

042_15 的 lowIC 五情景表中，recorded farmer template 在多数年份的籽粒产量偏低，并且实际管理为 0 mm 灌溉、586 kg/ha 施氮；这不适合作为不加说明的“真实 recorded farmer”比较项。

本轮固定 PPO 框架，仅切换到 originIC，以隔离“初始土壤条件”对四情景和 PPO 的影响。

## 使用方法

配置只在以下一个文件修改：

`configs/046_02_sya_originIC_binary_timing_ppo.json`

执行顺序：

```bash
cd /workspace/src
/opt/gym_dssat_pdi/bin/python run_sya_baselines_configured_046_03.py --config ../configs/046_02_sya_originIC_binary_timing_ppo.json --dry-run
/opt/gym_dssat_pdi/bin/python run_sya_baselines_configured_046_03.py --config ../configs/046_02_sya_originIC_binary_timing_ppo.json
/opt/gym_dssat_pdi/bin/python run_sya_ppo_configured_046_02.py --config ../configs/046_02_sya_originIC_binary_timing_ppo.json --dry-run
/opt/gym_dssat_pdi/bin/python run_sya_ppo_configured_046_02.py --config ../configs/046_02_sya_originIC_binary_timing_ppo.json
```

训练结束后才运行报告脚本。当前正式配置为 100K 训练、25K/50K/75K/100K 四个 checkpoint；2K smoke 使用 `--smoke`，输出到独立目录。
