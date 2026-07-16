# 021_09 SY2014 延长探索衰减的单变量验证

## 依据

021_08离线审计显示，15K–20K坍缩窗口的请求正氮比例由61.6%降至45.5%，正灌溉比例基本不变；同期epsilon由0.186降至0.05、动作熵下降。5K–10K与15K–20K均存在大量Q排序漂移，因此不优先修改target interval。

## 目标

只将`exploration_fraction`从0.35改为0.70，使50K全局训练计划下epsilon在20K仍高于0.05，判断探索衰减是否与氮动作坍缩有关。

## 固定不变

- SY2014、IC=2、seed0；
- reward及其系数；
- 9动作空间、水氮预算、单次上限、7日间隔和DAP窗口；
- DQN学习率、buffer、batch、n-step、gamma、target interval等所有其他参数；
- DSSAT输入、品种、土壤和天气；
- 50K全局探索时间轴修复和checkpoint/resume协议。

## 分级执行

1. 新建独立YAML，不能覆盖021_05；
2. dry-run；
3. 只跑到5K smoke，内部总计划仍为50K；
4. smoke通过条件：
   - epsilon约0.864，而非旧配置的0.729或0.05；
   - `_total_timesteps=50000`；
   - runtime audit通过，IC=2输入保持不变；
   - 无OOM、无旧结果覆盖；
5. smoke通过后才允许另建正式配置跑到25K；本prompt不自动授权跨seed或跨站点训练。

## 正式对照判定（仅在后续25K执行时）

- 重点比较15K、20K、25K的确定性评估；
- 若20K/25K仍保持有意义氮投入和高产，支持“探索过早降到下限参与坍缩”；
- 若坍缩只是推迟，或仍在epsilon较高时出现，则削弱该机制；
- 不以单个高产尖峰判定成功；
- 不同时修改target interval、reward或观测空间。

## 记录

保存YAML、dry-run、smoke日志、checkpoint、日值/季节CSV和中文实验记录。未经用户后续确认，不执行25K长训练或Git push。
