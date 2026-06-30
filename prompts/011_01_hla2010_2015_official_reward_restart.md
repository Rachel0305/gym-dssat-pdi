# 011_01 HLA 2010/2015 official reward PPO restart

## 背景

旧 PPO 主线存在多重混杂因素：IC=0、2004 极端干旱、旧品种参数、管理模式和动作通道未完全统一。旧结果保留为参考，不覆盖、不删除。

现在重新开一条干净主线：

- 站点：HLA
- 候选年份：2010、2015
- 初始条件：IC=1
- 品种参数：用户最新校准 HY0006
- 奖励函数：从 `gym_dssat_pdi` 官方原始 reward 开始
- 原则：先做低成本 smoke test，不直接长训练

## 官方 reward 来源

项目内副本：

`references/rewards.py`

核心形式：

```python
fertilization_reward = trnu * coef - penality * anfer
irrigation_reward = next_topwt - previous_topwt - 15 * amir
all_reward = [fertilization_reward, irrigation_reward]
```

本线使用项目内 monkeypatch 方式在训练进程中加载该文件，不修改 Docker/site-packages。

注意：官方 `all_reward` 返回两个子奖励组成的 list，而 Stable-Baselines3 PPO 需要标量 reward。因此 smoke 脚本临时使用：

```python
R = sum(all_reward)
```

这等价于把官方 fertilization reward 和 irrigation reward 相加，不引入新的产量终止项或胁迫惩罚。

## 本轮目标

优先级按顺序执行：

1. 确认官方 reward 可以在 SB3 PPO 进程中被加载并标量化。
2. 确认 gym/PDI 动作通道能让灌溉和施肥真正进入 DSSAT 生长模拟。
3. 只有动作通道验证通过后，才允许做极短 PPO training smoke。
4. 如果 smoke 通过，再考虑 5k steps。

## 管理通道设定

PPO 训练输入使用目标年份 null 模板，但打开 reported 管理通道：

- `IRRIG=R`
- `FERTI=R`
- `MI=1`
- `MF=1`

初始灌溉/施肥行保留为 0，占位用于 PDI/gym-DSSAT 写入动作。

## 011_01 当前执行结果与修复

第一次执行 2010 年动作通道 smoke 时，Python 层动作确实被发出：

- DAP 1：`anfer=165`
- DAP 49、70、95：`amir=10`

但 PDI/DSSAT 输出没有响应：

- `MgmtEvent.OUT` 中灌溉事件数：0
- `MgmtEvent.OUT` 中施肥事件数：0
- 最终 `GWAD=6956 kg/ha`，与 2010 null 情景一致

原因是 HLA 静态 MZX 没有官方 Jinja 占位符，`mode='all'` 无法自动渲染为 `IRRIG=L, FERTI=L`。

已按官方 `my_data/UFGA8201-HL.jinja2` 写法，在复制后的 input MZX 中插入：

```text
{{ wther }}
{{ plant }}
{{ irrig }}
{{ ferti }}
```

修复后重新执行 action smoke，`pdi_tmp_snapshot/fileX.MZX` 正确渲染为：

```text
1 ME              M     M     E     R     S     L     R     1     G     R     2
1 MA              R     L     L     R     M
```

`MgmtEvent.OUT` 中成功出现：

- 施氮 165 kg N/ha
- 灌溉 10 + 10 + 10 mm

最终结果：

| 指标 | 修复后 |
| --- | ---: |
| irrigation_events_mgmtevent | 3 |
| fertilizer_events_mgmtevent | 1 |
| irrigation_total_mgmtevent | 30 |
| fertilizer_total_mgmtevent | 165 |
| final_gwad | 7679 |
| final_cwad | 20665 |

因此当前结论是：

> 官方 reward 可以加载；插入 Jinja 占位符后，action channel 已通过验证。下一步可以做极短 PPO training smoke，但仍不能直接长训练。

## 安全规则

- 不做长训练。
- action channel 未通过前，不运行 PPO training smoke。
- 当前 2010 action channel 已通过，但正式训练前仍需先做短 PPO smoke。
- 不修改 Docker/site-packages。
- 不覆盖旧实验。
- 所有输出保存到新目录。

## 输出目录

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart`

已经保存：

- input 文件
- action smoke CSV
- `MgmtEvent.OUT` 快照
- `PlantGro.OUT` 快照
- `event_summary.json`
- `README.md`
