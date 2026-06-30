# 011_10 旧 output_hl 模型在当前 corrected HLA2010 环境中的复评估

## 目的

用户指出 `output_hl/irrigation` 和 `output_hl/fertilization/best_model.zip` 可能是以前“单独优化灌溉/施肥”时看起来比较合理的 PPO 模型，但不确定是否就是当时使用的模型。

本实验不重新训练，只做低成本复评估：

1. 读取旧模型 zip 元信息；
2. 判断是否能放入当前 corrected HLA2010 环境；
3. 若兼容，则在当前环境下评估其实际管理行为。

## 当前环境

- Docker 容器：`b2fd6726c8c1`
- Python：`/opt/gym_dssat_pdi/bin/python`
- 工作目录：`/workspace`
- 环境脚本：`src/run_hla_official_reward_restart_smoke.py`
- 当前 HLA2010 corrected 设置：
  - IC=1
  - 新校准品种参数
  - Jinja linked management 占位符已修正
  - mode=`all`
  - action names=`['amir', 'anfer']`
  - seasonal irrigation budget=120 mm
  - seasonal nitrogen budget=150 kg/ha

## 旧模型元信息

`output_hl/irrigation/best_model.zip`

- observation space: 24
- action space: 1
- last modified: 2026-04-23

`output_hl/fertilization/best_model.zip`

- observation space: 11
- action space: 1
- last modified: 2026-05-26

因此两者都不是当前 `all` 模式的 2-action 联合模型。

## 执行

旧灌溉模型的 24 维 observation 与当前 all-mode 环境一致，因此可以做适配评估：

- 旧模型 1D action -> 当前 `amir`
- 当前 `anfer` 固定为 -1.0，即安全 wrapper 后为 0 kg/ha N

命令：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/evaluate_old_hla_model_in_current_env.py --year 2010 --model output_hl/irrigation/best_model.zip --label old_output_hl_irrigation_best_model"
```

输出：

```text
DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/old_model_current_env_eval/2010/old_output_hl_irrigation_best_model
```

## 结果

事件汇总：

- irrigation total from MgmtEvent.OUT: 1.9 mm
- fertilizer total from MgmtEvent.OUT: 0 kg/ha
- irrigation events: 2
- fertilizer events: 0
- CSV safe irrigation total: 1.99 mm
- CSV safe nitrogen total: 0 kg/ha

非零灌溉发生在：

| DAP | irrigation |
| ---: | ---: |
| 10 | 0.95 mm |
| 17 | 1.04 mm |

## 解释

这个旧 `output_hl/irrigation/best_model.zip` 放到当前 corrected HLA2010 环境后，并没有表现出“正常灌溉”或“合理使用 I120 预算”，而是几乎不灌水。

因此：

1. 它很可能不是用户记忆中那个“合理灌溉模型”；
2. 或者它依赖旧环境设置，例如旧 IC、旧品种参数、旧模板、旧年份、旧 observation/reward/action 定义；
3. 不能把这个旧模型的历史表现直接外推到当前 corrected HLA2010 流程。

旧施肥模型 observation space 是 11，与当前 all-mode 24 维 observation 不兼容；试探 `mode=fertilization` 时环境卡住，因此本轮不强行复评估旧施肥模型，避免浪费算力和引入新的环境混杂。

## 结论

当前证据不支持“旧 output_hl 模型可以直接解决当前 corrected HLA2010 早期打满/训练行为问题”。

如果用户之后找到其他候选旧模型 zip，可以继续用同一个复评估入口检查；但必须记录该模型的 observation/action space、原始训练设置和当前评估环境是否一致。
