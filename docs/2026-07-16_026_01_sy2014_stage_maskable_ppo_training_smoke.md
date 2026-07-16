# 026_01 SY2014 阶段型 MaskablePPO 最小训练 smoke 记录

## 1. 目的与边界

本任务在 026_00 已验证的六阶段环境上，检查 `MaskablePPO` 的安装、动作掩码、训练更新、模型保存/加载和确定性评估链路。它不是性能实验，不用于判断 PPO 是否已经获得成功农田管理策略。

## 2. 依赖安装

经用户明确批准，在指定 Docker 容器中执行：

```text
/opt/gym_dssat_pdi/bin/python -m pip install --no-cache-dir sb3-contrib==2.8.0
```

结果：

- `stable-baselines3==2.8.0`；
- `sb3-contrib==2.8.0`；
- `MaskablePPO` 可由指定 Python 正常导入；
- 因虚拟环境 site-packages 不可写，新增包实际安装到容器用户目录 `/home/gymusr/.local/`，没有修改或升级 SB3、Gymnasium、Torch 等已有核心包。

## 3. 预注册配置

- 环境：`StageDecisionEnv026`；
- 算法：`sb3_contrib.MaskablePPO`；
- seed：1；
- 25 维固定缩放观测；
- 六个阶段、9 个离散水氮动作及预算/晚期施氮 mask；
- `MlpPolicy`，网络 `[32, 32]`；
- learning rate `3e-4`；
- `n_steps=12`，`batch_size=12`，`n_epochs=2`；
- `gamma=1.0`，`gae_lambda=1.0`，`ent_coef=0.0`；
- 总训练步数 24，即 4 个完整训练季和 2 次 rollout/update；
- 训练前、训练后各确定性评估 1 季。

## 4. 执行异常记录

第一次启动时桌面命令等待上限误设为约 1 秒，命令在创建结果目录前终止。

第二次启动时桌面等待上限设为 10 秒，桌面工具返回超时，但容器内 Python 进程并未终止，仍在正常运行。随后通过容器进程号确认只存在这一份训练进程，没有启动第三份重复训练；等待该进程自然结束后读取结果。

这两次均属于外层启动/等待参数问题，没有修改科学配置，也没有被计作 PPO 阴性结果。

## 5. Smoke 结果

状态：`completed`；分支：`A_smoke_passed`。

全部预注册工程检查通过：

| 检查 | 结果 |
|---|---|
| SB3 / sb3-contrib 均为 2.8.0 | 通过 |
| 训练步数精确为 24 | 通过 |
| 完成 4 个训练季 | 通过 |
| 完成至少 2 次 rollout | 通过（2 次） |
| masked action 请求 | 0 |
| 每季阶段动作数 | 6 |
| 所有记录动作均在 mask 内 | 通过 |
| 评估数值和模型参数有限 | 通过 |
| 模型成功保存 | 通过 |
| 重新加载后确定性动作一致 | 通过 |

总 DSSAT 季节数为 6：训练前评估 1 季、训练 4 季、训练后评估 1 季。

### 确定性评估

| 时点 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | 季节奖励 |
|---|---:|---:|---:|---:|
| 训练前随机初始化 | 11211.36 | 60 | 200 | 6.3634 |
| 训练 24 步后 | 9779.27 | 75 | 100 | 3.7963 |

训练前的随机初始化网络碰巧选到一个高产、低投入序列；它不是训练所得策略，不能作为 PPO 成功结果。极短训练后确定性策略变差同样不能作为 PPO 失败结论，因为 24 步只够验证两次更新，远不足以评价学习收敛。

四个训练季的轨迹均合法，产量分别为 11193.48、11191.73、11237.38 和 10997.92 kg/ha；由于训练交互包含随机采样，这些轨迹也不等于训练后的确定性策略。

## 6. 结论

1. SY2014 阶段型 MaskablePPO 的端到端工程链路已经跑通。
2. 动作掩码在训练、评估中均生效，没有依赖执行层裁剪制造动作别名。
3. 026_01 没有回答“PPO 能否稳定全面超过 expert 和 auto”；该问题需要另立正式、预注册的训练曲线和跨 seed 实验。
4. 下一步不应根据训练前偶然高产或训练后短期下降现场调参，也不应直接扩展站点。

## 7. 输出文件

- `prompts/026_01_sy2014_stage_maskable_ppo_training_smoke.md`
- `src/smoke_sy2014_stage_maskable_ppo_train_026_01.py`
- `benchmark_results/026_01/026_01_result.json`
- `benchmark_results/026_01/026_01_episode_summary.csv`
- `benchmark_results/026_01/026_01_stage_actions.csv`
- `benchmark_results/026_01/026_01_model.zip`（本地 smoke 模型，不提交 Git）
