# PPO / RL crop-management algorithm improvement literature table

目的：为后续“不要继续小修小补 PPO 参数，而是有依据地改算法/建模方式”整理文献证据。

| 文献标题 | 文献主要内容 | 文献重点段落位置 | 跟我们的关系 | 对我们的启发 |
|---|---|---|---|---|
| [A Comparative Study of Deep Reinforcement Learning for Crop Production Management](https://arxiv.org/abs/2411.04106) | 在 gym-DSSAT 中比较 PPO 和 DQN，任务包括 fertilization、irrigation、mixed management；结果显示 PPO 在单独施肥/灌溉任务较强，但 DQN 在 mixed management 中更强。 | Abstract；Numerical experiments / mixed management 结果部分。 | 直接对应我们的问题：单独水/氮可能好，但水氮联合后 PPO 表现不一定好。 | 不要默认 PPO 最适合水氮联合；可以考虑 DQN/离散动作/混合动作结构作为算法替代。 |
| [Optimizing Crop Management with Reinforcement Learning and Imitation Learning](https://www.ijcai.org/proceedings/2023/0691.pdf) | 用 DSSAT 同时优化氮肥和灌溉；先训练 full-observation RL 策略，再用 imitation learning 学 partial-observation 可部署策略。 | Abstract；Introduction；Method 中 RL + IL 框架；full observation / partial observation 说明。 | 和我们的研究目标最接近：DSSAT、水氮联合、RL、可部署策略。 | 可以从“PPO 从零学”改成“专家/规则策略预训练或模仿学习 + PPO 微调”，减少早期乱操作和不合理施氮。 |
| [Nitrogen management with reinforcement learning and crop growth models](https://www.cambridge.org/core/journals/environmental-data-science/article/nitrogen-management-with-reinforcement-learning-and-crop-growth-models/358749FAFAA4990B1448DAB7F48D641C) | 提出 CropGym，用作物模型训练氮肥管理 RL agent；目标是在产量和环境影响之间取得平衡，并与专家标准实践比较。 | Impact Statement；Methods；Results 中 RL 策略与标准实践比较。 | 对应我们 2015 年“不缺氮但 PPO 仍施满”的问题。 | 氮管理必须显式考虑边际收益和环境/成本；不能只看产量，否则 RL 可能学成过量施氮。 |
| [Deep reinforcement learning for irrigation scheduling using high-dimensional sensor feedback](https://journals.plos.org/water/article?id=10.1371/journal.pwat.0000169) | 用深度强化学习做灌溉调度；状态包括生育期、LAI、土壤水、累计降雨/灌溉；动作是 0/10/20/30/40 mm 五个离散灌溉量。 | Abstract；Methods 中 state/action design；Fig. 1 网络结构和动作定义。 | 对应我们连续动作容易“窗口内用满”的问题。 | 可以把动作空间离散化，或拆成“是否操作 + 操作量”，让 PPO 更容易学会“不操作”。 |
| [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528) | 提出 constrained RL：同时定义 reward 和约束，而不是把所有目标都塞进一个 reward 权重里。 | Abstract；Introduction；Problem formulation / constraints 部分。 | 对应我们 reward 权重难定、产量奖励和水氮成本难平衡的问题。 | 可把目标改成“最大化产量/收益，同时约束水、氮、淋失或成本”，而不是手工调 `产量系数/水氮惩罚`。 |
| [Penalized Proximal Policy Optimization for Safe Reinforcement Learning](https://www.ijcai.org/proceedings/2022/0520.pdf) | 在 PPO 框架下处理安全/约束问题，用 penalty 方法处理 cost constraints。 | Abstract；Introduction；Method 中 constrained / penalized PPO 部分。 | 比 CPO 更接近我们当前使用的 PPO 框架。 | 可以考虑 project-local `CustomPPO` 或 PPO-Lagrangian/P3O 思路，把水氮用量作为 cost，而不是继续手动调 reward。 |

## 简短结论

当前结果不建议继续靠小幅调 PPO 超参数或 reward 系数抢救。

更有文献依据的三条路线：

1. **模仿学习 / 专家预训练 + PPO 微调**  
   先让策略学会专家或规则管理的基本节奏，再用 PPO 优化。

2. **离散或混合动作 PPO**  
   把动作改成离散水氮量，或拆成“是否操作”和“操作多少”，避免连续动作在窗口内直接用满预算。

3. **Constrained PPO / PPO-Lagrangian / P3O**  
   把水、氮、环境损失设成约束或 cost，让算法处理权衡，而不是手工猜 reward 权重。

## 对当前项目的建议

优先级建议：

1. 先做文献汇报，说明“PPO 在 mixed water-nitrogen management 中可能不稳定”有文献支持。
2. 如果导师要求继续做 PPO 最优，优先尝试“专家预训练/模仿学习 + PPO 微调”或“离散动作 PPO”。
3. 如果导师更关心资源约束和成本合理性，再考虑 constrained PPO / PPO-Lagrangian。

不要把当前 windowed PPO 继续包装成最终最优策略；它目前更适合作为“加农业操作约束后，PPO 行为从病态变得可解释，但仍未解决氮素效率”的阶段性结果。
