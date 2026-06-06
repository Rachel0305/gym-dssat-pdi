# Reward revision candidates

本文件只给候选方案，不覆盖当前 reward。

## Candidate A: strong input cost reward

```text
reward = crop_growth_reward - irrigation_cost * amir - nitrogen_cost * anfer - excessive_input_penalty
excessive_input_penalty = max(0, total_irrigation - irrigation_limit) * excess_irrigation_cost
                        + max(0, total_n - nitrogen_limit) * excess_n_cost
```

优点：过程成本清晰，可以直接抑制每天大水大肥。缺点：成本系数需要参数扫描。

## Candidate B: terminal yield plus process cost

```text
daily_reward = - daily_irrigation_cost - daily_nitrogen_cost
terminal_reward = final_yield_value - total_irrigation_cost - total_nitrogen_cost
```

优点：经济解释更强。缺点：terminal reward 稀疏，训练可能更慢，需要更稳定的 PPO 设置。
