# 022_18 SY2014 保守参数路径可行性审计

## 1. 目的

022_17证明DAP65局部排序需要纠正，同时8个非别名DAP110状态均应选择action0。正式设计锚定/蒸馏loss之前，本任务利用022_15已有update0与update500 checkpoint，在二者线性参数路径上检查是否存在同时满足两类因果目标的区域。

本任务不训练DQN、不调用DSSAT、不保存插值checkpoint。

## 2. 固定方法

对三个seed分别计算：

```text
theta(alpha)=(1-alpha)*theta0+alpha*theta500
alpha=0.000,0.001,...,1.000
```

每个alpha检查：

1. 三个受控DAP65状态全部满足`Q(a1)>Q(a7)`；
2. 022_17确认的8个非别名DAP110状态全部选择action0。

同时记录MC训练loss及DAP1/DAP85未污染测试状态相对起点的argmax变化，但不把未经因果验证的所有状态强制冻结。

## 3. 结果

| Seed | 首次全部纠正DAP65的alpha | DAP110 8/8保护区间 | 同时可行区间 |
|---:|---:|---|---|
| 0 | 0.181 | 从未达到8/8 | 无 |
| 1 | 0.507 | 从未达到8/8 | 无 |
| 2 | 0.393 | 0.000–0.072 | 无 |

### 3.1 Seed0

旧checkpoint在alpha=0时只对8个DAP110状态中的6个选择action0，本身已包含2个受控证实的错误。DAP65在alpha=0.181后全部纠正，但DAP110从未达到8/8。

### 3.2 Seed1

旧checkpoint起点只有5/8个DAP110状态正确；路径中最多提高到6/8，但从未达到8/8。DAP65直到alpha=0.507才全部纠正。

### 3.3 Seed2

旧checkpoint起点为8/8正确，但该保护仅维持到alpha=0.072；DAP65需要到alpha=0.393才全部纠正，两区间不相交。

## 4. 判定

**C分支：0/3 seed存在可行线性参数区间。**

可以确认：

- 沿现有update0→500训练方向，简单早停或选择中间插值点不能同时解决DAP65与已验证DAP110目标；
- “贴住旧网络、其余不动”不是充分方案，因为seed0/1旧网络本身就在若干DAP110状态中选择了已证实较差的action1；
- 对seed2，即使旧网络DAP110全对，DAP110保护也在DAP65修复完成前先被破坏；
- 因此新的保守机制不能只是无差别复制旧Q输出，必须区分“旧网络正确部分”和“已有因果证据证明需要修正的旧错误”。

不能确认：

- 线性参数插值无可行区间不代表整个神经网络参数空间不存在可行解；
- 不能据此断言锚定、蒸馏或多目标监督一定失败；
- 不能把插值模型当作训练checkpoint或论文结果。

## 5. 对下一步的约束

Claude建议的“整体锚定旧网络”需要修正：当前证据不支持把旧网络所有决策都当教师真值。若继续，应采用**性能感知的受控约束集**：

- DAP65三个状态明确监督action1优于action7；
- DAP110八个非别名状态明确监督action0优于action1；
- 其他未经受控验证的状态只可做有限的软稳定性约束，不能冒充正确标签。

在正式训练前仍需另立一次纯离线梯度/可行性审计，预注册多组因果pairwise目标如何与MC loss组合及权重来源。不得直接将022_13 warm-start启动，也不得现场追加lambda扫描。

## 6. 输出

- `prompts/022_18_sy2014_conservative_parameter_path_feasibility_audit.md`
- `src/audit_sy2014_conservative_parameter_path_feasibility_022_18.py`
- `benchmark_results/022_18/022_18_parameter_path_audit.csv`
- `benchmark_results/022_18/022_18_seed_feasible_intervals.csv`
- `benchmark_results/022_18/022_18_result.json`
- `benchmark_results/022_18/022_18_conservative_path_feasibility.png/.svg`

当前未执行Git commit或push；022_13 warm-start仍暂停。
