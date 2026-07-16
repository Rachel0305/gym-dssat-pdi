# 022_19 SY2014 双因果pair+MC最终梯度兼容审计

## 1. 范围

最终组合仅含`MC + DAP65因果pair + DAP110因果pair`。不加入软锚定/蒸馏项，避免重新引入没有受控真值和固定权重来源的第四项。

两组pair均采用组内mean SmoothL1。权重规定每组因果梯度获得MC共享层梯度范数的一半，随后取三个seed推导值中位数作为统一配置。

## 2. 固定权重

```text
lambda65  = 0.003002436945892297
lambda110 = 0.06764715488665898
```

DAP65与DAP110共享层梯度余弦分别为0.346、0.165、0.224，三个seed均为正，不存在预注册定义下的直接方向冲突。

## 3. 虚拟步结果

| Seed | DAP65 loss | DAP110 loss | MC相对变化 | 兼容 |
|---:|---|---|---:|---|
| 0 | 0.149324→0.149278 | 0.001988→0.001986 | +0.0000075% | 是 |
| 1 | 0.349565→0.349473 | 0.002985→0.002984 | -0.0000516% | 是 |
| 2 | 0.214407→0.214337 | 0.000519→0.000515 | -0.0000145% | 是 |

## 4. 判定

**A分支：3/3 seed兼容。** 允许022_20唯一一次固定配置离线训练。该结果只说明局部梯度兼容，不保证多步训练成功。

DQN训练0步，DSSAT调用0次，未保存虚拟模型。当前未执行Git commit或push。

## 5. 文件

- `prompts/022_19_sy2014_dual_causal_pair_mc_gradient_audit.md`
- `src/audit_sy2014_dual_causal_pair_mc_gradient_022_19.py`
- `benchmark_results/022_19/022_19_dap65_causal_targets.csv`
- `benchmark_results/022_19/022_19_dap110_causal_targets.csv`
- `benchmark_results/022_19/022_19_seed_gradient_ratios.csv`
- `benchmark_results/022_19/022_19_fixed_weight_virtual_step_audit.csv`
- `benchmark_results/022_19/022_19_result.json`

