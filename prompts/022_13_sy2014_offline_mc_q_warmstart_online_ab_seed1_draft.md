# 022_13（待定草案）SY2014 离线 MC-Q warm-start 短在线 A/B seed1

> 状态：未执行。022_09已确认seed1关键施氮排序仍错；022_12离线梯度冲突审计及后续一次性离线排序训练完成前不得启动本草案。

## 1. 依据

022_08 在严格场景留出条件下证明离线 MC-Q 学习可行：seed0/1 全部预注册条件通过，seed2仅阶段聚合相关略低于阈值。现在检验离线学到的排序能否在短在线训练中保持。

## 2. A/B唯一变量

- Control：seed1随机初始化 Q 网络；
- Treatment：加载 `benchmark_results/022_08/checkpoints/offline_mc_q_seed1.pt`；
- 唯一差异为网络初始权重。

两个arm共同冻结：

- SY2014、IC=2、相同DSSAT输入；
- 022_03全部288条fixed-grid replay预填充；
- 022_08最小支持动作集合，DAP85/action5不得选择但其singleton训练样本仍保留在replay；
- 25→64→64→9、SmoothL1、Adam 1e-4；
- terminal-complete MC target/1000；
- replay capacity2000、batch32；第5季起每季6次更新；
- seed1；在线15季；epsilon沿原60季日程的前15季，不压缩衰减；
- checkpoint=0/5/10/15；
- I120/N300预算与裁剪不变；不加入PER/demo/TD/target network/reward修改。

每个arm独立重置相同Python/NumPy/Torch seed和在线RNG。不得复用022_05作为control，因为022_05的DAP85支持集包含singleton action5，不满足“只改初始化”的要求。

## 3. 训练前检查

1. 022_08结果为A且seed1全部条件通过；
2. warm checkpoint结构与当前Q网络逐层一致；
3. 两arm replay均从相同288条数据开始；
4. 两arm可选动作集合完全相同；
5. 冻结初始策略先独立确定性评估并保存；
6. 所有在线动作均不得越出最小支持集。

## 4. 判定

- **A warm-start有效且保持**：Treatment至少3/4 checkpoint通过主判据、season15通过、至少1个严格成功；且Treatment通过数严格多于Control。
- **B 有正信号但不足**：Treatment至少1个checkpoint通过主判据，或在产量不低于Control 99%的情况下显著改善WP/PFP或水氮投入，但不满足A；停止自动扩展。
- **C 无收益/退化**：Treatment 0/4通过且没有上述效率改善，或明显差于Control；停止warm-start分支。
- **D 实现失败**：任一冻结检查、动作支持、数据或输出验证失败。

“显著改善”在本任务固定为：相对同checkpoint Control，灌溉至少少15mm或施氮至少少50kg/ha，且产量≥Control的99%。不得事后改变。

## 5. 输出

- precheck JSON；
- 两arm checkpoint、训练季、stage action、update log；
- checkpoint汇总、逐日值、A/B差值表；
- PNG/SVG；
- 结果JSON和中文记录。
