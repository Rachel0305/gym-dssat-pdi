# 021_06 SY2014 离线 Q 排序与动作混叠审计

## 状态

- 新训练：**未进行**
- 15K/20K 确定性状态捕获：**completed**
- 5K–25K 固定状态 Q 值审计：**completed**
- wrapper 动作混叠审计：**completed**
- 因果根因确认：**尚未完成**

## 目的

021_05 修复探索率时间轴后，SY2014 seed0 在 15K 仍保持 10746 kg/ha、I120/N300，但 20K 变为 5711 kg/ha、I120/N0。本实验不训练、不改 reward、不改 IC，只判断：

1. 15K→20K 是否发生单步动作 Q 排序反转；
2. 预算、共享 7 日操作间隔和 DAP1–120 窗口是否造成请求动作与执行动作混叠；
3. 两种现象能否解释氮动作消失。

## 方法

### 固定状态

分别重放 15K 和 20K checkpoint 的确定性轨迹，从每条轨迹选取 DAP 1、8、15、22、29、36、50、80、121，共18个真实 pre-step 状态。保存原始25维观测、累计/剩余水氮预算、上次操作 DAP 和间隔许可状态。

同一个固定观测分别输入5K、10K、15K、20K、25K的online Q和target Q网络，计算9个动作的完整Q值和排名，因此checkpoint间比较没有混入状态变化。

### 动作定义

9个动作是单步笛卡尔积：`I∈{0,15,30}` mm × `N∈{0,50,100}` kg/ha。I120/N300是全季累计量，不是单步Q动作。

### 排序反转

在相同灌溉档位内定义：

`M = max(Q(N50), Q(N100)) - Q(N0)`。

15K与20K的M符号相反记为rank flip；若两端绝对边际都不小于各自9动作Q标准差的0.25倍，记为robust rank flip。0.25是预注册的描述性阈值，不是文献阈值。

### 动作混叠

将每个固定状态下的9个请求动作逐一通过正式wrapper相同的预算、窗口、单次上限和共享7日间隔规则，统计唯一执行动作数。脚本还把参考模型实际执行动作与规则重建结果逐步核对；全部一致，否则会直接报错停止。

## 结果一：online Q发生广泛排序变化

| 指标 | 结果 |
|---|---:|
| 固定状态数 | 18 |
| online rank flip | 28/54 |
| online robust rank flip | 19/54 |
| 无动作混叠状态中的online rank flip | 14 |
| 无动作混叠状态中的robust flip | 11 |
| target rank flip | 0/54 |
| online全局argmax改变 | 15/18状态 |
| 无动作混叠状态中的argmax改变 | 9 |

最直接的证据来自DAP1初始状态：此时预算完整、窗口开放、没有7日间隔限制，9个动作均可原样执行。

| 网络checkpoint | online argmax | 请求水 | 请求氮 |
|---:|---:|---:|---:|
| 15K | 动作8 | 30 mm | 100 kg/ha |
| 20K | 动作0 | 0 mm | 0 kg/ha |

因此，在完全没有wrapper裁剪的同一个初始状态上，online网络已经从高水高氮动作切换到no-op。wrapper动作混叠不能单独解释20K的氮动作消失。

## 结果二：15K与20K的target Q保持不变

对18个固定状态×9个动作比较，15K与20K checkpoint 的target Q最大绝对差为0，target氮偏好rank flip为0；同期online Q出现28次rank flip。

这把15K–20K观察到的动作排序变化定位到online网络一侧，但不能仅凭此结果宣称“Q过估计”或确定target update是因果根因。它只说明：在这一训练区间内，online网络发生了显著策略漂移，而target网络没有同步变化。

## 结果三：动作混叠真实存在且具有状态依赖

18个固定状态中：

- 9个状态出现动作混叠；
- 8个状态中，9个请求动作全部变成同一个执行动作；
- 15K轨迹DAP29时，水预算耗尽、氮仅余50，9个请求动作只对应2个执行动作；
- 15K轨迹DAP36以后预算耗尽，全部请求动作变为I0/N0；
- 20K轨迹DAP36、50、80处受共享7日间隔限制，全部请求动作变为I0/N0；
- DAP121在操作窗口外，全部请求动作变为I0/N0。

当前观测配置含`dap`和`totir`，其中`totir`可作为累计灌溉的代理；但不显式包含累计施氮量、剩余氮预算或`last_operation_dap`。因此agent不能完整观察决定动作裁剪结果的wrapper内部状态，存在部分可观测性/执行动作混叠风险。

## 综合判断

021_06排除了“氮坍缩完全由wrapper裁剪造成”这一单一解释：

1. online Q排序漂移真实存在；
2. 动作混叠也真实存在；
3. DAP1无混叠状态已出现动作8→动作0的argmax翻转，因此Q网络漂移是必要关注对象；
4. 后期预算耗尽和7日间隔又进一步造成大量动作混叠，可能污染replay buffer中的动作—结果对应关系；
5. 当前最安全的表述是“online Q漂移与部分可观测的动作混叠共同构成候选机制”，不能写成已确认因果。

## 对下一步的约束

暂不修改reward系数。021_07若继续，应保持单变量和低成本：

1. 先审计15K–20K之间target network实际更新时间、TD error和训练loss；
2. 检查target Q在该区间完全不变是否与`target_update_interval=10000`及checkpoint边界严格对应；
3. 单独设计“将剩余氮预算和距上次操作天数加入观测”的环境结构对照前，先确认这属于Markov状态修复，而不是为了结果调参；
4. 不得同时修改观测、target interval、reward scaling或成本系数。

## 输出文件

- `prompts/021_06_sy2014_offline_q_action_alias_audit.md`
- `src/audit_sy2014_q_action_alias_021_06.py`
- `src/finalize_sy2014_q_action_alias_021_06.py`
- `benchmark_results/021_06/021_06_fixed_state_manifest.csv`
- `benchmark_results/021_06/021_06_q_values_long.csv`
- `benchmark_results/021_06/021_06_nitrogen_preference_margins.csv`
- `benchmark_results/021_06/021_06_rank_flip_summary.csv`
- `benchmark_results/021_06/021_06_action_alias_long.csv`
- `benchmark_results/021_06/021_06_action_alias_summary.csv`
- `benchmark_results/021_06/021_06_online_argmax_by_fixed_state.csv`
- `benchmark_results/021_06/021_06_audit_summary.json`
- `benchmark_results/021_06/021_06_decisive_evidence_summary.json`
- `benchmark_results/021_06/021_06_q_margin_and_action_alias.png`

## Git

- 本地提交：**blocked**。当前运行环境只允许读取 `.git`，创建 `.git/index.lock` 时返回 `Permission denied`。
- 所有021_06文件已保存在项目目录，未丢失；恢复Git写权限后只需按清单选择性提交。
- push：未执行；未经用户明确确认不push。
