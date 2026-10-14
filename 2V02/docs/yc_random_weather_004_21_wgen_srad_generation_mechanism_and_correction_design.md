# YC WGEN SRAD生成机制审计与多变量修正设计（004_21）

## 结论

```text
runtime_srad_mechanism_status = PARTIALLY_VERIFIED
low_tail_loss_mechanism = TRUNCATION_OR_BOUND
recommended_correction_family = RUNTIME_PARAMETER_CHANGE
weather_only_pilot_ready = YES
```

`YES`表示可以进入**有前置源码核验门槛的 weather-only pilot 设计**，不表示现在已修改 runtime，也不表示本轮生成了天气。必须先取得与已安装 `4.8.0.24+2~a4ae650` 对应的准确源码并核验 `DailyWeatherCheck`；若无法闭合则 pilot 不得生成候选天气。

## 本轮边界与输入核验

本轮仅做源码、二进制旁证、冻结输入与已有诊断的只读审计。未生成天气，未运行 WGEN、DSSAT 作物模拟或 PPO，未修改 CLI、旧天气归档、canonical 配置，也未 commit/push。

执行中曾尝试一次 runtime 版本探测；`run_dssat` 因当前目录缺少其初始化所需的 paraconf 文件而立即报错退出（`Error in paraconf: No such file or directory`），未启动天气生成或作物模拟。该探测不作为任何科学证据。

只读 runner 验证了 frozen fitting CSV（3653行，SHA256 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`）与 `CNYC.CLI`（SHA256 `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`）。按 `RAIN>0` 为 wet day、`RAIN<=0` 为 dry day，以样本标准差重算每月 dry/wet SRAD mean/SD、全月均值；12个月共60个参数值均与 CLI 一位小数序列化相符。参数明细见 `srad_cli_parameter_audit.csv`。因此当前证据不支持“CLI参数拟合/序列化错误”作为低尾消失原因。

其中，`SDMN/SDSD`是dry-day SRAD均值/标准差，`SWMN/SWSD`是wet-day SRAD均值/标准差，进入 `WGENIN` 的逐日参数向量；`SAMN`是CLI月平均值，与fitting全日月均值核对，但不在该14项逐日WGEN参数向量中，不直接驱动日SRAD抽样。

## Runtime究竟怎样生成SRAD

官方 DSSAT `v4.8.0.24` 的 [`Weather/WGEN.for`](https://github.com/DSSAT/dssat-csm-os/blob/v4.8.0.24/Weather/WGEN.for)给出实际计算链：WGENIN 将 CLI 的 SDMN/SDSD/SWMN/SWSD 载入每日参数向量；WGENPM 将月参数线性插值并加周期修正；WGENGN 先按上一日湿/干状态，用 PDW/PWW Markov 转移生成降雨发生，再对 wet day 生成雨量。随后以 `RAIN>=0.0001` 分类 wet/dry：dry/wet 状态分别选择 SRAD 均值和标准差（并分别选择 Tmax 参数），Tmin 使用月均值和标准差。

温度和辐射不是各自独立抽样。源码用 Box–Muller 从同一随机流形成三个标准正态创新，并拒绝 `abs(z)>2.5` 的创新；再按 `X=B*E+A*XPRV` 形成共享且具有前一日记忆的三维潜在状态。其非对角 A/B 矩阵让 Tmax、Tmin、SRAD残差存在同时及滞后联系。降雨金额本身不是 SRAD 公式的连续输入；降雨通过 wet/dry 参数切换及连续共享随机流参与形成最终联合结构。

SRAD先按 `raw = X(3)*SRSD + SRMN` 变换，随后源码明确执行：

```text
RC = 0.8 * RADA(DOY, latitude)
SRMIN = 0.2 * RC
SRAD = min(max(raw, SRMIN), RC)
```

这同时构成下限和上限；越界值是**截断/夹取**，不是重抽样。换言之，SRAD至少为 `0.16*RADA`。在YC纬度36.83°、6–10月作物季日历下，按源码RADA公式计算的硬下限范围为 **3.3060–6.6870 MJ m⁻² d⁻¹**。因此 `<2`、`<2.46823896` 和 `<3` 的每日SRAD，在所有作物季日都不可能由该源码生成，理论概率严格为0。更高阈值仅对其整月硬下限高于阈值的月份给出零概率；其他情况涉及状态过程，本轮不猜数值概率。计算见 `theoretical_low_srad_probability.csv` 与 `parameter_and_bound_audit.json`。

由此可解释004_20的关键现象：2004–2013 fitting确实有低于2.468的SRAD，且源CLI参数与fitting重算一致；但无论湿/干条件均值、方差或seed如何取值，SRAD在输出端都被更高的天文辐射下限夹住。因此100套候选天气未生成这些低值不是“100次抽样碰巧没抽到”，也不是首先由CLI参数拟合错误造成，而是公共源码中的确定性下限机制足以解释。

## lower bound / 版本证据边界

公开tag源码证实 WGEN 的 `SRMIN` 下限和 `RC` 上限。容器中实际程序的debug路径指向 `dssat-pdi-4.8.0.24+2~a4ae650/Weather/WGEN.for`，二进制字符串还包含“SRAD低于0.1时设为0.1”的诊断文本；但准确匹配的 patched `WGEN.for`、`DailyWeatherCheck` 源码及历史 crop-executable SHA256未归档。004_17记下来的runtime wrapper hash不能代替作物可执行文件哈希。故本轮将机制定为 `PARTIALLY_VERIFIED`：算法硬下限有官方源码支持且有本机二进制旁证，但尚未证明安装补丁版与公开tag逐字节/行为完全一致，也未证实0.1检查在当前补丁中的具体调用效果。

## 联合结构：能表达，但未必保留 fitting 结构

004_20给出的描述性相关为：fitting `Spearman(SRAD,RAIN)≈-0.392`、`Spearman(SRAD,Tmax)≈0.668`；EXACT78分别约`-0.456`和`+0.438`。WGEN确有湿/干条件SRAD/Tmax参数和共享自回归残差，故生成相关会由发生状态、条件参数、共同创新及滞后潜状态共同形成。另一方面，残差矩阵是固定的，不是从YC fitting每日联合数据估计的；雨量值不直接进入SRAD的连续公式；下限还会选择性压平低SRAD值。运行机制可以生成非零相关，但这不保证复现fitting联合分布。上述相关差异是结构诊断而非因果证明。

## Correction方案比较与最小必要修改

候选比较见 `correction_candidate_comparison.csv`。CLI各月参数已经正确反映 fitting 的 wet/dry SRAD边际统计，继续改SDMN/SDSD/SWMN/SWSD既不能越过SRMIN，也会把已核验参数改离fitting。输出后单变量patch虽可制造低值，却会绕开WGEN潜在状态，可能产生SRAD与雨、温度不协调。直接上 SRAD–RAIN–Tmax 多变量模型则复杂且会同时改变多个气象边际。

因此，最小必要的首个候选是**仅在隔离runtime副本中测试SRAD下限规则**，保留CLI、fitting参数、降雨发生/雨量、Tmax/Tmin、RNG、seed/context和SRAD上限不变。候选下限暂不能无条件指定：仅当准确补丁版 `DailyWeatherCheck` 源码证实0.1 MJ m⁻² d⁻¹为兼容且不会重加旧天文下限时，才采用该值；否则停止并重新论证，不自行猜一个替代值。完整pilot设计和停止门槛见 `weather_only_pilot_design.md`。

## 对004_20问题的直接回答

1. **是否只是单日minimum？** 不是只有参数问题；本轮找到了可确定产生低尾缺失的输出下限。004_20另外报告了P1及wet/cool联合结构差异，故仍要在pilot中检查持续低辐射和联合指标，不能只看minimum。
2. **是否有显式clip/bound？** 有。公开tag中SRAD明确夹在`[0.2*RC, RC]`；`RC=0.8*RADA`，故floor=`0.16*RADA`。另有本机binary string显示0.1下限检查，但准确补丁源码仍未知。
3. **为何fitting有而WGEN无？** CLI统计复算正确，但runtime输出下限在作物季高于2.468，因此这些低值理论上被完全排除。
4. **SRAD–RAIN–Tmax如何关联？** 雨发生的Markov状态选择wet/dry条件SRAD/Tmax参数；三变量共享受限正态创新与滞后潜状态；雨量不作为连续SRAD协变量。固定残差矩阵和输出clamp可造成fitting与生成相关结构不同。
5. **该修哪个环节？最小修正？** 先审准确运行版，再只测试SRAD lower-bound规则，不改CLI边际参数，不做生成后逐日patch，也暂不上复杂多变量模型。
6. **可否做weather-only correction pilot？** 设计已就绪，但执行前置门槛尚未满足：需要补齐精确patched source/build hash并核验`DailyWeatherCheck`。本轮没有生成任何候选天气。
7. **是否有依据扩到160或重训PPO？** 没有。本轮只支持先做隔离的天气生成机制pilot；扩池和PPO训练都必须等待weather-only候选通过分布、联合结构、可复现性和归档QA。

## 分类与产物

```text
runtime_srad_mechanism_status = PARTIALLY_VERIFIED
low_tail_loss_mechanism = TRUNCATION_OR_BOUND
recommended_correction_family = RUNTIME_PARAMETER_CHANGE
weather_only_pilot_ready = YES  # exact patch/source preflight is mandatory
```

机器结论见 `results/yc_random_weather_ppo/004_21_yc_wgen_srad_generation_mechanism_and_correction_design/decision.json`。本目录还包含参数审计、理论硬界概率表、完整机制链、变量依赖链、候选方案CSV与weather-only pilot设计。没有修改既有实验资产，没有生成天气，没有运行WGEN/DSSAT/PPO，也没有commit/push。
