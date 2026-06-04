# 给 Codex 的任务 Prompt：五个站点初始土壤水分设定、PPT 记录与 Jinja2 模板修改

## 任务背景

我正在使用 `gym-DSSAT` / DSSAT 进行玉米农田水氮管理优化实验。最近发现：在真实历史天气条件下，部分站点几乎没有出现土壤水分胁迫，怀疑原因之一是 DSSAT 实验文件里的 `*INITIAL CONDITIONS` 初始土壤体积含水量 `SH2O` 设置过高。

之前多个站点曾经统一或近似统一设置为 `SH2O = 0.30`，但不同站点土壤的 `SLLL / SDUL / SSAT` 差异很大，所以同样的 `0.30` 在不同土壤中代表完全不同的水分状态。现在需要根据每个站点土壤文件中的 `SLLL` 和 `SDUL`，采用一个有文献支撑、可解释、统一的设定方法，重新设定五个站点的初始土壤水分。

本任务只设定一个主方案，暂时不做土壤水分敏感性测试。

---

## 你需要完成的工作

请完成两件事：

1. **生成一个 PowerPoint 文件**，记录本次初始土壤水分分析过程、依据、计算方法、当前问题、最终设定结果，并在 PPT 中标注参考文献。
2. **修改五个站点的 Jinja2 模板文件**，把 `*INITIAL CONDITIONS` 里的 `SH2O` 修改为本 prompt 中给出的最终结果。

五个 Jinja2 模板文件名称为：

```text
UFGA8201-HL.jinja2
UFGA8201-SY.jinja2
UFGA8201-LC.jinja2
UFGA8201-YC.jinja2
UFGA8201-FQ.jinja2
```

可以直接修改原始 Jinja2 文件，因为原始文件已经备份到 GitHub。

---

## 非常重要的安全要求

请严格遵守：

1. **只修改上述 5 个 Jinja2 模板文件，以及新生成 PPT 文件。**
2. **不要删除任何数据文件、结果文件、模型文件、天气文件、土壤文件或其他脚本。**
3. **不要运行任何格式化硬盘、清空目录、批量删除、重置仓库、强制覆盖仓库的命令。**
4. 禁止使用以下危险命令或等价操作：

```bash
rm -rf /
rm -rf *
rm -rf ./*
git reset --hard
git clean -fdx
format
mkfs
```

5. 修改前可以先检查文件内容，但不要移动整个项目目录。
6. 修改完成后，请输出修改摘要，说明每个站点修改了哪些 `SH2O` 值。

---

## 科学依据和设定原则

DSSAT 土壤文件中：

- `SLLL`：土壤水分下限，近似作物可吸水下限 / 萎蔫点。
- `SDUL`：土壤排水上限，近似田间持水量。
- `SSAT`：饱和含水量。
- `SH2O`：初始土壤体积含水量。

不能简单地让所有站点统一使用 `SH2O = 0.30`。更合理的做法是根据每层土壤的可利用水范围设定：

```text
SH2O = SLLL + f × (SDUL - SLLL)
```

其中：

```text
f = 0.55
```

表示初始土壤水分为该土层可利用水范围的 55%，属于“中等偏干 / 适中略偏干”的设定。该设定可以避免初始土壤过湿，同时又不是极端干旱情景。

本任务最终采用：

```text
SH2O = SLLL + 0.55 × (SDUL - SLLL)
```

所有结果保留两位小数，写入 Jinja2 模板。

---

## 需要在 PPT 中说明的分析过程

PPT 至少包括以下内容：

### 第 1 页：标题页

建议标题：

```text
五个站点 DSSAT 初始土壤水分设定分析
```

副标题可写：

```text
基于 SLLL、SDUL 与相对可利用水比例的 SH2O 修正方案
```

### 第 2 页：问题背景

说明：

- 原始实验中，多个站点初始土壤水分设为 `0.30`。
- 真实历史天气下，部分站点几乎没有出现明显 `SWFAC` 土壤水分胁迫。
- 怀疑原因：初始 `SH2O` 偏高，导致模拟初期根区水分过于充足。
- 但 `0.30` 是否偏高，不能直接判断，必须结合每层 `SLLL / SDUL / SSAT`。

### 第 3 页：DSSAT 土壤水分参数解释

说明：

- `SLLL`：土壤水分下限，近似作物无法继续有效吸水的下限。
- `SDUL`：排水上限，近似田间持水量。
- `SSAT`：饱和含水量。
- `SH2O`：初始土壤体积含水量。
- 作物可利用水范围近似为：`SDUL - SLLL`。

### 第 4 页：判断当前 SH2O 是否偏湿的方法

公式：

```text
相对可利用水比例 = (SH2O - SLLL) / (SDUL - SLLL)
```

解释：

```text
0%：接近萎蔫点，非常干
50%：中等水分
80% 以上：偏湿
100%：接近 SDUL / 田间持水量
超过 100%：高于 SDUL，过湿，可能产生排水
```

### 第 5 页：原始 SH2O 设置的问题总结

请使用表格展示：

| 站点 | 原始 SH2O 情况 | 判断 |
|---|---|---|
| 海伦 HL | 全层 0.30，约 68%–76% 可利用水 | 中等偏湿 |
| 沈阳 SY | 全层 0.30，约 70%–94% 可利用水 | 明显偏湿 |
| 栾城 LC | 0.23/0.25/0.28/0.30，约 56%–70% 可利用水 | 基本合理 |
| 禹城 YC | 0.30/0.30/0.31/0.31，约 73%–88% 可利用水 | 偏湿 |
| 封丘 FQ | 第一层 0.30，但 SDUL=0.21，相对可利用水约 200% | 第一层严重过湿 |

### 第 6 页：最终设定方法

说明本次采用：

```text
SH2O = SLLL + 0.55 × (SDUL - SLLL)
```

理由：

- 使用每层土壤自己的 `SLLL` 和 `SDUL`，避免不同土壤统一写 `0.30` 造成偏差。
- `f=0.55` 表示中等偏干，不是极端干旱，也不是田间持水量附近的偏湿状态。
- 更适合作为主实验的统一初始水分设定。
- 暂时不做敏感性测试，只采用该主方案。

### 第 7 页：最终 SH2O 结果表

请展示以下结果：

| 站点 | 土层 ICBL/cm | SLLL | SDUL | 新 SH2O |
|---|---:|---:|---:|---:|
| HL | 20 | 0.11 | 0.39 | 0.26 |
| HL | 40 | 0.11 | 0.38 | 0.26 |
| HL | 60 | 0.11 | 0.38 | 0.26 |
| HL | 90 | 0.11 | 0.36 | 0.25 |
| SY | 10 | 0.14 | 0.37 | 0.27 |
| SY | 20 | 0.15 | 0.32 | 0.24 |
| SY | 30 | 0.16 | 0.31 | 0.24 |
| SY | 40 | 0.14 | 0.32 | 0.24 |
| SY | 60 | 0.13 | 0.31 | 0.23 |
| SY | 100 | 0.13 | 0.35 | 0.25 |
| LC | 20 | 0.09 | 0.33 | 0.22 |
| LC | 40 | 0.11 | 0.36 | 0.25 |
| LC | 110 | 0.12 | 0.37 | 0.26 |
| LC | 150 | 0.07 | 0.40 | 0.25 |
| YC | 15 | 0.09 | 0.33 | 0.22 |
| YC | 30 | 0.11 | 0.36 | 0.25 |
| YC | 60 | 0.12 | 0.37 | 0.26 |
| YC | 90 | 0.07 | 0.40 | 0.25 |
| FQ | 30 | 0.12 | 0.21 | 0.17 |
| FQ | 70 | 0.24 | 0.31 | 0.28 |
| FQ | 100 | 0.15 | 0.36 | 0.27 |

### 第 8 页：最终结论

建议写：

- 不能继续所有站点统一使用 `SH2O=0.30`。
- 沈阳、禹城、封丘原始设定明显偏湿，尤其封丘 30 cm 层已经超过 `SDUL`。
- 海伦 `0.30` 虽未超过 `SDUL`，但属于中等偏湿，可能降低水分胁迫出现概率。
- 栾城原设置相对合理，但为保证方法统一，也改为基于 `0.55 × 可利用水` 的设定。
- 本研究最终采用 `SH2O = SLLL + 0.55 × (SDUL - SLLL)` 作为五个站点主实验的初始土壤水分设定。

### 第 9 页：参考文献

PPT 中必须标注参考文献。建议使用以下文献：

1. Jones, J. W., Hoogenboom, G., Porter, C. H., Boote, K. J., Batchelor, W. D., Hunt, L. A., Wilkens, P. W., Singh, U., Gijsman, A. J., & Ritchie, J. T. (2003). The DSSAT cropping system model. *European Journal of Agronomy*, 18(3–4), 235–265. https://doi.org/10.1016/S1161-0301(02)00107-7
2. DSSAT Foundation. Soil water balance overview. https://dssat.net/soil-water/
3. Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56. FAO. https://www.fao.org/4/x0490e/x0490e00.htm
4. Ritchie, J. T. (1998). Soil water balance and plant water stress. In G. Y. Tsuji, G. Hoogenboom, & P. K. Thornton (Eds.), *Understanding Options for Agricultural Production*. Springer.

在 PPT 正文中引用时，可以写成：

```text
DSSAT 使用 SLLL、SDUL 和 SSAT 描述各层土壤水分特征，并基于降雨、灌溉、径流、排水、蒸发和蒸腾计算土壤水分平衡（Jones et al., 2003; DSSAT Foundation, n.d.）。

FAO-56 将作物根区总可利用水定义为田间持水量与萎蔫点之间的差值，因此用 SDUL - SLLL 表示 DSSAT 中近似可利用水范围是合理的（Allen et al., 1998）。
```

---

## Jinja2 模板修改要求

请在每个模板文件中找到 `*INITIAL CONDITIONS` 部分，然后只修改 `SH2O` 列。不要改动 `SNH4` 和 `SNO3`，除非原文件中本来就是 `-99`。

### 1. 海伦站：`UFGA8201-HL.jinja2`

把 `SH2O` 改为：

```text
*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    MZ 07121   100     0     1     1   -99     0     0     0   100    15 -99
@C  ICBL  SH2O  SNH4  SNO3
 1    20  0.26  22.7  10.1
 1    40  0.26  10.8  11.3
 1    60  0.26  15.1  12.5
 1    90  0.25  8.20  8.10
```

### 2. 沈阳站：`UFGA8201-SY.jinja2`

把 `SH2O` 改为：

```text
*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    MZ 07121   100     0     1     1   -99  1000    .8     0   100    15 -99
@C  ICBL  SH2O  SNH4  SNO3
 1    10  0.27  4.57  2.60
 1    20  0.24  4.55  2.62
 1    30  0.24  4.07  2.54
 1    40  0.24  4.05  2.52
 1    60  0.23  3.82  2.00
 1   100  0.25  -99   -99
```

### 3. 栾城站：`UFGA8201-LC.jinja2`

把 `SH2O` 改为：

```text
*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    MZ 07121   100     0     1     1   -99  1000    .8     0   100    15 -99
@C  ICBL  SH2O  SNH4  SNO3
 1    20  0.22   4.0   5.3
 1    40  0.25   4.0   4.5
 1   110  0.26   4.0   6.2
 1   150  0.25   4.0  14.6
```

### 4. 禹城站：`UFGA8201-YC.jinja2`

把 `SH2O` 改为：

```text
*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    MZ 08153   100     0     1     1   -99  1000    .8     0   100    15 -99
@C  ICBL  SH2O  SNH4  SNO3
 1    15  0.22   4.0  10.0
 1    30  0.25   4.0   8.0
 1    60  0.26   4.0   6.0
 1    90  0.25   4.0   5.0
```

### 5. 封丘站：`UFGA8201-FQ.jinja2`

把 `SH2O` 改为：

```text
*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    MZ 07152   100     0     1     1   -99  1000    .8     0   100    15 -99
@C  ICBL  SH2O  SNH4  SNO3
 1    30  0.17   -99   -99
 1    70  0.28   -99   -99
 1   100  0.27   -99   -99
```

---

## 生成 PPT 的技术建议

可以使用 Python 的 `python-pptx` 生成 PPT，例如：

```bash
pip install python-pptx
```

建议输出文件名：

```text
initial_soil_water_analysis.pptx
```

PPT 风格要求：

- 中文为主。
- 表格清晰。
- 公式必须写清楚。
- 每页不要堆太多字。
- 参考文献至少在最后一页完整列出。
- 在正文涉及 DSSAT 土壤水分、FAO 可利用水概念的位置，用括号标注文献，例如 `(Jones et al., 2003; Allen et al., 1998)`。

---

## 修改完成后的检查要求

完成后请检查：

1. 五个 Jinja2 文件仍然存在。
2. 每个文件的 `*INITIAL CONDITIONS` 中 `SH2O` 已经变成目标值。
3. `SNH4` 和 `SNO3` 没有被误改。
4. 新生成了 PPT 文件 `initial_soil_water_analysis.pptx`。
5. 不要运行长时间训练，不要运行 DSSAT 批量实验，只做模板修改和 PPT 生成。

最后请给出简短总结：

```text
已完成：
1. 生成 PPT：initial_soil_water_analysis.pptx
2. 修改 UFGA8201-HL.jinja2：SH2O = 0.26/0.26/0.26/0.25
3. 修改 UFGA8201-SY.jinja2：SH2O = 0.27/0.24/0.24/0.24/0.23/0.25
4. 修改 UFGA8201-LC.jinja2：SH2O = 0.22/0.25/0.26/0.25
5. 修改 UFGA8201-YC.jinja2：SH2O = 0.22/0.25/0.26/0.25
6. 修改 UFGA8201-FQ.jinja2：SH2O = 0.17/0.28/0.27
```
