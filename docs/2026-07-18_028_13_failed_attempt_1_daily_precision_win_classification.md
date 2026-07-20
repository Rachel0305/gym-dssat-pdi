# 028_13 已筛选年份 MaskablePPO 导师总览

## 结论先行

- 17/17 个已筛选站点年都至少存在 1 个 seed，在产量、WP_ET、PFP_N 中至少一项严格超过四基线最大值。
- 其中 10/16 个已完成三 seed 的站点年达到 ≥2/3 初步稳定；LC2010 仅 1 seed，不能判定跨 seed 稳定。
- 五个站点均有成功候选；但当前证据不支持“所有年份、所有 seed 都成功”。
- 可称为统一阶段型 MaskablePPO 框架：主算法和训练超参数一致；不可称为单一通用模型，因为 scaler、输入、可执行阶段数和模型权重仍是站点专属。

## 五站点汇总

|site|screened years|years with candidate|years >=2/3 stable|not assessed|
|---|---:|---:|---:|---:|
|FQ|6|6|3|0|
|HLA|5|4|4|0|
|LC|1|1|0|1|
|SY|3|3|2|0|
|YC|2|2|1|0|

## 年份级结果

|site-year|representative seed|winning metrics|winner seeds|stability|yield|WP_ET|PFP_N|I|N|
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
|FQ2013|0|yield|1/3|candidate_exists_not_cross_seed_stable|7814|2.05|31.3|60|250|
|FQ2014|1|yield;WP_ET|2/3|initially_stable_2of3_or_better|8717|2.44|43.6|75|200|
|FQ2016|2|PFP_N|2/3|initially_stable_2of3_or_better|8012|2.41|80.1|75|100|
|FQ2019|2|WP_ET;PFP_N|3/3|initially_stable_2of3_or_better|8446|2.73|84.5|45|100|
|FQ2020|2|yield;WP_ET;PFP_N|1/3|candidate_exists_not_cross_seed_stable|9677|2.62|96.8|120|100|
|FQ2023|1|WP_ET|1/3|candidate_exists_not_cross_seed_stable|9100|2.65|30.3|60|300|
|HLA2007|0|yield|2/3|initially_stable_2of3_or_better|7987|1.59|NA|90|0|
|HLA2010|0|PFP_N|2/3|initially_stable_2of3_or_better|7854|1.69|157.1|90|50|
|HLA2015|1|yield;WP_ET;PFP_N|2/3|initially_stable_2of3_or_better|7653|1.53|51.0|60|150|
|HLA2016|1||2/3|initially_stable_2of3_or_better|7538|1.72|30.2|30|250|
|HLA2022|1|PFP_N|1/3|candidate_exists_not_cross_seed_stable|7934|1.68|79.3|90|100|
|LC2010|0|PFP_N|1/1|not_assessed_one_seed|8739|3.06|58.3|60|150|
|SY2012|1|PFP_N|1/3|candidate_exists_not_cross_seed_stable|10056|2.28|50.3|105|200|
|SY2014|0|yield;WP_ET;PFP_N|3/3|initially_stable_2of3_or_better|11205|2.31|56.0|60|200|
|SY2015|2|yield;WP_ET;PFP_N|2/3|initially_stable_2of3_or_better|11108|2.35|44.4|120|250|
|YC2008|2|PFP_N|1/3|candidate_exists_not_cross_seed_stable|8158|2.23|81.6|90|100|
|YC2014|2|PFP_N|2/3|initially_stable_2of3_or_better|9418|2.57|94.2|120|100|

## 解释边界

- “17/17存在候选”来自预注册代表 seed；没有隐藏其他失败 seed，全部 winner count 已列出。
- “接近”仍未设置人为容差；未领先指标的百分比差距保存在 CSV，交由导师判断。
- HLA2007 等 prepared/derived 输入变体及 FQ recorded_shifted 来源必须随结果一并说明。
- 每个年份的数据、图和模型哈希由 028_12 manifest 绑定。
