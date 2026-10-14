# NOAA/NCDC ISD-Lite 数据格式核验（任务 021）

## 来源身份

原始输入为 `china_isd_lite_2013.zip`。随包 `_数据说明.docx` 说明其底层来源是 NOAA/NCDC 公开 ISD-Lite FTP，并由资料提供方按年份重打包中国及港澳台站点。项目目录 `cma_daily_weather` 是历史目录名；本 ZIP 不是 CMA 日值数据。

包内有 415 个 2013 站点数据文件，ZIP 解压后总量约 83.97 MB。本审计只通过 ZIP 流式读取 HLA 和 FQA 最近 20 个中国候选站文件，没有整包解压。

## 时间戳与记录格式

- 站点文件名为 `USAF-WBAN-YYYY`；HLA 文件为 `507560-99999-2013`。
- 时间为 GMT/UTC。NCEI 技术说明指出 ISD-Lite 时间戳取最接近整点的观测时刻；站点实际记录频率不一定逐小时，随包说明称近年多为 3 小时、少数为 1 小时。
- 每行 61 个字符：年份列 1–4、月份 6–7、日期 9–10、小时 12–13；之后 8 个六字符气象字段连续排列。
- 降水 1 小时字段列 50–55，6 小时字段列 56–61；两者单位 mm，整数按 10 缩放，即原值除以 10。`-9999` 是缺测，`-1` 是 trace；trace 不赋任意毫米值。
- 随包 `isd-lite-format.txt` 有两处列范围与长度不一致：日期列把 9–11 写成长度 2，露点列把 20–24 写成长度 6。实际 61 字符行表明日期为 9–10、露点为 20–25；本审计按原始样例和固定宽度顺序解析，并校验日期和气象字段数。

## 累计窗和本地日

为可审计地处理重叠，本审计把观测时刻作为累计窗终点：1 小时字段表示 `[t−1h,t)`，6 小时字段表示 `[t−6h,t)`。随包格式文件说明了累计时长，但没有单列区间端点；该端点读法依据“在观测时刻测得累计深度”的 ISD 字段定义，作为本次审计假设明确记录。

所有日界转换到 Asia/Shanghai（UTC+08:00），目标日为本地 00:00–24:00。只有一组不重叠区间能完整覆盖 24 小时且所有有效路径给出唯一数值总量时，才输出有效日值；1 小时与 6 小时区间不相加重复计雨。跨本地午夜的区间不拆分。替代完整路径总量冲突，或完整路径依赖 trace 时，标为 unresolved。

## 参考资料

- 随包 `isd-lite-format.txt`、`isd-history.csv`、`isd-history.txt` 和 `_数据说明.docx`。
- [NCEI ISD-Lite Technical Document](https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-technical-document.pdf)：固定宽度子集、整点时间处理、缺测及 trace 编码、质量限制。
- [NCEI ISD-Lite Format](https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.pdf)：字段列、降水单位与缩放因子。
