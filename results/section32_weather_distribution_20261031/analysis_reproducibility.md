# 复现说明

运行scripts/analyze_section32_weather_distribution_137.py，仅需Python/pandas/numpy/matplotlib，串行读文件，无模拟导入。原始档案路径与SHA见source_sha256.json，不上传大批原始逐日文件。合并去重窗口值、所有暴露计数、排除记录和图数据均保存。计算中逐文件读取，缓存只保留小型日表，不并行、不加载模型。

窗口与阈值先写入analysis_contract.json。H等权年份，A图是去重窗口，抽样频次另表，不把重复序列当独立样本。
