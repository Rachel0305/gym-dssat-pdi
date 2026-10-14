# 058 闰年注入补充门槛

seed0的2K随机年份前缀中，两站都未出现2008；因此额外执行单进程、全日历诊断，使用057冻结2008/seed1001 WTH，WTHER=M读取已有天气。SY与LC的Weather.OUT均有连续366天，SRAD/TMAX/TMIN/RAIN每天均与源WTH一致，含2月29日。峰值约133MB。此诊断的作物管理输出不用于PPO科学比较，固定天气读取不涉及运行时天气RNG；复用诊断报告中的runtime_seed_verified字段不能作为随机生成seed证明，应以冻结源hash为准。

首次诊断失败：旧模板WSTA仍指CNSY0501，而只附加2008文件，DSSAT报Weather file not found。已保留失败日志与脚本备份，在新attempt002目录显式修正诊断模板天气文件名后，两站通过。未修改源模板或源WTH。

结合主smoke与本闰年门槛，冻结天气注入的技术证据完整。下一步可编写正式串行队列：SY seed0–7，再LC seed0–7；100K/seed，四个checkpoint十年历史验证，仍须启动前hash核验与资源限制。正式训练尚未启动。
