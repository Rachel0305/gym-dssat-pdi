# 046_04 SYA originIC 外部规则型 auto-N 记录

- 不训练网络；灌溉为 DSSAT native auto，氮肥为 gym-DSSAT 外部 NSTRES 规则。
- 规则：NSTRES >= 0.05，每次 25 kg/ha，间隔 7 天，季节上限 250 kg/ha，DAP90 后禁氮。
- 成功年份：10；失败年份：0；耗时：32.1 s。
- 若实际施氮仍为 0，代表该规则阈值未被触发，不调整阈值以凑出施肥。
