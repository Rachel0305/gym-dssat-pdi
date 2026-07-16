# 021_36 SY2014 1K在线轨迹 checkpoint Q排序离线审计

## 目的

解释021_35中0步优质、250/500步no-op坍缩、750步部分恢复、1000步完全恢复的轨迹，判断产量变化是否与固定oracle示范状态上的非零动作召回和Q-margin同步。

## 边界

- 只读取021_35的0/250/500/750/1000 checkpoint和轨迹表；
- 使用021_24冻结standardization和同一组160个oracle示范状态；
- 不训练、不调用DSSAT、不改任何文件定义；
- 对五个非零事件逐一保存target action、预测action和expert margin；
- 样本仅5个checkpoint，相关性只做描述，不写成因果。

## 预注册判据

- A：yield与nonzero recall的Spearman>=0.8，且yield与positive-margin fraction的Spearman>=0.8；Q排序变化与策略坍缩/恢复高度同步；
- B：上述两个相关系数绝对值均<0.3；固定示范状态Q指标不能解释闭环产量轨迹；
- C：其他情况；存在部分对应但不是单一解释。

## 输出

曲线CSV、事件CSV、相关性CSV、validation JSON、summary JSON、PNG/SVG和中文记录。不Git push。

