# 059 正式八种子队列

已获得用户明确授权。后台控制器按SY seed0–7、LC seed0–7顺序串行，每seed100K，四个checkpoint分别2014–2023十年历史验证；保持冻结训练合同和057独立天气抽样方案。058共享seed0 smoke与闰年注入检查作为同合同队列前置证据，运行器仍调用原smoke验证函数核验真实结果，没有为其他seed伪造smoke。

运行器：scripts/run_sylc_formal_queue_059.py。状态：results/sylc_external_weather_059/queue_status.json；控制器日志：results/sylc_external_weather_059_controller.log；单种子日志：results/sylc_external_weather_059/{site}_seed{seed}.log。

限制：线程1、单环境缓存、RSS1536MB、每seed7200秒。实际天气按固定日期偏移检查，复制前后冻结hash检查；每个种子完成后核验4模型及40验证结果、模型真实seed、年份/抽样一致、验证天气hash，并压缩备份。任何失败停止队列并保留输出，不跳过seed、不自行降标准、不覆盖已有模型。

正式结果：benchmark_results/059{site}s{seed}_frozen_weather_augmented_100k；完成备份：backups/sylc_059_formal/{site}/seed{seed}/formal_result.zip。重复渲染输入留在实验目录，备份不重复压缩这部分。

本记录说明启动与执行规则，不证明队列已完成。只有queue_status.json标记COMPLETE且有16份completion_audit.json及逐seed备份验证，才可报告训练完成。完成后应对全部八种子、固定历史年份汇总产量、灌溉、施氮及PFP_N，对照同合同未增强结果，不依据局部checkpoint择优或宣称改善。
