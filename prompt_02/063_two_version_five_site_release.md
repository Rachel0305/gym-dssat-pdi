# 063 五站双版本项目整理、补充YC ETCP与GitHub备份

用户确认边界并授权GitHub备份：根目录1V01保留五站原冻结80策略、FQ原2018输入/结果、YC当时WP_ET缺项；2V02保留相同训练模型，使用FQ修正2018验证并补齐YC两域seed0–7在2014–2023的精确ETCP/WP_ET。原始目录不覆盖。

完整项目包包含可运行代码、配置、源天气/土壤/品种、训练模型与checkpoint/验证表、逐日表/管理事件、独立PNG、审计和版本差异。重复render/cache/tmp、无关论文文件、失败大缓存不上传；受保护原输入复制，模型仅小型策略zip。保留重现路径与model/source sha256。各版本均有全文件manifest和完整性核验，明示不相同的数据来源与未统一checkpoint指标。V01当时缺项如实保留，不伪造补齐。

YC仅evaluation-only串行回放，1536MB、单模型600秒，旧R3/4/5精确30行可复用，其他seed/year核验原yield/I/N/步数及WTH/template hash，失败停。FQ历史seed7修正2018如缺则单独补回放，禁止重训或重选模型。修订只改变验证输入，2018旧端点不能声称和新端点一致。

GitHub上传只包含本次1V01/2V02及打包脚本/prompt/索引，不混入已有工作区修改，不push大文件/缓存，不force push。创建codex版本分支，逐文件大小检查、stage明确路径、提交后正常push并核对远端commit。用户本次已明确授权push，无需重复询问。
