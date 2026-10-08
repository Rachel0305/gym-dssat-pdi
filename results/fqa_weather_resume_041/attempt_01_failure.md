# 首次构建失败记录

首次运行 `build_static_fitting.py` 在 2013-03-31 的 FQA `SRAD=306.055` 停止，报错为 `candidate attempts to replace valid station (2013-03-31, SRAD)`。原因是新脚本最初只把负 SRAD 判为异常，遗漏了 008 源审计已经采用的 `SRAD > 60` 上限。首次运行在写出天气与 CLI 前停止，原始文件和旧结果未修改。随后把 SRAD 与 TMAX/TMIN 的物理阈值改为与 008 原脚本一致（SRAD <0 或 >60，温度 <-60 或 >50），第二次运行成功。
