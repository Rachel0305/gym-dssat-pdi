# 021_41 SY2014 在线demo n-step屏蔽1K A/B记录

## 单变量

Control复用021_35；Treatment仅删除在线混合batch中demo样本的n-step梯度，agent n-step的原始1/32缩放、agent TD1、demo margin及其他设置均不变。离线500次学习不变。

## 结果

| arm | checkpoint | yield_kg_ha | irrigation_mm | nitrogen_kg_ha | late_n_after_dap90_kg_ha | expert_efficiency_gate |
| --- | --- | --- | --- | --- | --- | --- |
| control_021_35_all_nstep | 0 | 11175.0 | 90.0 | 300.0 | 0.0 | True |
| control_021_35_all_nstep | 250 | 5408.0 | 0.0 | 0.0 | 0.0 | False |
| control_021_35_all_nstep | 500 | 5408.0 | 0.0 | 0.0 | 0.0 | False |
| control_021_35_all_nstep | 750 | 10603.0 | 30.0 | 200.0 | 0.0 | False |
| control_021_35_all_nstep | 1000 | 11175.0 | 75.0 | 300.0 | 0.0 | True |
| treatment_mask_demo_nstep | 0 | 11175.0 | 90.0 | 300.0 | 0.0 | True |
| treatment_mask_demo_nstep | 250 | 11175.0 | 90.0 | 300.0 | 0.0 | True |
| treatment_mask_demo_nstep | 500 | 11175.0 | 90.0 | 300.0 | 0.0 | True |
| treatment_mask_demo_nstep | 750 | 11205.0 | 90.0 | 200.0 | 0.0 | True |
| treatment_mask_demo_nstep | 1000 | 11205.0 | 90.0 | 200.0 | 0.0 | True |

Control通过数=1/4，Treatment通过数=4/4。预注册分支：**A**。屏蔽在线demo n-step显著提高优质策略保持性，可另立5K/多seed任务。

## 边界

仅seed0 1K；没有启动5K或多seed，也没有根据结果调其他权重。
