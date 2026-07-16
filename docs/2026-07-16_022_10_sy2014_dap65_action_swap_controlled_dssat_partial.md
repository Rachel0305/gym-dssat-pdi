# 022_10 SY2014 DAP65 action1 vs action7 受控DSSAT对照（阻塞记录）

## 当前状态

任务书与独立执行脚本已完成，输入方案已核对：

- Control动作：3/4/7/1/1/0；
- Treatment动作：3/4/7/7/1/0；
- 唯一差异为DAP65 action1换为action7；
- DAP65前累计I30/N200，两个动作均不会发生预算裁剪；
- 预期季节总量为I60/N200与I60/N300。

容器命令在脚本启动前被当前Codex权限配置拒绝：

```text
permission denied while trying to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine
```

因此：

- DSSAT前向季数：0；
- DQN训练：0；
- 没有科学结果；
- 不能判定action1或action7占优；
- 不得把本次记作阴性实验。

## 恢复方式

Docker API访问恢复后，原样执行：

```powershell
docker exec -w /workspace b2fd6726c8c1 /opt/gym_dssat_pdi/bin/python src/run_sy2014_dap65_action_swap_controlled_022_10.py
```

不得改变动作序列、输入或判据。正式输出目录 `benchmark_results/022_10/` 尚未创建，可安全原样运行。

## 已准备文件

- `prompts/022_10_sy2014_dap65_action1_vs_action7_controlled_dssat.md`
- `src/run_sy2014_dap65_action_swap_controlled_022_10.py`
- `prompts/022_11_sy2014_offline_mc_q_warmstart_online_ab_seed1_draft.md`（继续暂停）
