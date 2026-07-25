# 033_02 启用 IC 后的渲染 smoke 记录

## 任务性质

本任务只验证补丁后的安全渲染函数是否把 treatment 行中的 `IC/MI/MF` 显式启用；不跑 DSSAT，不训练。

## 输出

- CSV：`benchmark_results/033_02_enable_ic_render_smoke/033_02_render_smoke_rows.csv`

## 结果

- 通过：5/5

| 站点 | 年份 | IC | MI | MF | INITIAL CONDITIONS id | 判定 |
|---|---:|---:|---:|---:|---|---|
| HLA | 2007 | 1 | 1 | 1 | 1;2;3 | 通过 |
| SYA | 2014 | 1 | 1 | 1 | 1;2 | 通过 |
| LCA | 2010 | 1 | 1 | 1 | 1 | 通过 |
| FQA | 2008 | 1 | 1 | 1 | 1 | 通过 |
| YCA | 2014 | 1 | 1 | 1 | 1 | 通过 |

## 结论

- 若 5/5 均通过，则说明未来通过 `ppo_safe_rendering.safe_render_template` 生成的工作输入会启用当前模板中的第一组 `*INITIAL CONDITIONS`。
- 这只修复未来渲染链条，不会改变既有历史结果；既有结果若要使用 IC，必须重新生成输入并重跑。
