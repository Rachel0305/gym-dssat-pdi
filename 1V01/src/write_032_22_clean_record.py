from pathlib import Path

import pandas as pd


ROOT = Path("/workspaces/gym-dssat-pdi")
OUT = ROOT / "benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch"
DOC = ROOT / "docs/032_22_five_site_half_split_stress_aware_maskableppo_batch_record.md"
SITE_MAP = {"FQA": "FQ", "HLA": "HL", "LCA": "LC", "SYA": "SY", "YCA": "YC"}


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_空表_"
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = [str(row.get(col, "")).replace("\n", " ") for col in cols]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    train = pd.read_csv(OUT / "evaluation/032_22_training_checkpoint_inventory.csv", keep_default_na=False)
    eval_df = pd.read_csv(OUT / "evaluation/032_22_checkpoint_validation_summary.csv", keep_default_na=False)
    by = pd.read_csv(OUT / "evaluation/032_22_validation_summary_by_station_checkpoint.csv", keep_default_na=False)
    split = pd.read_csv(OUT / "configs/032_22_half_split_selection.csv", keep_default_na=False)
    if "site" not in split.columns:
        split["site"] = split["station_code"].map(SITE_MAP)

    best_rows = []
    for _, group in by.copy().groupby("station_code"):
        group = group.copy()
        group["any_metric_win_four_count"] = pd.to_numeric(group["any_metric_win_four_count"], errors="coerce")
        group["mean_gap_yield_vs_four_max"] = pd.to_numeric(group.get("mean_gap_yield_vs_four_max", 0), errors="coerce")
        best_rows.append(
            group.sort_values(
                ["any_metric_win_four_count", "mean_gap_yield_vs_four_max"],
                ascending=[False, False],
            ).iloc[0]
        )
    best = pd.DataFrame(best_rows)
    best_cols = [
        "station_code",
        "site",
        "checkpoint_step",
        "validation_years",
        "any_metric_win_four_count",
        "mean_final_grnwt",
        "mean_total_irrigation",
        "mean_total_n",
        "mean_gap_yield_vs_four_max",
        "mean_gap_wp_et_vs_four_max",
        "mean_gap_pfp_n_vs_four_max",
    ]
    for col in best_cols:
        if col not in best.columns:
            best[col] = ""

    lines: list[str] = []
    lines += ["# 032_22 五站点前半训练-后半验证自由时序 stress-aware MaskablePPO 批量实验记录", ""]
    lines += ["## 结论先说", ""]
    lines += [
        "- 状态：训练与验证均已完成。",
        "- 算法：MaskablePPO，seed=0；每个站点独立训练 100,000 timesteps。",
        "- 站点：FQ、HL、LC、SY、YC；没有跨站点迁移，也没有联合训练。",
        "- checkpoint：25,000、50,000、75,000、100,000。",
        "- 验证：每站点后半年份 10 年；共 5 × 4 × 10 = 200 条 checkpoint-year 验证，200/200 成功。",
        "- 注意：本轮是自由时序 stress-aware PPO 的五站点批量初筛，不是最终参数优化；只有 seed0。",
        "",
    ]
    lines += ["## 固定年份划分", "", md_table(split[["station_code", "site", "year", "split"]], max_rows=120), ""]
    lines += [
        "## 训练模型清单",
        "",
        md_table(train[["station_code", "site", "checkpoint_step", "run_status", "model_path", "model_sha256"]], max_rows=30),
        "",
    ]
    lines += ["## 站点-checkpoint 验证汇总", "", md_table(by, max_rows=80), ""]
    lines += ["## 每站点当前最佳 checkpoint（按 any_metric_win_four_count 优先）", "", md_table(best[best_cols], max_rows=20), ""]
    lines += ["## 目前直观看法", ""]
    lines += [
        "- LC 和 YC 在后半验证年份上表现最好：多个 checkpoint 达到 10/10 年至少一个指标超过四情景最高值。",
        "- HL 有部分 checkpoint 有信号，但不稳定；25k 的节氮倾向强，但平均产量较低，需要看逐年表和管理措施是否合理。",
        "- FQ 和 SY 在这套 seed0、100K、当前 reward/约束下没有出现“至少一个指标超过四情景最高值”的年份，需要后续诊断或换配置。",
        "- 因为本轮只有单 seed，不能直接说某站点已经稳定成功；它更像是为下一步选 checkpoint、画图和诊断提供候选。",
        "",
    ]
    lines += ["## 文件输出", ""]
    lines += [
        f"- 训练清单：`{(OUT / 'evaluation/032_22_training_checkpoint_inventory.csv').relative_to(ROOT).as_posix()}`",
        f"- 验证逐年表：`{(OUT / 'evaluation/032_22_checkpoint_validation_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- 站点汇总表：`{(OUT / 'evaluation/032_22_validation_summary_by_station_checkpoint.csv').relative_to(ROOT).as_posix()}`",
        f"- 日值输出目录：`{(OUT / 'daily_outputs').relative_to(ROOT).as_posix()}`",
        f"- 模型目录：`{(OUT / 'models').relative_to(ROOT).as_posix()}`",
        "",
    ]
    lines += ["## 非科学性问题与修复记录", ""]
    lines += [
        "- 第一次启动后发现 SB3 需要 Gymnasium API，脚本导入从 `gym` 改为 `gymnasium` 后环境初始化通过。",
        "- 首轮验证失败不是模型问题，而是 `daily_outputs/<station>` 目录没有创建；修复后复用已有模型重新验证，200/200 成功。",
        "- 最终汇总时发现 `site` 列缺失导致汇总函数报错；已根据 `station_code` 补映射后重新生成最终 CSV 和本记录。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(DOC)


if __name__ == "__main__":
    main()
