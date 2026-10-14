#!/usr/bin/env python3
"""Build SY2012/2015 per-seed five-scenario figures from frozen snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src import build_relaxed_success_five_scenario_daily_evidence_027_05 as old


OUT = ROOT / "benchmark_results" / "028_05_sy_crossyear_frozen_ppo_daily"
SOURCE = ROOT / "benchmark_results" / "026_07_attempt2"
DOC = ROOT / "docs" / "2026-07-18_028_05_sy2012_sy2015_frozen_crossyear_daily_completion.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gap_row(summary: pd.DataFrame, year: int, seed: int, checkpoint: int) -> dict[str, object]:
    rl = summary[summary["scenario"].eq("rl_candidate")].iloc[0]
    base = summary[~summary["scenario"].eq("rl_candidate")]
    result: dict[str, object] = {"site": "SY", "year": year, "seed": seed, "checkpoint": checkpoint}
    wins = []
    for label, col in {"yield": "final_grain_kg_ha", "WP_ET": "wp_et_kg_m3", "PFP_N": "pfp_n_kg_kg"}.items():
        vals = pd.to_numeric(base[col], errors="coerce").dropna()
        value_raw = pd.to_numeric(pd.Series([rl[col]]), errors="coerce").iloc[0]
        value = float(value_raw) if pd.notna(value_raw) else math.nan
        best = float(vals.max()) if not vals.empty else math.nan
        gap = value - best if math.isfinite(value) and math.isfinite(best) else math.nan
        pct = 100 * gap / best if math.isfinite(gap) and best else math.nan
        win = bool(math.isfinite(gap) and gap > 0)
        result.update({f"rl_{label}": value, f"baseline_max_{label}": best, f"gap_{label}": gap, f"gap_pct_{label}": pct, f"winning_{label}": win})
        if win:
            wins.append(label)
    result["advisor_any_metric_strict_winner"] = bool(wins)
    result["winning_metrics"] = ";".join(wins) if wins else "none"
    result["close_other_metrics_status"] = "not_adjudicated_no_advisor_tolerance"
    return result


def main() -> None:
    frozen = pd.read_csv(OUT / "028_05_sy_crossyear_frozen_ppo_summary.csv")
    all_daily, all_summary, all_checks, gaps, manifests = [], [], [], [], []
    figure_count = 0
    for row in frozen.sort_values(["year", "seed"]).to_dict("records"):
        year, seed = int(row["year"]), int(row["seed"])
        model_path = ROOT / str(row["source_model"])
        checkpoint = int(model_path.stem.split("_")[-1])
        case_dir = OUT / str(year) / f"seed{seed}"
        base_root = SOURCE / str(year)
        snapshots = {
            "null": base_root / "baseline_source" / "runs" / str(year) / "seed0" / "null" / "pdi_tmp_snapshot_eval",
            "recorded_farmer": base_root / "baseline_source" / "runs" / str(year) / "seed0" / "recorded" / "pdi_tmp_snapshot_eval",
            "dssat_auto": base_root / "baseline_source" / "runs" / str(year) / "seed0" / "dssat_auto" / "pdi_tmp_snapshot_eval",
            "official_extension_expert": base_root / "expert_source" / "runs" / str(year) / "seed0" / "transfer_official_extension_expert" / "pdi_tmp_snapshot_eval",
            "rl_candidate": case_dir / "snapshot",
        }
        missing = [str(p) for p in snapshots.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"SY{year}/seed{seed}: {missing}")
        case = old.Case("SY", "Shenyang", year, seed, checkpoint, model_path, snapshots, case_dir / "frozen_evidence.json", "SY2014 fixed model transferred with zero training")
        daily, summary, checks = old.build_case(case, algorithm="MaskablePPO")
        daily.to_csv(case_dir / "five_scenario_daily.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(case_dir / "five_scenario_summary.csv", index=False, encoding="utf-8-sig")
        check_frame = pd.DataFrame(checks)
        check_frame.to_csv(case_dir / "daily_evidence_checks.csv", index=False, encoding="utf-8-sig")
        if not bool(check_frame["passed"].all()):
            raise RuntimeError(f"SY{year}/seed{seed}: daily check failed")

        old.FIG = case_dir / "figures"
        old.FIG.mkdir(parents=True, exist_ok=True)
        dummy = old.ExistingPPOCase(
            "SY", "Shenyang", year, seed, checkpoint, model_path,
            SOURCE / f"026_07_sy{year}_four_baselines.csv",
            case_dir / "five_scenario_summary.csv", case_dir / "stage_actions.csv", ("checkpoint", checkpoint),
        )
        paths = old.plot_ppo_endpoints(summary, dummy)
        paths += old.plot_daily(daily, case, algorithm="MaskablePPO")
        actions = pd.read_csv(case_dir / "stage_actions.csv")
        irrigation = "executed_irrigation" if "executed_irrigation" in actions else "executed_executed_irrigation"
        nitrogen = "executed_nitrogen" if "executed_nitrogen" in actions else "executed_executed_nitrogen"
        actions = actions.rename(columns={irrigation: "executed_irrigation_mm", nitrogen: "executed_nitrogen_kg_ha"})
        paths += old.plot_ppo_stage_actions(actions, dummy)
        figure_count += len(paths)

        gap = gap_row(summary, year, seed, checkpoint)
        gaps.append(gap)
        manifest = {
            "site": "SY", "year": year, "seed": seed, "checkpoint": checkpoint,
            "model_sha256": sha256(model_path), "daily_sha256": sha256(case_dir / "five_scenario_daily.csv"),
            "summary_sha256": sha256(case_dir / "five_scenario_summary.csv"),
            "figure_paths": [str(p.relative_to(ROOT)) for p in paths], "training_calls": 0,
        }
        (case_dir / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifests.append(manifest)
        all_daily.append(daily); all_summary.append(summary); all_checks.append(check_frame)

    pd.concat(all_daily, ignore_index=True).to_csv(OUT / "028_05_all_cases_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_summary, ignore_index=True).to_csv(OUT / "028_05_all_cases_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_checks, ignore_index=True).to_csv(OUT / "028_05_all_daily_evidence_checks.csv", index=False, encoding="utf-8-sig")
    gap_frame = pd.DataFrame(gaps)
    gap_frame.to_csv(OUT / "028_05_advisor_any_metric_gap_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "028_05_evidence_manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 028_05 SY2012/SY2015 固定权重跨年日值证据补齐记录", "",
        "共完成6个固定权重跨年case、6个DSSAT季节、0次训练。所有模型哈希、动作序列和终值均复现026_07。", "",
        "第一次执行在SY2015 seed0因PFP_N为NA且检查函数未识别NA=NA而停止；部分输出完整保存在`benchmark_results/028_05_failed_attempt_1_nan_pfp_comparison`。修复只改变工程比较逻辑，不改变指标定义或模型结果。", "",
        "|年份|seed|checkpoint|领先指标|产量差%|WP_ET差%|PFP_N差%|", "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in gaps:
        def fmt(key: str) -> str:
            v = r[key]
            return "NA" if not isinstance(v, (int, float)) or not math.isfinite(float(v)) else f"{float(v):.3f}"
        lines.append(f"|{r['year']}|{r['seed']}|{r['checkpoint']}|{r['winning_metrics']}|{fmt('gap_pct_yield')}|{fmt('gap_pct_WP_ET')}|{fmt('gap_pct_PFP_N')}|")
    lines += ["", "‘接近’仍未设人为容差；表中百分比供导师判断。所有图与CSV由manifest绑定。"]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "cases": len(manifests), "figures": figure_count, "training_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
