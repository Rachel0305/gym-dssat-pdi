#!/usr/bin/env python3
"""Build 027_05-style five-scenario evidence from 028_04 frozen snapshots."""

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


OUT = ROOT / "benchmark_results" / "028_04_existing_yc_fq_lc_ppo_frozen_daily"
SOURCE = ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2"
DOC = ROOT / "docs" / "2026-07-18_028_04_existing_yc_fq_lc_ppo_frozen_daily_completion.md"
STATIONS = {"YC": "Yucheng", "FQ": "Fengqiu", "LC": "Luancheng"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gap_row(summary: pd.DataFrame, site: str, year: int, seed: int, checkpoint: int) -> dict[str, object]:
    rl = summary[summary["scenario"].eq("rl_candidate")].iloc[0]
    base = summary[~summary["scenario"].eq("rl_candidate")]
    columns = {
        "yield": "final_grain_kg_ha",
        "WP_ET": "wp_et_kg_m3",
        "PFP_N": "pfp_n_kg_kg",
    }
    out: dict[str, object] = {"site": site, "year": year, "seed": seed, "checkpoint": checkpoint}
    wins = []
    for label, column in columns.items():
        values = pd.to_numeric(base[column], errors="coerce").dropna()
        rv = pd.to_numeric(pd.Series([rl[column]]), errors="coerce").iloc[0]
        best = float(values.max()) if not values.empty else math.nan
        value = float(rv) if pd.notna(rv) else math.nan
        gap = value - best if math.isfinite(value) and math.isfinite(best) else math.nan
        pct = 100.0 * gap / best if math.isfinite(gap) and best != 0 else math.nan
        win = bool(math.isfinite(gap) and gap > 0)
        out.update({f"rl_{label}": value, f"baseline_max_{label}": best, f"gap_{label}": gap, f"gap_pct_{label}": pct, f"winning_{label}": win})
        if win:
            wins.append(label)
    out["advisor_any_metric_strict_winner"] = bool(wins)
    out["winning_metrics"] = ";".join(wins) if wins else "none"
    out["close_other_metrics_status"] = "not_adjudicated_no_advisor_tolerance"
    return out


def main() -> None:
    frozen = pd.read_csv(OUT / "028_04_frozen_ppo_summary.csv")
    all_daily = []
    all_summary = []
    all_checks = []
    all_gaps = []
    figures = []
    manifests = []

    for item in frozen.sort_values(["site", "seed"]).to_dict("records"):
        site, year = str(item["site"]), int(item["year"])
        seed, checkpoint = int(item["seed"]), int(item["checkpoint"])
        case_dir = OUT / site / f"seed{seed}_checkpoint{checkpoint}"
        baseline_root = SOURCE / site / "readiness" / "baseline_runs"
        snapshots = {
            "null": baseline_root / "null" / "pdi_tmp_snapshot_eval",
            "recorded_farmer": baseline_root / "recorded_farmer" / "pdi_tmp_snapshot_eval",
            "dssat_auto": baseline_root / "dssat_auto" / "pdi_tmp_snapshot_eval",
            "official_extension_expert": baseline_root / "official_extension_expert" / "pdi_tmp_snapshot_eval",
            "rl_candidate": case_dir / "snapshot",
        }
        missing = [str(p) for p in snapshots.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{site}/seed{seed}: missing snapshots {missing}")
        model_path = SOURCE / site / f"seed{seed}" / f"checkpoint_{checkpoint:06d}.zip"
        case = old.Case(
            site, STATIONS[site], year, seed, checkpoint, model_path, snapshots,
            case_dir / "frozen_evidence.json", "027_07 selected checkpoint; deterministic frozen reevaluation; no learn().",
        )
        daily, summary, checks = old.build_case(case, algorithm="MaskablePPO")
        daily.to_csv(case_dir / "five_scenario_daily.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(case_dir / "five_scenario_summary.csv", index=False, encoding="utf-8-sig")
        checks_df = pd.DataFrame(checks)
        checks_df.to_csv(case_dir / "daily_evidence_checks.csv", index=False, encoding="utf-8-sig")
        if not bool(checks_df["passed"].all()):
            raise RuntimeError(f"{site}/seed{seed}: daily evidence checks failed")

        old.FIG = case_dir / "figures"
        old.FIG.mkdir(parents=True, exist_ok=True)
        dummy = old.ExistingPPOCase(
            site, STATIONS[site], year, seed, checkpoint, model_path,
            SOURCE / site / "readiness" / "four_baseline_fresh_rerun.csv",
            case_dir / "five_scenario_summary.csv", case_dir / "stage_actions.csv", ("checkpoint", checkpoint),
        )
        figure_paths = old.plot_ppo_endpoints(summary, dummy)
        figure_paths += old.plot_daily(daily, case, algorithm="MaskablePPO")
        actions = pd.read_csv(case_dir / "stage_actions.csv")
        actions = actions.rename(columns={"executed_irrigation": "executed_irrigation_mm", "executed_nitrogen": "executed_nitrogen_kg_ha"})
        figure_paths += old.plot_ppo_stage_actions(actions, dummy)
        figures.extend(figure_paths)

        gap = gap_row(summary, site, year, seed, checkpoint)
        all_gaps.append(gap)
        all_daily.append(daily)
        all_summary.append(summary)
        all_checks.append(checks_df)
        manifest = {
            "site": site, "year": year, "seed": seed, "checkpoint": checkpoint,
            "model_sha256": sha256(model_path),
            "daily_sha256": sha256(case_dir / "five_scenario_daily.csv"),
            "summary_sha256": sha256(case_dir / "five_scenario_summary.csv"),
            "figure_paths": [str(p.relative_to(ROOT)) for p in figure_paths],
            "training_calls": 0,
        }
        (case_dir / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifests.append(manifest)

    pd.concat(all_daily, ignore_index=True).to_csv(OUT / "028_04_all_cases_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_summary, ignore_index=True).to_csv(OUT / "028_04_all_cases_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_checks, ignore_index=True).to_csv(OUT / "028_04_all_daily_evidence_checks.csv", index=False, encoding="utf-8-sig")
    gaps = pd.DataFrame(all_gaps)
    gaps.to_csv(OUT / "028_04_advisor_any_metric_gap_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "028_04_evidence_manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 028_04 YC/FQ/LC 现有 PPO 冻结日值证据补齐记录", "",
        "## 结果", "",
        "本任务完成4个既有选中模型的确定性冻结复评估和五情景证据整理，共运行4个DSSAT季节，训练调用为0。模型哈希、终值和资源投入均与027_07一致。", "",
        "|站点|年份|seed|checkpoint|领先指标|产量差%|WP_ET差%|PFP_N差%|", "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in all_gaps:
        def fmt(key: str) -> str:
            v = r[key]
            return "NA" if not isinstance(v, (int, float)) or not math.isfinite(float(v)) else f"{float(v):.3f}"
        lines.append(f"|{r['site']}|{r['year']}|{r['seed']}|{r['checkpoint']}|{r['winning_metrics']}|{fmt('gap_pct_yield')}|{fmt('gap_pct_WP_ET')}|{fmt('gap_pct_PFP_N')}|")
    lines += [
        "", "## 边界", "",
        "- YC seed0/2与LC seed0至少一项严格领先；FQ seed0为阴性对照。",
        "- 另外两项只报告与四基线最大值的差距；导师未定义‘接近’容差，因此不自动判定。",
        "- 图与CSV使用每个case的manifest绑定模型、日值、终值和图路径，防止数据图错配。",
        "- 本任务没有重训、没有重选checkpoint、没有重跑四基线。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "cases": len(manifests), "figures": len(figures), "training_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
