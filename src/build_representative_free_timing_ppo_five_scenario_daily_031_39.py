from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src import build_relaxed_success_five_scenario_daily_evidence_027_05 as old

OUT = ROOT / "benchmark_results" / "031_39_representative_free_timing_ppo_five_scenario_daily"
DOC = ROOT / "docs" / "031_39_representative_free_timing_ppo_five_scenario_daily_record.md"
SELECTED = ROOT / "benchmark_results" / "031_38_five_station_ppo_water_n_saving_summary" / "tables" / "031_38_selected_station_year_ppo_deltas_vs_official.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_clean_output() -> None:
    if OUT.exists():
        raise FileExistsError(f"Output directory already exists; remove or archive it manually if rerun is needed: {OUT}")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, values)) + " |" for values in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    ensure_clean_output()
    selected = pd.read_csv(SELECTED, keep_default_na=False)
    row = selected[(selected["station_code"].eq("LCA")) & (selected["year"].astype(str).eq("2010"))]
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one selected LCA2010 row, got {len(row)}")
    r = row.iloc[0]
    model_path = ROOT / str(r["model_path"])
    candidate_snapshot = ROOT / str(r["snapshot_path"])
    if not candidate_snapshot.exists():
        raise FileNotFoundError(candidate_snapshot)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    baseline_root = ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2" / "LC" / "readiness" / "baseline_runs"
    snapshots = {
        "null": baseline_root / "null" / "pdi_tmp_snapshot_eval",
        "recorded_farmer": baseline_root / "recorded_farmer" / "pdi_tmp_snapshot_eval",
        "dssat_auto": baseline_root / "dssat_auto" / "pdi_tmp_snapshot_eval",
        "official_extension_expert": baseline_root / "official_extension_expert" / "pdi_tmp_snapshot_eval",
        "rl_candidate": candidate_snapshot,
    }
    missing = [str(path) for path in snapshots.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    case = old.Case(
        site="LC",
        station="Luancheng",
        year=2010,
        seed=int(r["seed"]),
        checkpoint=int(r["checkpoint_step"]),
        model_path=model_path,
        snapshots=snapshots,
        selection_source=SELECTED,
        note="031_38 selected current free-timing MaskablePPO candidate; no training or checkpoint reselection in 031_39.",
    )
    daily, summary, checks = old.build_case(case, algorithm="MaskablePPO")
    checks_df = pd.DataFrame(checks)
    if not bool(checks_df["passed"].all()):
        raise RuntimeError("Daily evidence checks failed")

    daily_csv = OUT / "031_39_lca2010_current_free_timing_ppo_five_scenario_daily.csv"
    summary_csv = OUT / "031_39_lca2010_current_free_timing_ppo_five_scenario_summary.csv"
    checks_csv = OUT / "031_39_lca2010_daily_evidence_checks.csv"
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")

    old.FIG = OUT / "figures"
    old.FIG.mkdir(parents=True, exist_ok=True)
    daily_figs = old.plot_daily(daily, case, algorithm="MaskablePPO")

    # Reuse endpoint plotting by making a minimal ExistingPPOCase-like object.
    endpoint_case = old.ExistingPPOCase(
        site="LC",
        station="Luancheng",
        year=2010,
        seed=int(r["seed"]),
        checkpoint=int(r["checkpoint_step"]),
        model_path=model_path,
        baseline_path=summary_csv,
        checkpoint_summary_path=summary_csv,
        stage_actions_path=summary_csv,
        ppo_row_selector=("scenario", "rl_candidate"),
    )
    endpoint_figs = old.plot_ppo_endpoints(summary, endpoint_case)

    manifest = {
        "task": "031_39",
        "site": "LC",
        "station_code": "LCA",
        "year": 2010,
        "seed": int(r["seed"]),
        "checkpoint_step": int(r["checkpoint_step"]),
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": sha256(model_path),
        "candidate_snapshot": str(candidate_snapshot.relative_to(ROOT)).replace("\\", "/"),
        "training_calls": 0,
        "new_dssat_runs": 0,
        "daily_csv": str(daily_csv.relative_to(ROOT)).replace("\\", "/"),
        "summary_csv": str(summary_csv.relative_to(ROOT)).replace("\\", "/"),
        "checks_csv": str(checks_csv.relative_to(ROOT)).replace("\\", "/"),
        "figures": [str(p.relative_to(ROOT)).replace("\\", "/") for p in [*daily_figs, *endpoint_figs]],
    }
    (OUT / "031_39_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 031_39 representative free-timing PPO five-scenario daily record",
        "",
        "## Scope",
        "",
        "- No training.",
        "- No new DSSAT run.",
        "- Current 031-series free-timing MaskablePPO candidate for LCA2010.",
        "- Plot style reuses the 027_05 five-scenario daily process design.",
        "",
        "## Selected candidate",
        "",
        md_table(
            pd.DataFrame(
                [
                    {
                        "station_code": "LCA",
                        "site": "LC",
                        "year": 2010,
                        "seed": int(r["seed"]),
                        "checkpoint_step": int(r["checkpoint_step"]),
                        "final_grain_kg_ha": float(r["final_grain_kg_ha"]),
                        "irrigation_mm": float(r["irrigation_event_total_mm"]),
                        "nitrogen_kg_ha": float(r["nitrogen_event_total_kg_ha"]),
                        "winning_metrics": str(r.get("winning_metrics", "")),
                    }
                ]
            )
        ),
        "",
        "## Daily evidence checks",
        "",
        md_table(checks_df, 30),
        "",
        "## Outputs",
        "",
        md_table(pd.DataFrame({"path": manifest["figures"] + [manifest["daily_csv"], manifest["summary_csv"], manifest["checks_csv"]]}), 20),
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
