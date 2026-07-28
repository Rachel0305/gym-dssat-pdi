from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_missing_dssat_auto_completion_for_03134_031_36 as auto36


TASK = "032_18"
OUT = ROOT / "benchmark_results" / "032_18_lc_missing_dssat_auto_daily_completion"
DOC = ROOT / "docs" / "032_18_lc_missing_dssat_auto_daily_completion_record.md"
PROMPT = ROOT / "prompts" / "032_18_lc_missing_dssat_auto_daily_completion.md"
CONFIG_03136 = ROOT / "experiments" / "ppo_observed_years" / "config_031_36_missing_dssat_auto_completion_for_03134.yaml"
TARGETS = pd.DataFrame(
    [
        {"station_code": "LCA", "site": "LC", "year": 2008},
        {"station_code": "LCA", "site": "LC", "year": 2009},
        {"station_code": "LCA", "site": "LC", "year": 2011},
    ]
)


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, values)) + " |" for values in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def write_record(generated: pd.DataFrame, manifest: pd.DataFrame, daily: pd.DataFrame) -> None:
    status_counts = manifest.groupby("status").size().reset_index(name="n") if not manifest.empty else pd.DataFrame()
    daily_counts = daily.groupby(["year", "scenario"]).size().reset_index(name="daily_rows") if not daily.empty else pd.DataFrame()
    lines = [
        "# 032_18 LC missing DSSAT-auto daily completion record",
        "",
        "## Status",
        "",
        f"- Target rows: {len(TARGETS)}.",
        f"- Generated rows: {len(generated)}.",
        f"- Failed rows: {int((manifest['status'] == 'failed').sum()) if not manifest.empty and 'status' in manifest else 0}.",
        "- PPO/DQN training: 0.",
        "- Candidate model reselection: none.",
        "- Original DSSAT input files were not modified; rendered per-run templates and snapshots are stored under 032_18.",
        "",
        "## Target station-years",
        "",
        md_table(TARGETS),
        "",
        "## Manifest",
        "",
        md_table(manifest),
        "",
        "## Status counts",
        "",
        md_table(status_counts),
        "",
        "## Generated summary",
        "",
        md_table(generated[["station_code", "site", "year", "scenario", "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "max_water_stress", "max_nitrogen_stress"]] if not generated.empty else pd.DataFrame()),
        "",
        "## Daily row counts",
        "",
        md_table(daily_counts),
        "",
        "## Next use",
        "",
        "- 032_17 should read `benchmark_results/032_18_lc_missing_dssat_auto_daily_completion/evaluation/032_18_generated_dssat_auto_daily.csv` as a supplemental auto daily source.",
        "- This task only fills missing baseline evidence; it does not alter PPO candidate outputs.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    shutil.copyfile(CONFIG_03136, OUT / "configs" / CONFIG_03136.name)

    meta = load_yaml(CONFIG_03136)
    run_config = load_yaml(ROOT / meta["base_config"])
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")

    auto36.OUT = OUT
    env_config = auto36.build_env_config(run_config, meta, TARGETS)
    write_yaml(env_config, OUT / "configs" / "032_18_resolved_env_config.yaml")

    manifest_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    for row in TARGETS.itertuples(index=False):
        station = str(row.station_code)
        site = str(row.site)
        year = int(row.year)
        try:
            daily, summary = auto36.eval_dssat_auto(run_config, env_config, meta, station, site, year)
            summary["source_status"] = "generated_032_18_true_dssat_auto"
            summary["source_file"] = summary["source_file"].replace("031_36_missing_dssat_auto_completion_for_03134", "032_18_lc_missing_dssat_auto_daily_completion")
            daily_frames.append(daily)
            summaries.append(summary)
            manifest_rows.append(
                {
                    "station_code": station,
                    "site": site,
                    "year": year,
                    "scenario": "dssat_auto",
                    "status": "generated_032_18_true_dssat_auto",
                    "details": summary["source_file"],
                }
            )
            pd.DataFrame(summaries).to_csv(OUT / "evaluation" / "032_18_generated_dssat_auto_summary_partial.csv", index=False, encoding="utf-8-sig")
            print(f"generated {station}{year} dssat_auto", flush=True)
        except Exception as exc:
            manifest_rows.append(
                {
                    "station_code": station,
                    "site": site,
                    "year": year,
                    "scenario": "dssat_auto",
                    "status": "failed",
                    "details": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAILED {station}{year}: {type(exc).__name__}: {exc}", flush=True)

    generated = pd.DataFrame(summaries)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    generated.to_csv(OUT / "evaluation" / "032_18_generated_dssat_auto_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "evaluation" / "032_18_generated_dssat_auto_daily.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT / "evaluation" / "032_18_dssat_auto_coverage_manifest.csv", index=False, encoding="utf-8-sig")
    write_record(generated, manifest, daily)

    result = {
        "task": TASK,
        "target_rows": int(len(TARGETS)),
        "generated_rows": int(len(generated)),
        "failed_rows": int((manifest["status"] == "failed").sum()) if not manifest.empty else 0,
        "daily_rows": int(len(daily)),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "032_18_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["failed_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
