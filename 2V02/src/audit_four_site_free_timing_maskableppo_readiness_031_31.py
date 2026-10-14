from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_31_four_site_free_timing_maskableppo_readiness.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def station_year_inventory(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, group in pool.groupby("station_code"):
        years = sorted(pd.to_numeric(group["year"], errors="coerce").dropna().astype(int).unique().tolist())
        rows.append(
            {
                "station_code": station,
                "available_year_count": len(years),
                "first_year": min(years) if years else "",
                "last_year": max(years) if years else "",
                "available_years": ",".join(map(str, years)),
                "has_any_weather_after_2000": any(year >= 2000 for year in years),
            }
        )
    return pd.DataFrame(rows).sort_values("station_code")


def smoke_status(smoke: pd.DataFrame, station: str, year: int) -> dict[str, Any]:
    if smoke.empty:
        return {"smoke_rows": 0, "smoke_ok_rows": 0, "smoke_final_grnwt": "", "smoke_notes": "smoke csv missing or empty"}
    subset = smoke[
        smoke["station_code"].astype(str).eq(station)
        & pd.to_numeric(smoke["year"], errors="coerce").astype("Int64").eq(int(year))
    ].copy()
    ok = subset[subset.get("run_status", "").astype(str).eq("ok")] if not subset.empty and "run_status" in subset else subset
    final = pd.to_numeric(ok.get("final_grnwt", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "smoke_rows": int(len(subset)),
        "smoke_ok_rows": int(len(ok)),
        "smoke_final_grnwt": float(final.iloc[0]) if len(final) else "",
        "smoke_notes": "" if len(ok) else "no successful 031_20 smoke row",
    }


def main() -> None:
    cfg = load_yaml(CONFIG)
    out = ROOT / cfg["output_root"]
    if out.exists() and any(out.glob("evaluation/*.csv")):
        raise FileExistsError(f"Existing 031_31 outputs found, refusing overwrite: {out}")
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs").mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(ROOT / cfg["scenario_pool_csv"])
    smoke_path = ROOT / cfg["smoke_eval_csv"]
    smoke = pd.read_csv(smoke_path, keep_default_na=False) if smoke_path.exists() else pd.DataFrame()
    inventory = station_year_inventory(pool)

    readiness_rows = []
    for station, year in {**cfg.get("target_stations", {}), **cfg.get("reference_completed_station", {})}.items():
        year = int(year)
        pool_row = pool[
            pool["station_code"].astype(str).eq(station)
            & pd.to_numeric(pool["year"], errors="coerce").astype("Int64").eq(year)
        ]
        weather_file = str(pool_row["weather_file"].iloc[0]) if len(pool_row) else ""
        weather_path = ROOT / weather_file.replace("\\", "/") if weather_file else None
        smoke_info = smoke_status(smoke, station, year)
        is_reference = station in cfg.get("reference_completed_station", {})
        ready = bool(len(pool_row) == 1 and weather_path is not None and weather_path.exists() and smoke_info["smoke_ok_rows"] > 0)
        readiness_rows.append(
            {
                "station_code": station,
                "proposed_train_year": year,
                "role": "reference_completed" if is_reference else "target_next",
                "scenario_pool_row_count": int(len(pool_row)),
                "weather_file": weather_file,
                "weather_exists": bool(weather_path.exists()) if weather_path is not None else False,
                **smoke_info,
                "ready_for_checkpoint_selection": ready,
            }
        )
    readiness = pd.DataFrame(readiness_rows).sort_values(["role", "station_code"])

    plan_rows = []
    for row in readiness.itertuples(index=False):
        if row.role == "reference_completed":
            next_task = "already_completed_SYA_031_27_to_031_30"
        elif bool(row.ready_for_checkpoint_selection):
            next_task = "ready_for_checkpoint_selection"
        elif not bool(row.weather_exists):
            next_task = "blocked_missing_weather"
        else:
            next_task = "blocked_failed_or_missing_smoke"
        plan_rows.append(
            {
                "station_code": row.station_code,
                "proposed_train_year": int(row.proposed_train_year),
                "next_task": next_task,
                "recommended_order": "HLA,FQA,LCA,YCA" if next_task == "ready_for_checkpoint_selection" else "",
                "notes": "Use same frozen config as SYA; no tuning before first cross-station pass." if next_task == "ready_for_checkpoint_selection" else "",
            }
        )
    plan = pd.DataFrame(plan_rows)

    inventory.to_csv(out / "evaluation" / "031_31_station_year_inventory.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(out / "evaluation" / "031_31_train_year_readiness.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(out / "evaluation" / "031_31_next_task_plan.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 031_31 Four-site free-timing MaskablePPO readiness record",
        "",
        "## Scope",
        "",
        "- No training and no DSSAT rerun.",
        "- This audit prepares HLA/FQA/LCA/YCA for the same free-timing discrete MaskablePPO workflow used in SYA.",
        "- The SYA result is treated as completed reference, not retuned.",
        "",
        "## Station-year inventory",
        "",
        df_to_markdown(inventory, 20),
        "",
        "## Proposed train-year readiness",
        "",
        df_to_markdown(readiness, 20),
        "",
        "## Next task plan",
        "",
        df_to_markdown(plan, 20),
    ]
    doc = ROOT / "docs" / "031_31_four_site_free_timing_maskableppo_readiness_record.md"
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "ready_targets": int(readiness[readiness["role"].eq("target_next")]["ready_for_checkpoint_selection"].sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
