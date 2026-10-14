from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from ppo_safe_rendering import build_env_args, ensure_project_on_path, yyddd
from run_fq_yc_new_cultivar_forward_screening_013_01 import set_management_for_treatment
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_extension_expert_baseline_018_03 as extension
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as baseline_tools


CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_28_sy_all_year_four_baseline_completion.yaml"
SOURCE_SY_MZX = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY" / "CNSY1201.MZX"
SOURCE_RECORDED_TRNO = {2012: 1, 2014: 2, 2015: 3}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


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


def real_action(raw_env: Any, irrigation: float, nitrogen: float) -> np.ndarray:
    return normalize_action(
        raw_env.formator.action_names,
        raw_env.formator.action_space_dict,
        {"amir": float(irrigation), "anfer": float(nitrogen)},
    )


def single_summary_metrics(snapshot: Path, final_yield: float) -> dict[str, Any]:
    rows = baseline_tools.parse_summary_out(snapshot / "Summary.OUT")
    if not rows:
        raise ValueError(f"No Summary.OUT rows found in {snapshot}")
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = baseline_tools.num(row, "HWAM")
        if hwam is None:
            continue
        candidates.append((abs(float(hwam) - float(final_yield)), -idx, row))
    if not candidates:
        raise ValueError(f"No usable Summary.OUT HWAM row in {snapshot}")
    _, neg_idx, row = min(candidates, key=lambda item: (item[0], item[1]))
    ircm = float(baseline_tools.num(row, "IRCM") or 0.0)
    nicm = float(baseline_tools.num(row, "NICM") or 0.0)
    etcp = baseline_tools.num(row, "ETCP")
    ypem = baseline_tools.num(row, "YPEM")
    ypnam = baseline_tools.num(row, "YPNAM")
    if etcp is None or float(etcp) <= 0:
        raise ValueError(f"Invalid ETCP in {snapshot}: {etcp}")
    wp = float(ypem) * 0.1 if ypem is not None and float(ypem) >= 0 else float(final_yield) / float(etcp) / 10.0
    pfp = float(ypnam) if nicm > 0 and ypnam is not None and float(ypnam) >= 0 else math.nan
    return {
        "summary_row_index": int(-neg_idx),
        "grain_yield_kg_ha": float(baseline_tools.num(row, "HWAM") or final_yield),
        "etcp_mm": float(etcp),
        "actual_irrigation_mm": ircm,
        "actual_nitrogen_kg_ha": nicm,
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
        "summary_yield_abs_error_vs_env_grnwt": abs(float(baseline_tools.num(row, "HWAM") or final_yield) - float(final_yield)),
    }


def sy_years(env_config: dict[str, Any], station: str) -> list[dict[str, Any]]:
    years = []
    for item in env_config.get("observed_years", {}).get(station, []):
        year = int(item["year"])
        if year >= 2000:
            weather_text = str(item.get("weather_file", "")).replace("\\", "/")
            weather = ROOT / weather_text
            years.append({**item, "year": year, "weather_exists": weather.exists()})
    return sorted(years, key=lambda x: int(x["year"]))


def expert_schedule_for_sy(year: int, region: str) -> dict[int, dict[str, float]]:
    schedule = extension.build_region_schedule()
    schedule = schedule[schedule["region"].eq(region)].copy()
    schedule.insert(0, "site", "SY")
    schedule.insert(1, "station", "Shenyang")
    schedule.insert(2, "year", int(year))
    return extension.split_irrigation_events(schedule)


def yyddd_to_timestamp(yyddd_text: str) -> pd.Timestamp:
    text = str(yyddd_text).strip()
    if len(text) != 5 or not text.isdigit():
        raise ValueError(f"Invalid YYDDD date: {yyddd_text}")
    year = 2000 + int(text[:2])
    doy = int(text[2:])
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)


def extract_recorded_template_events() -> pd.DataFrame:
    text = SOURCE_SY_MZX.read_text(encoding="latin-1", errors="ignore")
    planting: dict[int, pd.Timestamp] = {}
    in_planting = False
    in_irrigation = False
    in_fertilizer = False
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("*"):
            in_planting = line.startswith("*PLANTING")
            in_irrigation = line.startswith("*IRRIGATION")
            in_fertilizer = line.startswith("*FERTILIZERS")
            continue
        if line.startswith("@") or not line.strip():
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        trno = int(parts[0])
        source_year = next((year for year, source_trno in SOURCE_RECORDED_TRNO.items() if source_trno == trno), None)
        if source_year is None:
            continue
        if in_planting and len(parts) >= 2:
            planting[source_year] = yyddd_to_timestamp(parts[1])
        elif in_irrigation and len(parts) >= 4:
            if len(parts[1]) != 5 or not parts[1].isdigit():
                continue
            try:
                amount = float(parts[3])
            except ValueError:
                amount = 0.0
            if amount > 0:
                rows.append(
                    {
                        "source_year": source_year,
                        "event_type": "irrigation",
                        "source_yyddd": parts[1],
                        "amount": amount,
                    }
                )
        elif in_fertilizer and len(parts) >= 6:
            if len(parts[1]) != 5 or not parts[1].isdigit():
                continue
            try:
                amount = float(parts[5])
            except ValueError:
                amount = 0.0
            if amount > 0:
                rows.append(
                    {
                        "source_year": source_year,
                        "event_type": "nitrogen",
                        "source_yyddd": parts[1],
                        "amount": amount,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["source_date"] = out["source_yyddd"].map(yyddd_to_timestamp)
    out["source_planting_date"] = out["source_year"].map(planting)
    out["source_dap"] = (out["source_date"] - out["source_planting_date"]).dt.days + 1
    out["source_dap_raw"] = out["source_dap"]
    out["source_dap"] = out["source_dap"].clip(lower=0)
    out["dap_clamped_to_episode_min"] = out["source_dap"] != out["source_dap_raw"]
    grouped = (
        out.groupby(["source_year", "event_type", "source_dap", "source_dap_raw", "dap_clamped_to_episode_min"], as_index=False)["amount"]
        .sum()
        .sort_values(["source_year", "source_dap", "event_type"])
    )
    return grouped


def recorded_template_schedule(source_year: int, templates: pd.DataFrame) -> dict[int, dict[str, float]]:
    sub = templates[templates["source_year"].eq(int(source_year))]
    actions: dict[int, dict[str, float]] = {}
    for row in sub.itertuples(index=False):
        dap = int(row.source_dap)
        actions.setdefault(dap, {"amir": 0.0, "anfer": 0.0})
        if str(row.event_type) == "irrigation":
            actions[dap]["amir"] += float(row.amount)
        elif str(row.event_type) == "nitrogen":
            actions[dap]["anfer"] += float(row.amount)
    return actions


def automatic_management_block(year: int, planting_date: str) -> str:
    # Uses the same SY automatic-management parameter pattern as CNSY1201.MZX.
    yy001 = f"{int(year) % 100:02d}001"
    return (
        "\n@  AUTOMATIC MANAGEMENT\n"
        "@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN\n"
        f" 1 PL          {yy001} {yy001}    40   100    30    40    10\n"
        "@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF\n"
        " 1 IR             30    50   100 GS000 IR001    10     1\n"
        "@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF\n"
        " 1 NI             30    50    25 FE001 GS000\n"
        "@N RESIDUES    RIPCN RTIME RIDEP\n"
        " 1 RE            100     1    20\n"
        "@N HARVEST     HFRST HLAST HPCNP HPCNR\n"
        " 1 HA              0 01001   100     0\n"
    )


def ensure_automatic_management(text: str, year: int, planting_date: str) -> str:
    if "@  AUTOMATIC MANAGEMENT" in text:
        return text
    block = automatic_management_block(year, planting_date)
    marker = "@N OUTPUTS"
    idx = text.find(marker)
    if idx < 0:
        return text + block
    next_star = text.find("\n*", idx + 1)
    if next_star < 0:
        return text.rstrip() + "\n" + block
    return text[:next_star].rstrip() + "\n" + block + "\n" + text[next_star:]


def make_raw_env_for_scenario(
    env_config: dict[str, Any],
    out_root: Path,
    station: str,
    year_info: dict[str, Any],
    scenario: str,
    seed: int,
) -> Any:
    ensure_project_on_path()
    year = int(year_info["year"])
    run_tag = f"031_28_{scenario}"
    env_args = build_env_args(
        station=station,
        year=year,
        planting_date=str(year_info["planting_date"]),
        seed=seed,
        config={**env_config, "paths": {**env_config["paths"], "output_root": str(out_root.relative_to(ROOT))}},
        run_tag=run_tag,
        evaluation=True,
        mode=env_config.get("runtime", {}).get("mode", "all"),
    )
    template = Path(env_args["fileX_template_path"])
    text = template.read_text(encoding="utf-8", errors="replace")
    if scenario == "dssat_auto":
        text = ensure_automatic_management(text, year, str(year_info["planting_date"]))
        text = set_management_for_treatment(text, 1, "A", "A")
        template.write_text(text, encoding="utf-8")
    elif scenario == "null" or scenario == "official_extension_expert" or scenario.startswith("recorded_farmer_template_"):
        text = set_management_for_treatment(text, 1, "L", "L")
        template.write_text(text, encoding="utf-8")
    else:
        raise ValueError(scenario)
    raw_env = extension.make_raw_env(env_args)
    return raw_env


def run_generated_scenario(
    env_config: dict[str, Any],
    out_root: Path,
    station: str,
    year_info: dict[str, Any],
    scenario: str,
    seed: int,
    max_steps: int,
    region: str,
    recorded_templates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    year = int(year_info["year"])
    raw_env = make_raw_env_for_scenario(env_config, out_root, station, year_info, scenario, seed)
    daily_rows: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    if scenario == "official_extension_expert":
        schedule = expert_schedule_for_sy(year, region)
    elif scenario.startswith("recorded_farmer_template_"):
        if recorded_templates is None:
            raise ValueError("recorded_templates is required for recorded template scenarios")
        source_year = int(scenario.rsplit("_", 1)[-1])
        schedule = recorded_template_schedule(source_year, recorded_templates)
    else:
        schedule = {}
    fired: set[int] = set()
    try:
        obs, info = raw_env.reset()
        for step in range(max_steps):
            state = latest_observation_dict(raw_env, obs, info)
            dap_raw = scalar(state.get("dap"), step + 1)
            dap = int(round(float(dap_raw))) if np.isfinite(float(dap_raw)) else step + 1
            if schedule and dap in schedule and dap not in fired:
                request = schedule[dap]
                fired.add(dap)
            else:
                request = {"amir": 0.0, "anfer": 0.0}
            daily_rows.append(
                {
                    "site": "SY",
                    "station_code": station,
                    "year": year,
                    "scenario": scenario,
                    "step": step,
                    "dap": dap,
                    "grnwt": float(scalar(state.get("grnwt"), math.nan)),
                    "topwt": float(scalar(state.get("topwt"), math.nan)),
                    "swfac": float(scalar(state.get("swfac"), math.nan)),
                    "nstres": float(scalar(state.get("nstres"), math.nan)),
                    "irrigation_action_mm": float(request.get("amir", 0.0)),
                    "nitrogen_action_kg_ha": float(request.get("anfer", 0.0)),
                }
            )
            obs, _, terminated, truncated, info = raw_env.step(
                real_action(raw_env, request.get("amir", 0.0), request.get("anfer", 0.0))
            )
            final_state = latest_observation_dict(raw_env, obs, info)
            if terminated or truncated:
                break
        else:
            raise RuntimeError(f"{station}{year}/{scenario}: did not terminate within {max_steps} steps")
        tmp = baseline_tools.snapshot_from_env(raw_env)
        snapshot = out_root / "snapshots" / station / str(year) / scenario
        if snapshot.exists():
            raise FileExistsError(f"Snapshot already exists, refusing overwrite: {snapshot}")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp, snapshot)
    finally:
        raw_env.close()
    final_yield = float(scalar(final_state.get("grnwt"), math.nan))
    final_biomass = float(scalar(final_state.get("topwt"), math.nan))
    daily = pd.DataFrame(daily_rows)
    metrics = single_summary_metrics(snapshot, final_yield)
    summary = {
        "site": "SY",
        "station_code": station,
        "year": year,
        "scenario": scenario,
        "source_status": "generated",
        "run_status": "ok",
        "planting_date": str(year_info["planting_date"]),
        "weather_file": str(year_info.get("weather_file", "")),
        "final_env_grnwt_kg_ha": final_yield,
        "final_env_topwt_kg_ha": final_biomass,
        "max_water_stress": float(daily["swfac"].max(skipna=True)),
        "mean_water_stress": float(daily["swfac"].mean(skipna=True)),
        "max_nitrogen_stress": float(daily["nstres"].max(skipna=True)),
        "mean_nitrogen_stress": float(daily["nstres"].mean(skipna=True)),
        "water_stress_days_gt_0p05": int((daily["swfac"] > 0.05).sum()),
        "nitrogen_stress_days_gt_0p05": int((daily["nstres"] > 0.05).sum()),
        "snapshot_path": str(snapshot.relative_to(ROOT)),
        **metrics,
    }
    return daily, summary


def existing_recorded_rows(config: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / config["existing_baseline_csv"]
    if not path.exists():
        return pd.DataFrame()
    source = pd.read_csv(path, keep_default_na=False)
    station_label = config.get("station_label_for_existing", "SY")
    mask = source["site"].astype(str).eq(station_label) & source["scenario"].astype(str).isin(
        ["recorded_farmer", "recorded"]
    )
    rows = source[mask].copy()
    if rows.empty:
        return rows
    rows["scenario"] = "recorded_farmer"
    rows["source_status"] = "reused_028_02"
    return rows


def run(config_path: Path, smoke: bool = False) -> None:
    config = load_yaml(config_path)
    task_id = str(config.get("task_id", "031_28"))
    out_root = ROOT / config["output_root"]
    if out_root.exists() and any(out_root.glob("evaluation/*.csv")):
        raise FileExistsError(f"Existing 031_28 outputs found, refusing overwrite: {out_root}")
    for sub in ["configs", "evaluation", "snapshots", "reports"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    env_config = load_yaml(ROOT / config["source_env_config"])
    write_yaml(env_config, out_root / "configs" / f"{task_id}_resolved_env_config.yaml")
    years = sy_years(env_config, config["station_code"])
    if smoke:
        years = [year for year in years if int(year["year"]) == 2005][:1]
    recorded_templates = extract_recorded_template_events()
    recorded_templates.to_csv(out_root / "evaluation" / f"{task_id}_recorded_template_events.csv", index=False, encoding="utf-8-sig")
    summaries: list[dict[str, Any]] = []
    all_daily: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for year_info in years:
        year = int(year_info["year"])
        if not bool(year_info["weather_exists"]):
            for scenario in config["scenarios_to_generate"]:
                manifest_rows.append(
                    {
                        "site": "SY",
                        "station_code": config["station_code"],
                        "year": year,
                        "scenario": scenario,
                        "coverage_status": "blocked_missing_weather",
                        "details": year_info.get("weather_file", ""),
                    }
                )
            continue
        for scenario in config["scenarios_to_generate"]:
            try:
                daily, summary = run_generated_scenario(
                    env_config=env_config,
                    out_root=out_root,
                    station=config["station_code"],
                    year_info=year_info,
                    scenario=scenario,
                    seed=int(config["seed"]),
                    max_steps=int(config["max_steps"]),
                    region=str(config["expert_region"]),
                    recorded_templates=recorded_templates,
                )
                summary["source_status"] = f"generated_{task_id}"
                summaries.append(summary)
                all_daily.append(daily)
                manifest_rows.append(
                    {
                        "site": "SY",
                        "station_code": config["station_code"],
                        "year": year,
                        "scenario": scenario,
                        "coverage_status": f"generated_{task_id}",
                        "details": summary["snapshot_path"],
                    }
                )
            except Exception as exc:
                manifest_rows.append(
                    {
                        "site": "SY",
                        "station_code": config["station_code"],
                        "year": year,
                        "scenario": scenario,
                        "coverage_status": "failed",
                        "details": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    existing_recorded = existing_recorded_rows(config)
    existing_recorded.to_csv(out_root / "evaluation" / "031_28_existing_reused_baselines.csv", index=False, encoding="utf-8-sig")
    recorded_years = set(pd.to_numeric(existing_recorded.get("year", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).tolist())
    for year_info in years:
        year = int(year_info["year"])
        if not any(str(s).startswith("recorded_farmer_template_") for s in config["scenarios_to_generate"]):
            manifest_rows.append(
                {
                    "site": "SY",
                    "station_code": config["station_code"],
                    "year": year,
                    "scenario": "recorded_farmer",
                    "coverage_status": "reused_028_02" if year in recorded_years else "blocked_missing_recorded_management_source",
                    "details": "Existing recorded-farmer management evidence reused from 028_02" if year in recorded_years else "No all-year recorded management source was found; not reconstructed.",
                }
            )
    summary_df = pd.DataFrame(summaries).sort_values(["year", "scenario"]) if summaries else pd.DataFrame()
    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows).sort_values(["year", "scenario"])
    summary_df.to_csv(out_root / "evaluation" / f"{task_id}_sy_baseline_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(out_root / "evaluation" / f"{task_id}_sy_baseline_daily.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(out_root / "evaluation" / f"{task_id}_sy_baseline_coverage_manifest.csv", index=False, encoding="utf-8-sig")

    generated_counts = manifest.groupby(["scenario", "coverage_status"]).size().reset_index(name="n")
    lines = [
        f"# {task_id} SY all-year baseline completion record",
        "",
        "## Scope",
        "",
        "- No PPO/DQN training was run.",
        "- 031_27 frozen PPO candidates were not changed.",
        "- Generated baselines use the same SYA all-year rendered DSSAT environment as 031_27.",
        "- True recorded-farmer management is not reconstructed where original management evidence is unavailable.",
        "- Recorded-template scenarios, if present, are counterfactual transfers of the 2012/2014/2015 recorded schedules.",
        "",
        "## Coverage summary",
        "",
        df_to_markdown(generated_counts, 80),
        "",
        "## Generated baseline metric preview",
        "",
        df_to_markdown(
            summary_df[
                [
                    "year",
                    "scenario",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "max_water_stress",
                    "max_nitrogen_stress",
                ]
            ]
            if not summary_df.empty
            else pd.DataFrame(),
            80,
        ),
        "",
        "## Interpretation boundary",
        "",
        f"{task_id} generated the requested scenarios listed in the manifest. Any `recorded_farmer_template_*` scenario is a transferred historical-management template, not a real recorded-farmer observation for the target year.",
    ]
    doc = ROOT / "docs" / f"{task_id}_sy_auto_and_recorded_template_completion_record.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "smoke": smoke, "summary_rows": len(summary_df), "daily_rows": len(daily_df)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()
