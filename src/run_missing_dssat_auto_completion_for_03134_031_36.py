from __future__ import annotations

import argparse
import json
import math
import re
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

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_extension_expert_baseline_018_03 as extension
import run_four_site_all_year_frozen_maskableppo_transfer_031_34 as transfer34
import run_missing_four_baseline_completion_for_03134_031_35 as complete35
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from ppo_safe_rendering import build_env_args, ensure_project_on_path
from run_fq_yc_new_cultivar_forward_screening_013_01 import set_management_for_treatment


CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_36_missing_dssat_auto_completion_for_03134.yaml"
OUT = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134"
DOC = ROOT / "docs" / "031_36_missing_dssat_auto_completion_for_03134_record.md"


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


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def target_missing_auto_years(meta: dict[str, Any], mode: str) -> pd.DataFrame:
    comp = pd.read_csv(ROOT / meta["template_aware_comparison_csv"], keep_default_na=False)
    target = str(meta.get("target_status", "baseline_incomplete_3rows"))
    miss = comp.loc[comp["baseline_comparison_status"].eq(target), ["station_code", "site", "year"]].drop_duplicates()
    miss["year"] = pd.to_numeric(miss["year"], errors="coerce").astype(int)
    if mode == "smoke":
        sm = meta["smoke"]
        miss = miss[miss["station_code"].eq(str(sm["station_code"])) & miss["year"].eq(int(sm["year"]))].copy()
    return miss.sort_values(["station_code", "year"]).reset_index(drop=True)


def build_env_config(run_config: dict[str, Any], meta: dict[str, Any], years: pd.DataFrame) -> dict[str, Any]:
    pool = pd.read_csv(ROOT / meta["scenario_pool_csv"])
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selection = pool.merge(years[["station_code", "year"]], on=["station_code", "year"], how="inner")
    selection["selected_for_train"] = False
    selection["selected_for_eval"] = True
    selection["selection_reason"] = "031_36_missing_dssat_auto_completion"
    return direct_ppo.build_env_config(run_config, selection)


def automatic_management_block(year: int) -> str:
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


def ensure_automatic_management(text: str, year: int) -> str:
    if "@  AUTOMATIC MANAGEMENT" in text:
        return text
    block = automatic_management_block(year)
    marker = "@N OUTPUTS"
    idx = text.find(marker)
    if idx < 0:
        return text.rstrip() + "\n" + block
    next_star = text.find("\n*", idx + 1)
    if next_star < 0:
        return text.rstrip() + "\n" + block
    return text[:next_star].rstrip() + "\n" + block + "\n" + text[next_star:]


def set_auto_treatment_one(text: str, year: int) -> str:
    text = ensure_automatic_management(text, year)
    try:
        return set_management_for_treatment(text, 1, "A", "A")
    except Exception:
        # Some rendered templates contain a slightly different management line spacing.
        lines = []
        changed = False
        in_management = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("@N MANAGEMENT") and "IRRIG" in stripped and "FERTI" in stripped:
                in_management = True
                lines.append(line)
                continue
            if in_management and re.match(r"^\s*1\s+MA\b", line):
                parts = line.split()
                if len(parts) >= 7:
                    parts[4] = "A"
                    parts[5] = "A"
                    lines.append(f"{int(parts[0]):2d} MA              R     {parts[4]}     {parts[5]}     R     M")
                    changed = True
                    continue
            if in_management and (stripped.startswith("@") or stripped.startswith("*")):
                in_management = False
            lines.append(line)
        if not changed:
            raise
        return "\n".join(lines) + "\n"


def make_auto_env(env_config: dict[str, Any], station: str, year: int, seed: int) -> Any:
    ensure_project_on_path()
    year_info = direct_ppo.find_year(env_config, station, int(year))
    env_args = build_env_args(
        station=station,
        year=int(year),
        planting_date=year_info["planting_date"],
        seed=int(seed),
        config=env_config,
        run_tag=f"{station}_{year}_031_36_dssat_auto",
        evaluation=True,
        mode=env_config.get("runtime", {}).get("mode", "all"),
    )
    template = Path(env_args["fileX_template_path"])
    text = template.read_text(encoding="utf-8", errors="replace")
    text = set_auto_treatment_one(text, int(year))
    template.write_text(text, encoding="utf-8")
    return extension.make_raw_env(env_args)


def real_action(raw_env: Any, irrigation: float = 0.0, nitrogen: float = 0.0) -> np.ndarray:
    return normalize_action(
        raw_env.formator.action_names,
        raw_env.formator.action_space_dict,
        {"amir": float(irrigation), "anfer": float(nitrogen)},
    )


def metrics_from_snapshot(snapshot: Path, final_yield: float) -> dict[str, Any]:
    rows = siteppo.parse_summary_out(snapshot / "Summary.OUT")
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = siteppo.num(row, "HWAM")
        if hwam is None:
            continue
        candidates.append((abs(float(hwam) - float(final_yield)), -idx, row))
    if not candidates:
        raise ValueError(f"No usable Summary.OUT row in {snapshot}")
    _, neg_idx, row = min(candidates, key=lambda item: (item[0], item[1]))
    ircm = float(siteppo.num(row, "IRCM") or 0.0)
    nicm = float(siteppo.num(row, "NICM") or 0.0)
    etcp = siteppo.num(row, "ETCP")
    ypem = siteppo.num(row, "YPEM")
    ypnam = siteppo.num(row, "YPNAM")
    if etcp is None or float(etcp) <= 0:
        raise ValueError(f"Invalid ETCP in {snapshot}: {etcp}")
    wp = float(ypem) * 0.1 if ypem is not None and float(ypem) >= 0 else float(final_yield) / float(etcp) / 10.0
    pfp = float(ypnam) if nicm > 0 and ypnam is not None and float(ypnam) >= 0 else math.nan
    return {
        "actual_irrigation_mm": ircm,
        "actual_nitrogen_kg_ha": nicm,
        "summary_irrigation_total": ircm,
        "summary_nitrogen_total": nicm,
        "etcp_mm": float(etcp),
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
        "summary_match_score": abs(float(siteppo.num(row, "HWAM") or final_yield) - float(final_yield)),
        "summary_row_index": int(-neg_idx),
    }


def eval_dssat_auto(run_config: dict[str, Any], env_config: dict[str, Any], meta: dict[str, Any], station: str, site: str, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = make_auto_env(env_config, station, int(year), int(meta["seed"]))
    weather = direct_ppo.weather_for_daily(run_config)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, int(year))["planting_date"])
        while not done and step_count < int(meta["max_steps"]):
            latest_pre = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap"), step_count + 1)
            dap = int(round(float(dap_raw))) if np.isfinite(float(dap_raw)) and float(dap_raw) > 0 else step_count + 1
            obs, reward, terminated, truncated, info = env.step(real_action(env, 0.0, 0.0))
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            rows.append(
                {
                    "station_code": station,
                    "site": site,
                    "year": int(year),
                    "scenario": "dssat_auto",
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "external_irrigation_action_mm": 0.0,
                    "external_nitrogen_action_kg_ha": 0.0,
                    "reward": float(reward),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(rows)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"DSSAT auto did not finish: {station}{year}")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        snapshot_tmp = siteppo.snapshot_from_env(env)
        snapshot = OUT / "snapshots" / station / str(year) / "dssat_auto"
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
        metrics = metrics_from_snapshot(snapshot, final_y)
        summary = {
            "station_code": station,
            "site": site,
            "year": int(year),
            "scenario": "dssat_auto",
            "grain_yield_kg_ha": final_y,
            "biomass_kg_ha": final_b,
            "max_water_stress": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nitrogen_stress": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "source_status": "generated_031_36_true_dssat_auto",
            "source_file": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            "requested_irrigation_mm": 0.0,
            "requested_nitrogen_kg_ha": 0.0,
            **metrics,
        }
        return daily, summary
    finally:
        env.close()


def dedupe_with_new_auto(existing: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    unified = pd.concat([existing, generated], ignore_index=True, sort=False)
    if unified.empty:
        return unified
    priority = {"generated_031_36_true_dssat_auto": 0, "generated_031_35_fixed_policy": 1, "reused_existing": 2}
    unified["_source_rank"] = unified["source_status"].map(priority).fillna(9)
    unified = unified.sort_values(["station_code", "year", "scenario", "_source_rank", "source_file"])
    unified = unified.drop_duplicates(["station_code", "year", "scenario"], keep="first")
    return unified.drop(columns=["_source_rank"])


def rebuild_comparison(meta: dict[str, Any], unified: pd.DataFrame, mode: str) -> pd.DataFrame:
    ppo = pd.read_csv(ROOT / meta["ppo_full_summary_csv"], keep_default_na=False)
    for col in ["final_grain_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg"]:
        if col in ppo.columns:
            ppo[col] = pd.to_numeric(ppo[col], errors="coerce")
    baseline_input = complete35.baseline_comparison_inputs(unified)
    compared = transfer34.add_baseline_comparison(ppo, baseline_input)
    compared.to_csv(OUT / "evaluation" / f"031_36_{mode}_ppo_vs_completed_template_aware_four_baselines.csv", index=False, encoding="utf-8-sig")
    return compared


def write_record(mode: str, missing: pd.DataFrame, generated: pd.DataFrame, manifest: pd.DataFrame, unified: pd.DataFrame, compared: pd.DataFrame) -> None:
    counts = compared.groupby(["station_code", "baseline_comparison_status"]).size().reset_index(name="n")
    ok = compared[compared["baseline_comparison_status"].eq("ok")]
    wins = ok.groupby("station_code").agg(
        compared_ok=("year", "count"),
        candidate_years=("year", "nunique"),
        any_metric_wins=("advisor_any_metric_strict_winner", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        yield_wins=("yield_strict_win", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        wp_wins=("wp_et_strict_win", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        pfp_wins=("pfp_n_strict_win", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
    ).reset_index() if not ok.empty else pd.DataFrame()
    lines = [
        "# 031_36 Missing true DSSAT-auto completion for 031_34/031_35 record",
        "",
        f"Mode: `{mode}`",
        "",
        "## Scope",
        "",
        "- No PPO/DQN training.",
        "- Only true `dssat_auto` gaps from 031_35 template-aware incomplete rows were generated.",
        "- Original MZX/WTH/SOL/CUL files were not modified; only rendered per-run FileX templates under 031_36 were edited.",
        "- `recorded_farmer_template_02705` remains a comparison-only transferred recorded-farmer surrogate where true yearly recorded farmer is unavailable.",
        "",
        "## Target missing auto station-years",
        "",
        md_table(missing, 120),
        "",
        "## DSSAT-auto coverage manifest",
        "",
        md_table(manifest.groupby(["station_code", "status"]).size().reset_index(name="n") if not manifest.empty else pd.DataFrame(), 80),
        "",
        "## Generated auto metric preview",
        "",
        md_table(generated[["station_code", "site", "year", "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "max_water_stress", "max_nitrogen_stress"]] if not generated.empty else pd.DataFrame(), 80),
        "",
        "## Rebuilt four-baseline comparison status",
        "",
        md_table(counts, 80),
        "",
        "## Wins among comparable rows",
        "",
        md_table(wins, 80),
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ensure_dirs()
    meta = load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    run_config = load_yaml(ROOT / meta["base_config"])
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    missing = target_missing_auto_years(meta, args.mode)
    missing.to_csv(OUT / "evaluation" / f"031_36_{args.mode}_missing_dssat_auto_station_years.csv", index=False, encoding="utf-8-sig")
    env_config = build_env_config(run_config, meta, missing)
    write_yaml(env_config, OUT / "configs" / f"031_36_{args.mode}_resolved_env_config.yaml")

    manifest_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    for row in missing.itertuples(index=False):
        station = str(row.station_code)
        site = str(row.site)
        year = int(row.year)
        try:
            daily, summary = eval_dssat_auto(run_config, env_config, meta, station, site, year)
            daily_frames.append(daily)
            summaries.append(summary)
            manifest_rows.append({"station_code": station, "site": site, "year": year, "scenario": "dssat_auto", "status": "generated_031_36_true_dssat_auto", "details": summary["source_file"]})
            pd.DataFrame(summaries).to_csv(OUT / "evaluation" / f"031_36_{args.mode}_generated_dssat_auto_summary_partial.csv", index=False, encoding="utf-8-sig")
        except Exception as exc:
            manifest_rows.append({"station_code": station, "site": site, "year": year, "scenario": "dssat_auto", "status": "failed", "details": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})

    generated = pd.DataFrame(summaries)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    generated.to_csv(OUT / "evaluation" / f"031_36_{args.mode}_generated_dssat_auto_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "evaluation" / f"031_36_{args.mode}_generated_dssat_auto_daily.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT / "evaluation" / f"031_36_{args.mode}_dssat_auto_coverage_manifest.csv", index=False, encoding="utf-8-sig")

    existing = pd.read_csv(ROOT / meta["03135_unified_baseline_csv"], keep_default_na=False)
    unified = dedupe_with_new_auto(existing, generated)
    unified.to_csv(OUT / "evaluation" / f"031_36_{args.mode}_completed_template_aware_unified_baseline_summary.csv", index=False, encoding="utf-8-sig")
    compared = rebuild_comparison(meta, unified, args.mode)
    write_record(args.mode, missing, generated, manifest, unified, compared)
    result = {
        "task": "031_36",
        "mode": args.mode,
        "target_rows": int(len(missing)),
        "generated_rows": int(len(generated)),
        "failed_rows": int((manifest["status"] == "failed").sum()) if not manifest.empty else 0,
        "comparison_rows": int(len(compared)),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / f"031_36_{args.mode}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not compared.empty:
        print(compared.groupby(["station_code", "baseline_comparison_status"]).size().reset_index(name="n").to_string(index=False))


if __name__ == "__main__":
    main()
