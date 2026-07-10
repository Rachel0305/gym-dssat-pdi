from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
from run_hla_official_reward_restart_smoke import (
    install_official_reward_module,
    parse_events as parse_hla_events,
    prepare_case_at as prepare_hla_case_at,
)

import run_hla_unified_dqn_long_train_015_09 as hla_base
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_base


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_03_multisite_extension_expert_baseline_record.md"
SCHEDULE_PATH = OUT_DIR / "018_03_extension_expert_schedule.csv"
SCENARIO = "extension_expert_fixed_dap"
MAX_SINGLE_IRRIGATION_MM = 50.0


CASES = [
    {"site": "HLA", "station": "Hailun", "year": 2010, "region": "northeast_greatwall_spring_maize"},
    {"site": "SY", "station": "Shenyang", "year": 2014, "region": "northeast_greatwall_spring_maize"},
    {"site": "YC", "station": "Yucheng", "year": 2014, "region": "huanghuai_fenwei_summer_maize"},
    {"site": "FQ", "station": "Fengqiu", "year": 2016, "region": "huanghuai_fenwei_summer_maize"},
    {"site": "LC", "station": "Luancheng", "year": 2010, "region": "huanghuai_fenwei_summer_maize"},
]


def midpoint(low: float, high: float) -> float:
    return (float(low) + float(high)) / 2.0


def kg_mu_to_kg_ha(value: float) -> float:
    return float(value) * 15.0


def fang_mu_to_mm(value: float) -> float:
    return float(value) * 1.5


def build_region_schedule() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    northeast = [
        ("sowing_base", "播种期/基肥", 0, (7.0, 8.0), (20.0, 30.0)),
        ("small_bell", "小喇叭口期", 30, (2.0, 2.5), (25.0, 35.0)),
        ("large_bell", "大喇叭口期", 50, (4.0, 5.0), (35.0, 40.0)),
        ("tasseling_silking", "抽雄散粉期", 65, (1.5, 2.0), (35.0, 40.0)),
        ("early_grain_filling", "灌浆初期", 85, (2.0, 2.5), (25.0, 30.0)),
        ("late_milk", "乳熟末期", 110, (1.5, 2.0), (15.0, 25.0)),
    ]
    huanghuai = [
        ("emergence_water", "出苗水", 7, (5.0, 6.0), (10.0, 20.0)),
        ("small_bell", "小喇叭口期", 30, (2.5, 3.0), (20.0, 30.0)),
        ("large_bell", "大喇叭口期", 45, (3.5, 4.0), (30.0, 35.0)),
        ("tasseling_silking", "抽雄散粉期", 60, (2.0, 2.5), (30.0, 35.0)),
        ("grain_filling", "灌浆期", 80, (2.0, 2.5), (25.0, 30.0)),
        ("milk_stage", "乳熟期", 100, (0.0, 0.0), (15.0, 25.0)),
    ]
    for region, source_table, stages in [
        ("northeast_greatwall_spring_maize", "表1 东北及长城沿线春玉米区", northeast),
        ("huanghuai_fenwei_summer_maize", "表3 华北黄淮和汾渭平原夏玉米区", huanghuai),
    ]:
        for code, stage, dap, n_range, i_range in stages:
            rows.append(
                {
                    "region": region,
                    "source_table": source_table,
                    "stage_code": code,
                    "stage_cn": stage,
                    "dap": dap,
                    "n_kg_mu_low": n_range[0],
                    "n_kg_mu_high": n_range[1],
                    "n_kg_ha_mid": kg_mu_to_kg_ha(midpoint(*n_range)),
                    "irrigation_fang_mu_low": i_range[0],
                    "irrigation_fang_mu_high": i_range[1],
                    "irrigation_mm_mid": fang_mu_to_mm(midpoint(*i_range)),
                }
            )
    return pd.DataFrame(rows)


def build_case_schedule() -> pd.DataFrame:
    region_schedule = build_region_schedule()
    out_rows = []
    for case in CASES:
        sub = region_schedule[region_schedule["region"].eq(case["region"])].copy()
        sub.insert(0, "site", case["site"])
        sub.insert(1, "station", case["station"])
        sub.insert(2, "year", case["year"])
        out_rows.append(sub)
    schedule = pd.concat(out_rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(SCHEDULE_PATH, index=False, encoding="utf-8-sig")
    return schedule


def split_irrigation_events(schedule: pd.DataFrame) -> dict[int, dict[str, float]]:
    actions: dict[int, dict[str, float]] = {}
    for _, row in schedule.sort_values("dap").iterrows():
        dap = int(row["dap"])
        irrigation = float(row["irrigation_mm_mid"])
        nitrogen = float(row["n_kg_ha_mid"])
        actions.setdefault(dap, {"amir": 0.0, "anfer": 0.0})
        actions[dap]["anfer"] += nitrogen
        remaining = irrigation
        offset = 0
        while remaining > 1e-9:
            amount = min(MAX_SINGLE_IRRIGATION_MM, remaining)
            action_dap = dap + offset
            actions.setdefault(action_dap, {"amir": 0.0, "anfer": 0.0})
            actions[action_dap]["amir"] += amount
            remaining -= amount
            offset += 1
    return actions


def make_raw_env(env_args: dict[str, Any]):
    return yc_base.make_raw_env(env_args)


def copy_aux_from_input(input_dir: Path) -> list[str]:
    return [
        str(p)
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT", ".CLI", ".PRM", ".WDB"}
    ]


def prepare_hla(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    hla_base.configure_shared_settings()
    install_official_reward_module()
    prepare_hla_case_at(int(case["year"]), run_dir)
    return json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))


def prepare_yc(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    yc_base.OUT_DIR = OUT_DIR / "yc2014_source"
    yc_base.SEED = 0
    yc_base.WATER_COST = 1.0
    yc_base.NITROGEN_COST = 5.0
    yc_base.IRRIGATION_BUDGET = 9999.0
    yc_base.NITROGEN_BUDGET = 9999.0
    yc_base.DAILY_IRRIGATION_CAP = 9999.0
    yc_base.DAILY_NITROGEN_CAP = 9999.0
    yc_base.MIN_INTERVAL_DAYS = 0
    source_run_dir = yc_base.prepare_case_for_scenario("dqn_extension_expert_fixed_dap_018_03")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(source_run_dir, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env_args["log_saving_path"] = str(run_dir / "extension_expert_fixed_dap.log")
    input_dir = run_dir / "input"
    filex = next(input_dir.glob("*.MZX"))
    env_args["fileX_template_path"] = str(filex)
    env_args["auxiliary_file_paths"] = copy_aux_from_input(input_dir)
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return env_args


def prepare_fq(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from run_fq_all_year_screen_and_dqn_transfer_014_01 import (
        INPUT_ROOT,
        MZX_NAME,
        TEMPLATE_TRNO,
        prepare_text_for_shifted_scenario,
    )

    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    text = prepare_text_for_shifted_scenario(int(case["year"]), "dqn_linked_free_daily")
    filex = input_dir / f"CNFQ{int(case['year'])}_extension_expert_018_03.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": TEMPLATE_TRNO,
        "auxiliary_file_paths": copy_aux_from_input(input_dir),
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return env_args


def prepare_sy(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import run_sy_local_dqn_train_cross_year_transfer_017_08 as sy

    sy.configure_globals()
    prepared = sy.prepare_run_dir(int(case["year"]), "dqn_extension_expert_fixed_dap_018_03", seed=0, root=OUT_DIR / "sy_source")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(prepared, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env_args["log_saving_path"] = str(run_dir / "pdi_gym.log")
    input_dir = run_dir / "input"
    filex = next(input_dir.glob("*.MZX"))
    env_args["fileX_template_path"] = str(filex)
    env_args["auxiliary_file_paths"] = copy_aux_from_input(input_dir)
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return env_args


def prepare_lc(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    from run_fq_yc_new_cultivar_forward_screening_013_01 import set_management_for_treatment
    from run_lc_fixed_input_year_screening_017_11 import OUT_DIR as LC_SCREEN_DIR, fixed_source_text

    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    text = set_management_for_treatment(fixed_source_text(), 3, "L", "L")
    filex = input_dir / "CNLC1001_extension_expert_018_03.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    source_input = LC_SCREEN_DIR / "runs" / "2010" / "null" / "input"
    if not source_input.exists():
        raise FileNotFoundError(f"LC 017_11 fixed input not found: {source_input}")
    for src in source_input.iterdir():
        if src.is_file() and src.suffix.upper() != ".MZX":
            shutil.copyfile(src, input_dir / src.name)

    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 3,
        "auxiliary_file_paths": copy_aux_from_input(input_dir),
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return env_args


PREPARE_FUNCS = {
    "HLA": prepare_hla,
    "YC": prepare_yc,
    "FQ": prepare_fq,
    "SY": prepare_sy,
    "LC": prepare_lc,
}


def run_fixed_schedule(case: dict[str, Any], env_args: dict[str, Any], schedule: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    actions = split_irrigation_events(schedule)
    env = make_raw_env(env_args)
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    snapshot_dir = run_dir / "pdi_tmp_snapshot_eval"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    try:
        obs, info = env.reset()
        fired_action_daps: set[int] = set()
        for step in range(420):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", step)) or 0))
            if dap_before in actions and dap_before not in fired_action_daps:
                real = actions[dap_before]
                fired_action_daps.add(dap_before)
            else:
                real = {"amir": 0.0, "anfer": 0.0}
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": case["site"],
                    "station": case["station"],
                    "year": int(case["year"]),
                    "region": case["region"],
                    "scenario": SCENARIO,
                    "step": step,
                    "dap_action": dap_before,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "rain_obs": scalar(latest.get("rain")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm_action": float(real.get("amir", 0.0)),
                    "fertilizer_kg_ha_action": float(real.get("anfer", 0.0)),
                    "raw_reward": repr(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if real.get("amir", 0.0) > 0 or real.get("anfer", 0.0) > 0:
                action_rows.append(
                    {
                        "site": case["site"],
                        "station": case["station"],
                        "year": int(case["year"]),
                        "scenario": SCENARIO,
                        "dap": dap_before,
                        "irrigation_mm_action": float(real.get("amir", 0.0)),
                        "fertilizer_kg_ha_action": float(real.get("anfer", 0.0)),
                    }
                )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot_dir, dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    actions_df = pd.DataFrame(action_rows)
    events = parse_hla_events(snapshot_dir / "MgmtEvent.OUT")
    plantgro = parse_dssat_table(snapshot_dir / "PlantGro.OUT") if (snapshot_dir / "PlantGro.OUT").exists() else pd.DataFrame()
    final_gwad = float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan
    final_cwad = float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan
    summary = {
        "site": case["site"],
        "station": case["station"],
        "year": int(case["year"]),
        "region": case["region"],
        "scenario": SCENARIO,
        "status": "ok",
        "final_gwad": final_gwad,
        "final_cwad": final_cwad,
        "rain_total_obs": float(pd.to_numeric(daily.get("rain_obs", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not daily.empty else np.nan,
        "action_irrigation_total": float(daily["irrigation_mm_action"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha_action"].sum()) if not daily.empty else 0.0,
        "event_irrigation_total": float(events.get("irrigation_total_mgmtevent", np.nan)) if isinstance(events, dict) else np.nan,
        "event_fertilizer_total": float(events.get("fertilizer_total_mgmtevent", np.nan)) if isinstance(events, dict) else np.nan,
        "max_water_stress": float(pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce").max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce").max()) if not daily.empty else np.nan,
        "final_dap": float(pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce").dropna().iloc[-1]) if not daily.empty and not pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce").dropna().empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "error": "",
    }
    return daily, actions_df, summary


def load_existing_baselines() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(site: str, station: str, year: int, scenario_label: str, grain: float, biomass: float, irrigation: float, nitrogen: float, source: str, max_w=np.nan, max_n=np.nan):
        rows.append(
            {
                "site": site,
                "station": station,
                "year": year,
                "scenario_label": scenario_label,
                "scenario_group": scenario_label,
                "grain_yield_kg_ha": grain,
                "biomass_kg_ha": biomass,
                "irrigation_mm": irrigation,
                "nitrogen_kg_ha": nitrogen,
                "max_water_stress": max_w,
                "max_nitrogen_stress": max_n,
                "source_file": source,
                "note": "existing baseline/result",
            }
        )

    # HLA2010 seed0 main comparison.
    path = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_final_dqn_four_scenario_015_16" / "hla_2010_four_scenario_final_dqn_seed0_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            add_row("HLA", "Hailun", 2010, str(r.get("label", "Null") if pd.notna(r.get("label", np.nan)) else "Null"), float(r["final_gwad"]), float(r["final_cwad"]), float(r["irrigation_total"]), float(r["fertilizer_total"]), str(path.relative_to(PROJECT_ROOT)), r.get("max_water_stress", np.nan), r.get("max_nitrogen_stress", np.nan))

    # YC2014 representative rows.
    path = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_formal_four_scenario_015_06" / "seed0_seed1_best" / "015_06_yc2014_formal_four_scenario_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[df.get("representative_for_figure", True).astype(str).str.lower().eq("true")]
        for _, r in df.iterrows():
            scen = r["scenario"] if pd.notna(r["scenario"]) else "null"
            label = {"null": "Null", "recorded": "Recorded/farmer practice", "dssat_auto": "DSSAT auto", "dqn_best": "DQN best checkpoint"}.get(str(scen), str(scen))
            add_row("YC", "Yucheng", 2014, label, float(r["final_grain_kg_ha"]), float(r["final_biomass_kg_ha"]), float(r["irrigation_total_mm"]), float(r["fertilizer_total_kg_ha"]), str(path.relative_to(PROJECT_ROOT)), r.get("max_water_stress", np.nan), r.get("max_nitrogen_stress", np.nan))

    # FQ2016 existing four-scenario.
    path = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_four_scenario_process_017_02" / "fq2016_four_scenario_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            label = {
                "null_zero": "Null",
                "recorded_shifted": "Recorded/farmer practice",
                "dssat_auto": "DSSAT auto",
                "dqn_seed1_best_reward": "DQN best checkpoint",
            }.get(str(r["scenario"]), str(r["scenario"]))
            add_row("FQ", "Fengqiu", 2016, label, float(r["final_grain_kg_ha"]), float(r["final_biomass_kg_ha"]), float(r["irrigation_total"]), float(r["fertilizer_total"]), str(path.relative_to(PROJECT_ROOT)), r.get("max_water_stress", np.nan), r.get("max_nitrogen_stress", np.nan))

    # SY2014 existing four-scenario.
    path = PROJECT_ROOT / "DSSAT_auto_validation" / "sy2014_dqn_resource_space_017_09" / "017_09_sy2014_four_scenario_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            label = str(r.get("scenario_label", r.get("scenario", "")))
            add_row("SY", "Shenyang", 2014, label, float(r["final_gwad"]), float(r["final_cwad"]), float(r["irrigation_total_mm"]), float(r["fertilizer_total_kg_ha"]), str(path.relative_to(PROJECT_ROOT)), r.get("max_water_stress", np.nan), r.get("max_nitrogen_stress", np.nan))

    # LC2010 seed0 checkpoint summary as DQN only, plus known constants from script if available.
    lc_path = PROJECT_ROOT / "DSSAT_auto_validation" / "lc2010_baseline_relative_dqn_smoke_017_12" / "seed0_5000steps" / "checkpoint_summary.csv"
    if lc_path.exists():
        df = pd.read_csv(lc_path)
        # Pick best total reward if available, otherwise last row.
        if "total_reward" in df.columns:
            r = df.sort_values("total_reward", ascending=False).iloc[0]
        else:
            r = df.iloc[-1]
        grain_col = "final_gwad" if "final_gwad" in df.columns else ("final_grain_kg_ha" if "final_grain_kg_ha" in df.columns else None)
        biomass_col = "final_cwad" if "final_cwad" in df.columns else ("final_biomass_kg_ha" if "final_biomass_kg_ha" in df.columns else None)
        irrigation_col = "irrigation_total" if "irrigation_total" in df.columns else ("irrigation_total_mm" if "irrigation_total_mm" in df.columns else None)
        nitrogen_col = "fertilizer_total" if "fertilizer_total" in df.columns else ("fertilizer_total_kg_ha" if "fertilizer_total_kg_ha" in df.columns else None)
        if grain_col and biomass_col and irrigation_col and nitrogen_col:
            add_row("LC", "Luancheng", 2010, "DQN best checkpoint", float(r[grain_col]), float(r[biomass_col]), float(r[irrigation_col]), float(r[nitrogen_col]), str(lc_path.relative_to(PROJECT_ROOT)), r.get("max_water_stress", np.nan), r.get("max_nitrogen_stress", np.nan))
        # Known diagnostic constants recorded in lc script.
        add_row("LC", "Luancheng", 2010, "Null", 8051.0, np.nan, 0.0, 0.0, "run_lc2010_baseline_relative_dqn_smoke_017_12.py constants", np.nan, np.nan)
        add_row("LC", "Luancheng", 2010, "Recorded/farmer practice", 8732.0, np.nan, np.nan, np.nan, "run_lc2010_baseline_relative_dqn_smoke_017_12.py constants", np.nan, np.nan)
        add_row("LC", "Luancheng", 2010, "DSSAT auto", 8738.0, np.nan, np.nan, np.nan, "run_lc2010_baseline_relative_dqn_smoke_017_12.py constants", np.nan, np.nan)

    return pd.DataFrame(rows)


def plot_summary(clean: pd.DataFrame, out_path: Path) -> None:
    if clean.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = [
        ("grain_yield_kg_ha", "Grain yield", "kg/ha"),
        ("irrigation_mm", "Irrigation", "mm"),
        ("nitrogen_kg_ha", "Nitrogen", "kg/ha"),
    ]
    for ax, (col, title, ylabel) in zip(axes, cols):
        data = clean.copy()
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data["x_label"] = data["site"] + "\n" + data["scenario_label"].astype(str).str.replace("Official extension expert fixed DAP", "Extension expert", regex=False).str.slice(0, 22)
        colors = data["scenario_label"].map(
            {
                "Null": "#444444",
                "Recorded expert": "#CC6677",
                "Recorded/farmer practice": "#CC6677",
                "DSSAT auto": "#DDCC77",
                "DQN best checkpoint": "#117733",
                "Official extension expert fixed DAP": "#4477AA",
            }
        ).fillna("#999999")
        ax.bar(np.arange(len(data)), data[col], color=colors, edgecolor="#222222", linewidth=0.5)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(data["x_label"], rotation=72, ha="right", fontsize=7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("018_03 official extension expert baseline across representative station-years", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_record(extension_summary: pd.DataFrame, clean: pd.DataFrame, failures: list[dict[str, Any]]) -> None:
    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "_empty_"
        view = df[cols].copy()
        for c in view.columns:
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x)))
        lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
        return "\n".join(lines)

    lines = [
        "# 018_03 五站点官方农技推广 expert baseline 扩展记录",
        "",
        "## 本轮原则",
        "",
        "- 只新增 `official extension expert fixed DAP` 情景。",
        "- 不修改 DQN 奖励函数。",
        "- 不训练 DQN。",
        "- 不改原始输入文件。",
        "- HLA/SY 使用 PDF 表 1；YC/FQ/LC 使用 PDF 表 3。",
        "- 采用固定 DAP 映射，不用事后模拟生育期。",
        "",
        "## Extension expert replay 结果",
        "",
        md_table(extension_summary, ["site", "station", "year", "status", "final_gwad", "final_cwad", "event_irrigation_total", "event_fertilizer_total", "max_water_stress", "max_nitrogen_stress", "error"]),
        "",
        "## 与已有情景的干净合并表",
        "",
        md_table(clean, ["site", "station", "year", "scenario_label", "grain_yield_kg_ha", "biomass_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "max_water_stress", "max_nitrogen_stress", "source_file"]),
        "",
        "## 初步解释",
        "",
        "这一步的目的不是证明 DQN 最优，而是把导师要求的官方推广 expert baseline 加入当前叙事。汇报时应区分：",
        "",
        "1. `Recorded/farmer practice`：历史记录/农民管理。",
        "2. `Official extension expert fixed DAP`：根据官方农技推广方案中值换算得到的固定 DAP 管理。",
        "3. `DSSAT auto`：DSSAT 原生自动管理。",
        "4. `DQN best checkpoint`：现有 DQN 结果，不在本轮重新训练。",
        "",
        "如果 DQN 与官方推广 expert 产量相近但投入更低，可解释为“用更低资源逼近高投入推广方案的产量平台”；如果 DQN 产量低但更省资源，则需由导师决定是否接受产量-资源权衡；如果 DQN 产量和效率均低于官方推广方案，则后续需要讨论奖励函数和约束设置。",
        "",
        "## 输出文件",
        "",
        f"- schedule: `{SCHEDULE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- summary: `{(OUT_DIR / '018_03_extension_expert_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- daily: `{(OUT_DIR / '018_03_extension_expert_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- events: `{(OUT_DIR / '018_03_extension_expert_events.csv').relative_to(PROJECT_ROOT)}`",
        f"- clean comparison: `{(OUT_DIR / '018_03_clean_multisite_comparison_with_extension_expert.csv').relative_to(PROJECT_ROOT)}`",
        f"- figure: `{(OUT_DIR / 'figures' / '018_03_extension_expert_summary.png').relative_to(PROJECT_ROOT)}`",
    ]
    if failures:
        lines.extend(["", "## 失败/异常记录", "", md_table(pd.DataFrame(failures), ["site", "year", "error"])])
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule = build_case_schedule()
    daily_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for case in CASES:
        run_dir = OUT_DIR / f"{case['site']}{case['year']}" / SCENARIO
        try:
            env_args = PREPARE_FUNCS[case["site"]](case, run_dir)
            case_schedule = schedule[(schedule["site"].eq(case["site"])) & (schedule["year"].eq(case["year"]))]
            daily, events, summary = run_fixed_schedule(case, env_args, case_schedule, run_dir)
            daily_frames.append(daily)
            event_frames.append(events)
            summaries.append(summary)
            print(f"OK {case['site']} {case['year']}: GWAD={summary['final_gwad']} I={summary['event_irrigation_total']} N={summary['event_fertilizer_total']}")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            failures.append({"site": case["site"], "year": case["year"], "error": err})
            summaries.append(
                {
                    "site": case["site"],
                    "station": case["station"],
                    "year": case["year"],
                    "region": case["region"],
                    "scenario": SCENARIO,
                    "status": "failed",
                    "final_gwad": np.nan,
                    "final_cwad": np.nan,
                    "rain_total_obs": np.nan,
                    "action_irrigation_total": np.nan,
                    "action_fertilizer_total": np.nan,
                    "event_irrigation_total": np.nan,
                    "event_fertilizer_total": np.nan,
                    "max_water_stress": np.nan,
                    "max_nitrogen_stress": np.nan,
                    "final_dap": np.nan,
                    "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                    "error": err,
                    "traceback": traceback.format_exc(limit=4),
                }
            )
            print(f"FAILED {case['site']} {case['year']}: {err}")

    extension_summary = pd.DataFrame(summaries)
    extension_summary["scenario_label"] = "Official extension expert fixed DAP"
    extension_summary["irrigation_mm"] = extension_summary["event_irrigation_total"].fillna(extension_summary["action_irrigation_total"])
    extension_summary["nitrogen_kg_ha"] = extension_summary["event_fertilizer_total"].fillna(extension_summary["action_fertilizer_total"])
    extension_summary["grain_yield_kg_ha"] = extension_summary["final_gwad"]
    extension_summary["biomass_kg_ha"] = extension_summary["final_cwad"]
    extension_summary["source_file"] = "018_03_extension_expert_summary.csv"
    extension_summary["note"] = "official extension schedule fixed DAP"

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    existing = load_existing_baselines()
    clean = pd.concat(
        [
            existing,
            extension_summary[
                [
                    "site",
                    "station",
                    "year",
                    "scenario_label",
                    "grain_yield_kg_ha",
                    "biomass_kg_ha",
                    "irrigation_mm",
                    "nitrogen_kg_ha",
                    "max_water_stress",
                    "max_nitrogen_stress",
                    "source_file",
                    "note",
                ]
            ],
        ],
        ignore_index=True,
    )
    order = {
        "Null": 0,
        "Recorded expert": 1,
        "Recorded/farmer practice": 1,
        "DSSAT auto": 2,
        "DQN best checkpoint": 3,
        "DQN ckpt15000": 3,
        "Official extension expert fixed DAP": 4,
    }
    clean["_order"] = clean["scenario_label"].map(order).fillna(9)
    clean = clean.sort_values(["site", "year", "_order", "scenario_label"]).drop(columns=["_order"])

    daily_all.to_csv(OUT_DIR / "018_03_extension_expert_daily.csv", index=False, encoding="utf-8-sig")
    events_all.to_csv(OUT_DIR / "018_03_extension_expert_events.csv", index=False, encoding="utf-8-sig")
    extension_summary.to_csv(OUT_DIR / "018_03_extension_expert_summary.csv", index=False, encoding="utf-8-sig")
    clean.to_csv(OUT_DIR / "018_03_clean_multisite_comparison_with_extension_expert.csv", index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "018_03_failures.csv", index=False, encoding="utf-8-sig")
    plot_summary(clean, OUT_DIR / "figures" / "018_03_extension_expert_summary.png")
    write_record(extension_summary, clean, failures)
    print(extension_summary[["site", "year", "status", "final_gwad", "final_cwad", "irrigation_mm", "nitrogen_kg_ha", "max_water_stress", "max_nitrogen_stress", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
