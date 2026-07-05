from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT,
    parse_dssat_table,
    prepare_text_for_scenario,
    set_management_for_treatment,
)
from run_yc2014_station_level3_true_model_transfer_016_04 import (
    MZX_NAME,
    SITE,
    STATION,
    TEMPLATE_TRNO,
    shift_yc2014_template_to_year,
    make_raw_env,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_cross_year_transfer_success_plots_016_11_expertfix"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-05_016_13_yc2014_expert_shifted_replay_for_selected_years.md"

YEARS = [2006, 2009, 2015, 2018]
EXPERT_EVENTS = [
    {"dap": 0, "amir": 0.0, "anfer": 96.0},
    {"dap": 43, "amir": 120.0, "anfer": 278.0},
]


def prepare_run_dir(year: int) -> tuple[Path, dict[str, Any]]:
    input_src = INPUT_ROOT / SITE
    run_dir = OUT_DIR / "runs" / str(year) / "expert_shifted"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    source = (input_src / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    shifted = shift_yc2014_template_to_year(source, year)
    text = set_management_for_treatment(shifted, TEMPLATE_TRNO, "L", "L")
    filex = input_dir / f"CNYC{year}_expert_shifted_replay.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    for src in input_src.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)

    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": TEMPLATE_TRNO,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir, env_args


def action_for_dap(dap: float) -> dict[str, float]:
    dap_i = int(round(float(dap)))
    for event in EXPERT_EVENTS:
        if dap_i == int(event["dap"]):
            return {"amir": float(event["amir"]), "anfer": float(event["anfer"])}
    return {"amir": 0.0, "anfer": 0.0}


def run_year(year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir, env_args = prepare_run_dir(year)
    env = make_raw_env(env_args)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(340):
            latest = latest_observation_dict(env, obs, info)
            dap = scalar(latest.get("dap"))
            real_action = action_for_dap(dap)
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, real_action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": year,
                    "scenario": "recorded_shifted",
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": scalar(reward),
                    "irrigation_mm": real_action["amir"],
                    "fertilizer_kg_ha": real_action["anfer"],
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = {
        "site": SITE,
        "station": STATION,
        "year": year,
        "scenario": "recorded_shifted",
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    return daily, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_daily = []
    all_summary = []
    for year in YEARS:
        daily, summary = run_year(year)
        all_daily.append(daily)
        all_summary.append(summary)
    pd.concat(all_daily, ignore_index=True).to_csv(OUT_DIR / "yc2014_expert_shifted_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_summary).to_csv(OUT_DIR / "yc2014_expert_shifted_summary.csv", index=False, encoding="utf-8-sig")
    DOC_PATH.write_text(
        "# 016_13 YC2014 expert 策略平移到代表年份\n\n"
        "已对 2006/2009/2015/2018 四个年份补跑 2014 recorded 管理日程平移结果。\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(all_summary).to_string(index=False))


if __name__ == "__main__":
    main()
