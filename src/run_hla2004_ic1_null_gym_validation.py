from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


VARIANT = os.environ.get("HLA2004_IC1_VARIANT", "sdate_04121")
OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / f"gym_ic1_null_validation_{VARIANT}"
SOURCE_RUN = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404"


def ensure_inputs() -> dict[str, Path]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rendered_dir = OUT_ROOT / "input"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    filex = rendered_dir / "CNHL0404_IC1_NULL.MZX"
    weather = rendered_dir / "CNHL0401.WTH"
    soil = rendered_dir / "SOIL.SOL"
    cultivar = PROJECT_ROOT / "my_data" / "MZCER048.CUL"

    shutil.copyfile(SOURCE_RUN / "CNHL0404.MZX", filex)
    shutil.copyfile(SOURCE_RUN / "CNHL0401.WTH", weather)
    shutil.copyfile(SOURCE_RUN / "SOIL.SOL", soil)

    text = filex.read_text(encoding="utf-8", errors="replace")
    if "Sim2004                    1  1  0  1" not in text:
        raise RuntimeError("CNHL0404 input does not appear to have IC=1 in the treatment line.")
    # Linux DSSAT/PDI rejects an initial-condition date before SDATE.
    # The Windows XBuild run accepted CNHL0404 with ICDAT=04121 and SDATE=04125,
    # but gym-DSSAT blocks on "Please press <ENTER>" in that case.  Test two
    # minimal variants without touching the source CNHL0404:
    #   sdate_04121: start simulation at the initial-condition date.
    #   icdat_04125: keep Windows SDATE but move ICDAT to SDATE.
    if VARIANT == "sdate_04121":
        text = text.replace(" 1 GE              1     1     S 04125  2150 DEFAULT SIMULATION CONTR",
                            " 1 GE              1     1     S 04121  2150 DEFAULT SIMULATION CONTR")
    elif VARIANT == "icdat_04125":
        text = text.replace(" 1    MZ 04121   100     0     1     1   -99     0     0     0   100    15 -99",
                            " 1    MZ 04125   100     0     1     1   -99     0     0     0   100    15 -99")
    else:
        raise RuntimeError(f"Unknown HLA2004_IC1_VARIANT={VARIANT}")
    filex.write_text(text, encoding="utf-8")
    return {"filex": filex, "weather": weather, "soil": soil, "cultivar": cultivar}


def run_episode() -> dict[str, object]:
    import gym
    from sb3_wrapper import GymDssatWrapper

    paths = ensure_inputs()
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env_args = {
        "log_saving_path": str(log_dir / "HLA_2004_CNHL0404_IC1_NULL_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(paths["filex"]),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(paths["cultivar"]), str(paths["weather"]), str(paths["soil"])],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (OUT_ROOT / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")

    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    records: list[dict[str, object]] = []
    step_count = 0
    cumulative_irrig = 0.0
    cumulative_n = 0.0
    tmp_snapshot = OUT_ROOT / "pdi_tmp_snapshot"
    if tmp_snapshot.exists():
        shutil.rmtree(tmp_snapshot)
    try:
        obs, info = env.reset()
        done = False
        while not done and step_count < 260:
            latest_before = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest_before.get("dap", step_count))))
            full_action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, full_action)
            obs, reward, terminated, truncated, info = env.step(norm)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = pd.Timestamp("2004-05-04") + pd.Timedelta(days=max(dap - 1, 0))
            cumulative_irrig += 0.0
            cumulative_n += 0.0
            records.append(
                {
                    "step_index": step_count,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "real_action_amir": 0.0,
                    "real_action_anfer": 0.0,
                    "cumulative_irrigation": cumulative_irrig,
                    "cumulative_n": cumulative_n,
                    "reward": float(reward),
                    "done": done,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "trnu": scalar(latest.get("trnu")),
                    "totir": scalar(latest.get("totir"), cumulative_irrig),
                    "tofer": scalar(latest.get("tofer"), cumulative_n),
                }
            )
            step_count += 1
    finally:
        try:
            tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
            if tmp_folder and Path(tmp_folder).exists():
                shutil.copytree(tmp_folder, tmp_snapshot, dirs_exist_ok=True)
            env.close()
        except Exception:
            pass

    daily = pd.DataFrame(records)
    daily_path = OUT_ROOT / "gym_ic1_null_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    last = daily.iloc[-1].to_dict() if len(daily) else {}
    summary = {
        "status": "ok" if len(daily) else "empty",
        "daily_steps": int(len(daily)),
        "final_grnwt": float(last.get("grnwt", np.nan)) if last else np.nan,
        "final_topwt": float(last.get("topwt", np.nan)) if last else np.nan,
        "total_irrigation": float(daily["real_action_amir"].sum()) if len(daily) else 0.0,
        "total_n": float(daily["real_action_anfer"].sum()) if len(daily) else 0.0,
        "max_swfac": float(pd.to_numeric(daily.get("swfac"), errors="coerce").max()) if len(daily) else np.nan,
        "max_nstres": float(pd.to_numeric(daily.get("nstres"), errors="coerce").max()) if len(daily) else np.nan,
        "daily_csv": str(daily_path.relative_to(PROJECT_ROOT)),
    }
    pd.DataFrame([summary]).to_csv(OUT_ROOT / "gym_ic1_null_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_ROOT / "gym_ic1_null_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run_episode()
