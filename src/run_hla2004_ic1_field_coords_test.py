from __future__ import annotations

import json
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


OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "gym_ic1_null_validation_field_coords"
SOURCE_RUN = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404"


def ensure_inputs() -> dict[str, Path]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    input_dir = OUT_ROOT / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    filex = input_dir / "CNHL0404_IC1_FIELD_COORDS.MZX"
    weather = input_dir / "CNHL0401.WTH"
    soil = input_dir / "SOIL.SOL"
    cultivar = PROJECT_ROOT / "my_data" / "MZCER048.CUL"

    shutil.copyfile(SOURCE_RUN / "CNHL0404.MZX", filex)
    shutil.copyfile(SOURCE_RUN / "CNHL0401.WTH", weather)
    shutil.copyfile(SOURCE_RUN / "SOIL.SOL", soil)

    text = filex.read_text(encoding="latin1", errors="ignore")
    if " 1 1 1 0 Sim2004                    1  1  0  1" not in text:
        raise RuntimeError("CNHL0404 does not look like the expected IC=1 input.")

    # PDI 4.8.0 warns that it cannot transfer FIELD coordinates when they are -99.
    # Keep all agronomic settings unchanged; only provide the station coordinates
    # already present in Summary/Weather metadata: lat=47.45, lon=126.90, elev=234.
    text = text.replace(
        " 1             -99             -99       -99               -99   -99   -99   -99   -99   -99",
        " 1          126.90           47.45       234               -99   -99   -99   -99   -99   -99",
    )
    filex.write_text(text, encoding="latin1")
    return {"filex": filex, "weather": weather, "soil": soil, "cultivar": cultivar}


def run_episode(max_steps: int = 260) -> dict[str, object]:
    import gym
    from sb3_wrapper import GymDssatWrapper

    paths = ensure_inputs()
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot = OUT_ROOT / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    env_args = {
        "log_saving_path": str(log_dir / "HLA_2004_CNHL0404_IC1_FIELD_COORDS_gym.log"),
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
    try:
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            records.append(
                {
                    "step": step,
                    "yrdoy": yrdoy,
                    "year": int(yrdoy // 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": scalar(reward),
                    "done": done,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            step += 1
    finally:
        tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp_folder and Path(tmp_folder).exists():
            shutil.copytree(tmp_folder, snapshot, dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(records)
    daily_path = OUT_ROOT / "gym_ic1_field_coords_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    last = daily.iloc[-1].to_dict() if len(daily) else {}
    summary = {
        "status": "ok" if len(daily) else "empty",
        "daily_steps": int(len(daily)),
        "final_doy": int(last.get("doy", -1)) if last else None,
        "final_dap": float(last.get("dap", np.nan)) if last else np.nan,
        "final_grnwt": float(last.get("grnwt", np.nan)) if last else np.nan,
        "final_topwt": float(last.get("topwt", np.nan)) if last else np.nan,
        "max_swfac": float(pd.to_numeric(daily.get("swfac"), errors="coerce").max()) if len(daily) else np.nan,
        "min_swfac": float(pd.to_numeric(daily.get("swfac"), errors="coerce").min()) if len(daily) else np.nan,
        "max_nstres": float(pd.to_numeric(daily.get("nstres"), errors="coerce").max()) if len(daily) else np.nan,
        "min_nstres": float(pd.to_numeric(daily.get("nstres"), errors="coerce").min()) if len(daily) else np.nan,
        "daily_csv": str(daily_path.relative_to(PROJECT_ROOT)),
    }
    pd.DataFrame([summary]).to_csv(OUT_ROOT / "gym_ic1_field_coords_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_ROOT / "gym_ic1_field_coords_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run_episode()
