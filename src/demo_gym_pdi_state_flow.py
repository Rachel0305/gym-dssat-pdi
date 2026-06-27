from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


WATCH_KEYS = [
    "yrdoy",
    "dap",
    "swfac",
    "nstres",
    "topwt",
    "grnwt",
    "xlai",
    "istage",
    "vstage",
    "trnu",
    "wtnup",
    "pltpop",
]


def as_plain(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def copy_case_inputs(source_run: Path, output_dir: Path, label: str) -> dict[str, Path]:
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    filex_source = source_run / "CNHL0404.MZX"
    weather_source = source_run / "CNHL0401.WTH"
    soil_source = source_run / "SOIL.SOL"
    cultivar_source = PROJECT_ROOT / "my_data" / "MZCER048.CUL"

    required = [filex_source, weather_source, soil_source, cultivar_source]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    copied = {
        "filex": input_dir / f"{label}.MZX",
        "weather": input_dir / "CNHL0401.WTH",
        "soil": input_dir / "SOIL.SOL",
        "cultivar": input_dir / "MZCER048.CUL",
    }
    shutil.copyfile(filex_source, copied["filex"])
    shutil.copyfile(weather_source, copied["weather"])
    shutil.copyfile(soil_source, copied["soil"])
    shutil.copyfile(cultivar_source, copied["cultivar"])
    return copied


def parse_plantgro(path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open("r", encoding="latin-1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if header and not stripped.startswith(("*", "!", "$")):
                values = stripped.split()
                if len(values) >= len(header):
                    rows.append(values[: len(header)])

    if header is None:
        raise ValueError(f"Cannot find PlantGro header in {path}")
    frame = pd.DataFrame(rows, columns=header)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="ignore")
    if "DAP" in frame.columns:
        frame = frame.drop_duplicates(subset=["DAP"], keep="last").reset_index(drop=True)
    return frame


def run_demo(source_run: Path, output_dir: Path, max_steps: int) -> None:
    import gym
    from gym_dssat_pdi.envs.utils import utils as pdi_utils
    from sb3_wrapper import GymDssatWrapper

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = copy_case_inputs(source_run, output_dir, output_dir.name)

    raw_post_records: list[dict[str, Any]] = []
    original_post_treat = pdi_utils._post_treat_state

    def spy_post_treat(state, cultivar="maize"):
        raw = dict(state) if state else {}
        post = original_post_treat(state, cultivar)
        record: dict[str, Any] = {"call_index": len(raw_post_records), "cultivar": cultivar}
        for prefix, src in [("raw", raw), ("post", post if post else {})]:
            for key in WATCH_KEYS:
                record[f"{prefix}_{key}"] = as_plain(src.get(key))
        raw_post_records.append(record)
        return post

    pdi_utils._post_treat_state = spy_post_treat

    env_args = {
        "log_saving_path": str(output_dir / "gym_pdi_state_flow.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(paths["filex"]),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(paths["cultivar"]), str(paths["weather"]), str(paths["soil"])],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (output_dir / "env_args.json").write_text(
        json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    gym_records: list[dict[str, Any]] = []
    tmp_snapshot = output_dir / "pdi_tmp_snapshot"

    try:
        env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
        obs, info = env.reset()
        done = False
        step_index = 0
        while not done and step_index < max_steps:
            latest_before = latest_observation_dict(env, obs, info)
            action = {name: 0.0 for name in env.formator.action_names}
            normalized = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(normalized)
            done = bool(terminated or truncated)
            latest_after = latest_observation_dict(env, obs, info)
            gym_records.append(
                {
                    "step_index": step_index,
                    "dap_before": scalar(latest_before.get("dap")),
                    "post_dap": scalar(latest_after.get("dap")),
                    "post_topwt": scalar(latest_after.get("topwt")),
                    "post_grnwt": scalar(latest_after.get("grnwt")),
                    "post_swfac": scalar(latest_after.get("swfac")),
                    "post_nstres": scalar(latest_after.get("nstres")),
                    "reward": scalar(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                }
            )
            step_index += 1

        tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp_folder and Path(tmp_folder).exists():
            shutil.copytree(tmp_folder, tmp_snapshot, dirs_exist_ok=True)
        env.close()
    finally:
        pdi_utils._post_treat_state = original_post_treat

    raw_post = pd.DataFrame(raw_post_records)
    gym_post = pd.DataFrame(gym_records)
    raw_post.to_csv(output_dir / "01_raw_state_to_post_state.csv", index=False, encoding="utf-8-sig")
    gym_post.to_csv(output_dir / "02_gym_step_post_state.csv", index=False, encoding="utf-8-sig")

    plantgro_path = tmp_snapshot / "PlantGro.OUT"
    if plantgro_path.exists():
        plantgro = parse_plantgro(plantgro_path)
        plantgro.to_csv(output_dir / "03_pdi_PlantGro_parsed.csv", index=False, encoding="utf-8-sig")
    else:
        plantgro = pd.DataFrame()

    comparison = pd.DataFrame()
    if not plantgro.empty and not gym_post.empty and "DAP" in plantgro.columns:
        comparison = plantgro[["DAP", "CWAD", "GWAD", "WSPD", "NSTD"]].copy()
        comparison = comparison.merge(
            gym_post[["post_dap", "post_topwt", "post_grnwt", "post_swfac", "post_nstres"]],
            left_on="DAP",
            right_on="post_dap",
            how="inner",
        )
        comparison["topwt_minus_CWAD"] = comparison["post_topwt"] - comparison["CWAD"]
        comparison["grnwt_minus_GWAD"] = comparison["post_grnwt"] - comparison["GWAD"]
        comparison["swfac_minus_WSPD"] = comparison["post_swfac"] - comparison["WSPD"]
        comparison["nstres_minus_NSTD"] = comparison["post_nstres"] - comparison["NSTD"]
        comparison.to_csv(output_dir / "04_PlantGro_vs_gym_post_state_by_DAP.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_run": str(source_run),
        "output_dir": str(output_dir),
        "raw_post_rows": int(len(raw_post)),
        "gym_step_rows": int(len(gym_post)),
        "plantgro_rows_after_dedup": int(len(plantgro)),
        "final_gym_post_topwt": float(gym_post["post_topwt"].dropna().iloc[-1]) if len(gym_post) else None,
        "final_gym_post_grnwt": float(gym_post["post_grnwt"].dropna().iloc[-1]) if len(gym_post) else None,
        "final_plantgro_CWAD": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro else None,
        "final_plantgro_GWAD": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro else None,
        "max_abs_topwt_minus_CWAD": float(comparison["topwt_minus_CWAD"].abs().max()) if len(comparison) else None,
        "max_abs_grnwt_minus_GWAD": float(comparison["grnwt_minus_GWAD"].abs().max()) if len(comparison) else None,
        "max_abs_swfac_minus_WSPD": float(comparison["swfac_minus_WSPD"].abs().max()) if len(comparison) else None,
        "max_abs_nstres_minus_NSTD": float(comparison["nstres_minus_NSTD"].abs().max()) if len(comparison) else None,
    }
    (output_dir / "00_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nKey outputs:")
    print(output_dir / "01_raw_state_to_post_state.csv")
    print(output_dir / "02_gym_step_post_state.csv")
    print(output_dir / "03_pdi_PlantGro_parsed.csv")
    print(output_dir / "04_PlantGro_vs_gym_post_state_by_DAP.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the gym-DSSAT/PDI state flow.")
    parser.add_argument(
        "--source-run",
        type=Path,
        default=PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404_IC0_null",
        help="Folder containing CNHL0404.MZX, CNHL0401.WTH, and SOIL.SOL.",
    )
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "HLA_2004"
        / "gym_pdi_state_flow_demo"
        / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()
    run_demo(args.source_run.resolve(), args.output_dir.resolve(), args.max_steps)


if __name__ == "__main__":
    main()
