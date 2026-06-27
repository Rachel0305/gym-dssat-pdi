from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
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


OUT_ROOT = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "pdi_field_coordinate_diagnosis_ic0_null"
)


def unique_dir(prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_ROOT / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def patch_single_year_filex(src: Path, dst: Path, *, fixed_coords: bool) -> None:
    text = src.read_text(encoding="latin1", errors="ignore")
    text = re.sub(r"(\n\s*1 GE\s+)\d+(\s+1\s+S\s+04125)", r"\g<1>1\g<2>", text)
    # For a single 2004 run under PDI, use the explicit 2004 weather station.
    text = text.replace(" 1 CNHL2004 CNHL       ", " 1 CNHL2004 CNHL0401   ")
    if fixed_coords:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("@L ...........XCRD"):
                if i + 1 < len(lines):
                    # XCRD is longitude, YCRD is latitude in DSSAT FILEX field section.
                    lines[i + 1] = (
                        " 1         126.900          47.450       234               -99   -99   -99   -99   -99   -99"
                    )
                break
        text = "\n".join(lines) + "\n"
    dst.write_text(text, encoding="latin1")


def parse_plantgro(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    cols: list[str] | None = None
    with path.open("r", encoding="latin1", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.startswith("@"):
                cols = s[1:].split()
                continue
            if cols and re.match(r"^\s*\d{4}\s+\d+", s):
                parts = s.split()
                if len(parts) >= len(cols):
                    rows.append(dict(zip(cols, parts[: len(cols)])))
    df = pd.DataFrame(rows)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def run_case(case_name: str, fixed_coords: bool) -> dict[str, object]:
    import gym
    from sb3_wrapper import GymDssatWrapper

    case_dir = unique_dir(case_name)
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    src_run = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0405_IC0_null"
    filex = input_dir / f"{case_name}.MZX"
    patch_single_year_filex(src_run / "CNHL0405.MZX", filex, fixed_coords=fixed_coords)
    weather = input_dir / "CNHL0401.WTH"
    soil = input_dir / "SOIL.SOL"
    cultivar = PROJECT_ROOT / "my_data" / "MZCER048.CUL"
    shutil.copyfile(src_run / "CNHL0401.WTH", weather)
    shutil.copyfile(src_run / "SOIL.SOL", soil)

    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    env_args = {
        "log_saving_path": str(log_dir / f"{case_name}_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(cultivar), str(weather), str(soil)],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")

    records: list[dict[str, object]] = []
    tmp_snapshot = case_dir / "pdi_tmp_snapshot"
    env = None
    try:
        env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < 260:
            full_action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, full_action)
            obs, reward, terminated, truncated, info = env.step(norm)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            records.append(
                {
                    "step_index": step,
                    "post_dap": scalar(latest.get("dap")),
                    "post_topwt": scalar(latest.get("topwt")),
                    "post_grnwt": scalar(latest.get("grnwt")),
                    "post_xlai": scalar(latest.get("xlai")),
                    "post_swfac": scalar(latest.get("swfac")),
                    "post_nstres": scalar(latest.get("nstres")),
                    "reward": scalar(reward),
                    "done": done,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            step += 1
        tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp_folder and Path(tmp_folder).exists():
            shutil.copytree(tmp_folder, tmp_snapshot, dirs_exist_ok=True)
    finally:
        if env is not None:
            env.close()

    daily = pd.DataFrame(records)
    daily.to_csv(case_dir / f"{case_name}_gym_daily_post.csv", index=False, encoding="utf-8-sig")
    pg = parse_plantgro(tmp_snapshot / "PlantGro.OUT") if (tmp_snapshot / "PlantGro.OUT").exists() else pd.DataFrame()
    pg.to_csv(case_dir / f"{case_name}_pdi_PlantGro_parsed.csv", index=False, encoding="utf-8-sig")
    warn = (tmp_snapshot / "WARNING.OUT").read_text(encoding="latin1", errors="ignore") if (tmp_snapshot / "WARNING.OUT").exists() else ""
    summary = {
        "case_name": case_name,
        "fixed_coords": fixed_coords,
        "case_dir": str(case_dir.relative_to(PROJECT_ROOT)),
        "daily_steps": int(len(daily)),
        "final_gym_topwt": float(pd.to_numeric(daily.get("post_topwt"), errors="coerce").dropna().iloc[-1]) if len(daily) else np.nan,
        "final_gym_grnwt": float(pd.to_numeric(daily.get("post_grnwt"), errors="coerce").dropna().iloc[-1]) if len(daily) else np.nan,
        "final_pdi_cwad": float(pd.to_numeric(pg.get("CWAD"), errors="coerce").dropna().iloc[-1]) if len(pg) else np.nan,
        "final_pdi_gwad": float(pd.to_numeric(pg.get("GWAD"), errors="coerce").dropna().iloc[-1]) if len(pg) else np.nan,
        "warning_lat_lon_elev": any(s in warn for s in ["Error reading latitude", "Error reading longitude", "Error reading elevation"]),
        "warning_field_transfer": "Error transferring variable" in warn,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(case_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return summary


def prepare_windows_rerun_package() -> Path:
    package_dir = OUT_ROOT / "windows_rerun_package_pdi_filex"
    package_dir.mkdir(parents=True, exist_ok=True)
    src = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "raw_post_state_diagnostics" / "windows_input_ic0_null" / "pdi_tmp_snapshot"
    for name in ["fileX.MZX", "CNHL0401.WTH", "SOIL.SOL", "MZCER048.CUL"]:
        if (src / name).exists():
            shutil.copyfile(src / name, package_dir / name)
    readme = """# Windows rerun package for PDI-generated fileX

Purpose: run the PDI-generated `fileX.MZX` in Windows DSSAT 4.8.5 to test whether the 412 kg/ha yield follows the FILEX input or the PDI/DSSAT runtime.

Expected interpretation:
- If Windows DSSAT with this file gives approximately 412 kg/ha, the cause is likely FILEX/input formatting.
- If Windows DSSAT still gives approximately 248 kg/ha, the cause is more likely DSSAT version/runtime/PDI execution differences.

Files copied from the PDI tmp snapshot:
- fileX.MZX
- CNHL0401.WTH
- SOIL.SOL
- MZCER048.CUL

Do not overwrite original DSSAT files; copy these into a temporary DSSAT experiment folder before running.
"""
    (package_dir / "README_windows_rerun_package.md").write_text(readme, encoding="utf-8")
    return package_dir


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    package = prepare_windows_rerun_package()
    summaries = [
        run_case("pdi_ic0_null_original_coords_missing", fixed_coords=False),
        run_case("pdi_ic0_null_fixed_field_coords", fixed_coords=True),
    ]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_ROOT / "pdi_field_coordinate_diagnosis_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"windows_rerun_package": str(package.relative_to(PROJECT_ROOT)), "summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
