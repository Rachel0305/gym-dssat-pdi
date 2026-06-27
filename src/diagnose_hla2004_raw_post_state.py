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


def _copy_inputs(source_run: Path, out_root: Path, label: str) -> dict[str, Path]:
    input_dir = out_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    filex = input_dir / f"{label}.MZX"
    weather = input_dir / "CNHL0401.WTH"
    soil = input_dir / "SOIL.SOL"
    cultivar = PROJECT_ROOT / "my_data" / "MZCER048.CUL"
    shutil.copyfile(source_run / "CNHL0404.MZX", filex)
    if (source_run / "CNHL0401.WTH").exists():
        shutil.copyfile(source_run / "CNHL0401.WTH", weather)
    else:
        shutil.copyfile(PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404" / "CNHL0401.WTH", weather)
    if (source_run / "SOIL.SOL").exists():
        shutil.copyfile(source_run / "SOIL.SOL", soil)
    else:
        shutil.copyfile(PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404" / "SOIL.SOL", soil)
    return {"filex": filex, "weather": weather, "soil": soil, "cultivar": cultivar}


def run_case(case_name: str, source_run: Path, max_steps: int = 260) -> dict[str, object]:
    import gym
    from gym_dssat_pdi.envs.utils import utils as pdi_utils
    from sb3_wrapper import GymDssatWrapper

    out_root = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "raw_post_state_diagnostics" / case_name
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = _copy_inputs(source_run, out_root, case_name)

    raw_post_records: list[dict[str, object]] = []
    original_post_treat = pdi_utils._post_treat_state

    def wrapped_post_treat(state, cultivar="maize"):
        raw = dict(state) if state else {}
        post = original_post_treat(state, cultivar)
        rec: dict[str, object] = {"cultivar": cultivar}
        for prefix, src in [("raw", raw), ("post", post if post else {})]:
            for key in [
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
            ]:
                val = src.get(key)
                try:
                    if hasattr(val, "item"):
                        val = val.item()
                except Exception:
                    pass
                rec[f"{prefix}_{key}"] = val
        raw_post_records.append(rec)
        return post

    pdi_utils._post_treat_state = wrapped_post_treat
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env_args = {
        "log_saving_path": str(log_dir / f"{case_name}_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(paths["filex"]),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(paths["cultivar"]), str(paths["weather"]), str(paths["soil"])],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (out_root / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")

    records: list[dict[str, object]] = []
    tmp_snapshot = out_root / "pdi_tmp_snapshot"
    try:
        env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", step_count))))
            full_action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, full_action)
            obs, reward, terminated, truncated, info = env.step(norm)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            records.append(
                {
                    "step_index": step_count,
                    "dap_before": dap_before,
                    "reward": scalar(reward),
                    "done": done,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "post_dap": scalar(latest.get("dap")),
                    "post_topwt": scalar(latest.get("topwt")),
                    "post_grnwt": scalar(latest.get("grnwt")),
                    "post_xlai": scalar(latest.get("xlai")),
                    "post_swfac": scalar(latest.get("swfac")),
                    "post_nstres": scalar(latest.get("nstres")),
                    "post_trnu": scalar(latest.get("trnu")),
                }
            )
            step_count += 1
        tmp_folder = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp_folder and Path(tmp_folder).exists():
            shutil.copytree(tmp_folder, tmp_snapshot, dirs_exist_ok=True)
        env.close()
    finally:
        pdi_utils._post_treat_state = original_post_treat

    daily = pd.DataFrame(records)
    raw_post = pd.DataFrame(raw_post_records)
    daily.to_csv(out_root / f"{case_name}_gym_daily_post.csv", index=False, encoding="utf-8-sig")
    raw_post.to_csv(out_root / f"{case_name}_raw_post_state.csv", index=False, encoding="utf-8-sig")

    summary = {
        "case": case_name,
        "status": "ok" if len(daily) else "empty",
        "daily_steps": int(len(daily)),
        "raw_post_records": int(len(raw_post)),
        "final_post_grnwt": float(pd.to_numeric(daily.get("post_grnwt"), errors="coerce").dropna().iloc[-1]) if len(daily) else np.nan,
        "final_post_topwt": float(pd.to_numeric(daily.get("post_topwt"), errors="coerce").dropna().iloc[-1]) if len(daily) else np.nan,
        "max_post_swfac": float(pd.to_numeric(daily.get("post_swfac"), errors="coerce").max()) if len(daily) else np.nan,
        "max_post_nstres": float(pd.to_numeric(daily.get("post_nstres"), errors="coerce").max()) if len(daily) else np.nan,
        "max_raw_swfac": float(pd.to_numeric(raw_post.get("raw_swfac"), errors="coerce").max()) if len(raw_post) else np.nan,
        "min_raw_swfac": float(pd.to_numeric(raw_post.get("raw_swfac"), errors="coerce").min()) if len(raw_post) else np.nan,
        "max_raw_nstres": float(pd.to_numeric(raw_post.get("raw_nstres"), errors="coerce").max()) if len(raw_post) else np.nan,
        "min_raw_nstres": float(pd.to_numeric(raw_post.get("raw_nstres"), errors="coerce").min()) if len(raw_post) else np.nan,
        "output_dir": str(out_root.relative_to(PROJECT_ROOT)),
    }
    pd.DataFrame([summary]).to_csv(out_root / f"{case_name}_summary.csv", index=False, encoding="utf-8-sig")
    (out_root / f"{case_name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    cases = {
        "windows_input_ic0_null": PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404_IC0_null",
        "windows_input_ic1_null": PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404",
    }
    summaries = []
    for name, src in cases.items():
        summaries.append(run_case(name, src))
    out = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "raw_post_state_diagnostics"
    pd.DataFrame(summaries).to_csv(out / "raw_post_state_diagnostics_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
