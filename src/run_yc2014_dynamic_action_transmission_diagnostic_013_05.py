from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
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
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT,
    SITE_CONFIG,
    parse_dssat_table,
    prepare_text_for_scenario,
    set_management_for_treatment,
)


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2014_dynamic_action_transmission_013_05"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_013_05_yc2014_dynamic_action_transmission_record.md"

SITE = "YC"
YEAR = 2014
TRNO = SITE_CONFIG[SITE]["treatments"][YEAR]
MZX_NAME = SITE_CONFIG[SITE]["mzx"]
RAW_FILES = ["PlantGro.OUT", "PlantN.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]

FORCED_DAPS = {
    43: {"amir": 30.0, "anfer": 100.0},
    50: {"amir": 30.0, "anfer": 100.0},
    57: {"amir": 30.0, "anfer": 100.0},
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_events(path: Path) -> dict[str, Any]:
    out = {
        "mgmt_event_irrigation_total": 0.0,
        "mgmt_event_fertilizer_total": 0.0,
        "mgmt_event_irrigation_count": 0,
        "mgmt_event_fertilizer_count": 0,
    }
    if not path.exists():
        return out
    seen = set()
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw:
            continue
        m = re.search(r"([-+]?\d+(?:\.\d*)?)\s*(mm|kg(?:\[[A-Za-z]+\])?/ha|kg)", raw)
        if not m:
            continue
        key = raw.strip()
        if key in seen:
            continue
        seen.add(key)
        amount = float(m.group(1))
        unit = m.group(2)
        if unit == "mm":
            out["mgmt_event_irrigation_total"] += amount
            out["mgmt_event_irrigation_count"] += 1
        elif "kg" in unit:
            out["mgmt_event_fertilizer_total"] += amount
            out["mgmt_event_fertilizer_count"] += 1
    return out


def prepare_case(scenario: str, linked: bool) -> Path:
    input_src = INPUT_ROOT / SITE
    run_dir = OUT_DIR / scenario
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    source = (input_src / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    if linked:
        text = set_management_for_treatment(source, TRNO, "L", "L")
    else:
        text = prepare_text_for_scenario(source, TRNO, "null")

    filex = input_dir / f"{scenario}.MZX"
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
        "experiment_number": TRNO,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def run_episode(scenario: str, linked: bool, forced: bool) -> dict[str, Any]:
    import gym
    from sb3_wrapper import GymDssatWrapper

    run_dir = prepare_case(scenario, linked)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = run_dir / "pdi_tmp_snapshot"
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(220):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(float(scalar(latest_before.get("dap", 0.0)) or 0.0)))
            real = dict(FORCED_DAPS.get(dap_before, {"amir": 0.0, "anfer": 0.0})) if forced else {"amir": 0.0, "anfer": 0.0}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest_after = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "scenario": scenario,
                    "step": step,
                    "dap_before": dap_before,
                    "amir_action": real["amir"],
                    "anfer_action": real["anfer"],
                    "dap_after": scalar(latest_after.get("dap")),
                    "grnwt": scalar(latest_after.get("grnwt")),
                    "topwt": scalar(latest_after.get("topwt")),
                    "swfac": scalar(latest_after.get("swfac")),
                    "nstres": scalar(latest_after.get("nstres")),
                    "done": bool(terminated or truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    daily.to_csv(run_dir / "daily_action_trace.csv", index=False, encoding="utf-8-sig")

    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    summary: dict[str, Any] = {
        "scenario": scenario,
        "linked_management": linked,
        "forced_action": forced,
        "steps": len(daily),
        "action_irrigation_total": float(daily["amir_action"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["anfer_action"].sum()) if not daily.empty else 0.0,
        "final_gwad": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and plantgro["GWAD"].notna().any() else np.nan,
        "final_cwad": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and plantgro["CWAD"].notna().any() else np.nan,
        "final_dap": float(plantgro["DAP"].dropna().iloc[-1]) if "DAP" in plantgro.columns and plantgro["DAP"].notna().any() else np.nan,
    }
    summary.update(parse_events(snapshot / "MgmtEvent.OUT"))
    for name in RAW_FILES:
        summary[f"sha256_{name}"] = sha256_file(snapshot / name)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}" if not float(val).is_integer() else f"{int(val)}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = [
        ("static_null_zero_action", False, False),
        ("static_forced_action", False, True),
        ("linked_null_zero_action", True, False),
        ("linked_forced_action", True, True),
    ]
    rows = []
    for scenario, linked, forced in scenarios:
        print(f"running {scenario}", flush=True)
        rows.append(run_episode(scenario, linked, forced))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "013_05_yc2014_dynamic_action_transmission_summary.csv", index=False, encoding="utf-8-sig")

    display_cols = [
        "scenario",
        "linked_management",
        "forced_action",
        "action_irrigation_total",
        "action_fertilizer_total",
        "mgmt_event_irrigation_total",
        "mgmt_event_fertilizer_total",
        "final_gwad",
        "final_cwad",
    ]
    doc = [
        "# 013_05 YC2014 dynamic action transmission diagnostic",
        "",
        "## Summary",
        "",
        md_table(summary[display_cols]),
        "",
        "## Conclusion template",
        "",
        "- If static_forced_action has action total > 0 but MgmtEvent total = 0, static N/N management blocks dynamic actions.",
        "- If linked_forced_action has MgmtEvent total > 0 and yield changes, PDI action transmission works when management is L/L.",
        "- DQN/PPO training inputs must use IRRIG=L and FERTI=L for dynamic actions.",
        "",
    ]
    DOC_PATH.write_text("\n".join(doc), encoding="utf-8")
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
