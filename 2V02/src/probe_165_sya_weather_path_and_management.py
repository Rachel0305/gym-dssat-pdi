"""165: minimal probe separating weather-path and management effects.

Runs only three input roots with a fixed PPO-like action schedule:
canonical 2005 weather, v2 original, and v2 early_dry.  This is a diagnostic
probe, not a training or model-selection experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_164_sya_weather_v2_fixed_management_gate as gate


OUT = ROOT / "benchmark_results" / "165_sya_weather_path_and_management_probe"
STATION = "SYA"
SOURCE_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
V2_ROOT = ROOT / "DSSAT_auto_validation" / "sya_weather_scenario_bank_163_v2"


def ppo_like_management() -> dict[str, dict[int, float]]:
    irrigation = {1: 45.0}
    irrigation.update({dap: 15.0 for dap in [8, 15, 31, 38, 45, 52, 59, 66, 73, 80, 91, 98, 105]})
    nitrogen = {1: 80.0, 9: 80.0, 16: 80.0}
    return {"irrigation": irrigation, "nitrogen": nitrogen}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    env_config = gate.make_env_config(OUT)
    management = ppo_like_management()
    roots = {
        "canonical_2005": SOURCE_ROOT,
        "v2_original": V2_ROOT / "original",
        "v2_early_dry": V2_ROOT / "early_dry",
    }
    rows = []
    for label, root in roots.items():
        _daily, summary = gate.run_policy(env_config, root, label, "ppo_like_fixed", management, OUT)
        rows.append(summary)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "165_probe_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "completed_minimal_probe",
        "training_executed": False,
        "policy": "fixed_schedule_approx_162_action_sequence",
        "roots": {key: str(value.relative_to(ROOT)) for key, value in roots.items()},
        "results": result.to_dict(orient="records"),
    }
    (OUT / "165_probe_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "docs" / "165_sya_weather_path_and_management_probe_record.md").write_text(
        "\n".join(
            [
                "# 165 SYA weather path and management probe",
                "",
                "- Training executed: no.",
                "- Fixed policy: approximate 162 action sequence; no PPO inference.",
                "- Purpose: compare canonical 2005, v2 original, and v2 early_dry input roots.",
                f"- Summary: `{(OUT / '165_probe_summary.csv').relative_to(ROOT).as_posix()}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
