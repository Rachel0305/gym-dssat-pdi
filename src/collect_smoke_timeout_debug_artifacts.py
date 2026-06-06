from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "smoke_tests_debug"
os.environ["SMOKE_TEST_OUTPUT_ROOT"] = str(DEBUG_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smoke_test_agents import POLICIES, clip_real_action, normalize_real_action
from run_smoke_tests import build_env_args, load_config


EVAL_DIR = DEBUG_ROOT / "evaluation"
LOG_DIR = DEBUG_ROOT / "logs"
REVIEW_DIR = DEBUG_ROOT / "rendered_inputs_review"


def scalar_bound(value) -> float:
    try:
        return float(value)
    except Exception:
        return float(value.flatten()[0])


def collect_action_space() -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    rows = []
    config = load_config()
    cases = [
        ("HLA", 2007, "fixed_low_input"),
        ("LCA", 2010, "fixed_low_input"),
        ("SYA", 2014, "fixed_low_input"),
        ("YCA", 2014, "null_zero"),
    ]
    for station, year, policy_name in cases:
        meta = next(r for r in config["observed_years"] if r["station"] == station and int(r["year"]) == year)
        env_args = build_env_args(station, year, meta["planting_date"], int(config.get("seed", 123)))
        env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
        try:
            spaces = getattr(env.formator.action_space_dict, "spaces", env.formator.action_space_dict)
            policy = POLICIES[policy_name]
            scheduled_daps = sorted(set(policy.nitrogen_by_dap) | set(policy.irrigation_by_dap) | {1, 30, 60, 90})
            for dap in scheduled_daps:
                real_action = policy.action_for_dap(dap, env.formator.action_names)
                clipped, notes = clip_real_action(real_action, env.formator.action_space_dict)
                normalized = normalize_real_action(clipped, env.formator.action_names, env.formator.action_space_dict)
                norm_by_name = dict(zip(env.formator.action_names, normalized))
                for action_name in env.formator.action_names:
                    space = spaces[action_name]
                    low = scalar_bound(space.low)
                    high = scalar_bound(space.high)
                    rows.append(
                        {
                            "station": station,
                            "year": year,
                            "policy_name": policy_name,
                            "action_name": action_name,
                            "action_space_low": low,
                            "action_space_high": high,
                            "scheduled_dap": dap,
                            "scheduled_real_action": real_action.get(action_name, 0.0),
                            "clipped_real_action": clipped.get(action_name, 0.0),
                            "normalized_action": float(norm_by_name[action_name]),
                            "is_within_bounds": low <= clipped.get(action_name, 0.0) <= high and -1 <= float(norm_by_name[action_name]) <= 1,
                            "notes": ";".join(notes),
                        }
                    )
        finally:
            env.close()
    pd.DataFrame(rows).to_csv(EVAL_DIR / "action_space_and_policy_check.csv", index=False, encoding="utf-8-sig")


def parse_template_checks(case_id: str, station: str, year: int, policy: str, path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    checks = []
    items = {
        "wth_section_present": "@N METHODS" in text,
        "irrigation_section_present": "*IRRIGATION AND WATER MANAGEMENT" in text,
        "fertilizer_section_present": "*FERTILIZERS (INORGANIC)" in text,
        "simulation_controls_present": "*SIMULATION CONTROLS" in text,
        "planting_section_present": "*PLANTING DETAILS" in text,
        "mi_mf_enabled": bool(re.search(r"Sim\d{4}\s+1\s+1\s+0\s+0\s+1\s+1\s+1", text)),
        "no_original_pre_start_yca_irrigation": " 1 08163 IR003   120" not in text and " 1 14163 IR003   120" not in text,
        "safe_zero_irrigation_row_present": bool(re.search(r"@I IDATE\s+IROP IRVAL\s*\n\s*1\s+\d{5}\s+IR001\s+0", text)),
        "safe_zero_fertilizer_row_present": bool(re.search(r"@F FDATE.+\n\s*1\s+\d{5}\s+FE005\s+AP002\s+5\s+0", text)),
    }
    for item, ok in items.items():
        checks.append(
            {
                "case_id": case_id,
                "station": station,
                "year": year,
                "policy_name": policy,
                "file_type": "jinja2",
                "file_path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
                "exists": path.exists(),
                "check_item": item,
                "status": "pass" if ok else "fail",
                "details": "",
            }
        )
    return checks


def collect_rendered_input_checks() -> None:
    rows = []
    cases = [
        ("HLA_2007_fixed_low_input", "HLA", 2007, "fixed_low_input"),
        ("LCA_2010_fixed_low_input", "LCA", 2010, "fixed_low_input"),
        ("SYA_2014_fixed_low_input", "SYA", 2014, "fixed_low_input"),
        ("YCA_2014_null_zero", "YCA", 2014, "null_zero"),
    ]
    for case_id, station, year, policy in cases:
        rendered_dir = DEBUG_ROOT / "rendered_inputs" / station / str(year)
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for file in rendered_dir.glob("*"):
            target = REVIEW_DIR / f"{case_id}_{file.name}"
            if file.is_file():
                target.write_bytes(file.read_bytes())
        jinja = rendered_dir / f"{station}_{year}_smoke.jinja2"
        rows.extend(parse_template_checks(case_id, station, year, policy, jinja))
        for wth in rendered_dir.glob("*.WTH"):
            lines = wth.read_text(encoding="utf-8", errors="replace").splitlines()
            rows.append(
                {
                    "case_id": case_id,
                    "station": station,
                    "year": year,
                    "policy_name": policy,
                    "file_type": "WTH",
                    "file_path": str(wth.relative_to(PROJECT_ROOT)),
                    "exists": wth.exists(),
                    "check_item": "weather_year_present",
                    "status": "pass" if any(str(year) in line[:7] for line in lines) else "fail",
                    "details": f"line_count={len(lines)}",
                }
            )
    pd.DataFrame(rows).to_csv(EVAL_DIR / "rendered_input_check.csv", index=False, encoding="utf-8-sig")


def collect_log_checks() -> None:
    rows = []
    original_log_dir = PROJECT_ROOT / "Leave_One_experiments" / "smoke_tests" / "logs"
    cases = [
        ("LCA_2010_fixed_low_input", "LCA", 2010, "fixed_low_input"),
        ("SYA_2014_fixed_low_input", "SYA", 2014, "fixed_low_input"),
        ("YCA_2014_null_zero", "YCA", 2014, "null_zero"),
    ]
    keywords_error = ["STOP", "Fortran runtime error", "Error key", "not found", "prior to the start"]
    for case_id, station, year, policy in cases:
        for phase, source_dir in [("before_fix", original_log_dir), ("after_fix", LOG_DIR)]:
            log = source_dir / f"{station}_{year}.log"
            text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            last_lines = "\n".join(text.splitlines()[-50:])
            out = LOG_DIR / f"{case_id}_{phase}_last50.txt"
            out.write_text(last_lines, encoding="utf-8")
            rows.append(
                {
                    "case_id": case_id,
                    "station": station,
                    "year": year,
                    "policy_name": policy,
                    "log_file": str(log.relative_to(PROJECT_ROOT)) if log.exists() else str(log),
                    "exists": log.exists(),
                    "last_lines_path": str(out.relative_to(PROJECT_ROOT)),
                    "detected_error_keywords": ";".join(k for k in keywords_error if k in text),
                    "detected_warning_keywords": "WARNING.OUT" if "WARNING.OUT" in text else "",
                    "notes": phase,
                }
            )
    pd.DataFrame(rows).to_csv(EVAL_DIR / "dssat_log_check.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    collect_action_space()
    collect_rendered_input_checks()
    collect_log_checks()
    print(EVAL_DIR / "action_space_and_policy_check.csv")
    print(EVAL_DIR / "rendered_input_check.csv")
    print(EVAL_DIR / "dssat_log_check.csv")


if __name__ == "__main__":
    main()
