"""164: fixed-management physical response gate for the corrected SYA weather bank.

This is deliberately not a PPO experiment.  It runs two fixed management
policies across the six 2005-date weather variants from task 163, while
switching only the scenario-specific input root.  The purpose is to test the
DSSAT weather pathway before any augmented PPO training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


TASK_ID = "164"
TASK_NAME = "sya_weather_v2_fixed_management_gate"
DEFAULT_CONFIG = ROOT / "configs" / "163_sya_weather_scenario_bank_v2.json"
BASE_ENV_CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_all_year_direct_action_safe_ppo.yaml"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
STATION = "SYA"
SITE = "SY"
WEATHER_NAME = "CNSY0501.WTH"
DATA_RE = re.compile(
    r"^\s*(\d{7})\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*$"
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_weather(path: Path) -> dict[int, tuple[int, float, float, float, float]]:
    rows: dict[int, tuple[int, float, float, float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DATA_RE.match(line)
        if not match:
            continue
        date_text, srad, tmax, tmin, rain = match.groups()
        rows[int(date_text[-3:])] = (
            int(date_text[:4]),
            float(srad),
            float(tmax),
            float(tmin),
            float(rain),
        )
    if len(rows) < 300:
        raise ValueError(f"Too few weather rows: {path} ({len(rows)})")
    return rows


def expected_window(planting: int, window: str) -> tuple[int | None, int | None]:
    if window == "none":
        return None, None
    bounds = {"dap_1_30": (1, 30), "dap_31_60": (31, 60), "dap_61_90": (61, 90)}
    start, end = bounds[window]
    return planting + start - 1, planting + end - 1


def input_audit(cfg: dict[str, Any], manifest: pd.DataFrame) -> pd.DataFrame:
    source_weather = ROOT / str(cfg["source_input_root"]) / SITE / WEATHER_NAME
    source = parse_weather(source_weather)
    rows: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        target_weather = ROOT / str(item.target_weather)
        target = parse_weather(target_weather)
        start, end = expected_window(int(item.planting_doy), str(item.window))
        date_years = sorted({v[0] for v in target.values()})
        temp_srad_mismatch = 0
        nonwindow_rain_changes = 0
        ratio_errors = 0
        changed_rain_rows = 0
        expected_changed_rows = 0
        for doy in sorted(set(source) & set(target)):
            _, srad0, tmax0, tmin0, rain0 = source[doy]
            _, srad1, tmax1, tmin1, rain1 = target[doy]
            if not (np.isclose(srad0, srad1) and np.isclose(tmax0, tmax1) and np.isclose(tmin0, tmin1)):
                temp_srad_mismatch += 1
            in_window = start is not None and end is not None and start <= doy <= end
            if in_window:
                expected_changed_rows += 1
                if not np.isclose(rain1, max(0.0, rain0 * float(item.rain_multiplier)), atol=0.11):
                    ratio_errors += 1
            elif not np.isclose(rain0, rain1):
                nonwindow_rain_changes += 1
            if not np.isclose(rain0, rain1):
                changed_rain_rows += 1
        rows.append(
            {
                "variant": str(item.variant),
                "source_weather": rel(source_weather),
                "target_weather": rel(target_weather),
                "source_sha256": sha256(source_weather),
                "target_sha256": sha256(target_weather),
                "target_date_years": ",".join(map(str, date_years)),
                "changed_rain_rows": changed_rain_rows,
                "expected_window_rows": expected_changed_rows,
                "nonwindow_rain_changes": nonwindow_rain_changes,
                "temperature_or_radiation_mismatch_rows": temp_srad_mismatch,
                "rain_ratio_errors": ratio_errors,
                "source_total_rain_mm": float(sum(v[4] for v in source.values())),
                "target_total_rain_mm": float(sum(v[4] for v in target.values())),
                "passed": bool(
                    date_years == [2005]
                    and temp_srad_mismatch == 0
                    and nonwindow_rain_changes == 0
                    and ratio_errors == 0
                    and (str(item.variant) == "original" or changed_rain_rows > 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def make_env_config(output_root: Path) -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(BASE_ENV_CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["paths"]["output_root"] = rel(output_root)
    cfg["runtime"]["max_steps"] = 260
    selection = pd.DataFrame(
        [
            {
                "station_code": STATION,
                "station_name": SITE,
                "year": 2005,
                "weather_file": WEATHER_NAME,
                "scenario_type": "164_fixed_management_weather_gate",
            }
        ]
    )
    return direct_ppo.build_env_config(cfg, selection)


def run_policy(
    env_config: dict[str, Any],
    scenario_root: Path,
    variant: str,
    policy_name: str,
    management: dict[str, dict[int, float]],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = scenario_root
    env = None
    records: list[dict[str, Any]] = []
    total_i = 0.0
    total_n = 0.0
    start_time = time.time()
    try:
        env = direct_ppo.make_base_env(env_config, STATION, 2005, 0, f"{variant}_{policy_name}", evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(env_config["runtime"]["max_steps"]):
            before = latest_observation_dict(env, obs, info)
            dap_raw = scalar(before.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            action_real = {
                "amir": float(management.get("irrigation", {}).get(dap, 0.0)),
                "anfer": float(management.get("nitrogen", {}).get(dap, 0.0)),
            }
            total_i += action_real["amir"]
            total_n += action_real["anfer"]
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, action_real)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            records.append(
                {
                    "variant": variant,
                    "policy": policy_name,
                    "year": 2005,
                    "dap": dap,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "totir": scalar(latest.get("totir")),
                    "tofer": scalar(latest.get("tofer")),
                    "irrigation_requested_mm": action_real["amir"],
                    "nitrogen_requested_kg_ha": action_real["anfer"],
                    "reward": float(reward),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        daily_path = output_root / "daily_outputs" / f"{variant}_{policy_name}_daily.csv"
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
        nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
        completed = bool(len(daily) and bool(daily["done"].iloc[-1]))
        final_grain = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
        final_biomass = float(pd.to_numeric(daily.get("topwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
        summary = {
            "variant": variant,
            "policy": policy_name,
            "simulation_year": 2005,
            "episode_completed": completed,
            "episode_length": int(len(daily)),
            "final_grnwt": final_grain,
            "final_topwt": final_biomass,
            "total_irrigation_mm_requested": total_i,
            "total_n_kg_ha_requested": total_n,
            "mean_swfac": float(swfac.mean()) if len(swfac) else np.nan,
            "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
            "swfac_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
            "mean_nstres": float(nstres.mean()) if len(nstres) else np.nan,
            "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
            "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
            "daily_csv": rel(daily_path),
            "elapsed_seconds": time.time() - start_time,
            "error": "" if completed else "episode_not_completed",
        }
        return daily, summary
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    manifest_root = ROOT / str(cfg["manifest_output_root"])
    manifest_candidates = sorted(manifest_root.glob("*.csv"))
    if len(manifest_candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one scenario manifest in {manifest_root}, found {manifest_candidates}")
    manifest_path = manifest_candidates[0]
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    audit = input_audit(cfg, manifest)
    audit.to_csv(OUT / "164_weather_input_audit.csv", index=False, encoding="utf-8-sig")
    env_config = make_env_config(OUT)
    policies = {
        "rainfed_N": {"irrigation": {}, "nitrogen": {1: 50.0, 30: 50.0, 60: 50.0}},
        "limited_I_N": {"irrigation": {35: 30.0, 65: 30.0}, "nitrogen": {1: 50.0, 30: 50.0, 60: 50.0}},
    }
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        scenario_root = ROOT / str(item.scenario_input_root)
        for policy_name, management in policies.items():
            try:
                _daily, summary = run_policy(env_config, scenario_root, str(item.variant), policy_name, management, OUT)
                rows.append(summary)
                if not summary["episode_completed"]:
                    failures.append(summary)
            except Exception as exc:
                failures.append({"variant": str(item.variant), "policy": policy_name, "error": repr(exc)})
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / "164_fixed_management_summary.csv", index=False, encoding="utf-8-sig")
    generation_passed = bool(len(audit) == len(manifest) and bool(audit["passed"].all()))
    completed_passed = bool(len(result_df) == len(manifest) * len(policies) and bool(result_df["episode_completed"].all()))
    collapse_passed = bool(len(result_df) and (pd.to_numeric(result_df["final_grnwt"], errors="coerce") > 1000.0).all())
    rainfed = result_df[result_df["policy"].eq("rainfed_N")].copy()
    limited = result_df[result_df["policy"].eq("limited_I_N")].copy()
    rainfed_response = bool(len(rainfed) and float(pd.to_numeric(rainfed["final_grnwt"], errors="coerce").max() - pd.to_numeric(rainfed["final_grnwt"], errors="coerce").min()) > 1.0)
    limited_response = bool(len(limited) and float(pd.to_numeric(limited["final_grnwt"], errors="coerce").max() - pd.to_numeric(limited["final_grnwt"], errors="coerce").min()) > 1.0)
    response_passed = bool(generation_passed and completed_passed and collapse_passed and rainfed_response)
    summary = {
        "status": "completed_fixed_management_gate",
        "task": f"{TASK_ID}_{TASK_NAME}",
        "source_task": "163_sya_weather_scenario_bank_v2",
        "training_executed": False,
        "scenario_count": int(len(manifest)),
        "policy_count": len(policies),
        "generation_audit_passed": generation_passed,
        "all_episodes_completed": completed_passed,
        "no_low_yield_collapse": collapse_passed,
        "rainfed_yield_response": rainfed_response,
        "limited_irrigation_yield_response": limited_response,
        "physical_response_passed": response_passed,
        "outputs": {
            "weather_audit": rel(OUT / "164_weather_input_audit.csv"),
            "fixed_management_summary": rel(OUT / "164_fixed_management_summary.csv"),
        },
        "failures": failures,
    }
    (OUT / "164_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {TASK_ID} SYA corrected-weather fixed-management gate",
        "",
        "- Scope: SYA, one cultivar/soil input set, source and simulation year 2005.",
        "- Training executed: no.",
        "- Weather factor: RAIN only; six scenarios; independent input roots.",
        "- Fixed policies: rainfed_N and limited_I_N.",
        f"- Generation audit passed: `{generation_passed}`.",
        f"- All episodes completed: `{completed_passed}`.",
        f"- No low-yield collapse: `{collapse_passed}`.",
        f"- Rainfed yield response detected: `{rainfed_response}`.",
        f"- Physical response gate passed: `{response_passed}`.",
        "",
        f"- Weather audit: `{rel(OUT / '164_weather_input_audit.csv')}`.",
        f"- Summary: `{rel(OUT / '164_fixed_management_summary.csv')}`.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    global OUT, DOC
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    OUT = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    DOC = ROOT / "docs" / f"{OUT.name}_record.md"
    print(json.dumps(run(config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
