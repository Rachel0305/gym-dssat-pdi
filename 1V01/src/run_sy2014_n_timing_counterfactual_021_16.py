from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared


EVENT_SOURCE = ROOT / "benchmark_results" / "021_15" / "021_15_management_events_with_phenology.csv"
ENV_SOURCE = (
    ROOT
    / "benchmark_results"
    / "021_14"
    / "021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1"
    / "dqn_env_args.json"
)
OUT_ROOT = ROOT / "benchmark_results" / "021_16"
REFERENCE_YIELDS = {
    "i_early__n_early": 11046.28125,
    "i_late__n_late": 8120.24609375,
}

SCENARIOS = {
    "i_early__n_early": ("unscaled_021_10", "unscaled_021_10"),
    "i_early__n_late": ("unscaled_021_10", "scaled_0p1_021_14"),
    "i_late__n_early": ("scaled_0p1_021_14", "unscaled_021_10"),
    "i_late__n_late": ("scaled_0p1_021_14", "scaled_0p1_021_14"),
}


def load_patterns() -> dict[str, dict[str, dict[int, float]]]:
    frame = pd.read_csv(EVENT_SOURCE)
    frame = frame[pd.to_numeric(frame["checkpoint"], errors="coerce").eq(10000)].copy()
    patterns: dict[str, dict[str, dict[int, float]]] = {}
    for protocol in ("unscaled_021_10", "scaled_0p1_021_14"):
        subset = frame[frame["protocol"].eq(protocol)]
        patterns[protocol] = {}
        for operation in ("irrigation", "nitrogen"):
            rows = subset[subset["operation"].eq(operation)]
            schedule = {
                int(round(float(row.event_dap))): float(row.amount)
                for row in rows.itertuples(index=False)
            }
            patterns[protocol][operation] = schedule
    return patterns


def make_env_args(run_dir: Path) -> dict[str, Any]:
    env_args = json.loads(ENV_SOURCE.read_text(encoding="utf-8"))
    env_args["log_saving_path"] = str(run_dir / "pdi_gym.log")
    return env_args


def copy_snapshot(env: Any, destination: Path) -> None:
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if not tmp or not Path(tmp).exists():
        raise RuntimeError("PDI temporary output directory is unavailable")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {destination}")
    shutil.copytree(Path(tmp), destination)


def parse_management_events(snapshot: Path) -> pd.DataFrame:
    path = snapshot / "MgmtEvent.OUT"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(columns=["raw"])
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" in raw or "Fertil" in raw or "Nitrogen" in raw:
            rows.append({"raw": raw.rstrip()})
    return pd.DataFrame(rows)


def run_one(name: str, patterns: dict[str, dict[str, dict[int, float]]]) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise KeyError(name)
    run_dir = OUT_ROOT / name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)

    i_protocol, n_protocol = SCENARIOS[name]
    irrigation = patterns[i_protocol]["irrigation"]
    nitrogen = patterns[n_protocol]["nitrogen"]
    schedule: dict[int, dict[str, float]] = {}
    for dap, amount in irrigation.items():
        schedule.setdefault(dap, {"amir": 0.0, "anfer": 0.0})["amir"] += amount
    for dap, amount in nitrogen.items():
        schedule.setdefault(dap, {"amir": 0.0, "anfer": 0.0})["anfer"] += amount

    (run_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env_args = make_env_args(run_dir)
    (run_dir / "env_args.json").write_text(
        json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    env = shared.make_raw_env(env_args)
    daily_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    fired: set[int] = set()
    try:
        obs, info = env.reset()
        for step in range(420):
            before = latest_observation_dict(env, obs, info)
            dap_action = int(round(scalar(before.get("dap", step)) or 0))
            if dap_action in schedule and dap_action not in fired:
                real = schedule[dap_action]
                fired.add(dap_action)
            else:
                real = {"amir": 0.0, "anfer": 0.0}
            action = normalize_action(
                env.formator.action_names, env.formator.action_space_dict, real
            )
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            row = {
                "scenario": name,
                "step": step,
                "dap_action": dap_action,
                "dap": scalar(latest.get("dap")),
                "yrdoy": yrdoy,
                "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                "rain": scalar(latest.get("rain")),
                "grnwt": scalar(latest.get("grnwt")),
                "topwt": scalar(latest.get("topwt")),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "irrigation_mm_action": float(real["amir"]),
                "fertilizer_kg_ha_action": float(real["anfer"]),
                "raw_reward": repr(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            daily_rows.append(row)
            if real["amir"] > 0 or real["anfer"] > 0:
                action_rows.append(row.copy())
            if terminated or truncated:
                break
    finally:
        copy_snapshot(env, run_dir / "pdi_tmp_snapshot_eval")
        env.close()

    daily = pd.DataFrame(daily_rows)
    actions = pd.DataFrame(action_rows)
    daily.to_csv(run_dir / "daily_values.csv", index=False)
    actions.to_csv(run_dir / "requested_actions.csv", index=False)
    snapshot = run_dir / "pdi_tmp_snapshot_eval"
    parse_management_events(snapshot).to_csv(run_dir / "mgmt_event_raw.csv", index=False)
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    plantgro.to_csv(run_dir / "plantgro_parsed.csv", index=False)
    gwad = pd.to_numeric(plantgro.get("GWAD", pd.Series(dtype=float)), errors="coerce").dropna()
    cwad = pd.to_numeric(plantgro.get("CWAD", pd.Series(dtype=float)), errors="coerce").dropna()
    final_gwad = float(gwad.iloc[-1]) if not gwad.empty else np.nan
    final_cwad = float(cwad.iloc[-1]) if not cwad.empty else np.nan
    total_i = float(actions.get("irrigation_mm_action", pd.Series(dtype=float)).sum())
    total_n = float(actions.get("fertilizer_kg_ha_action", pd.Series(dtype=float)).sum())
    missing_daps = sorted(set(schedule) - fired)
    reference = REFERENCE_YIELDS.get(name, np.nan)
    summary = {
        "scenario": name,
        "irrigation_pattern": i_protocol,
        "nitrogen_pattern": n_protocol,
        "final_gwad": final_gwad,
        "final_cwad": final_cwad,
        "requested_irrigation_total": total_i,
        "requested_nitrogen_total": total_n,
        "expected_irrigation_total": float(sum(irrigation.values())),
        "expected_nitrogen_total": float(sum(nitrogen.values())),
        "missing_schedule_daps": json.dumps(missing_daps),
        "reference_checkpoint_gwad": reference,
        "absolute_reproduction_error": abs(final_gwad - reference) if np.isfinite(reference) else np.nan,
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS),
        default=list(SCENARIOS),
    )
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns()
    manifest = {
        "event_source": str(EVENT_SOURCE.relative_to(ROOT)),
        "env_source": str(ENV_SOURCE.relative_to(ROOT)),
        "patterns": patterns,
    }
    manifest_path = OUT_ROOT / "input_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summaries = [run_one(name, patterns) for name in args.scenarios]
    current = pd.DataFrame(summaries)
    summary_path = OUT_ROOT / "021_16_summary.csv"
    if summary_path.exists():
        prior = pd.read_csv(summary_path)
        current = pd.concat([prior, current], ignore_index=True)
        current = current.drop_duplicates(subset=["scenario"], keep="last")
    current.to_csv(summary_path, index=False)
    print(current.to_string(index=False))


if __name__ == "__main__":
    main()
