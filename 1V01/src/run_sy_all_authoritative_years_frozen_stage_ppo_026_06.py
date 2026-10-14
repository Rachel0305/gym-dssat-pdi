from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calculate_five_site_wue_nue_from_summary_019_10 import num, parse_summary_out
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    set_management_for_treatment,
    set_treatment_pointers,
    zero_target_reported_rows,
)
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import FixedObservationScaler
from stage_decision_env_ppo_026 import StageDecisionEnv026
import run_extension_expert_baseline_018_03 as expert
import run_sy_local_dqn_train_cross_year_transfer_017_08 as sy


OUT = ROOT / "benchmark_results" / "026_06"
SMOKE_OUT = ROOT / "benchmark_results" / "026_06_smoke_2015_null"
INPUT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY"
MZX = INPUT / "CNSY1201.MZX"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
REUSE_2012 = ROOT / "benchmark_results" / "026_05"
YEARS = (2012, 2014, 2015)
FRESH_YEARS = (2014, 2015)
EXPECTED_TREATMENTS = {2012: (1, 1), 2014: (2, 2), 2015: (3, 1)}
MODELS = {
    0: ROOT / "benchmark_results" / "026_03" / "checkpoint_000120.zip",
    1: ROOT / "benchmark_results" / "026_02" / "checkpoint_000060.zip",
    2: ROOT / "benchmark_results" / "026_04" / "checkpoint_000240.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def treatment_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    text = MZX.read_text(encoding="latin-1", errors="ignore")
    lines = text.splitlines()
    ic_dates: dict[int, int] = {}
    planting_dates: dict[int, int] = {}
    simulation_dates: dict[int, int] = {}
    section = ""
    for raw in lines:
        if raw.startswith("*"):
            if raw.startswith("*INITIAL CONDITIONS"):
                section = "initial"
            elif raw.startswith("*PLANTING DETAILS"):
                section = "planting"
            elif raw.startswith("*SIMULATION CONTROLS"):
                section = "simulation"
            else:
                section = ""
            continue
        parts = raw.split()
        if section == "initial" and len(parts) >= 3 and parts[0].isdigit() and parts[1] == "MZ" and parts[2].isdigit():
            ic_dates[int(parts[0])] = int(parts[2])
        elif section == "planting" and len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            planting_dates[int(parts[0])] = int(parts[1])
        elif section == "simulation" and len(parts) >= 6 and parts[0].isdigit() and parts[1] == "GE" and parts[5].isdigit():
            simulation_dates[int(parts[0])] = int(parts[5])
    pattern = re.compile(r"^\s*(\d+)\s+\d+\s+\d+\s+\d+\s+Sim(\d{4})\s+(.+)$")
    for raw in lines:
        match = pattern.match(raw)
        if not match:
            continue
        treatment = int(match.group(1))
        year = int(match.group(2))
        factors = match.group(3).split()
        ic_pointer = int(factors[3]) if len(factors) >= 4 else -1
        weather = INPUT / f"CNSY{year % 100:02d}01.WTH"
        expected = EXPECTED_TREATMENTS.get(year)
        icdat = ic_dates.get(ic_pointer)
        sdate = simulation_dates.get(treatment)
        pdate = planting_dates.get(treatment)
        dates_same_year = bool(icdat and sdate and pdate and icdat // 1000 == sdate // 1000 == pdate // 1000 == year % 100)
        dates_ordered = bool(icdat and sdate and pdate and sdate <= icdat <= pdate)
        rows.append(
            {
                "year": year,
                "treatment": treatment,
                "ic_pointer": ic_pointer,
                "expected_treatment": expected[0] if expected else np.nan,
                "expected_ic_pointer": expected[1] if expected else np.nan,
                "registered_authoritative_year": year in EXPECTED_TREATMENTS,
                "treatment_matches": bool(expected and treatment == expected[0]),
                "ic_pointer_matches": bool(expected and ic_pointer == expected[1]),
                "weather_exists": weather.exists(),
                "weather_sha256": sha256(weather) if weather.exists() else "",
                "mzx_sha256": sha256(MZX),
                "sdate": sdate,
                "icdat": icdat,
                "pdate": pdate,
                "dates_same_year": dates_same_year,
                "dates_ordered": dates_ordered,
                "date_chain_pass": bool(dates_same_year and dates_ordered),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def metrics_from_snapshot(snapshot: Path, final_yield: float, irrigation: float, nitrogen: float) -> dict[str, Any]:
    rows = parse_summary_out(snapshot / "Summary.OUT")
    scored = []
    for index, candidate in enumerate(rows):
        hwam, ircm = num(candidate, "HWAM"), num(candidate, "IRCM")
        if hwam is None or ircm is None:
            continue
        score = abs(hwam - final_yield) + abs(ircm - irrigation)
        scored.append((score, -index, candidate, index))
    if not scored:
        raise ValueError("No Summary.OUT row contains numeric HWAM/IRCM")
    score, _, row, row_index = min(scored, key=lambda item: (item[0], item[1]))
    if abs(float(num(row, "HWAM")) - final_yield) > 2.0:
        raise ValueError(f"Yield mismatch: expected {final_yield}, got {num(row, 'HWAM')}")
    if abs(float(num(row, "IRCM")) - irrigation) > 2.0:
        raise ValueError(f"Irrigation mismatch: expected {irrigation}, got {num(row, 'IRCM')}")
    ircm, nicm = num(row, "IRCM"), num(row, "NICM")
    etcp, ypem, ypnam = num(row, "ETCP"), num(row, "YPEM"), num(row, "YPNAM")
    wp = ypem * 0.1 if ypem is not None and ypem >= 0 else final_yield / float(etcp) / 10.0
    pfp = ypnam if nicm is not None and nicm > 0 and ypnam is not None and ypnam >= 0 else math.nan
    return {
        "summary_irrigation_total": ircm,
        "summary_nitrogen_total": nicm,
        "etcp_mm": etcp,
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
        "summary_match_score": score,
        "summary_row_index": row_index,
        "management_or_action_n_total": nitrogen,
        "n_accounting_difference_vs_summary": nitrogen - float(nicm) if nicm is not None else math.nan,
    }


class TransferStageEnv(StageDecisionEnv026):
    def __init__(self, raw_env: Any, scaler_path: Path) -> None:
        gym.Env.__init__(self)
        self.raw_env = raw_env
        self.scaler = FixedObservationScaler(scaler_path)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)
        self.obs = None
        self.state: dict[str, Any] = {}
        self.stage_index = 0
        self.used_i = 0.0
        self.used_n = 0.0
        self.executed = []
        self.stage_rows: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None
        self.invalid_attempts = 0

    def step(self, action_index: int):
        mask = self.action_masks()
        index = int(action_index)
        if index < 0 or index >= len(mask) or not bool(mask[index]):
            self.invalid_attempts += 1
        return super().step(index)


def prepare_linked_run(year: int, scenario: str, seed: int, root: Path) -> tuple[Path, dict[str, Any]]:
    preparation_scenario = scenario if scenario.startswith(("dqn", "transfer")) else f"transfer_{scenario}"
    run_dir = sy.prepare_run_dir(year, preparation_scenario, seed=seed, root=root)
    input_dir = run_dir / "input"
    filex = input_dir / "CNSY1201.MZX"
    trno = sy.YEARS[year]
    source = MZX.read_text(encoding="latin-1", errors="ignore")
    text = set_treatment_pointers(source, trno, "1", "1")
    text = set_management_for_treatment(text, trno, "L", "L")
    text = zero_target_reported_rows(text, trno)
    filex.write_text(text, encoding="latin-1", errors="ignore")
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env_args["fileX_template_path"] = str(filex)
    (run_dir / "env_args.json").write_text(json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, env_args


def run_baselines(year: int) -> list[dict[str, Any]]:
    sy.OUT_DIR = OUT / str(year) / "baseline_source"
    sy.configure_globals()
    rows: list[dict[str, Any]] = []
    for scenario in ("null", "recorded", "dssat_auto"):
        _, _, summary = sy.run_zero_action(year, scenario)
        snapshot = ROOT / summary["run_dir"] / "pdi_tmp_snapshot_eval"
        metrics = metrics_from_snapshot(snapshot, summary["final_gwad"], summary["irrigation_total"], summary["fertilizer_total"])
        rows.append({"year": year, "scenario": scenario, **summary, **metrics})

    case = {"site": "SY", "station": "Shenyang", "year": year, "region": "northeast_greatwall_spring_maize"}
    schedule = expert.build_region_schedule()
    schedule = schedule[schedule["region"].eq(case["region"])].copy()
    expert_run, env_args = prepare_linked_run(year, "official_extension_expert", 0, OUT / str(year) / "expert_source" / "runs")
    _, _, summary = expert.run_fixed_schedule(case, env_args, schedule, expert_run)
    snapshot = expert_run / "pdi_tmp_snapshot_eval"
    irrigation = float(summary["action_irrigation_total"])
    nitrogen = float(summary["action_fertilizer_total"])
    metrics = metrics_from_snapshot(snapshot, summary["final_gwad"], irrigation, nitrogen)
    rows.append(
        {
            "year": year,
            "scenario": "official_extension_expert",
            "site": "SY",
            "station": "Shenyang",
            "final_gwad": summary["final_gwad"],
            "final_cwad": summary["final_cwad"],
            "irrigation_total": irrigation,
            "fertilizer_total": nitrogen,
            "run_dir": str(expert_run.relative_to(ROOT)),
            **metrics,
        }
    )
    return rows


def local_targets(baselines: pd.DataFrame) -> dict[str, float]:
    main = baselines[baselines["scenario"].isin(["dssat_auto", "official_extension_expert"])].copy()
    positive_n = main[pd.to_numeric(main["summary_nitrogen_total"], errors="coerce") > 0]
    if len(main) != 2 or positive_n.empty:
        raise ValueError("Missing DSSAT auto or positive-N official expert baseline")
    return {
        "yield_min": float(main["final_gwad"].max()),
        "wp_et_min": float(main["WP_ET_kg_m3"].max()),
        "pfp_n_min": float(positive_n["PFP_N_kg_kg"].max()),
    }


def run_frozen_model(year: int, seed: int, model_path: Path, targets: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario = f"frozen_ppo_seed{seed}"
    run_dir, env_args = prepare_linked_run(year, scenario, seed, OUT / str(year) / "ppo_runs")
    env = TransferStageEnv(sy.make_raw_env(env_args), SCALER)
    model_hash_before = sha256(model_path)
    actions: list[dict[str, Any]] = []
    total_reward = 0.0
    try:
        model = MaskablePPO.load(model_path, device="cpu")
        obs, info = env.reset()
        done = False
        while not done:
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            stage_index = env.stage_index
            dap = int(info.get("dap", env._dap()))
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            executed = env.stage_rows[-1]
            actions.append(
                {
                    "year": year,
                    "seed": seed,
                    "stage_index": stage_index,
                    "dap": dap,
                    "action_index": int(action),
                    "mask_valid": bool(mask[int(action)]),
                    **executed,
                }
            )
            done = bool(terminated or truncated)
        if env.last_result is None:
            raise RuntimeError("Frozen PPO season ended without result")
        tmp = Path(getattr(env.raw_env.unwrapped, "_tmp_folder"))
        metrics = metrics_from_snapshot(tmp, env.last_result["final_yield"], env.last_result["irrigation_total"], env.last_result["nitrogen_total"])
        result = {
            "year": year,
            "scenario": scenario,
            "seed": seed,
            "source_model": str(model_path.relative_to(ROOT)),
            "model_sha256": model_hash_before,
            "action_sequence": ",".join(str(row["action_index"]) for row in actions),
            "final_gwad": env.last_result["final_yield"],
            "final_cwad": env.last_result["final_biomass"],
            "irrigation_total": env.last_result["irrigation_total"],
            "fertilizer_total": env.last_result["nitrogen_total"],
            "frozen_training_objective_return": total_reward,
            "invalid_action_attempts": env.invalid_attempts,
            **metrics,
        }
        result["yield_pass"] = result["final_gwad"] >= targets["yield_min"]
        result["wp_et_pass"] = result["WP_ET_kg_m3"] >= targets["wp_et_min"]
        result["pfp_n_pass"] = result["PFP_N_kg_kg"] >= targets["pfp_n_min"]
        result["local_primary_pass"] = bool(result["yield_pass"] and result["wp_et_pass"] and result["pfp_n_pass"])
    finally:
        env.close()
    if sha256(model_path) != model_hash_before:
        raise RuntimeError(f"Frozen model hash changed: {model_path}")
    return result, actions


def load_reused_2012() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    payload = json.loads((REUSE_2012 / "026_05_result.json").read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("branch") != "A_fixed_models_transfer":
        raise ValueError("026_05 is not a completed A-branch source")
    baselines = pd.read_csv(REUSE_2012 / "026_05_sy2012_four_baselines.csv")
    models = pd.read_csv(REUSE_2012 / "026_05_frozen_ppo_transfer_summary.csv")
    actions = pd.read_csv(REUSE_2012 / "026_05_frozen_ppo_stage_actions.csv")
    for frame in (baselines, models, actions):
        frame["year"] = 2012
    if set(payload["model_hashes"].values()) != {sha256(path) for path in MODELS.values()}:
        raise ValueError("Current checkpoint hashes differ from 026_05")
    return baselines, models, actions, payload


def run_smoke() -> None:
    if SMOKE_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {SMOKE_OUT}")
    audit = treatment_audit()
    row = audit[audit["year"].eq(2015)].iloc[0]
    if not bool(row["date_chain_pass"]):
        raise ValueError(f"SY2015 date-chain audit failed before DSSAT: SDATE={row['sdate']} ICDAT={row['icdat']} PDATE={row['pdate']}")
    sy.OUT_DIR = SMOKE_OUT
    sy.configure_globals()
    _, _, summary = sy.run_zero_action(2015, "null")
    snapshot = ROOT / summary["run_dir"] / "pdi_tmp_snapshot_eval"
    metrics = metrics_from_snapshot(snapshot, summary["final_gwad"], summary["irrigation_total"], summary["fertilizer_total"])
    payload = {"status": "passed", "year": 2015, "scenario": "null", "summary": summary, "metrics": metrics, "training_steps": 0}
    (SMOKE_OUT / "026_06_smoke_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    missing = [str(path) for path in [MZX, SCALER, *MODELS.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    audit = treatment_audit()
    audit_years = set(audit.loc[audit["registered_authoritative_year"], "year"].astype(int))
    audit_pass = (
        audit_years == set(YEARS)
        and bool(audit.loc[audit["registered_authoritative_year"], ["treatment_matches", "ic_pointer_matches", "weather_exists", "date_chain_pass"]].all().all())
    )
    OUT.mkdir(parents=True)
    audit.to_csv(OUT / "026_06_sy_treatment_year_input_audit.csv", index=False)
    if not audit_pass:
        payload = {
            "status": "partial",
            "branch": "C_input_or_execution_blocked",
            "eligible_years": list(YEARS),
            "blocking_reason": "SDATE/ICDAT/PDATE chain is not year-aligned for every authoritative treatment",
            "training_steps": 0,
            "dssat_calls_in_formal_run": 0,
            "next_step_allowed": False,
        }
        (OUT / "026_06_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    model_hashes = {str(seed): sha256(path) for seed, path in MODELS.items()}

    all_baselines: list[pd.DataFrame] = []
    all_models: list[pd.DataFrame] = []
    all_actions: list[pd.DataFrame] = []
    reused_baselines, reused_models, reused_actions, reused_payload = load_reused_2012()
    all_baselines.append(reused_baselines)
    all_models.append(reused_models)
    all_actions.append(reused_actions)

    year_payloads: dict[str, Any] = {
        "2012": {
            "source": "benchmark_results/026_05",
            "reused": True,
            "targets": reused_payload["local_targets"],
            "pass_count": int(reused_payload["local_primary_pass_count"]),
        }
    }

    for year in FRESH_YEARS:
        baseline_frame = pd.DataFrame(run_baselines(year))
        if set(baseline_frame["scenario"]) != {"null", "recorded", "dssat_auto", "official_extension_expert"}:
            raise ValueError(f"{year}: four-baseline set is incomplete")
        targets = local_targets(baseline_frame)
        model_rows: list[dict[str, Any]] = []
        action_rows: list[dict[str, Any]] = []
        for seed, model_path in MODELS.items():
            result, actions = run_frozen_model(year, seed, model_path, targets)
            model_rows.append(result)
            action_rows.extend(actions)
        model_frame = pd.DataFrame(model_rows)
        action_frame = pd.DataFrame(action_rows)
        baseline_frame.to_csv(OUT / f"026_06_sy{year}_four_baselines.csv", index=False)
        model_frame.to_csv(OUT / f"026_06_sy{year}_frozen_ppo_summary.csv", index=False)
        action_frame.to_csv(OUT / f"026_06_sy{year}_frozen_ppo_stage_actions.csv", index=False)
        all_baselines.append(baseline_frame)
        all_models.append(model_frame)
        all_actions.append(action_frame)
        year_payloads[str(year)] = {
            "source": "fresh_026_06",
            "reused": False,
            "targets": targets,
            "pass_count": int(model_frame["local_primary_pass"].sum()),
        }

    baselines = pd.concat(all_baselines, ignore_index=True, sort=False)
    models = pd.concat(all_models, ignore_index=True, sort=False)
    actions = pd.concat(all_actions, ignore_index=True, sort=False)
    baselines.to_csv(OUT / "026_06_sy_all_years_four_baselines.csv", index=False)
    models.to_csv(OUT / "026_06_sy_all_years_frozen_ppo_summary.csv", index=False)
    actions.to_csv(OUT / "026_06_sy_all_years_frozen_ppo_stage_actions.csv", index=False)
    pd.concat([baselines.assign(group="baseline"), models.assign(group="frozen_ppo")], ignore_index=True, sort=False).to_csv(
        OUT / "026_06_sy_all_years_all_scenarios.csv", index=False
    )

    recorded_yield = baselines.loc[baselines["scenario"].eq("recorded"), ["year", "final_gwad"]].rename(columns={"final_gwad": "recorded_yield"})
    models = models.merge(recorded_yield, on="year", how="left")
    models["yield_ge_recorded"] = models["final_gwad"] >= models["recorded_yield"]
    matrix = models.pivot(index="year", columns="seed", values="local_primary_pass").reset_index()
    matrix.columns = ["year", *[f"seed{int(col)}_local_primary" for col in matrix.columns[1:]]]
    matrix["pass_count"] = matrix.filter(like="_local_primary").sum(axis=1).astype(int)
    matrix["year_transfer_pass"] = matrix["pass_count"] >= 2
    matrix.to_csv(OUT / "026_06_sy_year_seed_pass_matrix.csv", index=False)

    engineering_checks = {
        "authoritative_years_exactly_2012_2014_2015": audit_years == set(YEARS),
        "all_treatment_ic_weather_checks_pass": audit_pass,
        "2012_reused_from_completed_026_05": reused_payload.get("branch") == "A_fixed_models_transfer",
        "all_three_models_each_year": bool(models.groupby("year")["seed"].nunique().eq(3).all()),
        "all_models_hash_unchanged": all(sha256(MODELS[int(seed)]) == digest for seed, digest in model_hashes.items()),
        "zero_training_steps": True,
        "zero_invalid_actions": int(pd.to_numeric(models["invalid_action_attempts"], errors="coerce").sum()) == 0,
        "all_model_metrics_finite": bool(
            np.isfinite(models[["final_gwad", "irrigation_total", "fertilizer_total", "WP_ET_kg_m3", "PFP_N_kg_kg"]].to_numpy(dtype=float)).all()
        ),
    }
    engineering_pass = all(engineering_checks.values())
    all_years_pass = bool(matrix["year_transfer_pass"].all())
    branch = "A_SY_all_years_transfer" if engineering_pass and all_years_pass else ("B_partial_year_transfer" if engineering_pass else "C_input_or_execution_blocked")
    payload = {
        "status": "completed",
        "branch": branch,
        "eligible_years": list(YEARS),
        "weather_only_years_excluded": [year for year in range(2001, 2024) if year not in YEARS],
        "model_hashes": model_hashes,
        "year_results": year_payloads,
        "engineering_checks": engineering_checks,
        "all_years_at_least_two_of_three": all_years_pass,
        "training_steps": 0,
        "scientific_success_claimed": False,
        "next_step_allowed": bool(engineering_pass),
    }
    (OUT / "026_06_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        run_smoke()
    else:
        main()
