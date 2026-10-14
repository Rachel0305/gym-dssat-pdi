from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_daily_original_reward_dqn_smoke_031_02 as dqn031


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "031_04_free_daily_original_reward_5k_train"
CONFIG_PPO = ROOT / "experiments" / "ppo_observed_years" / "config_031_04_free_daily_original_reward_5k_ppo.yaml"
CONFIG_DQN = ROOT / "experiments" / "ppo_observed_years" / "config_031_04_free_daily_original_reward_5k_dqn.yaml"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_04_single_smoke_SY2014_free_daily_original_reward_5k"
    return row


def ensure_root() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def run_ppo(selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = OUT / "ppo"
    direct_ppo.OUTPUT_ROOT = out
    direct_ppo.DOC_MD = out / "031_04_ppo_legacy_report.md"
    direct_ppo.DOC_PPT = out / "031_04_ppo_legacy_report.pptx"
    direct_ppo.ensure_dirs()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_PPO, out / "configs" / CONFIG_PPO.name)
    config = direct_ppo.load_yaml(CONFIG_PPO)
    selection.to_csv(out / "configs" / "031_04_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, out / "configs" / "031_04_resolved_env_config.yaml")
    train_summary = direct_ppo.train_station_models(config, env_config, selection, debug=False)
    eval_summary = direct_ppo.evaluate_models(config, env_config, selection, train_summary)
    diagnosis = direct_ppo.build_reasonableness(config, eval_summary) if not eval_summary.empty else pd.DataFrame()
    diagnosis.to_csv(out / "evaluation" / "ppo_decision_reasonableness_diagnosis.csv", index=False, encoding="utf-8-sig")
    return train_summary, eval_summary, diagnosis


def run_dqn(selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = OUT / "dqn"
    dqn031.OUT = out
    dqn031.ensure_dirs()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_DQN, out / "configs" / CONFIG_DQN.name)
    config = direct_ppo.load_yaml(CONFIG_DQN)
    selection.to_csv(out / "configs" / "031_04_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = out
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, out / "configs" / "031_04_resolved_env_config.yaml")
    train_summary = dqn031.train_model(config, env_config, selection)
    eval_summary = dqn031.evaluate_model(config, env_config, selection, train_summary)
    diagnosis = dqn031.build_diagnosis(config, eval_summary) if not eval_summary.empty else pd.DataFrame()
    return train_summary, eval_summary, diagnosis


def load_random_baseline() -> pd.DataFrame:
    path = ROOT / "benchmark_results" / "031_03_free_daily_original_reward_random_baseline" / "evaluation" / "031_03_random_noop_baseline_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["source"] = "031_03"
        return df
    return pd.DataFrame()


def combined_eval(ppo_eval: pd.DataFrame, dqn_eval: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not ppo_eval.empty:
        ppo = ppo_eval[ppo_eval["split"].eq("eval")].copy()
        ppo["source"] = "031_04"
        ppo["policy_name"] = "ppo_5k"
        rows.append(ppo)
    if not dqn_eval.empty:
        dqn = dqn_eval[dqn_eval["split"].eq("eval")].copy()
        dqn["source"] = "031_04"
        dqn["policy_name"] = "dqn_5k"
        rows.append(dqn)
    baseline = load_random_baseline()
    if not baseline.empty:
        rows.append(baseline)
    if not rows:
        return pd.DataFrame()
    raw = pd.concat(rows, ignore_index=True, sort=False)
    cols = [
        "source",
        "policy_name",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "profit_simple",
        "irrigation_event_count",
        "n_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
        "daily_csv_path",
    ]
    return raw[[c for c in cols if c in raw.columns]]


def main() -> None:
    ensure_root()
    selection = make_sy2014_selection()

    ppo_train, ppo_eval, ppo_diag = run_ppo(selection)
    dqn_train, dqn_eval, dqn_diag = run_dqn(selection)

    comparison = combined_eval(ppo_eval, dqn_eval)
    comparison_path = OUT / "031_04_ppo_dqn_5k_vs_baselines.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    result = {
        "task": "031_04_free_daily_original_reward_5k_train",
        "training_or_dssat_run": True,
        "site_year": "SYA2014",
        "seed": 0,
        "timesteps": 5000,
        "ppo_train": ppo_train.to_dict(orient="records"),
        "ppo_eval": ppo_eval.to_dict(orient="records"),
        "ppo_diagnosis": ppo_diag.to_dict(orient="records"),
        "dqn_train": dqn_train.to_dict(orient="records"),
        "dqn_eval": dqn_eval.to_dict(orient="records"),
        "dqn_diagnosis": dqn_diag.to_dict(orient="records"),
        "comparison": str(comparison_path.relative_to(ROOT)),
        "reward_formula": "delta_GRNWT - 1.0*irrigation - 5.0*nitrogen",
        "interpretation_scope": "SYA2014 seed0 5k training check only; no reward tuning.",
    }
    (OUT / "031_04_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# 031_04 Free-daily original-reward 5k training record",
        "",
        "## Scope",
        "",
        "- SYA2014 seed0 only.",
        "- PPO continuous action and DQN 3x3 discrete action.",
        "- 5,000 timesteps each.",
        "- Same free-daily original reward as 031_01/031_02/031_03.",
        "- No expert DAP windows; no 7-day minimum interval.",
        "",
        "## PPO training summary",
        "",
        ppo_train.to_string(index=False),
        "",
        "## PPO eval summary",
        "",
        ppo_eval.to_string(index=False),
        "",
        "## DQN training summary",
        "",
        dqn_train.to_string(index=False),
        "",
        "## DQN eval summary",
        "",
        dqn_eval.to_string(index=False),
        "",
        "## Combined comparison",
        "",
        comparison.to_string(index=False) if not comparison.empty else "No comparison rows.",
        "",
        "## Interpretation boundary",
        "",
        "This is a longer single-site-year smoke. It can show whether 5k training escapes the random cap-saturation baseline, but it cannot establish cross-year or cross-seed stability.",
    ]
    (OUT / "031_04_free_daily_original_reward_5k_train_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
