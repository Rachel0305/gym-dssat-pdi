from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_01_free_daily_original_reward_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_01_free_daily_original_reward_smoke"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_01_single_smoke_SY2014_free_daily_original_reward"
    return row


def main() -> None:
    # Force all reused direct-PPO helper outputs into the 031_01 folder.
    direct_ppo.OUTPUT_ROOT = OUT
    direct_ppo.DOC_MD = OUT / "031_01_legacy_direct_ppo_report.md"
    direct_ppo.DOC_PPT = OUT / "031_01_legacy_direct_ppo_report.pptx"

    direct_ppo.ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    selection = make_sy2014_selection()
    selection.to_csv(OUT / "configs" / "031_01_sy2014_smoke_selection.csv", index=False, encoding="utf-8-sig")

    env_config = direct_ppo.build_env_config(config, selection)
    env_config_path = OUT / "configs" / "031_01_resolved_env_config.yaml"
    direct_ppo.write_yaml(env_config, env_config_path)

    train_summary = direct_ppo.train_station_models(config, env_config, selection, debug=False)
    eval_summary = direct_ppo.evaluate_models(config, env_config, selection, train_summary)
    if not eval_summary.empty and "run_status" in eval_summary.columns:
        figures = direct_ppo.plot_representative_outputs(config, env_config, eval_summary)
        diagnosis = direct_ppo.build_reasonableness(config, eval_summary)
    else:
        figures = pd.DataFrame()
        diagnosis = pd.DataFrame()

    result = {
        "task": "031_01_free_daily_original_reward_smoke",
        "training_or_dssat_run": True,
        "config": str(CONFIG.relative_to(ROOT)),
        "selection": str((OUT / "configs" / "031_01_sy2014_smoke_selection.csv").relative_to(ROOT)),
        "env_config": str(env_config_path.relative_to(ROOT)),
        "train_summary": str((OUT / "evaluation" / "training_run_summary.csv").relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "ppo_direct_eval_summary.csv").relative_to(ROOT)),
        "figures_csv": str((OUT / "figures" / "representative_figure_index.csv").relative_to(ROOT)),
        "diagnosis": str((OUT / "evaluation" / "ppo_decision_reasonableness_diagnosis.csv").relative_to(ROOT)),
        "train_status": train_summary.to_dict(orient="records"),
        "eval_status": eval_summary.to_dict(orient="records"),
        "n_figures": int(len(figures)),
        "n_diagnosis_rows": int(len(diagnosis)),
        "free_daily": True,
        "min_interval_days": 1,
        "reward_formula": "delta_GRNWT - 1.0*irrigation - 5.0*nitrogen",
    }
    (OUT / "031_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# 031_01 Free-daily original-reward smoke record",
        "",
        "## Run",
        "",
        "- Smoke case: SYA2014 seed0.",
        "- Free daily decision: yes.",
        "- Expert DAP windows: no.",
        "- Minimum operation interval: 1 day.",
        "- Reward: `delta_GRNWT - 1.0 * irrigation - 5.0 * nitrogen`.",
        "- Training timesteps: 512.",
        "",
        "## Training summary",
        "",
        train_summary.to_markdown(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_markdown(index=False) if not eval_summary.empty else "No evaluation rows.",
        "",
        "## Decision diagnosis",
        "",
        diagnosis.to_markdown(index=False) if not diagnosis.empty else "No diagnosis rows.",
        "",
        "## Outputs",
        "",
        f"- Train summary: `{result['train_summary']}`",
        f"- Eval summary: `{result['eval_summary']}`",
        f"- Daily outputs: `benchmark_results/031_01_free_daily_original_reward_smoke/daily_outputs/`",
        f"- Figures: `benchmark_results/031_01_free_daily_original_reward_smoke/figures/`",
    ]
    (OUT / "031_01_free_daily_original_reward_smoke_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
