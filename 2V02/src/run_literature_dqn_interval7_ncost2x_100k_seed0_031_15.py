from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_literature_dqn_interval7_ncost2x_031_13 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_15_literature_dqn_interval7_ncost2x_100k_seed0.yaml"
OUT = ROOT / "benchmark_results" / "031_15_literature_dqn_interval7_ncost2x_100k_seed0"
DOC = ROOT / "docs" / "031_15_literature_dqn_interval7_ncost2x_100k_seed0_record.md"
SEEDS = [0]


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs", "tensorboard/SYA"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    direct_ppo.OUTPUT_ROOT = OUT
    base.OUT = OUT
    base.DOC = DOC
    base.CONFIG = CONFIG
    base.SEEDS = SEEDS

    selection = base.make_sy2014_selection()
    selection["selection_reason"] = "031_15_literature_dqn_interval7_ncost2x_100k_seed0"
    selection_path = OUT / "configs" / "031_15_sy2014_selection.csv"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
    env_config = direct_ppo.build_env_config(config, selection)
    env_config_path = OUT / "configs" / "031_15_resolved_env_config.yaml"
    direct_ppo.write_yaml(env_config, env_config_path)

    year = 2014
    train_rows = []
    eval_rows = []
    for seed in SEEDS:
        train_row = base.train_seed(config, env_config, seed, year)
        train_row["task"] = "031_15_literature_dqn_interval7_ncost2x_100k_seed0"
        train_rows.append(train_row)
        if train_row["run_status"] == "ok":
            eval_row = base.evaluate_seed(config, env_config, seed, year, ROOT / train_row["model_path"])
            eval_row["task"] = "031_15_literature_dqn_interval7_ncost2x_100k_seed0"
            eval_row["policy_name"] = "lit_dqn_interval7_ncost2x_100k_seed0"
            eval_rows.append(eval_row)
        else:
            eval_rows.append(
                {
                    "task": "031_15_literature_dqn_interval7_ncost2x_100k_seed0",
                    "algorithm": "literature_DQN_interval7_ncost2x",
                    "seed": seed,
                    "split": "eval",
                    "run_status": "failed",
                    "notes": train_row.get("notes", ""),
                }
            )

    train_summary = pd.DataFrame(train_rows)
    eval_summary = pd.DataFrame(eval_rows)
    train_path = OUT / "evaluation" / "training_run_summary.csv"
    eval_path = OUT / "evaluation" / "eval_summary.csv"
    train_summary.to_csv(train_path, index=False, encoding="utf-8-sig")
    eval_summary.to_csv(eval_path, index=False, encoding="utf-8-sig")

    prior_rows = []
    for prior in [
        ROOT / "benchmark_results" / "031_13_literature_dqn_interval7_ncost2x" / "evaluation" / "eval_summary.csv",
        ROOT / "benchmark_results" / "031_12_dqn_nitrogen_counterfactual_audit" / "evaluation" / "031_12_nitrogen_counterfactual_summary.csv",
    ]:
        if prior.exists():
            df = pd.read_csv(prior)
            df["source_file"] = str(prior.relative_to(ROOT))
            prior_rows.append(df)
    comparison = eval_summary.copy()
    comparison["source_file"] = str(eval_path.relative_to(ROOT))
    if prior_rows:
        comparison = pd.concat(prior_rows + [comparison], ignore_index=True, sort=False)
    comparison_path = OUT / "evaluation" / "031_15_with_prior_context.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 031_15 Literature DQN interval7 ncost2x 100k seed0 record",
        "",
        "## Scope",
        "",
        "- SYA2014 only.",
        "- Seed0 only.",
        "- Same DQN/reward/action constraints as 031_13.",
        "- Single changed variable relative to 031_13: total_timesteps 5000 -> 100000.",
        "- This is not a training-step scan and not a new nitrogen-cost scan.",
        "",
        "## Training summary",
        "",
        train_summary.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        "- If seed0 still uses N250 with early-heavy nitrogen, longer training alone did not fix the learned ranking.",
        "- If seed0 moves toward staged N200 while preserving yield, then 5k was likely too short and seed1/2 should be tested next under the same 100k setup.",
        "",
        "## Files",
        "",
        f"- Training summary: `{train_path.relative_to(ROOT)}`",
        f"- Evaluation summary: `{eval_path.relative_to(ROOT)}`",
        f"- Prior-context comparison: `{comparison_path.relative_to(ROOT)}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "task": "031_15_literature_dqn_interval7_ncost2x_100k_seed0",
        "training_run": True,
        "site_year": "SYA2014",
        "seeds": SEEDS,
        "changed_variable": "total_timesteps 5000 -> 100000",
        "record_md": str(DOC.relative_to(ROOT)),
        "eval_summary": str(eval_path.relative_to(ROOT)),
        "comparison": str(comparison_path.relative_to(ROOT)),
    }
    result_path = OUT / "031_15_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if len(eval_summary):
        print(eval_summary.to_string(index=False))


if __name__ == "__main__":
    main()

