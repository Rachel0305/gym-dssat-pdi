from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path

import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_literature_aligned_ppo_dqn_ddqn_sy2014_smoke_031_22 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_23_double_dueling_dqn_sy2014_seeds1_2_repeat.yaml"
OUT = ROOT / "benchmark_results" / "031_23_double_dueling_dqn_sy2014_seeds1_2_repeat"
DOC = ROOT / "docs" / "031_23_double_dueling_dqn_sy2014_seeds1_2_repeat_record.md"


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "models/SYA", "daily_outputs/SYA", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def prepare_seed_config(base_config: dict, seed: int) -> dict:
    config = dict(base_config)
    config["seed"] = int(seed)
    config["total_timesteps"] = int(config.get("total_timesteps", 20000))
    paths = dict(config["paths"])
    paths["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["paths"] = paths
    return config


def run_seed(seed: int, base_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = prepare_seed_config(base_config, seed)
    selection = base.make_sy2014_selection()
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    train_df, eval_df, _ = base.train_ddqn(config, env_config)
    train_df["seed"] = int(seed)
    eval_df["seed"] = int(seed)
    train_df["seed_repeat"] = int(seed)
    eval_df["seed_repeat"] = int(seed)
    return train_df, eval_df


def write_record(train: pd.DataFrame, evals: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# 031_23 Double-Dueling DQN SY2014 seed1/seed2 repeat",
        "",
        "## Scope",
        "",
        "- Repeats only the 031_22 mask-aware Double-Dueling DQN configuration.",
        "- SYA2014, seeds 1 and 2, 20k timesteps each.",
        "- No reward/action/constraint/hyperparameter changes.",
        "- Final model only; no checkpoint selection.",
        "",
        "## Training summary",
        "",
        train.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        evals[
            [
                "seed",
                "final_grnwt",
                "total_irrigation",
                "total_n",
                "profit_simple",
                "PFP_N",
                "early_dap1_10_n",
                "nstres_days_gt_0p05",
                "action_sequence",
            ]
        ].to_string(index=False),
        "",
        "## With 031_22 seed0 context",
        "",
        comparison[
            [
                "comparison_label",
                "seed",
                "final_grnwt",
                "total_irrigation",
                "total_n",
                "profit_simple",
                "PFP_N",
                "early_dap1_10_n",
                "nstres_days_gt_0p05",
                "action_sequence",
            ]
        ].to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        "- This is still same-year training/evaluation, not cross-year transfer.",
        "- If seed1/2 remain promising, next step should be frozen transfer to SY2012/SY2015.",
        "- If seed1/2 collapse or front-load N, seed0 should be treated as insufficient evidence.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    meta = direct_ppo.load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    base_config_path = ROOT / meta["base_config"]
    base_config = direct_ppo.load_yaml(base_config_path)
    base_config["total_timesteps"] = int(meta.get("total_timesteps", base_config.get("total_timesteps", 20000)))
    shutil.copyfile(base_config_path, OUT / "configs" / base_config_path.name)

    train_rows = []
    eval_rows = []
    for seed in [int(s) for s in meta["seeds"]]:
        try:
            train_df, eval_df = run_seed(seed, base_config)
        except Exception:
            train_df = pd.DataFrame(
                [
                    {
                        "algorithm": "mask_aware_double_dueling_DQN",
                        "seed": seed,
                        "run_status": "failed",
                        "notes": traceback.format_exc()[-2500:],
                    }
                ]
            )
            eval_df = pd.DataFrame(
                [
                    {
                        "algorithm": "mask_aware_double_dueling_DQN",
                        "seed": seed,
                        "run_status": "failed",
                        "notes": traceback.format_exc()[-2500:],
                    }
                ]
            )
        train_rows.append(train_df)
        eval_rows.append(eval_df)

    train = pd.concat(train_rows, ignore_index=True, sort=False)
    evals = pd.concat(eval_rows, ignore_index=True, sort=False)
    train.to_csv(OUT / "evaluation" / "031_23_ddqn_training_summary.csv", index=False, encoding="utf-8-sig")
    evals.to_csv(OUT / "evaluation" / "031_23_ddqn_eval_summary.csv", index=False, encoding="utf-8-sig")

    old_path = ROOT / "benchmark_results" / "031_22_literature_aligned_ppo_dqn_ddqn_sy2014_smoke" / "evaluation" / "031_22_algorithm_comparison_summary.csv"
    old = pd.read_csv(old_path)
    ddqn0 = old[old["comparison_label"].eq("mask_aware_DoubleDuelingDQN_031_22_20k_seed0")].copy()
    ddqn0["comparison_label"] = "mask_aware_DoubleDuelingDQN_031_22_20k_seed0"
    evals2 = evals.copy()
    evals2["comparison_label"] = evals2["seed"].map(lambda s: f"mask_aware_DoubleDuelingDQN_031_23_20k_seed{int(s)}")
    comparison = pd.concat([ddqn0, evals2], ignore_index=True, sort=False)
    comparison.to_csv(OUT / "evaluation" / "031_23_ddqn_seed0_1_2_comparison.csv", index=False, encoding="utf-8-sig")
    write_record(train, evals, comparison)
    result = {
        "task": "031_23_double_dueling_dqn_sy2014_seeds1_2_repeat",
        "record_md": str(DOC.relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "031_23_ddqn_eval_summary.csv").relative_to(ROOT)),
        "comparison": str((OUT / "evaluation" / "031_23_ddqn_seed0_1_2_comparison.csv").relative_to(ROOT)),
    }
    (OUT / "031_23_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        comparison[
            [
                "comparison_label",
                "seed",
                "final_grnwt",
                "total_irrigation",
                "total_n",
                "profit_simple",
                "PFP_N",
                "early_dap1_10_n",
                "nstres_days_gt_0p05",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
