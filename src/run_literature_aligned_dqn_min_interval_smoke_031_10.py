from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_literature_aligned_dqn_free_timing_smoke_031_09 as lit


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_10_literature_aligned_dqn_min_interval_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_10_literature_aligned_dqn_min_interval_smoke"
DOC = ROOT / "docs" / "031_10_literature_aligned_dqn_min_interval_smoke_record.md"


def write_record(train_summary: pd.DataFrame, eval_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# 031_10 Literature-aligned DQN with 7-day minimum interval smoke record",
        "",
        "## Scope",
        "",
        "- SYA2014 seed0 only.",
        "- Algorithm: SB3 DQN with the same literature-aligned reconstruction as 031_09.",
        "- Single changed variable from 031_09: same-resource minimum operation interval is 7 days instead of 1 day.",
        "- Free daily timing remains enabled; expert DAP windows are not used.",
        "- Training timesteps: 5,000.",
        "- Action grid: irrigation `{0,6,12,18,24}` mm x nitrogen `{0,40,80,120,160}` kg/ha.",
        "- Reward: terminal `0.158*Y - 0.79*N - 1.1*W`; non-terminal `-0.79*N - 1.1*W`.",
        "",
        "## Training summary",
        "",
        train_summary.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_string(index=False) if not eval_summary.empty else "No evaluation rows.",
        "",
        "## Comparison rows",
        "",
        comparison.to_string(index=False) if not comparison.empty else "No comparison rows.",
        "",
        "## Interpretation boundary",
        "",
        "This is a one-seed, one-site-year smoke. It can test whether a 7-day hard interval reduces early dump/action clipping relative to 031_09, but it cannot establish cross-seed or cross-year success.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Reuse 031_09 implementation, but route all constants to 031_10 paths.
    lit.CONFIG = CONFIG
    lit.OUT = OUT
    lit.DOC = DOC
    lit.ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)

    config = direct_ppo.load_yaml(CONFIG)
    selection = lit.make_sy2014_selection()
    selection["selection_reason"] = "031_10_literature_aligned_dqn_min_interval_smoke"
    selection_path = OUT / "configs" / "031_10_sy2014_selection.csv"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")

    direct_ppo.OUTPUT_ROOT = OUT
    direct_ppo.ensure_dirs()
    env_config = direct_ppo.build_env_config(config, selection)
    env_config_path = OUT / "configs" / "031_10_resolved_env_config.yaml"
    direct_ppo.write_yaml(env_config, env_config_path)

    train_summary = lit.train_model(config, env_config, selection)
    eval_summary = lit.evaluate_model(config, env_config, selection, train_summary)

    refs = lit.load_reference_rows()
    # Add 031_09 as the main direct comparator if available.
    path_03109 = ROOT / "benchmark_results" / "031_09_literature_aligned_dqn_free_timing_smoke" / "031_09_literature_aligned_dqn_vs_references.csv"
    if path_03109.exists():
        refs = pd.concat([pd.read_csv(path_03109), refs], ignore_index=True, sort=False) if not refs.empty else pd.read_csv(path_03109)
    lit_eval = eval_summary[eval_summary["split"].eq("eval")].copy() if not eval_summary.empty else pd.DataFrame()
    lit_eval["source"] = "031_10"
    comparison = pd.concat([lit_eval, refs], ignore_index=True, sort=False) if not refs.empty else lit_eval
    comparison_path = OUT / "031_10_literature_aligned_dqn_min_interval_vs_references.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    result = {
        "task": "031_10_literature_aligned_dqn_min_interval_smoke",
        "training_run": True,
        "site_year": "SYA2014",
        "seed": int(config["seed"]),
        "timesteps": int(config["total_timesteps"]),
        "single_change_from_031_09": "same-resource minimum operation interval changed from 1 day to 7 days",
        "config": str(CONFIG.relative_to(ROOT)),
        "selection": str(selection_path.relative_to(ROOT)),
        "env_config": str(env_config_path.relative_to(ROOT)),
        "train_summary": str((OUT / "evaluation" / "training_run_summary.csv").relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "literature_aligned_dqn_eval_summary.csv").relative_to(ROOT)),
        "comparison": str(comparison_path.relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
        "scope": "one-seed min-interval smoke; not full method validation",
    }
    (OUT / "031_10_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_record(train_summary, eval_summary, comparison)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not lit_eval.empty:
        print(lit_eval.to_string(index=False))


if __name__ == "__main__":
    main()

