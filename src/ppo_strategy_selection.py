from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppo_safe_rendering import DEFAULT_CONFIG, PROJECT_ROOT, load_yaml


def normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_v = values.min()
    max_v = values.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        score = pd.Series(0.5, index=series.index)
    else:
        score = (values - min_v) / (max_v - min_v)
    return score if higher_is_better else 1.0 - score


def select_best_policies(config_path: Path = DEFAULT_CONFIG) -> Path:
    config = load_yaml(config_path)
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    summary_path = output_root / "evaluation" / "ppo_evaluation_summary.csv"
    out = output_root / "strategy_selection" / "best_policy_by_site.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
        return out
    df = pd.read_csv(summary_path)
    ok = df[(df["run_status"] == "ok") & (df["eval_year"] != df["train_year"])].copy()
    rows = []
    for station, group in ok.groupby("station"):
        agg = group.groupby(["policy_name", "train_year", "train_year_label", "model_path"], as_index=False).agg(
            mean_yield=("final_grnwt", "mean"),
            std_yield=("final_grnwt", "std"),
            mean_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_n_fertilizer=("total_n_fertilizer", "mean"),
            validation_years=("eval_year", lambda s: ",".join(str(int(v)) for v in sorted(s))),
        )
        if agg.empty:
            continue
        agg["std_yield"] = agg["std_yield"].fillna(0.0)
        score = (
            normalize(agg["mean_yield"], True)
            + normalize(agg["mean_reward"], True)
            - normalize(agg["std_yield"], True)
            - normalize(agg["mean_irrigation"], True)
            - normalize(agg["mean_n_fertilizer"], True)
        )
        agg["stability_score"] = score
        best = agg.sort_values("stability_score", ascending=False).iloc[0]
        n_years = len(config["observed_years"][station])
        rows.append(
            {
                "station": station,
                "best_policy_name": best["policy_name"],
                "best_train_year": int(best["train_year"]),
                "best_train_year_type": best["train_year_label"],
                "validation_years": best["validation_years"],
                "mean_yield": best["mean_yield"],
                "std_yield": best["std_yield"],
                "mean_reward": best["mean_reward"],
                "std_reward": best["std_reward"],
                "mean_irrigation": best["mean_irrigation"],
                "mean_n_fertilizer": best["mean_n_fertilizer"],
                "stability_score": best["stability_score"],
                "reason": "single_seed_cross_year_score; limited_two_year_cross_validation" if n_years == 2 else "single_seed_cross_year_score",
                "model_path": best["model_path"],
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    print(select_best_policies().relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
