from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmark_results" / "021_47" / "021_47_agent_sample_draws.csv"
OUT = ROOT / "benchmark_results" / "021_48"
BONUS = 1620.0
HUBER_DELTA = 1.0


def huber_grad_wrt_q(q: pd.Series, target: pd.Series) -> np.ndarray:
    return np.clip(q.to_numpy(float) - target.to_numpy(float), -HUBER_DELTA, HUBER_DELTA)


def bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def empirical_fit_table(direct: pd.DataFrame) -> pd.DataFrame:
    per_update = (
        direct.groupby(["global_index", "update"], as_index=False)
        .agg(
            chosen_q=("chosen_q", "mean"),
            target_1=("target_1", "mean"),
            target_1_no_bonus=("target_1_no_bonus", "mean"),
            draws=("draw_position", "size"),
            origin_dap=("origin_dap", "first"),
        )
        .sort_values(["global_index", "update"])
    )
    rows: list[dict[str, float | int]] = []
    for global_index, g in per_update.groupby("global_index", sort=True):
        g = g.sort_values("update").reset_index(drop=True)
        n = len(g)
        if n < 20:
            continue
        k = max(1, math.ceil(0.1 * n))
        q_initial = float(g.head(k)["chosen_q"].mean())
        q_final = float(g.tail(k)["chosen_q"].mean())
        target = float(g["target_1"].mean())
        target_no_bonus = float(g["target_1_no_bonus"].mean())
        residual_initial = target - q_initial
        residual_final = target - q_final
        closure = (residual_initial - residual_final) / residual_initial
        slope = float(np.polyfit(g["update"].to_numpy(float), g["chosen_q"].to_numpy(float), 1)[0])
        extrapolated = residual_final / slope if slope > 0 else np.nan
        rows.append(
            {
                "global_index": int(global_index),
                "origin_dap": int(float(g["origin_dap"].iloc[0])),
                "unique_updates": n,
                "draws": int(g["draws"].sum()),
                "first_update": int(g["update"].min()),
                "last_update": int(g["update"].max()),
                "q_initial_decile_mean": q_initial,
                "q_final_decile_mean": q_final,
                "target_actual": target,
                "target_no_bonus": target_no_bonus,
                "residual_initial": residual_initial,
                "residual_final": residual_final,
                "residual_closure_fraction": closure,
                "q_ols_slope_per_update": slope,
                "descriptive_remaining_updates_if_linear": extrapolated,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["direct_bonus"] = bool_col(df["direct_bonus"])
    df["nstep_contains_bonus"] = bool_col(df["nstep_contains_bonus"])

    bonus_df = df.loc[df["nstep_contains_bonus"]].copy()
    bonus_df["target_n_no_bonus"] = bonus_df["target_n"] - bonus_df["discounted_bonus_in_nstep"]
    bonus_df["residual_n_actual"] = bonus_df["target_n"] - bonus_df["chosen_q"]
    bonus_df["residual_n_no_bonus"] = bonus_df["target_n_no_bonus"] - bonus_df["chosen_q"]
    bonus_df["huber_grad_n_actual"] = huber_grad_wrt_q(bonus_df["chosen_q"], bonus_df["target_n"])
    bonus_df["huber_grad_n_no_bonus"] = huber_grad_wrt_q(bonus_df["chosen_q"], bonus_df["target_n_no_bonus"])
    bonus_df["no_bonus_n_still_linear"] = bonus_df["residual_n_no_bonus"].abs() > HUBER_DELTA
    bonus_df["n_grad_identical"] = np.isclose(
        bonus_df["huber_grad_n_actual"], bonus_df["huber_grad_n_no_bonus"], atol=0.0, rtol=0.0
    )

    direct = bonus_df.loc[bonus_df["direct_bonus"]].copy()
    direct["target_1_no_bonus"] = direct["target_1"] - BONUS
    direct["residual_1_actual"] = direct["target_1"] - direct["chosen_q"]
    direct["residual_1_no_bonus"] = direct["target_1_no_bonus"] - direct["chosen_q"]
    direct["huber_grad_1_actual"] = huber_grad_wrt_q(direct["chosen_q"], direct["target_1"])
    direct["huber_grad_1_no_bonus"] = huber_grad_wrt_q(direct["chosen_q"], direct["target_1_no_bonus"])
    direct["no_bonus_1_still_linear"] = direct["residual_1_no_bonus"].abs() > HUBER_DELTA
    direct["grad_1_identical"] = np.isclose(
        direct["huber_grad_1_actual"], direct["huber_grad_1_no_bonus"], atol=0.0, rtol=0.0
    )

    # Algebraic validation: subtracting the recorded discounted bonus must be exact up to float CSV precision.
    identity_error = float(
        np.max(
            np.abs(
                (bonus_df["target_n_no_bonus"] + bonus_df["discounted_bonus_in_nstep"])
                - bonus_df["target_n"]
            )
        )
    )

    fit = empirical_fit_table(direct)
    if fit.empty:
        raise RuntimeError("No direct bonus transition has at least 20 unique updates")

    category_rows = []
    for name, g in [("direct_bonus", direct), ("nstep_contains_bonus", bonus_df)]:
        suffix = "1" if name == "direct_bonus" else "n"
        category_rows.append(
            {
                "category": name,
                "draws": len(g),
                "unique_transitions": int(g["global_index"].nunique()),
                "target_actual_mean": float(g[f"target_{suffix}"].mean()),
                "target_no_bonus_mean": float(g[f"target_{suffix}_no_bonus"].mean()),
                "residual_actual_mean": float(g[f"residual_{suffix}_actual"].mean()),
                "residual_no_bonus_mean": float(g[f"residual_{suffix}_no_bonus"].mean()),
                "bonus_share_of_actual_residual_mean": float(
                    (g["discounted_bonus_in_nstep"] / g[f"residual_{suffix}_actual"]).mean()
                ),
                "no_bonus_still_huber_linear_fraction": float(g[f"no_bonus_{suffix}_still_linear"].mean()),
                "huber_gradient_identical_fraction": float(g[f"{'grad_1' if suffix == '1' else 'n_grad'}_identical"].mean()),
            }
        )
    categories = pd.DataFrame(category_rows)

    direct_linear = float(direct["no_bonus_1_still_linear"].mean())
    direct_same_grad = float(direct["grad_1_identical"].mean())
    mean_closure = float(fit["residual_closure_fraction"].mean())
    if (1 - direct_linear) > 0.5 or (1 - direct_same_grad) > 0.5:
        branch = "A"
    elif direct_linear >= 0.95 and direct_same_grad >= 0.95 and mean_closure < 0.01:
        branch = "B"
    else:
        branch = "C"

    keep_cols = [
        "update", "env_step", "draw_position", "global_index", "origin_env_step", "episode",
        "origin_dap", "action", "direct_bonus", "discounted_bonus_in_nstep", "importance_weight",
        "chosen_q", "target_1", "target_n", "target_n_no_bonus", "residual_n_actual",
        "residual_n_no_bonus", "huber_grad_n_actual", "huber_grad_n_no_bonus",
        "no_bonus_n_still_linear", "n_grad_identical",
    ]
    bonus_df[keep_cols].to_csv(OUT / "021_48_bonus_sample_counterfactual.csv", index=False)
    direct.to_csv(OUT / "021_48_direct_bonus_counterfactual.csv", index=False)
    fit.to_csv(OUT / "021_48_direct_transition_empirical_fit.csv", index=False)
    categories.to_csv(OUT / "021_48_category_summary.csv", index=False)

    summary = {
        "status": "completed",
        "branch": branch,
        "input": str(INPUT.relative_to(ROOT)),
        "identity_error": identity_error,
        "direct_draws": int(len(direct)),
        "direct_unique_transitions": int(direct["global_index"].nunique()),
        "direct_no_bonus_still_huber_linear_fraction": direct_linear,
        "direct_huber_gradient_identical_fraction": direct_same_grad,
        "direct_actual_target_mean": float(direct["target_1"].mean()),
        "direct_no_bonus_target_mean": float(direct["target_1_no_bonus"].mean()),
        "direct_actual_residual_mean": float(direct["residual_1_actual"].mean()),
        "direct_no_bonus_residual_mean": float(direct["residual_1_no_bonus"].mean()),
        "direct_bonus_share_of_actual_residual_mean": float((BONUS / direct["residual_1_actual"]).mean()),
        "direct_empirical_mean_residual_closure_fraction": mean_closure,
        "direct_empirical_median_residual_closure_fraction": float(fit["residual_closure_fraction"].median()),
        "note": "Counterfactual holds the sampled rows fixed; PER probabilities and importance weights are not counterfactual training weights.",
    }
    (OUT / "021_48_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(2)
    actual_targets = [categories.loc[0, "target_actual_mean"], categories.loc[1, "target_actual_mean"]]
    no_bonus_targets = [categories.loc[0, "target_no_bonus_mean"], categories.loc[1, "target_no_bonus_mean"]]
    width = 0.34
    axes[0].bar(x - width / 2, actual_targets, width, color="#b2182b", label="Actual target")
    axes[0].bar(x + width / 2, no_bonus_targets, width, color="#2166ac", label="Counterfactual: no +1620")
    axes[0].set_xticks(x, ["Direct terminal", "5-step contains bonus"])
    axes[0].set_ylabel("Mean TD target")
    axes[0].set_title("Removing bonus lowers target, but target remains large")
    axes[0].legend(frameon=False, loc="lower right")

    fx = np.arange(len(fit))
    axes[1].bar(fx, fit["residual_closure_fraction"] * 100, color="#4d9221")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="1% preregistered bound")
    axes[1].set_xticks(fx, fit["global_index"].astype(str))
    axes[1].set_xlabel("Direct terminal transition global index")
    axes[1].set_ylabel("Observed residual closed (%)")
    axes[1].set_title("Same-transition Q fitting during 1K audit")
    axes[1].legend(frameon=False)
    fig.suptitle("SY2014 021_48: terminal bonus Huber counterfactual audit", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "021_48_terminal_bonus_huber_counterfactual.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "021_48_terminal_bonus_huber_counterfactual.svg", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
