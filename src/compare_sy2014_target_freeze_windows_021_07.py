"""021_07 补充：离线比较 SY2014 两个 target 冻结窗口。

只读取 021_06 已保存的固定状态 Q 值和动作混叠摘要；不训练、不调用 DSSAT。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
Q_PATH = ROOT / "benchmark_results/021_06/021_06_q_values_long.csv"
ALIAS_PATH = ROOT / "benchmark_results/021_06/021_06_action_alias_summary.csv"
OUT_DIR = ROOT / "benchmark_results/021_07"
OUT_CSV = OUT_DIR / "021_07_freeze_window_comparison.csv"
OUT_JSON = OUT_DIR / "021_07_freeze_window_comparison_summary.json"
OUT_PNG = OUT_DIR / "021_07_freeze_window_comparison.png"

WINDOWS = ((5000, 10000, "5K→10K"), (15000, 20000, "15K→20K"))


def ranking(values: dict[int, float]) -> str:
    return ">".join(f"N{n}" for n, _ in sorted(values.items(), key=lambda x: (-x[1], x[0])))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(Q_PATH)
    alias = pd.read_csv(ALIAS_PATH)[["state_id", "unique_executed_action_count"]]

    records: list[dict[str, object]] = []
    state_argmax: list[dict[str, object]] = []
    for start, end, label in WINDOWS:
        for network, q_col in (("online", "online_q"), ("target", "target_q")):
            for (state_id, source, target_dap, actual_dap, irrigation), group in q.groupby(
                ["state_id", "source_checkpoint", "target_dap", "actual_dap", "requested_irrigation"],
                sort=False,
            ):
                a = group[group.evaluated_checkpoint == start]
                b = group[group.evaluated_checkpoint == end]
                if len(a) != 3 or len(b) != 3:
                    raise RuntimeError(f"missing Q rows: {label} {network} {state_id} I{irrigation}")
                va = {int(r.requested_nitrogen): float(getattr(r, q_col)) for r in a.itertuples()}
                vb = {int(r.requested_nitrogen): float(getattr(r, q_col)) for r in b.itertuples()}
                std_a = float(a[f"{network}_q_std_9"].iloc[0])
                std_b = float(b[f"{network}_q_std_9"].iloc[0])
                margin_a = max(va[50], va[100]) - va[0]
                margin_b = max(vb[50], vb[100]) - vb[0]
                sign_flip = (margin_a < 0 < margin_b) or (margin_b < 0 < margin_a)
                robust = sign_flip and abs(margin_a) >= 0.25 * std_a and abs(margin_b) >= 0.25 * std_b
                records.append(
                    {
                        "window": label,
                        "start_checkpoint": start,
                        "end_checkpoint": end,
                        "network": network,
                        "state_id": state_id,
                        "source_checkpoint": source,
                        "target_dap": target_dap,
                        "actual_dap": actual_dap,
                        "irrigation_group": irrigation,
                        "ranking_start": ranking(va),
                        "ranking_end": ranking(vb),
                        "full_ranking_changed": ranking(va) != ranking(vb),
                        "n50_vs_n100_changed": (va[50] > va[100]) != (vb[50] > vb[100]),
                        "n0_vs_n50_changed": (va[0] > va[50]) != (vb[0] > vb[50]),
                        "n0_vs_n100_changed": (va[0] > va[100]) != (vb[0] > vb[100]),
                        "nitrogen_margin_start": margin_a,
                        "nitrogen_margin_end": margin_b,
                        "nitrogen_sign_flip": sign_flip,
                        "robust_sign_flip": robust,
                        "q_mae": float(np.mean([abs(vb[n] - va[n]) for n in (0, 50, 100)])),
                    }
                )

            for state_id, group in q.groupby("state_id", sort=False):
                a = group[group.evaluated_checkpoint == start]
                b = group[group.evaluated_checkpoint == end]
                if len(a) != 9 or len(b) != 9:
                    raise RuntimeError(f"missing action rows: {label} {state_id}")
                state_argmax.append(
                    {
                        "window": label,
                        "network": network,
                        "state_id": state_id,
                        "argmax_changed": int(a.loc[a[q_col].idxmax(), "action_index"])
                        != int(b.loc[b[q_col].idxmax(), "action_index"]),
                    }
                )

    detail = pd.DataFrame(records).merge(alias, on="state_id", how="left", validate="many_to_one")
    detail.to_csv(OUT_CSV, index=False)
    argmax = pd.DataFrame(state_argmax)

    summaries: list[dict[str, object]] = []
    for (window, network), x in detail.groupby(["window", "network"], sort=False):
        unaliased = x[x.unique_executed_action_count == 9]
        ax = argmax[(argmax.window == window) & (argmax.network == network)]
        summaries.append(
            {
                "window": window,
                "network": network,
                "comparisons": int(len(x)),
                "full_ranking_changes": int(x.full_ranking_changed.sum()),
                "n50_vs_n100_changes": int(x.n50_vs_n100_changed.sum()),
                "nitrogen_sign_flips": int(x.nitrogen_sign_flip.sum()),
                "robust_sign_flips": int(x.robust_sign_flip.sum()),
                "unaliased_comparisons": int(len(unaliased)),
                "unaliased_full_ranking_changes": int(unaliased.full_ranking_changed.sum()),
                "global_argmax_changes": int(ax.argmax_changed.sum()),
                "mean_q_mae": float(x.q_mae.mean()),
            }
        )
    summary = pd.DataFrame(summaries)
    OUT_JSON.write_text(
        json.dumps({"source": "021_06 fixed-state Q values", "summary": summaries}, indent=2),
        encoding="utf-8",
    )

    online = summary[summary.network == "online"].set_index("window")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    labels = [w[2] for w in WINDOWS]
    axes[0].bar(labels, online.loc[labels, "full_ranking_changes"], color=["#4C78A8", "#E45756"])
    axes[0].set_title("Online full N-ranking changes")
    axes[0].set_ylabel("Count / 54")
    axes[1].bar(labels, online.loc[labels, "robust_sign_flips"], color=["#4C78A8", "#E45756"])
    axes[1].set_title("Robust N vs no-N sign flips")
    axes[1].set_ylabel("Count / 54")
    axes[2].bar(labels, online.loc[labels, "mean_q_mae"], color=["#4C78A8", "#E45756"])
    axes[2].set_title("Mean Q drift")
    axes[2].set_ylabel("Mean absolute Q difference")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("SY2014 fixed-state comparison of two target-freeze windows")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"saved: {OUT_CSV}")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
