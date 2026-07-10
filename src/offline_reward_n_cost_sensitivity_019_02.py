from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "reward_sensitivity_019_02"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-10_019_02_reward_n_cost_sensitivity_offline_rescore.md"

WATER_COST = 1.0
N_COSTS = [5.0, 10.0, 20.0]

DATASETS = [
    {
        "site": "HLA",
        "station": "Hailun",
        "year": 2010,
        "seed": 0,
        "null": 6956.0,
        "path": "DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed0_50000steps/checkpoint_summary.csv",
        "source": "015_12 HLA2010 seed0",
    },
    {
        "site": "HLA",
        "station": "Hailun",
        "year": 2010,
        "seed": 1,
        "null": 6956.0,
        "path": "DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps/checkpoint_summary.csv",
        "source": "015_12 HLA2010 seed1",
    },
    {
        "site": "YC",
        "station": "Yucheng",
        "year": 2014,
        "seed": 0,
        "null": 7825.0,
        "path": "DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/seed0/015_04_yc2014_unified_dqn_checkpoint_summary.csv",
        "source": "015_04 YC2014 seed0",
    },
    {
        "site": "YC",
        "station": "Yucheng",
        "year": 2014,
        "seed": 1,
        "null": 7825.0,
        "path": "DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_seed1_015_05/seed1/015_05_yc2014_unified_dqn_checkpoint_seed1_summary.csv",
        "source": "015_05 YC2014 seed1",
    },
    {
        "site": "FQ",
        "station": "Fengqiu",
        "year": 2016,
        "seed": 0,
        "null": 7066.0,
        "path": "DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed0_50000steps/checkpoint_summary.csv",
        "source": "015_14 FQ2016 seed0",
    },
    {
        "site": "FQ",
        "station": "Fengqiu",
        "year": 2016,
        "seed": 1,
        "null": 7066.0,
        "path": "DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/checkpoint_summary.csv",
        "source": "015_14 FQ2016 seed1",
    },
    {
        "site": "SY",
        "station": "Shenyang",
        "year": 2014,
        "seed": 0,
        "null": 2769.0,
        "path": "DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08/017_08_sy_dqn_checkpoint_summary.csv",
        "source": "017_08 SY2014 seed0",
    },
    {
        "site": "SY",
        "station": "Shenyang",
        "year": 2014,
        "seed": 1,
        "null": 2769.0,
        "path": "DSSAT_auto_validation/sy2014_seed1_minimal_reproduction_018_08/018_08_formal_checkpoint_summary.csv",
        "source": "018_08 SY2014 seed1",
    },
    {
        "site": "LC",
        "station": "Luancheng",
        "year": 2010,
        "seed": None,
        "null": 8051.0,
        "path": "DSSAT_auto_validation/extension_expert_baseline_018_03/018_06_lc2010_seed_stability_audit/018_06_lc2010_seed_checkpoint_table.csv",
        "source": "018_06 LC2010 seed0/seed1",
    },
]


def pick_col(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"Missing any of columns: {names}")


def load_one(meta: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / meta["path"]
    df = pd.read_csv(path)
    checkpoint_col = pick_col(df, ["checkpoint_step", "checkpoint"])
    yield_col = pick_col(df, ["final_grain_kg_ha", "final_gwad"])
    biomass_col = pick_col(df, ["final_biomass_kg_ha", "final_cwad"])
    irrigation_col = pick_col(df, ["action_irrigation_total", "irrigation_total"])
    nitrogen_col = pick_col(df, ["action_fertilizer_total", "fertilizer_total"])
    water_stress_col = pick_col(df, ["max_water_stress"])
    n_stress_col = pick_col(df, ["max_nitrogen_stress"])

    seed_col = "seed" if "seed" in df.columns else None
    out = pd.DataFrame(
        {
            "site": meta["site"],
            "station": meta["station"],
            "year": meta["year"],
            "seed": df[seed_col] if seed_col else meta["seed"],
            "checkpoint": pd.to_numeric(df[checkpoint_col], errors="coerce"),
            "gwad": pd.to_numeric(df[yield_col], errors="coerce"),
            "cwad": pd.to_numeric(df[biomass_col], errors="coerce"),
            "irrigation": pd.to_numeric(df[irrigation_col], errors="coerce").fillna(0.0),
            "nitrogen": pd.to_numeric(df[nitrogen_col], errors="coerce").fillna(0.0),
            "max_water_stress": pd.to_numeric(df[water_stress_col], errors="coerce"),
            "max_nitrogen_stress": pd.to_numeric(df[n_stress_col], errors="coerce"),
            "null_gwad": meta["null"],
            "source": meta["source"],
            "source_path": meta["path"],
        }
    )
    out = out.dropna(subset=["checkpoint", "gwad"]).copy()
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    return out


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if not v.is_integer() else f"{int(v)}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def interpret(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, year), group in best.groupby(["site", "year"]):
        base = group[group["n_cost"].eq(5.0)]
        high = group[group["n_cost"].eq(20.0)]
        if base.empty or high.empty:
            continue
        base_n = float(base["nitrogen"].mean())
        high_n = float(high["nitrogen"].mean())
        base_y = float(base["gwad"].mean())
        high_y = float(high["gwad"].mean())
        if high_n < base_n and high_y >= base_y - 100:
            status = "n_cost_promising"
            note = "提高N cost会选择更低施氮checkpoint，且平均产量损失不大。"
        elif high_n < base_n and high_y < base_y - 100:
            status = "n_cost_reduces_n_but_costs_yield"
            note = "提高N cost会降低施氮，但产量损失较明显，需要谨慎。"
        elif high_n == base_n:
            status = "n_cost_no_selection_change"
            note = "现有checkpoint里，提高N cost没有改变最佳施氮选择；若要改变策略，需要重新训练或改动作/约束。"
        else:
            status = "needs_review"
            note = "结果不符合简单模式，需要人工检查。"
        rows.append(
            {
                "site": site,
                "year": int(year),
                "base_mean_n_at_cost5": base_n,
                "mean_n_at_cost20": high_n,
                "base_mean_gwad_at_cost5": base_y,
                "mean_gwad_at_cost20": high_y,
                "status": status,
                "interpretation": note,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = pd.concat([load_one(meta) for meta in DATASETS], ignore_index=True)

    rescored_rows = []
    for n_cost in N_COSTS:
        tmp = all_rows.copy()
        tmp["n_cost"] = n_cost
        tmp["water_cost"] = WATER_COST
        tmp["yield_gain"] = (tmp["gwad"] - tmp["null_gwad"]).clip(lower=0.0)
        tmp["offline_reward"] = tmp["yield_gain"] - WATER_COST * tmp["irrigation"] - n_cost * tmp["nitrogen"]
        rescored_rows.append(tmp)
    rescored = pd.concat(rescored_rows, ignore_index=True)

    sort_cols = ["offline_reward", "gwad", "cwad", "checkpoint"]
    best = (
        rescored.sort_values(sort_cols, ascending=[False, False, False, True])
        .groupby(["site", "station", "year", "seed", "n_cost"], dropna=False, as_index=False)
        .head(1)
        .sort_values(["site", "year", "seed", "n_cost"])
        .reset_index(drop=True)
    )
    site_interpret = interpret(best)

    rescored.to_csv(OUT_DIR / "019_02_all_checkpoint_rescore.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "019_02_best_by_site_seed_ncost.csv", index=False, encoding="utf-8-sig")
    site_interpret.to_csv(OUT_DIR / "019_02_site_level_interpretation.csv", index=False, encoding="utf-8-sig")

    view_cols = ["site", "year", "seed", "n_cost", "checkpoint", "gwad", "irrigation", "nitrogen", "offline_reward"]
    lines = [
        "# 019_02 Reward N Cost Sensitivity Offline Rescore",
        "",
        "## Conclusion",
        "",
        "This run only rescored existing checkpoints. It did not train DQN and did not call DSSAT.",
        "",
        "## Best Checkpoint Under Each N Cost",
        "",
        markdown_table(best[view_cols]),
        "",
        "## Site-Level Interpretation",
        "",
        markdown_table(site_interpret),
        "",
        "## Important Limitation",
        "",
        "Offline rescoring can show whether existing checkpoints would be selected differently, but it cannot prove a newly trained policy would learn the same behavior. If a station shows promise here, the next step is a small retraining smoke under the new cost.",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
