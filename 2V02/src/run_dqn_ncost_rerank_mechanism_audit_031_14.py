from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_SUMMARY = ROOT / "benchmark_results" / "031_12_dqn_nitrogen_counterfactual_audit" / "evaluation" / "031_12_nitrogen_counterfactual_summary.csv"
OUT = ROOT / "benchmark_results" / "031_14_dqn_ncost_rerank_mechanism_audit"
DOC = ROOT / "docs" / "031_14_dqn_ncost_rerank_mechanism_audit_record.md"

N_COSTS = [0.79, 1.58, 2.37, 3.16]
PRIMARY_N_COST = 1.58
YIELD_COEF = 0.158
WATER_COST = 1.1


def lit_score(row: pd.Series, n_cost: float) -> float:
    return (
        YIELD_COEF * float(row["final_grnwt"])
        - WATER_COST * float(row["total_irrigation"])
        - float(n_cost) * float(row["total_n"])
    )


def ensure_dirs() -> None:
    (OUT / "evaluation").mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def format_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def main() -> None:
    ensure_dirs()
    if not IN_SUMMARY.exists():
        raise FileNotFoundError(IN_SUMMARY)

    base = pd.read_csv(IN_SUMMARY)
    required = {"seed", "variant", "final_grnwt", "total_irrigation", "total_n", "profit_simple", "PFP_N"}
    missing = required.difference(base.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    base = base.copy()
    base["seed"] = base["seed"].astype(int)
    rows = []
    for n_cost in N_COSTS:
        tmp = base.copy()
        tmp["n_cost"] = float(n_cost)
        tmp["lit_score"] = tmp.apply(lambda r: lit_score(r, n_cost), axis=1)
        tmp["rank_within_seed"] = tmp.groupby("seed")["lit_score"].rank(method="first", ascending=False).astype(int)
        rows.append(tmp)
    rerank = pd.concat(rows, ignore_index=True)
    rerank_path = OUT / "evaluation" / "031_14_variant_rerank_by_ncost.csv"
    rerank.to_csv(rerank_path, index=False, encoding="utf-8-sig")

    winners = (
        rerank.sort_values(["n_cost", "seed", "rank_within_seed"])
        .groupby(["n_cost", "seed"], as_index=False)
        .first()
        [["n_cost", "seed", "variant", "lit_score", "final_grnwt", "total_irrigation", "total_n", "profit_simple", "PFP_N"]]
        .rename(columns={"variant": "winning_variant"})
    )
    winners_path = OUT / "evaluation" / "031_14_winners_by_ncost_seed.csv"
    winners.to_csv(winners_path, index=False, encoding="utf-8-sig")

    # Break-even c_N for stage_spread_N200 vs original_replay.
    be_rows = []
    for seed, sdf in base.groupby("seed"):
        original = sdf[sdf["variant"].eq("original_replay")]
        n200 = sdf[sdf["variant"].eq("stage_spread_N200")]
        if len(original) != 1 or len(n200) != 1:
            be_rows.append({"seed": int(seed), "status": "missing_pair", "break_even_n_cost": np.nan})
            continue
        o = original.iloc[0]
        s = n200.iloc[0]
        denom = float(o["total_n"]) - float(s["total_n"])
        numer = YIELD_COEF * (float(o["final_grnwt"]) - float(s["final_grnwt"])) - WATER_COST * (
            float(o["total_irrigation"]) - float(s["total_irrigation"])
        )
        be = numer / denom if abs(denom) > 1e-12 else np.nan
        score_original_079 = lit_score(o, 0.79)
        score_n200_079 = lit_score(s, 0.79)
        score_original_158 = lit_score(o, 1.58)
        score_n200_158 = lit_score(s, 1.58)
        be_rows.append(
            {
                "seed": int(seed),
                "status": "ok",
                "break_even_n_cost": be,
                "original_yield": float(o["final_grnwt"]),
                "n200_yield": float(s["final_grnwt"]),
                "yield_delta_n200_minus_original": float(s["final_grnwt"]) - float(o["final_grnwt"]),
                "original_n": float(o["total_n"]),
                "n200_n": float(s["total_n"]),
                "score_delta_n200_minus_original_at_0p79": score_n200_079 - score_original_079,
                "score_delta_n200_minus_original_at_1p58": score_n200_158 - score_original_158,
            }
        )
    breakeven = pd.DataFrame(be_rows)
    breakeven_path = OUT / "evaluation" / "031_14_stage_spread_N200_vs_original_breakeven.csv"
    breakeven.to_csv(breakeven_path, index=False, encoding="utf-8-sig")

    primary = rerank[rerank["n_cost"].eq(PRIMARY_N_COST)].copy()
    primary_winners = winners[winners["n_cost"].eq(PRIMARY_N_COST)].copy()
    n200_rank = primary[primary["variant"].eq("stage_spread_N200")][["seed", "rank_within_seed", "lit_score"]].rename(
        columns={"rank_within_seed": "stage_spread_N200_rank", "lit_score": "stage_spread_N200_score"}
    )
    original_rank = primary[primary["variant"].eq("original_replay")][["seed", "rank_within_seed", "lit_score"]].rename(
        columns={"rank_within_seed": "original_replay_rank", "lit_score": "original_replay_score"}
    )
    primary_join = primary_winners.merge(n200_rank, on="seed", how="left").merge(original_rank, on="seed", how="left")
    n200_outranks_original_count = int((primary_join["stage_spread_N200_rank"] < primary_join["original_replay_rank"]).sum())
    n_seed = int(primary_join["seed"].nunique())
    if n200_outranks_original_count >= max(2, n_seed):
        branch = "B_reward_ranking_already_favors_stage_spread_N200"
    elif n200_outranks_original_count == 0:
        branch = "A_nitrogen_penalty_still_too_weak"
    else:
        branch = "Mixed_partial_reward_ranking_change"

    aggregate = (
        rerank.groupby(["n_cost", "variant"], as_index=False)
        .agg(
            mean_score=("lit_score", "mean"),
            min_score=("lit_score", "min"),
            max_score=("lit_score", "max"),
            mean_rank=("rank_within_seed", "mean"),
            mean_yield=("final_grnwt", "mean"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_n=("total_n", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
        )
        .sort_values(["n_cost", "mean_rank", "mean_score"], ascending=[True, True, False])
    )
    aggregate_path = OUT / "evaluation" / "031_14_variant_aggregate_by_ncost.csv"
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 031_14 DQN nitrogen-cost rerank mechanism audit record",
        "",
        "## Scope",
        "",
        "- No training.",
        "- No new DSSAT calls.",
        "- Input: 031_12 DSSAT counterfactual variants for SYA2014 seeds 0/1/2.",
        "- Purpose: test whether 031_13 failure is explained by nitrogen penalty still being too weak, or by DQN not learning/using the better full-season ranking.",
        "",
        "## Reward used for reranking",
        "",
        "```text",
        f"R_lit(c_N) = {YIELD_COEF} * final_grnwt - {WATER_COST} * total_irrigation - c_N * total_n",
        "```",
        "",
        "Primary checked nitrogen costs: 0.79 and 1.58. Values 2.37 and 3.16 are diagnostic only.",
        "",
        "## Winner by nitrogen cost and seed",
        "",
        winners.to_string(index=False),
        "",
        "## stage_spread_N200 vs original_replay break-even",
        "",
        breakeven.to_string(index=False),
        "",
        f"## Primary branch at c_N={PRIMARY_N_COST}",
        "",
        primary_join.to_string(index=False),
        "",
        f"- stage_spread_N200 outranks original_replay in {n200_outranks_original_count}/{n_seed} seeds.",
        f"- Branch: `{branch}`.",
        "",
        "## Aggregate by variant",
        "",
        aggregate.to_string(index=False),
        "",
        "## Interpretation",
        "",
    ]
    if branch == "B_reward_ranking_already_favors_stage_spread_N200":
        lines.extend(
            [
                "- At the doubled nitrogen cost used in 031_13, the already-simulated DSSAT counterfactuals rank the lower-N staged policy above the original learned N250 replay.",
                "- Therefore 031_13 is not cleanly explained by nitrogen penalty still being too weak.",
                "- The more likely issue is that 5k DQN training did not recover this full-season ranking in its learned Q/action policy.",
                "- This points to learning/exploration/value-estimation rather than simply needing a larger N-cost constant.",
            ]
        )
    elif branch == "A_nitrogen_penalty_still_too_weak":
        lines.extend(
            [
                "- At the doubled nitrogen cost used in 031_13, the lower-N staged policy still does not beat the original learned N250 replay.",
                "- Therefore the 031_13 behavior is consistent with nitrogen penalty still being too weak under this reward family.",
            ]
        )
    else:
        lines.extend(
            [
                "- The evidence is mixed across seeds.",
                "- A single larger N-cost value should not be chosen post hoc from this table; this task is diagnostic, not a parameter scan.",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Rerank table: `{rerank_path.relative_to(ROOT)}`",
            f"- Winners: `{winners_path.relative_to(ROOT)}`",
            f"- Break-even table: `{breakeven_path.relative_to(ROOT)}`",
            f"- Aggregate table: `{aggregate_path.relative_to(ROOT)}`",
        ]
    )

    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "task": "031_14_dqn_ncost_rerank_mechanism_audit",
        "training_run": False,
        "new_dssat_calls": False,
        "branch": branch,
        "n200_outranks_original_count_at_1p58": n200_outranks_original_count,
        "n_seed": n_seed,
        "record_md": str(DOC.relative_to(ROOT)),
        "rerank_csv": str(rerank_path.relative_to(ROOT)),
        "winners_csv": str(winners_path.relative_to(ROOT)),
        "breakeven_csv": str(breakeven_path.relative_to(ROOT)),
    }
    result_path = OUT / "031_14_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

