"""Read-only 1K/2K action-collapse diagnostic for HL and YC.

No PPO model is loaded and no DSSAT simulation is invoked. Existing daily CSVs
are read and new audit tables are written only under the isolated 087 output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "087_00_hl_yc_early_checkpoint_collapse_audit"
YEARS = list(range(2014, 2024))
CHECKPOINTS = [1000, 2000]
CASES = [
    {"case": "HL_seed0", "station": "HLA", "seed": 0, "root": ROOT / "benchmark_results/078_00_hla_lowIC_stability_tuning_B_nepochs3_smoke2k"},
    {"case": "HL_seed1", "station": "HLA", "seed": 1, "root": ROOT / "benchmark_results/085_00_hla_lowIC_nepochs3_seed1_corrected_smoke_B_nepochs3_smoke2k"},
    {"case": "YC_seed0", "station": "YCA", "seed": 0, "root": ROOT / "benchmark_results/079_00_yca_lowIC_stability_tuning_B_nepochs3_smoke2k"},
    {"case": "YC_seed1", "station": "YCA", "seed": 1, "root": ROOT / "benchmark_results/086_00_yca_lowIC_nepochs3_seed1_corrected_smoke_B_nepochs3_smoke2k"},
]


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = pd.Series(values).value_counts(normalize=True).to_numpy(dtype=float)
    return float(-sum(p * math.log(p, 2) for p in counts if p > 0))


def value(row: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(row.get(column, pd.Series(default, index=row.index)), errors="coerce").fillna(default)


def audit_file(path: Path, case: dict, checkpoint: int, year: int) -> dict:
    daily = pd.read_csv(path, keep_default_na=False)
    safe_i = value(daily, "safe_action_amir")
    safe_n = value(daily, "safe_action_anfer")
    dap = value(daily, "dap")
    positive = (safe_i > 1e-9) | (safe_n > 1e-9)
    pairs = [f"I{float(i):g}/N{float(n):g}" for i, n in zip(safe_i[positive], safe_n[positive])]
    events = [f"DAP{int(d)}:{pair}" for d, pair in zip(dap[positive], pairs)]
    discrete = value(daily, "discrete_action_index").astype(int).astype(str).tolist()
    swfac = value(daily, "swfac", default=np.nan)
    nstres = value(daily, "nstres", default=np.nan)
    grnwt = value(daily, "grnwt", default=np.nan)
    cum_i = value(daily, "season_cumulative_irrigation", default=np.nan)
    cum_n = value(daily, "season_cumulative_n", default=np.nan)
    return {
        "case": case["case"],
        "station": case["station"],
        "seed": case["seed"],
        "checkpoint": checkpoint,
        "year": year,
        "daily_rows": int(len(daily)),
        "positive_actions": int(positive.sum()),
        "after_dap1_actions": int((positive & (dap > 1)).sum()),
        "distinct_positive_pairs": int(len(set(pairs))),
        "positive_action_entropy_bits": entropy(pairs),
        "all_action_entropy_bits": entropy(discrete),
        "action_pairs": ";".join(sorted(set(pairs))),
        "action_signature": "|".join(events),
        "final_yield_kg_ha": float(grnwt.iloc[-1]) if len(grnwt) else np.nan,
        "season_irrigation_mm": float(cum_i.iloc[-1]) if len(cum_i) else np.nan,
        "season_n_kg_ha": float(cum_n.iloc[-1]) if len(cum_n) else np.nan,
        "min_swfac": float(swfac.min()) if swfac.notna().any() else np.nan,
        "min_nstres": float(nstres.min()) if nstres.notna().any() else np.nan,
        "source_csv": path.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    detail: list[dict] = []
    missing: list[str] = []
    for case in CASES:
        for checkpoint in CHECKPOINTS:
            for year in YEARS:
                path = case["root"] / "daily_outputs" / case["station"] / f"{case['station']}_{year}_seed{case['seed']}_ckpt{checkpoint}_daily.csv"
                if not path.exists():
                    missing.append(path.relative_to(ROOT).as_posix())
                    continue
                detail.append(audit_file(path, case, checkpoint, year))
    detail_df = pd.DataFrame(detail)
    if missing:
        raise FileNotFoundError("Missing daily CSV: " + missing[0])
    summaries: list[dict] = []
    for (case, checkpoint), sub in detail_df.groupby(["case", "checkpoint"]):
        signatures = sub["action_signature"].astype(str)
        pair_union = {piece for text in sub["action_pairs"] for piece in str(text).split(";") if piece}
        summaries.append({
            "case": case,
            "checkpoint": int(checkpoint),
            "years": int(len(sub)),
            "distinct_crossyear_signatures": int(signatures.nunique()),
            "mean_positive_actions": float(sub["positive_actions"].mean()),
            "mean_after_dap1_actions": float(sub["after_dap1_actions"].mean()),
            "mean_distinct_positive_pairs": float(sub["distinct_positive_pairs"].mean()),
            "mean_positive_entropy_bits": float(sub["positive_action_entropy_bits"].mean()),
            "mean_all_action_entropy_bits": float(sub["all_action_entropy_bits"].mean()),
            "union_positive_pairs": ";".join(sorted(pair_union)),
            "mean_yield_kg_ha": float(sub["final_yield_kg_ha"].mean()),
            "mean_irrigation_mm": float(sub["season_irrigation_mm"].mean()),
            "mean_n_kg_ha": float(sub["season_n_kg_ha"].mean()),
            "mean_min_swfac": float(sub["min_swfac"].mean()),
            "mean_min_nstres": float(sub["min_nstres"].mean()),
        })
    summary_df = pd.DataFrame(summaries).sort_values(["case", "checkpoint"])
    transitions: list[dict] = []
    for case, sub in detail_df.groupby("case"):
        left = sub[sub["checkpoint"].eq(1000)].set_index("year")
        right = sub[sub["checkpoint"].eq(2000)].set_index("year")
        common = left.index.intersection(right.index)
        transitions.append({
            "case": case,
            "years": int(len(common)),
            "signature_changed_years": int(sum(left.loc[y, "action_signature"] != right.loc[y, "action_signature"] for y in common)),
            "signature_changed_fraction": float(sum(left.loc[y, "action_signature"] != right.loc[y, "action_signature"] for y in common) / len(common)),
            "mean_delta_positive_actions_2k_minus_1k": float((right.loc[common, "positive_actions"] - left.loc[common, "positive_actions"]).mean()),
            "mean_delta_positive_entropy_bits_2k_minus_1k": float((right.loc[common, "positive_action_entropy_bits"] - left.loc[common, "positive_action_entropy_bits"]).mean()),
            "mean_delta_yield_kg_ha_2k_minus_1k": float((right.loc[common, "final_yield_kg_ha"] - left.loc[common, "final_yield_kg_ha"]).mean()),
            "mean_delta_irrigation_mm_2k_minus_1k": float((right.loc[common, "season_irrigation_mm"] - left.loc[common, "season_irrigation_mm"]).mean()),
        })
    transition_df = pd.DataFrame(transitions)
    crossseed: list[dict] = []
    for station, seed0_case, seed1_case in [("HL", "HL_seed0", "HL_seed1"), ("YC", "YC_seed0", "YC_seed1")]:
        for checkpoint in CHECKPOINTS:
            a = detail_df[(detail_df["case"].eq(seed0_case)) & detail_df["checkpoint"].eq(checkpoint)].set_index("year")
            b = detail_df[(detail_df["case"].eq(seed1_case)) & detail_df["checkpoint"].eq(checkpoint)].set_index("year")
            common = a.index.intersection(b.index)
            crossseed.append({
                "station": station,
                "checkpoint": checkpoint,
                "years": int(len(common)),
                "same_signature_years": int(sum(a.loc[y, "action_signature"] == b.loc[y, "action_signature"] for y in common)),
                "same_signature_fraction": float(sum(a.loc[y, "action_signature"] == b.loc[y, "action_signature"] for y in common) / len(common)),
                "mean_abs_yield_difference_seed1_minus_seed0": float((b.loc[common, "final_yield_kg_ha"] - a.loc[common, "final_yield_kg_ha"]).abs().mean()),
                "mean_yield_difference_seed1_minus_seed0": float((b.loc[common, "final_yield_kg_ha"] - a.loc[common, "final_yield_kg_ha"]).mean()),
                "mean_positive_action_difference_seed1_minus_seed0": float((b.loc[common, "positive_actions"] - a.loc[common, "positive_actions"]).mean()),
            })
    crossseed_df = pd.DataFrame(crossseed)
    OUT.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(OUT / "087_00_action_detail.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "087_00_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    transition_df.to_csv(OUT / "087_00_within_seed_transition.csv", index=False, encoding="utf-8-sig")
    crossseed_df.to_csv(OUT / "087_00_crossseed_comparison.csv", index=False, encoding="utf-8-sig")
    result = {
        "task": "087_00_hl_yc_early_checkpoint_collapse_audit",
        "status": "completed_read_only",
        "training_started": False,
        "dssat_started": False,
        "source_cases": [case["case"] for case in CASES],
        "checkpoints": CHECKPOINTS,
        "years": YEARS,
        "detail_csv": (OUT / "087_00_action_detail.csv").relative_to(ROOT).as_posix(),
        "checkpoint_summary_csv": (OUT / "087_00_checkpoint_summary.csv").relative_to(ROOT).as_posix(),
        "transition_csv": (OUT / "087_00_within_seed_transition.csv").relative_to(ROOT).as_posix(),
        "crossseed_csv": (OUT / "087_00_crossseed_comparison.csv").relative_to(ROOT).as_posix(),
    }
    (OUT / "087_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "087_00_record.md").write_text(
        "\n".join([
            "# 087 HL/YC early-checkpoint collapse audit",
            "",
            "- Read-only audit of existing 1K/2K daily CSVs; no PPO model loading, training, or DSSAT execution.",
            "- Cases: HL/YC seed0 and corrected true seed1; years 2014--2023.",
            f"- Checkpoint summary: `{result['checkpoint_summary_csv']}`.",
            f"- Within-seed transitions: `{result['transition_csv']}`.",
            f"- Cross-seed comparison: `{result['crossseed_csv']}`.",
        ]) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
