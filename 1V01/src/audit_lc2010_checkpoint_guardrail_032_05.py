from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "benchmark_results" / "032_04_lc2010_stress_aware_ppo_multiseed_200k" / "evaluation" / "032_04_checkpoint_eval_summary.csv"
PROMPT = ROOT / "prompts" / "032_05_lc2010_checkpoint_guardrail_audit.md"
OUT = ROOT / "benchmark_results" / "032_05_lc2010_checkpoint_guardrail_audit"
DOC = ROOT / "docs" / "032_05_lc2010_checkpoint_guardrail_audit_record.md"

EXPERT_YIELD = 8739.0
YIELD_FLOORS = {
    "95pct_expert": 0.95 * EXPERT_YIELD,
    "90pct_expert": 0.90 * EXPERT_YIELD,
    "85pct_expert": 0.85 * EXPERT_YIELD,
}
NSTRESS_CEILINGS = {
    "strict_nstress": 0.05,
    "moderate_nstress": 0.15,
    "loose_nstress": 0.30,
}


def ensure_dirs() -> None:
    for rel in ["configs", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def load_eval() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    df = df[df["run_status"].astype(str).str.startswith("ok")].copy()
    numeric_cols = [
        "seed",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "reward_stress_aware_sum",
        "max_swfac",
        "max_nstres",
        "swfac_days_gt_0p001",
        "nstres_days_gt_0p001",
        "irrigation_event_count",
        "n_event_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_rows = []
    seeds = sorted(df["seed"].dropna().unique())
    for y_label, y_floor in YIELD_FLOORS.items():
        for n_label, n_ceiling in NSTRESS_CEILINGS.items():
            for seed in seeds:
                g = df[df["seed"].eq(seed)].copy()
                eligible = g[(g["final_grnwt"] >= y_floor) & (g["max_nstres"] <= n_ceiling)].copy()
                base = {
                    "yield_guardrail": y_label,
                    "yield_floor": y_floor,
                    "nstress_guardrail": n_label,
                    "nstress_ceiling": n_ceiling,
                    "seed": int(seed),
                    "eligible_count": int(len(eligible)),
                    "total_checkpoints": int(len(g)),
                }
                rows.append(base)
                if eligible.empty:
                    selected_rows.append({**base, "selection_status": "no_eligible_checkpoint"})
                    continue
                best = eligible.sort_values(["reward_stress_aware_sum", "checkpoint_step"], ascending=[False, True]).iloc[0].to_dict()
                selected_rows.append(
                    {
                        **base,
                        "selection_status": "selected",
                        "checkpoint_step": int(best["checkpoint_step"]),
                        "final_grnwt": float(best["final_grnwt"]),
                        "yield_gap_to_expert": float(best["final_grnwt"] - EXPERT_YIELD),
                        "total_irrigation": float(best["total_irrigation"]),
                        "total_n": float(best["total_n"]),
                        "PFP_N": float(best["PFP_N"]) if pd.notna(best.get("PFP_N")) else None,
                        "reward_stress_aware_sum": float(best["reward_stress_aware_sum"]),
                        "max_swfac": float(best["max_swfac"]),
                        "max_nstres": float(best["max_nstres"]),
                        "swfac_days_gt_0p001": int(best["swfac_days_gt_0p001"]),
                        "nstres_days_gt_0p001": int(best["nstres_days_gt_0p001"]),
                        "irrigation_event_count": int(best["irrigation_event_count"]),
                        "n_event_count": int(best["n_event_count"]),
                        "action_sequence": best.get("action_sequence", ""),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(selected_rows)


def summarize_pair(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (y_label, n_label), g in selected.groupby(["yield_guardrail", "nstress_guardrail"]):
        ok = g[g["selection_status"].eq("selected")].copy()
        rows.append(
            {
                "yield_guardrail": y_label,
                "nstress_guardrail": n_label,
                "selected_seeds": int(len(ok)),
                "total_seeds": int(g["seed"].nunique()),
                "mean_yield": float(ok["final_grnwt"].mean()) if not ok.empty else None,
                "mean_irrigation": float(ok["total_irrigation"].mean()) if not ok.empty else None,
                "mean_n": float(ok["total_n"].mean()) if not ok.empty else None,
                "mean_max_nstres": float(ok["max_nstres"].mean()) if not ok.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["yield_guardrail", "nstress_guardrail"])


def write_record(counts: pd.DataFrame, selected: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    cols = [
        "yield_guardrail",
        "nstress_guardrail",
        "seed",
        "selection_status",
        "eligible_count",
        "checkpoint_step",
        "final_grnwt",
        "yield_gap_to_expert",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "reward_stress_aware_sum",
        "max_nstres",
        "nstres_days_gt_0p001",
        "action_sequence",
    ]
    lines = [
        "# 032_05 LC2010 checkpoint-selection guardrail audit record",
        "",
        "## Scope",
        "",
        "- Pure offline audit of 032_04 checkpoint evaluation results.",
        "- No training, no DSSAT rerun, no reward change.",
        f"- Expert yield reference: `{EXPERT_YIELD}` kg/ha.",
        "",
        "## Guardrail pair summary",
        "",
        pair_summary.to_string(index=False),
        "",
        "## Selected checkpoint table",
        "",
        selected[[c for c in cols if c in selected.columns]].to_string(index=False),
        "",
        "## Interpretation",
        "",
        "- A guardrail pair is useful only if it selects eligible checkpoints for at least 2/3 seeds.",
        "- Strict nitrogen-stress guardrails test whether high-yield candidates also keep NSTRES near zero.",
        "- Loose guardrails are diagnostic only; they should not be used to claim agronomic reliability if selected policies still have sustained NSTRES.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    df = load_eval()
    df.to_csv(OUT / "tables" / "032_05_input_checkpoint_eval_copy.csv", index=False, encoding="utf-8-sig")
    counts, selected = audit(df)
    pair_summary = summarize_pair(selected)
    counts.to_csv(OUT / "tables" / "032_05_guardrail_eligibility_counts.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "tables" / "032_05_guardrail_selected_checkpoints.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(OUT / "tables" / "032_05_guardrail_pair_summary.csv", index=False, encoding="utf-8-sig")
    write_record(counts, selected, pair_summary)
    result = {
        "task": "032_05_lc2010_checkpoint_guardrail_audit",
        "training_run": False,
        "dssat_run": False,
        "input": str(IN_CSV.relative_to(ROOT)),
        "selected_table": str((OUT / "tables" / "032_05_guardrail_selected_checkpoints.csv").relative_to(ROOT)),
        "pair_summary": str((OUT / "tables" / "032_05_guardrail_pair_summary.csv").relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
    }
    (OUT / "032_05_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
