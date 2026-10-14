"""105_00 read-only FQ checkpoint selection from existing replay tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/105_00_fqa_checkpoint_selection.json"
SOURCE = ROOT / "benchmark_results/098_00_fqa_ncost25_early_checkpoint_replay_2k_5k_10k/tables"
SEED1_RESULT = ROOT / "benchmark_results/102_00_fqa_originIC_ncost25_entropy002_smoke2k/102_00_result.json"
OUT = ROOT / "benchmark_results/105_00_fqa_checkpoint_selection"
DOC = ROOT / "docs/105_00_fqa_checkpoint_selection_record.md"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    tables = []
    for checkpoint in cfg["candidate_checkpoints"]:
        path = SOURCE / f"ckpt{checkpoint}" / f"098_00_ckpt{checkpoint}_season_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, keep_default_na=False)
        df["checkpoint_step"] = checkpoint
        tables.append(df)
    all_rows = pd.concat(tables, ignore_index=True)
    for column in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]:
        all_rows[column] = pd.to_numeric(all_rows[column], errors="coerce")
    summary = all_rows.groupby(["checkpoint_step", "scenario"], as_index=False).agg(
        years=("year", "count"),
        mean_yield=("grain_yield_kg_ha", "mean"),
        mean_wp_et=("WP_ET_kg_m3", "mean"),
        mean_pfp_n=("PFP_N_kg_kg", "mean"),
        mean_irrigation=("actual_irrigation_mm", "mean"),
        mean_n=("actual_nitrogen_kg_ha", "mean"),
    )
    official = summary[summary["scenario"].eq("official_extension_expert")].set_index("checkpoint_step")
    ppo = summary[summary["scenario"].eq("ppo")].set_index("checkpoint_step")
    decisions = []
    for checkpoint in cfg["candidate_checkpoints"]:
        p = ppo.loc[checkpoint]
        o = official.loc[checkpoint]
        decisions.append({
            "checkpoint_step": checkpoint,
            "ppo_mean_yield": float(p["mean_yield"]),
            "official_mean_yield": float(o["mean_yield"]),
            "ppo_mean_wp_et": float(p["mean_wp_et"]),
            "official_mean_wp_et": float(o["mean_wp_et"]),
            "ppo_mean_pfp_n": float(p["mean_pfp_n"]),
            "official_mean_pfp_n": float(o["mean_pfp_n"]),
            "yield_guardrail_pass": bool(float(p["mean_yield"]) >= float(o["mean_yield"])),
            "wp_et_guardrail_pass": bool(float(p["mean_wp_et"]) >= float(o["mean_wp_et"])),
        })
    selected = next((row["checkpoint_step"] for row in decisions if row["yield_guardrail_pass"] and row["wp_et_guardrail_pass"]), None)

    yearly = all_rows[all_rows["scenario"].ne("ppo")].copy()
    ppo_year = all_rows[all_rows["scenario"].eq("ppo")][["checkpoint_step", "year", "grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].copy()
    baseline = all_rows[all_rows["scenario"].ne("ppo")]
    wins = []
    for checkpoint in cfg["candidate_checkpoints"]:
        for year in sorted(ppo_year[ppo_year["checkpoint_step"].eq(checkpoint)]["year"].unique()):
            p = ppo_year[(ppo_year["checkpoint_step"].eq(checkpoint)) & (ppo_year["year"].eq(year))].iloc[0]
            b = baseline[(baseline["checkpoint_step"].eq(checkpoint)) & (baseline["year"].eq(year))]
            wins.append({
                "checkpoint_step": checkpoint,
                "year": int(year),
                "yield_win": bool(float(p["grain_yield_kg_ha"]) >= float(b["grain_yield_kg_ha"].max())),
                "wp_et_win": bool(float(p["WP_ET_kg_m3"]) >= float(b["WP_ET_kg_m3"].max())),
                "pfp_n_win": bool(float(p["PFP_N_kg_kg"]) >= float(pd.to_numeric(b["PFP_N_kg_kg"], errors="coerce").max())) if pd.to_numeric(b["PFP_N_kg_kg"], errors="coerce").notna().any() else False,
            })
    summary.to_csv(OUT / "105_00_checkpoint_scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(decisions).to_csv(OUT / "105_00_checkpoint_decision_table.csv", index=False, encoding="utf-8-sig")
    wins_df = pd.DataFrame(wins)
    wins_df.to_csv(OUT / "105_00_checkpoint_yearly_wins.csv", index=False, encoding="utf-8-sig")
    seed1 = json.loads(SEED1_RESULT.read_text(encoding="utf-8")) if SEED1_RESULT.exists() else {"status": "missing"}
    selected_wins = wins_df[wins_df["checkpoint_step"].eq(selected)] if selected is not None else pd.DataFrame()
    selected_yearly_wins = {
        "years": int(len(selected_wins)),
        "yield_wins": int(selected_wins["yield_win"].sum()) if not selected_wins.empty else 0,
        "wp_et_wins": int(selected_wins["wp_et_win"].sum()) if not selected_wins.empty else 0,
        "pfp_n_wins": int(selected_wins["pfp_n_win"].sum()) if not selected_wins.empty else 0,
    }
    result = {
        "task_id": cfg["task_id"],
        "mode": "read_only_selection",
        "training_run": False,
        "replay_run": False,
        "candidate_checkpoints": cfg["candidate_checkpoints"],
        "selection_rule": cfg["selection_rule"],
        "selected_checkpoint": selected,
        "decisions": decisions,
        "selected_yearly_wins": selected_yearly_wins,
        "seed1_gate": seed1.get("action_gate", {}),
        "cross_seed_stable": bool(seed1.get("action_gate", {}).get("mechanism_pass", False)),
        "paper_status": "exploratory_only_seed1_gate_failed" if selected is not None and not seed1.get("action_gate", {}).get("mechanism_pass", False) else "not_selected",
        "source_root": rel(SOURCE),
        "output_root": rel(OUT),
    }
    (OUT / "105_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 105_00 FQ checkpoint selection",
        "",
        "- 只读复用 098 五情景 replay 表；未训练、未重新运行 DSSAT。",
        f"- 按预注册规则选择的最早 checkpoint：`{selected}`。",
        "- seed=1 稳定性 gate 未通过，因此该 checkpoint 只能作为探索性结果，不能替换正式论文结果。",
        "",
        "## Checkpoint decisions",
        "",
        pd.DataFrame(decisions).to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- PFP_N 单独报告，不用单一指标选择 checkpoint。",
        "- 不将 checkpoint 选择解释为跨 seed 稳定性证明。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
