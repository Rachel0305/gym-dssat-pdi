from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as m


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "032_11_lc_multiyear_free_timing_ppo_training_length.md"
OUT = ROOT / "benchmark_results" / "032_11_lc_multiyear_free_timing_ppo_training_length"
DOC = ROOT / "docs" / "032_11_lc_multiyear_free_timing_ppo_training_length_record.md"
TOTAL_TIMESTEPS = 100000
CHECKPOINT_STEPS = [25000, 50000, 75000, 100000]
ORIGINAL_MAKE_SELECTION = m.make_selection


def install_032_11_globals() -> None:
    m.PROMPT = PROMPT
    m.OUT = OUT
    m.DOC = DOC
    m.TOTAL_TIMESTEPS = TOTAL_TIMESTEPS
    m.CHECKPOINT_STEPS = CHECKPOINT_STEPS


def make_selection_032_11() -> pd.DataFrame:
    rows = ORIGINAL_MAKE_SELECTION()
    rows["selection_reason"] = "032_11_lc_multiyear_free_timing_ppo_training_length"
    return rows


def write_record_032_11(config: dict, train_df: pd.DataFrame, reset_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    eval_cols = [
        "year",
        "checkpoint_step",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "reward_stress_aware_sum",
        "max_swfac",
        "max_nstres",
        "swfac_days_gt_0p05",
        "nstres_days_gt_0p05",
        "irrigation_event_count",
        "n_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "action_sequence",
    ]
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy() if not eval_df.empty else pd.DataFrame()
    by_ckpt = (
        ok.groupby("checkpoint_step", as_index=False)
        .agg(
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
            mean_reward=("reward_stress_aware_sum", "mean"),
            max_nstres=("max_nstres", "max"),
            max_swfac=("max_swfac", "max"),
        )
        if not ok.empty
        else pd.DataFrame()
    )
    sampled_all_years = bool((pd.to_numeric(reset_df.get("episode_count", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).all())
    all_ckpts_saved = bool(train_df["run_status"].astype(str).isin(["ok", "ok_existing"]).all())
    all_evals_complete = bool(len(ok) == len(CHECKPOINT_STEPS) * len(m.TRAIN_YEARS))
    early_pattern_rows = []
    if not ok.empty:
        for step, group in ok.groupby("checkpoint_step"):
            early_pattern_rows.append(
                {
                    "checkpoint_step": int(step),
                    "mean_first_irrigation_dap": float(pd.to_numeric(group["first_irrigation_dap"], errors="coerce").mean()),
                    "mean_first_n_dap": float(pd.to_numeric(group["first_n_dap"], errors="coerce").mean()),
                    "max_first_irrigation_dap": float(pd.to_numeric(group["first_irrigation_dap"], errors="coerce").max()),
                    "max_first_n_dap": float(pd.to_numeric(group["first_n_dap"], errors="coerce").max()),
                    "mean_irrigation_event_count": float(pd.to_numeric(group["irrigation_event_count"], errors="coerce").mean()),
                    "mean_n_event_count": float(pd.to_numeric(group["n_event_count"], errors="coerce").mean()),
                }
            )
    early_df = pd.DataFrame(early_pattern_rows)
    lines = [
        "# 032_11 LC multi-year free-timing MaskablePPO training-length record",
        "",
        "## Status",
        "",
        f"- Run pass: `{sampled_all_years and all_ckpts_saved and all_evals_complete}`.",
        f"- All train years sampled: `{sampled_all_years}`.",
        f"- All checkpoints saved: `{all_ckpts_saved}`.",
        f"- All checkpoint-year evaluations complete: `{all_evals_complete}`.",
        "",
        "## Scope",
        "",
        "- Station: LCA / LC.",
        f"- Training years: {', '.join(map(str, m.TRAIN_YEARS))}.",
        "- Algorithm: MaskablePPO only.",
        "- Weather forecast: not included.",
        f"- Seed: {m.SEED}.",
        f"- Total timesteps: {TOTAL_TIMESTEPS}.",
        f"- Checkpoints: {', '.join(map(str, CHECKPOINT_STEPS))}.",
        "- Reward/actions/constraints inherited unchanged from 032_10 / 032_00.",
        "",
        "## Training checkpoint inventory",
        "",
        train_df.to_string(index=False),
        "",
        "## Training year sampling",
        "",
        reset_df.to_string(index=False),
        "",
        "## Mean performance by checkpoint across LC2005-2010",
        "",
        by_ckpt.to_string(index=False) if not by_ckpt.empty else "No successful evaluations.",
        "",
        "## Early-action pattern by checkpoint",
        "",
        early_df.to_string(index=False) if not early_df.empty else "No successful evaluations.",
        "",
        "## Per-year checkpoint evaluation",
        "",
        eval_df[[c for c in eval_cols if c in eval_df.columns]].to_string(index=False) if not eval_df.empty else "No eval rows.",
        "",
        "## Interpretation boundary",
        "",
        "- This task only tests whether longer training helps the same no-forecast multi-year setup.",
        "- It is not a final cross-year model-selection result.",
        "- It does not evaluate LC2011-2020 or LC2021-2023.",
        "- It does not tune reward, constraints, or checkpoint-selection rules in-place.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    install_032_11_globals()
    m.make_selection = make_selection_032_11
    m.write_record = write_record_032_11

    m.ensure_dirs()
    shutil.copyfile(m.CONFIG, OUT / "configs" / m.CONFIG.name)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    config = m.load_config()
    selection = m.make_selection()
    selection.to_csv(OUT / "configs" / "032_11_lc_train_year_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "032_11_resolved_env_config.yaml")

    train_df, reset_df = m.train(config, env_config)
    train_df.to_csv(OUT / "evaluation" / "032_11_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")

    eval_rows = []
    for _, row in train_df.iterrows():
        if str(row["run_status"]) not in {"ok", "ok_existing"}:
            continue
        for year in m.TRAIN_YEARS:
            eval_rows.append(m.evaluate_checkpoint(config, env_config, row, year))
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUT / "evaluation" / "032_11_train_year_checkpoint_eval_summary.csv", index=False, encoding="utf-8-sig")

    write_record_032_11(config, train_df, reset_df, eval_df)
    result = {
        "task": "032_11_lc_multiyear_free_timing_ppo_training_length",
        "training_run": True,
        "station": m.STATION,
        "train_years": m.TRAIN_YEARS,
        "seed": m.SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoints": CHECKPOINT_STEPS,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "eval_summary": str((OUT / "evaluation" / "032_11_train_year_checkpoint_eval_summary.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "032_11_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not eval_df.empty:
        print(eval_df[["year", "checkpoint_step", "run_status", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "max_nstres", "action_sequence"]].to_string(index=False))


if __name__ == "__main__":
    main()
