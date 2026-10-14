"""094_00: isolated FQ entropy-coefficient 2K smoke."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


CONFIG = ROOT / "configs/094_00_fqa_entropy_smoke.json"
PROMPT = ROOT / "prompts/094_00_fqa_entropy_smoke.md"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
STATION, SITE = "FQA", "FQ"
LEVELS_I, LEVELS_N = [0.0, 15.0, 30.0, 45.0], [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def signature(daily: pd.DataFrame) -> str:
    i = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    n = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
    return "; ".join(sorted({f"I{float(a):g}/N{float(b):g}" for a, b in zip(i, n) if a > 1e-9 or b > 1e-9}))


def patch_training_config(out: Path) -> tuple[Path, dict[str, Any]]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    base["ppo"].update(cfg["fixed_ppo"])
    base["ppo"].update(cfg["ppo_override"])
    effective = ROOT / "configs/094_00_effective_entropy002_ppo_config.yaml"
    effective.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return effective, cfg


def audit(summary: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for rec in summary.itertuples(index=False):
        daily_path = ROOT / str(rec.daily_csv_path)
        daily = pd.read_csv(daily_path)
        i = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
        n = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
        dap = pd.to_numeric(daily["dap"], errors="coerce").fillna(0.0)
        prev_i = pd.to_numeric(daily["previous_cumulative_irrigation"], errors="coerce")
        prev_n = pd.to_numeric(daily["previous_cumulative_n"], errors="coerce")
        cum_i = pd.to_numeric(daily["season_cumulative_irrigation"], errors="coerce")
        cum_n = pd.to_numeric(daily["season_cumulative_n"], errors="coerce")
        positive = (i > 1e-9) | (n > 1e-9)
        transmitted = (~positive) | (np.isclose(cum_i - prev_i, i, atol=1e-6) & np.isclose(cum_n - prev_n, n, atol=1e-6))
        rows.append({
            "checkpoint_step": int(rec.checkpoint_step), "year": int(rec.year), "daily_path": str(rec.daily_csv_path),
            "final_grnwt": float(rec.final_grnwt), "total_irrigation": float(rec.total_irrigation), "total_n": float(rec.total_n),
            "action_signature": signature(daily), "positive_action_pair_count": int(len(set(f"I{float(a):g}/N{float(b):g}" for a, b in zip(i[positive], n[positive])))),
            "positive_action_rows_after_dap1": int((positive & (dap > 1)).sum()),
            "raw_to_safe_mismatch_rows": int((~np.isclose(pd.to_numeric(daily["raw_action_amir"], errors="coerce"), i, atol=1e-6) | ~np.isclose(pd.to_numeric(daily["raw_action_anfer"], errors="coerce"), n, atol=1e-6)).sum()),
            "safe_to_dssat_mismatch_rows": int((positive & ~transmitted).sum()),
            "off_grid_rows": int((~np.isclose(i.to_numpy()[:, None], np.asarray(LEVELS_I)[None, :], atol=1e-6).any(axis=1)).sum() + (~np.isclose(n.to_numpy()[:, None], np.asarray(LEVELS_N)[None, :], atol=1e-6).any(axis=1)).sum()),
        })
    audit_df = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    final = audit_df[audit_df["checkpoint_step"].eq(2000)].copy()
    non_2018 = final[final["year"].ne(2018)]
    sig_counts = final["action_signature"].value_counts()
    all_pairs = set("; ".join(final["action_signature"]).split("; ")) - {""}
    gate = {
        "validation_rows": int(len(final)), "expected_validation_rows": 10,
        "raw_to_safe_mismatch_rows": int(final["raw_to_safe_mismatch_rows"].sum()),
        "safe_to_dssat_mismatch_rows": int(final["safe_to_dssat_mismatch_rows"].sum()),
        "off_grid_rows": int(final["off_grid_rows"].sum()),
        "nonzero_action_pair_count": int(len(all_pairs)),
        "positive_action_rows_after_dap1": int(final["positive_action_rows_after_dap1"].sum()),
        "cross_year_distinct_signatures": int(final["action_signature"].nunique()),
        "minority_signature_year_count": int(sig_counts.iloc[-1]) if len(sig_counts) > 1 else 0,
        "non_2018_cross_year_distinct_signatures": int(non_2018["action_signature"].nunique()),
    }
    gate["mechanism_pass"] = bool(
        gate["validation_rows"] == 10 and gate["raw_to_safe_mismatch_rows"] == 0 and gate["safe_to_dssat_mismatch_rows"] == 0
        and gate["off_grid_rows"] == 0 and gate["nonzero_action_pair_count"] >= 3 and gate["positive_action_rows_after_dap1"] > 0
        and gate["cross_year_distinct_signatures"] > 1 and gate["minority_signature_year_count"] >= 2
        and gate["non_2018_cross_year_distinct_signatures"] > 1
    )
    audit_df.to_csv(out / "audits/094_00_action_audit.csv", index=False, encoding="utf-8-sig")
    return audit_df, gate


def observed_kwargs(model_path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(model_path), device="cpu")
    return {"learning_rate": model.learning_rate, "gamma": model.gamma, "gae_lambda": model.gae_lambda,
            "n_steps": model.n_steps, "batch_size": model.batch_size, "n_epochs": model.n_epochs,
            "ent_coef": model.ent_coef, "clip_range": model.clip_range, "policy_kwargs": model.policy_kwargs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    if args.dry_run:
        print(json.dumps({"task_id": cfg["task_id"], "station": STATION, "input_root": rel(INPUT_ROOT), "override": cfg["ppo_override"], "training_steps": 2000, "training_allowed": True, "external_n_allowed": False, "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    (out / "configs").mkdir(parents=True, exist_ok=False)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    effective_yaml, cfg = patch_training_config(out)
    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_config, old_out, old_doc = batch.CONFIG, batch.OUT, batch.DOC
    old_root, old_names = ppo_safe_rendering.MULTISITE_INPUT_ROOT, dict(batch.SITE_NAMES)
    try:
        engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT = cfg["task_id"], cfg["task_name"], out
        engine.BASE_DOC, engine.PROMPT, engine.LOWIC_INPUT_ROOT = ROOT / "docs/094_00_fqa_entropy_smoke_record.md", PROMPT, INPUT_ROOT
        engine.STATION, engine.SITES = STATION, [STATION]
        engine.BINARY_IRRIGATION_LEVELS, engine.BINARY_NITROGEN_LEVELS = LEVELS_I, LEVELS_N
        batch.CONFIG, batch.OUT, batch.DOC = effective_yaml, out, engine.BASE_DOC
        batch.SITE_NAMES[STATION] = SITE
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        engine.run_training(2000, [1000, 2000], suffix="")
        source = out / "evaluation/042_10_checkpoint_validation_summary.csv"
        summary = pd.read_csv(source, keep_default_na=False)
        summary.to_csv(out / "evaluation/094_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        _, gate = audit(summary, out)
        final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)]
        inv = pd.read_csv(out / "evaluation/042_10_training_checkpoint_inventory.csv")
        model_path = ROOT / str(inv[pd.to_numeric(inv["checkpoint_step"], errors="coerce").eq(2000)].iloc[0].model_path)
        observed = observed_kwargs(model_path)
        result = {"task_id": cfg["task_id"], "status": "completed_smoke", "training_run": True, "external_n_included": False,
                  "effective_yaml": rel(effective_yaml), "effective_kwargs_from_loaded_model": observed, "validation_rows": int(len(final)),
                  "mean_final_grnwt": float(pd.to_numeric(final["final_grnwt"], errors="coerce").mean()),
                  "mean_total_irrigation": float(pd.to_numeric(final["total_irrigation"], errors="coerce").mean()),
                  "mean_total_n": float(pd.to_numeric(final["total_n"], errors="coerce").mean()),
                  "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred", "action_gate": gate,
                  "output_root": rel(out), "next_step_allowed": gate["mechanism_pass"]}
        (out / "094_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out / "094_00_manifest.json").write_text(json.dumps({"config": cfg, "effective_kwargs": observed, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        doc_lines = ["# 094_00 FQ entropy regularization 2K smoke", "", "- 唯一干预：`ent_coef 0.01 -> 0.02`。", "- 无 external-N；WP_ET 未由 2K evaluator 导出。", "", "## Gate", ""]
        doc_lines.extend(f"- `{k}`: `{v}`" for k, v in gate.items())
        doc_lines.extend(["", "- 只有 `mechanism_pass=True` 才允许后续 25K；否则停止。", ""])
        (ROOT / "docs/094_00_fqa_entropy_smoke_record.md").write_text("\n".join(doc_lines), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        for key, value in old.items():
            setattr(engine, key, value)
        batch.CONFIG, batch.OUT, batch.DOC = old_config, old_out, old_doc
        batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_names)
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


if __name__ == "__main__":
    main()
