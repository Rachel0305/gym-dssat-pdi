"""095_00: isolated FQ nitrogen-cost reward 2K smoke."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_094_fqa_entropy_smoke as base


CONFIG = ROOT / "configs/095_00_fqa_reward_ncost25_smoke.json"
PROMPT = ROOT / "prompts/095_00_fqa_reward_ncost25_smoke.md"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"


def patch_training_config() -> tuple[Path, dict]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    effective["ppo"].update(cfg["fixed_ppo"])
    effective["reward"].update(cfg["reward_override"])
    effective_path = ROOT / "configs/095_00_effective_reward_ncost25_ppo_config.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    return effective_path, cfg


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    if args.dry_run:
        print(json.dumps({"task_id": cfg["task_id"], "station": "FQA", "input_root": base.rel(INPUT_ROOT), "reward_override": cfg["reward_override"], "training_steps": 2000, "training_allowed": True, "external_n_allowed": False, "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    (out / "configs").mkdir(parents=True, exist_ok=False)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    effective_yaml, cfg = patch_training_config()
    engine, batch = base.engine, base.batch
    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_config, old_out, old_doc = batch.CONFIG, batch.OUT, batch.DOC
    old_root, old_names = base.ppo_safe_rendering.MULTISITE_INPUT_ROOT, dict(batch.SITE_NAMES)
    try:
        engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT = cfg["task_id"], cfg["task_name"], out
        engine.BASE_DOC, engine.PROMPT, engine.LOWIC_INPUT_ROOT = ROOT / "docs/095_00_fqa_reward_ncost25_smoke_record.md", PROMPT, INPUT_ROOT
        engine.STATION, engine.SITES = "FQA", ["FQA"]
        engine.BINARY_IRRIGATION_LEVELS, engine.BINARY_NITROGEN_LEVELS = base.LEVELS_I, base.LEVELS_N
        batch.CONFIG, batch.OUT, batch.DOC = effective_yaml, out, engine.BASE_DOC
        batch.SITE_NAMES["FQA"] = "FQ"
        base.ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        engine.run_training(2000, [1000, 2000], suffix="")
        source = out / "evaluation/042_10_checkpoint_validation_summary.csv"
        summary = pd.read_csv(source, keep_default_na=False)
        summary.to_csv(out / "evaluation/095_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        _, gate = base.audit(summary, out)
        if (out / "audits/094_00_action_audit.csv").exists():
            shutil.copy2(out / "audits/094_00_action_audit.csv", out / "audits/095_00_action_audit.csv")
        final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)]
        inv = pd.read_csv(out / "evaluation/042_10_training_checkpoint_inventory.csv")
        model_path = ROOT / str(inv[pd.to_numeric(inv["checkpoint_step"], errors="coerce").eq(2000)].iloc[0].model_path)
        observed = base.observed_kwargs(model_path)
        result = {"task_id": cfg["task_id"], "status": "completed_smoke", "training_run": True, "external_n_included": False,
                  "reward_override": cfg["reward_override"], "effective_yaml": base.rel(effective_yaml), "effective_kwargs_from_loaded_model": observed,
                  "validation_rows": int(len(final)), "mean_final_grnwt": float(pd.to_numeric(final["final_grnwt"], errors="coerce").mean()),
                  "mean_total_irrigation": float(pd.to_numeric(final["total_irrigation"], errors="coerce").mean()),
                  "mean_total_n": float(pd.to_numeric(final["total_n"], errors="coerce").mean()), "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred",
                  "action_gate": gate, "output_root": base.rel(out), "next_step_allowed": gate["mechanism_pass"]}
        (out / "095_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out / "095_00_manifest.json").write_text(json.dumps({"config": cfg, "effective_kwargs": observed, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        lines = ["# 095_00 FQ low-N reward 2K smoke", "", "- 唯一 reward 干预：`nitrogen_cost 1.58 -> 2.5`。", "- 无 external-N；WP_ET 未由 2K evaluator 导出。", "", "## Gate", ""]
        lines.extend(f"- `{k}`: `{v}`" for k, v in gate.items())
        lines.extend(["", "- 只有 `mechanism_pass=True` 才允许后续 25K；否则停止。", ""])
        (ROOT / "docs/095_00_fqa_reward_ncost25_smoke_record.md").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        for key, value in old.items():
            setattr(engine, key, value)
        batch.CONFIG, batch.OUT, batch.DOC = old_config, old_out, old_doc
        batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_names)
        base.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


if __name__ == "__main__":
    main()
