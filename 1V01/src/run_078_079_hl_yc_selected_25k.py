"""Selected HL/YC 25K stability follow-up runner.

The runner is deliberately limited to the already selected candidate in each
formal config. It requires the matching 2K smoke and refuses >25K training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_078_079_hl_yc_stability_tuning as common


BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if int(cfg.get("training", {}).get("total_timesteps", 0)) != 25_000:
        raise ValueError("Formal selected runner is fixed at 25,000 timesteps.")
    if list(map(int, cfg.get("training", {}).get("checkpoint_steps", []))) != [5_000, 10_000, 25_000]:
        raise ValueError("Formal checkpoints must be [5000, 10000, 25000].")
    if cfg.get("promotion_policy", {}).get("no_50k_75k_100k") is not True:
        raise ValueError("Formal config must explicitly forbid 50K/75K/100K.")
    if cfg.get("candidate", {}).get("id") not in {"A_clip010", "B_nepochs3"}:
        raise ValueError("Only the registered selected candidates are allowed.")
    return cfg


def smoke_check(cfg: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(cfg["smoke_result"])
    checks: dict[str, Any] = {"smoke_result_exists": path.exists(), "smoke_result_path": rel(path)}
    if not path.exists():
        checks["next_step_allowed"] = False
        return checks
    result = json.loads(path.read_text(encoding="utf-8"))
    gate = result.get("action_gate", {})
    checks.update({
        "smoke_technical_pass": result.get("next_step_allowed") is True and gate.get("technical_pass") is True,
        "smoke_station_match": result.get("station_code") == cfg.get("station_code"),
        "smoke_candidate_match": result.get("candidate") == cfg.get("candidate", {}).get("id"),
        "smoke_input_profile": result.get("input_root", "").endswith("013_lowIC_manual"),
    })
    checks["next_step_allowed"] = all(value for key, value in checks.items() if key not in {"smoke_result_path", "next_step_allowed"})
    return checks


def model_kwargs(model_path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device="cpu")

    def widths(seq: Any) -> list[int]:
        return [int(layer.out_features) for layer in seq if hasattr(layer, "out_features")]

    extractor = model.policy.mlp_extractor
    return {
        "learning_rate": model.learning_rate,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "ent_coef": model.ent_coef,
        "clip_range": model.clip_range,
        "policy_kwargs": model.policy_kwargs,
        "policy_net_widths": widths(extractor.policy_net),
        "value_net_widths": widths(extractor.value_net),
    }


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    smoke = smoke_check(cfg)
    if not smoke["next_step_allowed"]:
        raise RuntimeError("Matching 2K smoke verification failed: " + json.dumps(smoke, ensure_ascii=False))
    candidate = {"id": cfg["candidate"]["id"], "override": cfg["candidate"]["override"]}
    pf = common.preflight(cfg, candidate)
    if not pf["next_step_allowed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    out = ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    (out / "configs").mkdir()
    shutil.copy2(config_path, out / "configs" / config_path.name)
    if cfg.get("prompt"):
        prompt_name = str(cfg["prompt"]).replace("prompts/", "")
    elif cfg["station_code"] == "HLA":
        prompt_name = "078_01_hl_lowIC_nepochs3_25k.md"
    elif cfg["candidate"]["id"] == "B_nepochs3":
        prompt_name = "079_02_yc_lowIC_nepochs3_25k.md"
    else:
        prompt_name = "079_01_yc_lowIC_clip010_25k.md"
    prompt = ROOT / "prompts" / prompt_name
    shutil.copy2(prompt, out / "configs" / prompt.name)
    effective_yaml = ROOT / "configs" / f"{cfg['task_id']}_effective_ppo_config.yaml"
    yaml_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    yaml_cfg["seed"] = int(cfg["seed"])
    yaml_cfg["ppo"].update(pf["effective_ppo"])
    effective_yaml.write_text(yaml.safe_dump(yaml_cfg, sort_keys=False), encoding="utf-8")

    engine = common.engine
    batch = common.batch
    ppo_safe_rendering = common.ppo_safe_rendering
    input_root = common.input_root(cfg)
    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"] if hasattr(engine, key)}
    old_config, old_out, old_doc, old_seed = batch.CONFIG, batch.OUT, batch.DOC, batch.SEED
    old_root, old_names = ppo_safe_rendering.MULTISITE_INPUT_ROOT, dict(batch.SITE_NAMES)
    start = time.perf_counter()
    rss0 = psutil.Process().memory_info().rss
    try:
        engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT = cfg["task_id"], cfg["task_name"], out
        engine.BASE_DOC = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
        engine.PROMPT = prompt
        engine.LOWIC_INPUT_ROOT = input_root
        engine.STATION, engine.SITES = cfg["station_code"], [cfg["station_code"]]
        engine.BINARY_IRRIGATION_LEVELS, engine.BINARY_NITROGEN_LEVELS = common.LEVELS_I, common.LEVELS_N
        batch.CONFIG, batch.OUT, batch.DOC = effective_yaml, out, engine.BASE_DOC
        batch.SEED = int(cfg["seed"])
        batch.SITE_NAMES[cfg["station_code"]] = cfg["site"]
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
        engine.run_training(25_000, [5_000, 10_000, 25_000], suffix="")

        summary_path = out / "evaluation" / "042_10_checkpoint_validation_summary.csv"
        summary = pd.read_csv(summary_path)
        checkpoint_results: dict[str, Any] = {}
        (out / "audits").mkdir(exist_ok=True)
        for step in [5_000, 10_000, 25_000]:
            final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(step)].copy()
            audit, gate = common.audit_rows(final)
            audit_path = out / "audits" / f"{cfg['task_id']}_ckpt{step}_action_audit.csv"
            audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
            metrics = final[["final_grnwt", "total_irrigation", "total_n", "PFP_N"]].mean(numeric_only=True).to_dict()
            checkpoint_results[str(step)] = {
                "validation_rows": int(len(final)),
                "metrics_mean": metrics,
                "action_gate": gate,
                "action_audit": rel(audit_path),
                "WP_ET_status": "not_available_in_25K_training_validation_summary; full replay required",
            }
        inventory = pd.read_csv(out / "evaluation" / "042_10_training_checkpoint_inventory.csv")
        model_row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(25_000)].iloc[0]
        observed = model_kwargs(ROOT / str(model_row["model_path"]))
        final_gate = checkpoint_results["25000"]["action_gate"]
        result = {
            **pf,
            "mode": "completed_formal25k",
            "output_root": rel(out),
            "effective_yaml": rel(effective_yaml),
            "smoke_verification": smoke,
            "effective_kwargs_from_loaded_25k_model": observed,
            "checkpoints": checkpoint_results,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "rss_delta_bytes": psutil.Process().memory_info().rss - rss0,
            "full_replay_required": True,
            "next_step_allowed": bool(final_gate.get("technical_pass")),
        }
        (out / f"{cfg['task_id']}_formal25k_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out / f"{cfg['task_id']}_run_manifest.json").write_text(json.dumps({"status": "completed_formal25k", "config": cfg, "preflight": pf, "smoke_verification": smoke, "observed_25k_kwargs": observed}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return result
    finally:
        for key, value in old.items():
            setattr(engine, key, value)
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = old_config, old_out, old_doc, old_seed
        batch.SITE_NAMES.clear()
        batch.SITE_NAMES.update(old_names)
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    if args.dry_run:
        cfg = load_config(config_path)
        candidate = {"id": cfg["candidate"]["id"], "override": cfg["candidate"]["override"]}
        print(json.dumps({"config": str(config_path), "preflight": common.preflight(cfg, candidate), "smoke_verification": smoke_check(cfg), "mode": "dry_run", "max_timesteps": 25000}, indent=2, ensure_ascii=False))
        return
    print(json.dumps(run(config_path), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
