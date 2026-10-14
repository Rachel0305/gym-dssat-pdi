"""074: isolated HL seed1 25K four-scenario replay.

The native DSSAT automatic-irrigation snapshot is read from the frozen four
baseline rebuild.  The external-N scenario is deliberately not loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
import run_sya_ppo_configured_046_02 as base04602
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env

HLA_REPORT = ROOT / "src" / "054_hla_lowIC_site_transfer" / "run_054_02_hla_lowIC_five_scenario_figures.py"
spec = importlib.util.spec_from_file_location("hla_report_074", HLA_REPORT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {HLA_REPORT}")
hla = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hla)

PLAN = ROOT / "configs" / "074_hla_lowIC_lr1e4_seed1_ckpt25k_four_scenario_replay.json"
PROMPT = ROOT / "prompts" / "2026-08-14_hl_seed1_lr1e4_25k_four_scenario_replay.md"
INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
SCENARIOS = ["recorded_farmer_template", "official_extension_expert", "dssat_auto", "rl_candidate"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cfg() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def out_root(c: dict[str, Any]) -> Path:
    return ROOT / c["output_root"]


def model_path(c: dict[str, Any]) -> Path:
    return ROOT / c["ppo_run_dir"] / "models" / "HLA" / "HLA_half_split_stress_aware_maskableppo_seed1_ckpt25000.zip"


def baseline_snapshot(c: dict[str, Any], year: int, scenario: str) -> Path:
    return ROOT / c["baseline_root"] / "snapshots" / "HLA" / str(year) / scenario


def preflight(c: dict[str, Any]) -> dict[str, Any]:
    out = out_root(c)
    issues: list[str] = []
    if out.exists() and any(out.iterdir()):
        issues.append("isolated output root already exists and is non-empty")
    if not model_path(c).exists():
        issues.append("25K PPO model missing")
    summary = ROOT / c["saved_summary"]
    if not summary.exists():
        issues.append("073 saved validation summary missing")
    if not (INPUT_ROOT / "HL" / "CNHL0701_corrected_IC123.MZX").exists():
        issues.append("HL lowIC MZX missing")
    for year in c["validation_years"]:
        for scenario in ["recorded_farmer_template", "official_extension_expert", "dssat_auto"]:
            snap = baseline_snapshot(c, year, scenario)
            if not (snap / "Summary.OUT").exists():
                issues.append(f"missing native four-baseline snapshot {scenario}/{year}")
    return {"task": f"{c['task_id']}_{c['task_name']}", "output_root": rel(out), "model": rel(model_path(c)), "input_root": rel(INPUT_ROOT), "issues": issues, "next_step_allowed": not issues}


def setup_env(c: dict[str, Any], out: Path):
    old = {
        "config": batch.CONFIG, "out": batch.OUT, "doc": batch.DOC, "seed": batch.SEED,
        "sites": list(batch.SITES), "site_names": dict(batch.SITE_NAMES),
        "input_root": ppo_safe_rendering.MULTISITE_INPUT_ROOT,
        "task_id": base04602.TASK_ID if hasattr(base04602, "TASK_ID") else None,
    }
    batch.CONFIG = ROOT / "configs" / "073_hla_seed1_lr1e4_effective.yaml"
    batch.OUT = out
    batch.DOC = out / "074_record.md"
    batch.SEED = int(c["seed"])
    batch.SITES = ["HLA"]
    batch.SITE_NAMES["HLA"] = "HL"
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
    config = batch.load_config()
    selection = batch.build_selection(batch.load_split())
    envcfg = batch.direct_ppo.build_env_config(config, selection)
    envcfg["paths"]["output_root"] = rel(out)
    return old, config, envcfg


def restore_env(old: dict[str, Any]) -> None:
    batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = old["config"], old["out"], old["doc"], old["seed"]
    batch.SITES = old["sites"]
    batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old["site_names"])
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["input_root"]


def replay_ppo(c: dict[str, Any], config: Any, envcfg: dict[str, Any], model: Any, year: int, out: Path, expected: float) -> tuple[Path, dict[str, Any]]:
    from sb3_contrib.common.maskable.utils import get_action_masks
    env = batch.base.make_env(config, envcfg, "HLA", int(year), int(c["seed"]), f"HLA_{year}_074_replay", evaluation=True)
    done = False; steps = 0
    try:
        obs, _ = env.reset()
        while not done and steps < 260:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated); steps += 1
        if not done:
            raise RuntimeError(f"HLA{year}: PPO replay did not finish within 260 steps")
        temp = snapshot_from_env(env)
        target = out / "snapshots" / "HLA" / str(year) / "rl_candidate"
        if target.exists():
            raise RuntimeError(f"refusing to overwrite {target}")
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(temp, target)
    finally:
        env.close()
    hla.STATION = "HLA"; hla.SITE = "HLA"
    daily = hla.daily_from_snapshot(target, year, "rl_candidate")
    endpoint = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
    diff = endpoint - float(expected)
    if not np.isclose(endpoint, expected, rtol=0.0, atol=float(c["endpoint_tolerance_kg_ha"])):
        raise RuntimeError(f"HLA{year}: endpoint mismatch saved={expected:.6f}, replay={endpoint:.6f}, diff={diff:.6f}")
    return target, {"year": int(year), "saved_endpoint_kg_ha": float(expected), "replay_endpoint_kg_ha": endpoint, "difference_kg_ha": diff, "episode_steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=["dry-run", "smoke", "replay"], required=True)
    c = cfg(); args = ap.parse_args(); pf = preflight(c)
    if args.phase == "dry-run":
        print(json.dumps(pf, ensure_ascii=False, indent=2)); return
    if not pf["next_step_allowed"]:
        raise RuntimeError("; ".join(pf["issues"]))
    out = out_root(c); out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PLAN, out / "074_config.json"); shutil.copy2(PROMPT, out / "074_prompt.md")
    old, config, envcfg = setup_env(c, out)
    hla.STATION = "HLA"; hla.SITE = "HLA"
    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(model_path(c)), device="cpu")
        summary = pd.read_csv(ROOT / c["saved_summary"], keep_default_na=False)
        years = [2014] if args.phase == "smoke" else list(map(int, c["validation_years"]))
        endpoint_rows: list[dict[str, Any]] = []; metric_rows: list[dict[str, Any]] = []; daily_rows: list[pd.DataFrame] = []
        for year in years:
            row = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(int(c["checkpoint"])) & pd.to_numeric(summary["year"], errors="coerce").eq(year)]
            if len(row) != 1: raise RuntimeError(f"expected one 073 endpoint row for HLA/{year}")
            expected = float(pd.to_numeric(row.iloc[0]["final_grnwt"], errors="coerce"))
            ppo_snap, endpoint = replay_ppo(c, config, envcfg, model, year, out, expected); endpoint_rows.append(endpoint)
            smap = {s: (ppo_snap if s == "rl_candidate" else baseline_snapshot(c, year, s)) for s in SCENARIOS}
            frames = [hla.daily_from_snapshot(smap[s], year, s) for s in SCENARIOS]
            combined = pd.concat(frames, ignore_index=True); daily_rows.append(combined)
            for scenario in SCENARIOS:
                sub = combined[combined["scenario"].eq(scenario)].sort_values("dap"); last = sub.iloc[-1]
                m = baseline.metrics_from_snapshot(smap[scenario], float(last["grain_yield_kg_ha"]))
                metric_rows.append({"year": year, "scenario": scenario, "final_grain_kg_ha": float(last["grain_yield_kg_ha"]), "irrigation_mm": float(m["actual_irrigation_mm"]), "nitrogen_kg_ha": float(m["actual_nitrogen_kg_ha"]), "etcp_mm": float(m["etcp_mm"]), "WP_ET_kg_m3": float(m["WP_ET_kg_m3"]), "PFP_N_kg_kg": float(m["PFP_N_kg_kg"]) if pd.notna(m["PFP_N_kg_kg"]) else np.nan})
        ep = pd.DataFrame(endpoint_rows); md = pd.DataFrame(metric_rows)
        (out / "evaluation").mkdir(parents=True, exist_ok=True); (out / "daily_outputs").mkdir(parents=True, exist_ok=True)
        ep.to_csv(out / "evaluation" / "074_endpoint_audit.csv", index=False, encoding="utf-8-sig")
        md.to_csv(out / "evaluation" / "074_four_scenario_season_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(daily_rows, ignore_index=True).to_csv(out / "daily_outputs" / "074_four_scenario_daily.csv", index=False, encoding="utf-8-sig")
        ppo = md[md.scenario.eq("rl_candidate")].set_index("year")
        base = md[~md.scenario.eq("rl_candidate")].groupby("year").max(numeric_only=True)
        gap = ppo[["final_grain_kg_ha","WP_ET_kg_m3","PFP_N_kg_kg"]].join(base[["final_grain_kg_ha","WP_ET_kg_m3","PFP_N_kg_kg"]], lsuffix="_ppo", rsuffix="_best_baseline")
        gap["yield_gap_vs_best_baseline"] = gap.final_grain_kg_ha_ppo - gap.final_grain_kg_ha_best_baseline
        gap["WP_ET_gap_vs_best_baseline"] = gap.WP_ET_kg_m3_ppo - gap.WP_ET_kg_m3_best_baseline
        gap["PFP_N_gap_vs_best_baseline"] = gap.PFP_N_kg_kg_ppo - gap.PFP_N_kg_kg_best_baseline
        gap.to_csv(out / "evaluation" / "074_ppo_gaps_vs_four_baseline_max.csv", encoding="utf-8-sig")
        result = {"task": f"{c['task_id']}_{c['task_name']}", "phase": args.phase, "preflight": pf, "checkpoint": c["checkpoint"], "years": years, "main_scenarios": c["main_scenarios"], "excluded_scenarios": c["excluded_scenarios"], "model_sha256": sha256(model_path(c)), "endpoint_gate": bool((ep["difference_kg_ha"].abs() <= c["endpoint_tolerance_kg_ha"]).all()), "metrics": rel(out / "evaluation" / "074_four_scenario_season_metrics.csv"), "gaps": rel(out / "evaluation" / "074_ppo_gaps_vs_four_baseline_max.csv"), "next_step_allowed": True}
        (out / "074_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        restore_env(old)


if __name__ == "__main__":
    main()
