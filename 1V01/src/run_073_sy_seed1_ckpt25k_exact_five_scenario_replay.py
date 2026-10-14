"""073: exact-contract SY 25K ten-year five-scenario replay.

This is intentionally isolated from 046_06.  PPO snapshots use the same
032_22 raw 16-action environment as the 070 direct evaluator; the four frozen
baseline snapshots are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
import run_sya_ppo_configured_046_02 as base04602
from build_sya_configured_five_scenario_daily_plots_046_06 import daily_from_snapshot
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env

PLAN = ROOT / "configs" / "073_sy_seed1_ckpt25k_exact_five_scenario_replay.json"
PROMPT = ROOT / "prompts" / "2026-08-13_sy_seed1_ckpt25k_exact_five_scenario_replay.md"
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert", "rl_candidate"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cfg() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def output_root(c: dict[str, Any]) -> Path:
    return ROOT / c["output_root"]


def paths(c: dict[str, Any]) -> dict[str, Path]:
    base = ROOT / c["baseline_root"] / "snapshots" / "SYA"
    auto = ROOT / c["external_auto_root"] / "snapshots" / "SYA"
    return {
        "null": base,
        "recorded_farmer_template": base,
        "official_extension_expert": base,
        "dssat_auto_external_n": auto,
    }


def preflight(c: dict[str, Any]) -> dict[str, Any]:
    out = output_root(c)
    summary = ROOT / c["saved_summary"]
    model = ROOT / c["ppo_run_dir"] / "models" / "SYA" / f"SYA_half_split_stress_aware_maskableppo_seed1_ckpt{int(c['checkpoint'])}.zip"
    input_root = base04602.INPUT_PROFILES[c["input_profile"]]
    issues: list[str] = []
    if out.exists() and any(out.iterdir()):
        issues.append("isolated output root already exists")
    for label, root in paths(c).items():
        for year in c["validation_years"]:
            name = "dssat_auto_irrigation_external_n_rule" if label == "dssat_auto_external_n" else label
            snap = root / str(year) / name
            if not (snap / "Summary.OUT").exists():
                issues.append(f"missing frozen {label} snapshot for {year}")
    if not summary.exists():
        issues.append("saved 070 summary missing")
    if not model.exists():
        issues.append("25K PPO model missing")
    if not input_root.exists():
        issues.append("originIC input root missing")
    return {"task": f"{c['task_id']}_{c['task_name']}", "output_root": rel(out), "model": rel(model), "input_root": rel(input_root), "issues": issues, "next_step_allowed": not issues}


def snapshot_map(c: dict[str, Any], out: Path, year: int, ppo_snapshot: Path) -> dict[str, Path]:
    roots = paths(c)
    return {
        "null": roots["null"] / str(year) / "null",
        "recorded_farmer_template": roots["recorded_farmer_template"] / str(year) / "recorded_farmer_template",
        "dssat_auto_external_n": roots["dssat_auto_external_n"] / str(year) / "dssat_auto_irrigation_external_n_rule",
        "official_extension_expert": roots["official_extension_expert"] / str(year) / "official_extension_expert",
        "rl_candidate": ppo_snapshot,
    }


def replay_one(c: dict[str, Any], config: dict[str, Any], envcfg: dict[str, Any], model: Any, year: int, out: Path, expected: float) -> tuple[Path, dict[str, Any]]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = batch.base.make_env(config, envcfg, "SYA", int(year), int(c["seed"]), f"SYA_{year}_073_exact_replay", evaluation=True)
    done = False
    steps = 0
    try:
        obs, _ = env.reset()
        while not done and steps < 260:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"SYA{year}: replay did not finish within 260 steps")
        temp = snapshot_from_env(env)
        target = out / "snapshots" / "SYA" / str(year) / "rl_candidate"
        if target.exists():
            raise RuntimeError(f"refusing to overwrite isolated target {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp, target)
    finally:
        env.close()
    daily = daily_from_snapshot(target, int(year), "rl_candidate")
    endpoint = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
    diff = endpoint - float(expected)
    if not np.isclose(endpoint, expected, rtol=0.0, atol=float(c["endpoint_tolerance_kg_ha"])):
        raise RuntimeError(f"SYA{year}: endpoint mismatch saved={expected:.6f}, replay={endpoint:.6f}, diff={diff:.6f}")
    return target, {"year": int(year), "saved_endpoint_kg_ha": float(expected), "replay_endpoint_kg_ha": endpoint, "difference_kg_ha": diff, "episode_steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["dry-run", "replay"], required=True)
    parser.add_argument("--checkpoint", type=int, default=None)
    args = parser.parse_args()
    c = cfg()
    if args.checkpoint is not None:
        c["checkpoint"] = int(args.checkpoint)
        c["output_root"] = c["output_root"].replace("ckpt25k", f"ckpt{int(args.checkpoint)//1000}k")
    pf = preflight(c)
    if args.phase == "dry-run":
        print(json.dumps(pf, ensure_ascii=False, indent=2))
        return
    if not pf["next_step_allowed"]:
        raise RuntimeError("; ".join(pf["issues"]))
    out = output_root(c)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PLAN, out / "073_config.json")
    shutil.copy2(PROMPT, out / "073_prompt.md")
    old = (batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, dict(batch.SITE_NAMES), ppo_safe_rendering.MULTISITE_INPUT_ROOT, list(batch.SITES))
    try:
        batch.CONFIG = ROOT / "configs" / "070_00_sya_effective.yaml"
        batch.OUT = out
        batch.DOC = out / "073_record.md"
        batch.SEED = int(c["seed"])
        batch.SITES = ["SYA"]
        batch.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04602.INPUT_PROFILES[c["input_profile"]]
        config = batch.load_config()
        selection = batch.build_selection(batch.load_split())
        envcfg = batch.direct_ppo.build_env_config(config, selection)
        envcfg["paths"]["output_root"] = rel(out)
        model_path = ROOT / c["ppo_run_dir"] / "models" / "SYA" / f"SYA_half_split_stress_aware_maskableppo_seed1_ckpt{int(c['checkpoint'])}.zip"
        summary = pd.read_csv(ROOT / c["saved_summary"], keep_default_na=False)
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(model_path), device="cpu")
        endpoint_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        daily_rows: list[pd.DataFrame] = []
        for year in c["validation_years"]:
            row = summary[(pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(int(c["checkpoint"]))) & (pd.to_numeric(summary["year"], errors="coerce").eq(int(year)))]
            if len(row) != 1:
                raise RuntimeError(f"expected one saved 070 summary row for SYA/{year}/25K, found {len(row)}")
            expected = float(pd.to_numeric(row.iloc[0]["final_grnwt"], errors="coerce"))
            ppo_snap, endpoint = replay_one(c, config, envcfg, model, int(year), out, expected)
            endpoint_rows.append(endpoint)
            smap = snapshot_map(c, out, int(year), ppo_snap)
            frames = [daily_from_snapshot(smap[scenario], int(year), scenario) for scenario in SCENARIOS]
            combined = pd.concat(frames, ignore_index=True)
            daily_rows.append(combined)
            for scenario in SCENARIOS:
                sub = combined[combined["scenario"].eq(scenario)]
                last = sub.iloc[-1]
                m = baseline.metrics_from_snapshot(smap[scenario], float(last["grain_yield_kg_ha"]))
                metric_rows.append({"year": int(year), "scenario": scenario, "final_grain_kg_ha": float(last["grain_yield_kg_ha"]), "final_biomass_kg_ha": float(last["biomass_kg_ha"]), "irrigation_mm": float(m["actual_irrigation_mm"]), "nitrogen_kg_ha": float(m["actual_nitrogen_kg_ha"]), "etcp_mm": float(m["etcp_mm"]), "WP_ET_kg_m3": float(m["WP_ET_kg_m3"]), "PFP_N_kg_kg": float(m["PFP_N_kg_kg"]) if pd.notna(m["PFP_N_kg_kg"]) else np.nan})
        endpoint_df = pd.DataFrame(endpoint_rows)
        metric_df = pd.DataFrame(metric_rows)
        daily_df = pd.concat(daily_rows, ignore_index=True)
        (out / "evaluation").mkdir(parents=True, exist_ok=True)
        (out / "daily_outputs").mkdir(parents=True, exist_ok=True)
        endpoint_df.to_csv(out / "evaluation" / "073_endpoint_audit.csv", index=False, encoding="utf-8-sig")
        metric_df.to_csv(out / "evaluation" / "073_five_scenario_metrics.csv", index=False, encoding="utf-8-sig")
        daily_df.to_csv(out / "daily_outputs" / "073_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
        result = {"task": f"{c['task_id']}_{c['task_name']}", "preflight": pf, "contract": c["contract"], "checkpoint": int(c["checkpoint"]), "years": c["validation_years"], "endpoint_audit": rel(out / "evaluation" / "073_endpoint_audit.csv"), "metrics": rel(out / "evaluation" / "073_five_scenario_metrics.csv"), "daily": rel(out / "daily_outputs" / "073_five_scenario_daily.csv"), "model_sha256": sha256(model_path), "endpoint_gate": bool((endpoint_df["difference_kg_ha"].abs() <= float(c["endpoint_tolerance_kg_ha"])).all()), "next_step_allowed": True}
        (out / "073_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = old[:4]
        batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old[4])
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old[5]
        batch.SITES = old[6]


if __name__ == "__main__":
    main()
