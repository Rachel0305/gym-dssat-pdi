"""072: replay a SY checkpoint with the exact 070 direct-evaluator contract.

The old 046_06 replay path imported the 042_10/040_36 lowIC wrapper.  This
isolated runner uses the same 032_22 wrapper/configuration as 070 evaluation
and only adds a DSSAT snapshot after the episode, without training.
"""
from __future__ import annotations

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
import run_sya_ppo_configured_046_02 as base04602
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env
from build_sya_configured_five_scenario_daily_plots_046_06 import daily_from_snapshot

PLAN = ROOT / "configs" / "072_sy_replay_contract_closure.json"


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


def preflight(c: dict[str, Any]) -> dict[str, Any]:
    summary = ROOT / c["expected_summary"]
    model = ROOT / c["ppo_run_dir"] / "models" / "SYA" / "SYA_half_split_stress_aware_maskableppo_seed1_ckpt25000.zip"
    input_root = base04602.INPUT_PROFILES[c["input_profile"]]
    issues = []
    if output_root(c).exists() and any(output_root(c).iterdir()):
        issues.append("isolated output root already exists")
    if not summary.exists():
        issues.append("expected saved summary missing")
    if not model.exists():
        issues.append("checkpoint model missing")
    if not input_root.exists():
        issues.append("originIC input root missing")
    return {"task": c["task_id"] + "_" + c["task_name"], "summary": rel(summary), "model": rel(model), "input_root": rel(input_root), "issues": issues, "next_step_allowed": not issues}


def runtime_setup(c: dict[str, Any], out: Path):
    ep = ROOT / "configs" / "070_00_sya_effective.yaml"
    old = {
        "batch_config": batch.CONFIG,
        "batch_out": batch.OUT,
        "batch_doc": batch.DOC,
        "batch_seed": batch.SEED,
        "batch_sites": dict(batch.SITE_NAMES),
        "safe_root": ppo_safe_rendering.MULTISITE_INPUT_ROOT,
    }
    batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = ep, out, ROOT / c["prompt"], int(c["seed"])
    batch.SITES = ["SYA"]
    batch.SITE_NAMES["SYA"] = "SY"
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04602.INPUT_PROFILES[c["input_profile"]]
    return old


def restore(old) -> None:
    batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = old["batch_config"], old["batch_out"], old["batch_doc"], old["batch_seed"]
    batch.SITE_NAMES.clear()
    batch.SITE_NAMES.update(old["batch_sites"])
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["safe_root"]


def main() -> None:
    c = cfg()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["dry-run", "audit"], required=True)
    args = parser.parse_args()
    pf = preflight(c)
    if args.phase == "dry-run":
        print(json.dumps(pf, ensure_ascii=False, indent=2))
        return
    if not pf["next_step_allowed"]:
        raise RuntimeError("; ".join(pf["issues"]))
    out = output_root(c)
    out.mkdir(parents=True, exist_ok=True)
    old = runtime_setup(c, out)
    try:
        config = batch.load_config()
        selection = batch.build_selection(batch.load_split())
        envcfg = batch.direct_ppo.build_env_config(config, selection)
        envcfg["paths"]["output_root"] = rel(out)
        model_path = ROOT / c["ppo_run_dir"] / "models" / "SYA" / "SYA_half_split_stress_aware_maskableppo_seed1_ckpt25000.zip"
        saved_df = pd.read_csv(ROOT / c["expected_summary"], keep_default_na=False)
        saved = saved_df[(pd.to_numeric(saved_df.checkpoint_step, errors="coerce").eq(c["checkpoint"])) & (pd.to_numeric(saved_df.year, errors="coerce").eq(c["year"]))].iloc[0]
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        model = MaskablePPO.load(str(model_path), device="cpu")
        env = batch.base.make_env(config, envcfg, "SYA", int(c["year"]), int(c["seed"]), "SYA_2014_072_exact_replay", evaluation=True)
        actions = []
        done = False
        steps = 0
        try:
            obs, info = env.reset()
            while not done and steps < 260:
                mask = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ai = dict(getattr(env, "last_action_info", {}))
                if float(ai.get("safe_action_amir", 0.0)) > 1e-9 or float(ai.get("safe_action_anfer", 0.0)) > 1e-9:
                    actions.append({"step": steps, "dap": int(round(float(getattr(env, "last_obs_dict", {}).get("dap", steps + 1)))), "irrigation": ai.get("safe_action_amir"), "nitrogen": ai.get("safe_action_anfer")})
                steps += 1
            if not done:
                raise RuntimeError("exact replay did not finish within 260 steps")
            snapshot_tmp = snapshot_from_env(env)
            snapshot = out / "snapshot" / "SYA" / str(c["year"]) / "rl_candidate"
            if snapshot.exists():
                shutil.rmtree(snapshot)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(snapshot_tmp, snapshot)
        finally:
            env.close()
        daily = daily_from_snapshot(snapshot, int(c["year"]), "rl_candidate")
        replay_yield = float(pd.to_numeric(daily.grain_yield_kg_ha, errors="coerce").iloc[-1])
        saved_yield = float(pd.to_numeric(saved.final_grnwt, errors="coerce"))
        result = {
            "task": c["task_id"] + "_" + c["task_name"],
            "checkpoint": c["checkpoint"],
            "year": c["year"],
            "saved_training_endpoint_kg_ha": saved_yield,
            "exact_replay_endpoint_kg_ha": replay_yield,
            "difference_kg_ha": replay_yield - saved_yield,
            "within_0p5_kg_ha": bool(np.isclose(replay_yield, saved_yield, rtol=0.0, atol=0.5)),
            "episode_steps": steps,
            "action_sequence": actions,
            "input_root": rel(base04602.INPUT_PROFILES[c["input_profile"]]),
            "model_sha256": sha256(model_path),
            "snapshot": rel(snapshot),
            "next_step_allowed": bool(np.isclose(replay_yield, saved_yield, rtol=0.0, atol=0.5)),
            "contract_note": "Uses batch.load_config and batch.base.StressAwareDiscreteWrapper exactly as 070 direct evaluator; no 042_10/040_36 wrapper.",
        }
        (out / "072_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        restore(old)


if __name__ == "__main__":
    main()
