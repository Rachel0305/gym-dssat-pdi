"""Replay completed FQ 100K policies on the already isolated corrected 2018 WTH.

The original WTH, training configurations, and old validation outputs are never
modified. Run one seed at a time to keep DSSAT/torch memory bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hl_fq_8seed_cross_site.analyze_hl_fq_8seed import config_and_output, selected_summary
import run_088_fq2018_input_quality_correction as task088

BASE = task088.base03
CORRECTED = ROOT / "benchmark_results/088_00_fq2018_input_quality_correction/corrected_input"
MANIFEST = ROOT / "benchmark_results/088_00_fq2018_input_quality_correction/088_00_correction_manifest.json"
OUT = ROOT / "results/hl_fq_8seed_cross_site/055_03_native_auto_corrected2018/replays/FQ"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed
    if seed not in range(7):
        raise ValueError("Only the seven completed FQ seeds are eligible")
    correction = json.loads(MANIFEST.read_text(encoding="utf-8"))
    corrected_wth = CORRECTED / "FQ/CNFQ1801.WTH"
    original_wth = ROOT / correction["source_weather"]
    if sha256(corrected_wth) != correction["corrected_weather_sha256"]:
        raise ValueError("Corrected WTH hash changed")
    if sha256(original_wth) != correction["source_weather_sha256"]:
        raise ValueError("Original WTH hash changed")

    target_root = OUT / f"seed{seed}"
    metrics_file = target_root / "corrected2018_ppo_metrics.json"
    if metrics_file.exists():
        print(f"SKIP FQ seed{seed}: existing corrected 2018 replay metrics", flush=True)
        return
    cfg_path, ppo_root = config_and_output("FQ", seed)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    source = selected_summary(ppo_root, "FQA", seed)[2018]
    model = ROOT / source["model_path"]
    if not model.is_file():
        raise FileNotFoundError(model)
    target_root.mkdir(parents=True, exist_ok=True)

    old_values = {key: getattr(BASE.engine, key) for key in [
        "TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT",
        "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS",
    ]}
    old_weather_root = BASE.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_site_names = dict(BASE.base03222.SITE_NAMES)
    env = None
    try:
        BASE.ppo_safe_rendering.MULTISITE_INPUT_ROOT = CORRECTED
        BASE.engine.TASK_ID = str(cfg["task_id"])
        BASE.engine.TASK_NAME = str(cfg["task_name"])
        BASE.engine.BASE_OUT = ppo_root
        BASE.engine.BASE_DOC = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
        BASE.engine.PROMPT = BASE.PROMPT
        BASE.engine.LOWIC_INPUT_ROOT = CORRECTED
        BASE.engine.STATION = "FQA"
        BASE.engine.SITES = ["FQA"]
        BASE.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        BASE.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        BASE.base03222.SITE_NAMES["FQA"] = "FQ"
        BASE.engine.patch_base_module(ppo_root, BASE.engine.BASE_DOC, int(cfg["training"]["total_timesteps"]),
                                      list(map(int, cfg["training"]["checkpoint_steps"])))
        run_config = BASE.base03222.load_config()
        selection = BASE.base03222.build_selection(BASE.base03222.load_split())
        env_config = BASE.direct_ppo.build_env_config(run_config, selection)
        env_config["paths"]["output_root"] = target_root.relative_to(ROOT).as_posix()
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
        policy = MaskablePPO.load(str(model), device="cpu")
        env = BASE.base03222.base.make_env(run_config, env_config, "FQA", 2018, seed,
                                            f"FQA_2018_05503_corrected_seed{seed}", evaluation=True)
        obs, _info = env.reset()
        done = False
        for steps in range(260):
            action, _ = policy.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            if done:
                break
        if not done:
            raise RuntimeError(f"FQ seed{seed} corrected2018 exceeded 260 steps")
        source_snapshot = BASE.snapshot_from_env(env)
        snapshot = target_root / "snapshot"
        if snapshot.exists():
            raise FileExistsError(snapshot)
        shutil.copytree(source_snapshot, snapshot)
        daily = BASE.daily_from_snapshot(snapshot, 2018, f"ppo_seed_{seed}")
        final_yield = float(daily.grain_yield_kg_ha.iloc[-1])
        measured = BASE.baseline.metrics_from_snapshot(snapshot, final_yield)
        warnings = (snapshot / "WARNING.OUT").read_text(encoding="utf-8", errors="replace").lower()
        result = {
            "site": "FQ", "seed": seed, "year": 2018, "checkpoint_step": 100000,
            "original_weather_yield_kg_ha": float(source["final_grnwt"]),
            "corrected_yield_kg_ha": final_yield,
            "corrected_irrigation_mm": measured["actual_irrigation_mm"],
            "corrected_n_kg_ha": measured["actual_nitrogen_kg_ha"],
            "corrected_etcp_mm": measured["etcp_mm"],
            "corrected_WP_ET_kg_m3": measured["WP_ET_kg_m3"],
            "corrected_PFP_N_kg_kg": measured["PFP_N_kg_kg"] if math.isfinite(float(measured["PFP_N_kg_kg"])) else None,
            "last_dap": int(daily.dap.max()),
            "slow_grain_filling_warning": "slowed grain filling" in warnings,
            "precipitation_error": "precipitation data error" in warnings,
            "model_path": model.relative_to(ROOT).as_posix(),
            "corrected_weather_sha256": correction["corrected_weather_sha256"],
            "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
        }
        daily.to_csv(target_root / "corrected2018_ppo_daily.csv", index=False, encoding="utf-8-sig")
        metrics_file.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"DONE FQ seed{seed} corrected2018: yield={final_yield:.1f}, I={result['corrected_irrigation_mm']:.1f}, N={result['corrected_n_kg_ha']:.1f}, WP_ET={result['corrected_WP_ET_kg_m3']:.2f}", flush=True)
    finally:
        if env is not None:
            env.close()
        BASE.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_weather_root
        BASE.base03222.SITE_NAMES.clear()
        BASE.base03222.SITE_NAMES.update(old_site_names)
        for key, value in old_values.items():
            setattr(BASE.engine, key, value)


if __name__ == "__main__":
    main()
