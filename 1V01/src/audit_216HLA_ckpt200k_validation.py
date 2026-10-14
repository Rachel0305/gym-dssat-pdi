"""Audit the saved 216HLA checkpoint-200000 without resuming training.

The 216HLA formal run stopped before 500K because of the project memory
guardrail.  This script performs a separate deterministic replay of the
saved 200K model on the ten original 2014--2023 HLA validation years.  It
writes to a fresh audit directory and never modifies the failed training run.
"""

from __future__ import annotations

import gc
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
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events
from run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04 import ProjectProfitDiscreteWrapper
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env


def load_054_helpers():
    source = ROOT / "src" / "054_hla_lowIC_site_transfer" / "run_054_03_hla_lowIC_five_scenario_figures.py"
    spec = importlib.util.spec_from_file_location("hla_054_metric_helpers_for_216", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIG054 = load_054_helpers()

TASK_ID = "216HLA"
TASK_NAME = "hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k"
SOURCE_ROOT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_ckpt200k_validation_replay"
CONFIG_PATH = SOURCE_ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
MODEL_PATH = SOURCE_ROOT / "models" / "HLA" / "HLA_half_split_stress_aware_maskableppo_seed0_ckpt200000.zip"
PROMPT = ROOT / "prompts" / "2026-09-03_hla_simple_profit_weather_augmentation_500k.md"
RECORD = OUT / "216HLA_ckpt200k_validation_replay_record.md"
STATION = "HLA"
SITE = "HLA"
YEARS = list(range(2014, 2024))
CHECKPOINT = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def setup_modules(cfg: dict[str, Any]) -> dict[str, Any]:
    old_engine = {
        key: getattr(engine, key)
        for key in [
            "TASK_ID",
            "TASK_NAME",
            "BASE_OUT",
            "BASE_DOC",
            "PROMPT",
            "LOWIC_INPUT_ROOT",
            "STATION",
            "SITES",
            "BINARY_IRRIGATION_LEVELS",
            "BINARY_NITROGEN_LEVELS",
        ]
    }
    old_base = {
        "SITES": list(base03222.SITES),
        "SITE_NAMES": dict(base03222.SITE_NAMES),
        "SEED": base03222.SEED,
        "OUT": base03222.OUT,
        "DOC": base03222.DOC,
        "PROMPT": base03222.PROMPT,
        "TOTAL_TIMESTEPS": base03222.TOTAL_TIMESTEPS,
        "CHECKPOINT_STEPS": list(base03222.CHECKPOINT_STEPS),
        "load_split": base03222.load_split,
        "build_selection": base03222.build_selection,
        "wrapper": base03222.base.StressAwareDiscreteWrapper,
    }
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT

    lowic_root = ROOT / str(cfg["source_input_root"])
    engine.TASK_ID = TASK_ID
    engine.TASK_NAME = TASK_NAME
    engine.BASE_OUT = OUT
    engine.BASE_DOC = RECORD
    engine.PROMPT = PROMPT
    engine.LOWIC_INPUT_ROOT = lowic_root
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
    engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))

    base03222.OUT = OUT
    base03222.DOC = RECORD
    base03222.PROMPT = PROMPT
    base03222.SITES = [STATION]
    base03222.SITE_NAMES.clear()
    base03222.SITE_NAMES.update({STATION: SITE})
    base03222.SEED = int(cfg.get("seed", 0))
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = lowic_root

    engine.patch_base_module(OUT, RECORD, int(cfg["training"]["total_timesteps"]), [CHECKPOINT])
    base03222.base.StressAwareDiscreteWrapper = ProjectProfitDiscreteWrapper
    return {"engine": old_engine, "base": old_base, "root": old_root}


def restore_modules(old: dict[str, Any]) -> None:
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["root"]
    base03222.base.StressAwareDiscreteWrapper = old["base"]["wrapper"]
    base03222.SITE_NAMES.clear()
    base03222.SITE_NAMES.update(old["base"]["SITE_NAMES"])
    for key, value in old["base"].items():
        if key not in {"SITE_NAMES", "wrapper"}:
            setattr(base03222, key, value)
    for key, value in old["engine"].items():
        setattr(engine, key, value)


def action_signature(daily: pd.DataFrame) -> str:
    rows = daily[
        daily["irrigation_executed_mm"].gt(0)
        | daily["nitrogen_executed_kg_ha"].gt(0)
    ].sort_values(["doy", "dap"])
    parts = []
    for _, row in rows.iterrows():
        parts.append(
            f"DAP{int(row['dap'])} I{float(row['irrigation_executed_mm']):g}/N{float(row['nitrogen_executed_kg_ha']):g}"
        )
    return "; ".join(parts)


def replay_year(cfg: dict[str, Any], model: Any, year: int) -> dict[str, Any]:
    old = setup_modules(cfg)
    env = None
    try:
        config = engine.load_config()
        selection = base03222.build_selection(base03222.load_split())
        env_config = direct_ppo.build_env_config(config, selection)
        env_config["paths"]["output_root"] = rel(OUT)
        env = base03222.base.make_env(
            config,
            env_config,
            STATION,
            int(year),
            int(cfg.get("seed", 0)),
            f"HLA_{year}_216HLA_ckpt200k_replay",
            evaluation=True,
        )
        from sb3_contrib.common.maskable.utils import get_action_masks

        obs, _info = env.reset()
        done = False
        steps = 0
        while not done and steps < 260:
            action, _ = model.predict(
                obs,
                action_masks=get_action_masks(env),
                deterministic=True,
            )
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"HLA{year} replay did not finish within 260 steps")

        tmp = snapshot_from_env(env)
        target = OUT / "snapshots" / STATION / str(year) / "rl_candidate"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(tmp, target)
        daily = FIG054.daily_from_snapshot(target, year, "rl_candidate")
        final_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        metrics = FIG054.baseline.metrics_from_snapshot(target, final_yield)
        total_i = float(daily["irrigation_executed_mm"].sum())
        total_n = float(daily["nitrogen_executed_kg_ha"].sum())
        signature = action_signature(daily)
        event_rows = daily[
            daily["irrigation_executed_mm"].gt(0)
            | daily["nitrogen_executed_kg_ha"].gt(0)
        ]
        return {
            "station": STATION,
            "year": int(year),
            "checkpoint_step": CHECKPOINT,
            "replay_steps": int(steps),
            "grain_yield_kg_ha": final_yield,
            "actual_irrigation_mm_daily_sum": total_i,
            "actual_nitrogen_kg_ha_daily_sum": total_n,
            "summary_irrigation_total": float(metrics["actual_irrigation_mm"]),
            "summary_nitrogen_total": float(metrics["actual_nitrogen_kg_ha"]),
            "etcp_mm": float(metrics["etcp_mm"]),
            "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
            "PFP_N_kg_kg": None if pd.isna(metrics["PFP_N_kg_kg"]) else float(metrics["PFP_N_kg_kg"]),
            "simple_profit": final_yield - 1.1 * total_i - 1.58 * total_n,
            "positive_action_rows": int(len(event_rows)),
            "irrigation_event_count": int(event_rows["irrigation_executed_mm"].gt(0).sum()),
            "nitrogen_event_count": int(event_rows["nitrogen_executed_kg_ha"].gt(0).sum()),
            "action_signature": signature,
            "snapshot": rel(target),
        }
    finally:
        if env is not None:
            env.close()
        restore_modules(old)
        gc.collect()


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit directory: {OUT}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "logs").mkdir()
    shutil.copy2(CONFIG_PATH, OUT / "configs" / CONFIG_PATH.name)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)

    import sb3_contrib

    old = setup_modules(cfg)
    try:
        model = sb3_contrib.MaskablePPO.load(str(MODEL_PATH), device="cpu")
    finally:
        restore_modules(old)

    rows: list[dict[str, Any]] = []
    for year in YEARS:
        row = replay_year(cfg, model, year)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "logs" / "216HLA_ckpt200k_validation_rows.csv", index=False, encoding="utf-8-sig")

    frame = pd.DataFrame(rows)
    numeric = [
        "grain_yield_kg_ha",
        "actual_irrigation_mm_daily_sum",
        "actual_nitrogen_kg_ha_daily_sum",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "simple_profit",
        "positive_action_rows",
        "irrigation_event_count",
        "nitrogen_event_count",
    ]
    means = {key: float(pd.to_numeric(frame[key], errors="coerce").mean()) for key in numeric}
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}_ckpt200k_validation_replay",
        "status": "ok",
        "source_run_status": "failed_after_memory_stop_at_224352_steps",
        "checkpoint_step": CHECKPOINT,
        "model_path": rel(MODEL_PATH),
        "validation_weather": "original_lowIC_2014_2023",
        "rows": len(rows),
        "means": means,
        "unique_annual_action_signatures": int(frame["action_signature"].nunique()),
        "all_years_total_irrigation_240": bool(np.allclose(frame["actual_irrigation_mm_daily_sum"], 240.0)),
        "all_years_total_n_240": bool(np.allclose(frame["actual_nitrogen_kg_ha_daily_sum"], 240.0)),
        "all_years_metrics_present": bool(frame[["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].notna().all().all()),
        "code_sha256": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
        "output_root": rel(OUT),
    }
    (OUT / "216HLA_ckpt200k_validation_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record = [
        "# 216HLA checkpoint 200K validation replay",
        "",
        "- This is a deterministic replay audit only; it does not resume or extend training.",
        "- The source 500K run stopped at 224352 steps after the memory guardrail reached approximately 2048 MB.",
        f"- Model: `{rel(MODEL_PATH)}`.",
        "- Validation: original lowIC HLA weather, 2014--2023, 10 episodes, deterministic MaskablePPO actions.",
        "- Reward contract: terminal yield minus water and nitrogen costs; stress-relief terms are zero.",
        "- The CSV and JSON outputs are exploratory evidence for the saved 200K checkpoint, not a replacement for a completed formal run.",
        "",
        "## Mean metrics",
        "",
        json.dumps(means, indent=2, ensure_ascii=False),
        "",
        f"- Unique annual action signatures: {result['unique_annual_action_signatures']}.",
        f"- All years total irrigation = 240 mm: {result['all_years_total_irrigation_240']}.",
        f"- All years total nitrogen = 240 kg/ha: {result['all_years_total_n_240']}.",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
