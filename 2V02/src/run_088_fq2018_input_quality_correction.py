"""088_00: isolate and test the FQ2018 weather-sign correction.

This runner creates a corrected FQ input copy, never edits the source input,
then reuses the audited 051_02 native-DSSAT four-baseline evaluator for only
2018. It deliberately does not train or replay PPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import importlib.util

_BASE051_PATH = ROOT / "src" / "051_fqa_originIC_site_transfer" / "run_051_02_fqa_originIC_four_baselines_static_level1.py"
_BASE051_SPEC = importlib.util.spec_from_file_location("base051_088", _BASE051_PATH)
if _BASE051_SPEC is None or _BASE051_SPEC.loader is None:
    raise ImportError(f"Cannot load {_BASE051_PATH}")
base051 = importlib.util.module_from_spec(_BASE051_SPEC)
_BASE051_SPEC.loader.exec_module(base051)

_BASE03_PATH = ROOT / "src" / "051_fqa_originIC_site_transfer" / "run_051_03_fqa_originIC_five_scenario_figures.py"
_BASE03_SPEC = importlib.util.spec_from_file_location("base03_088", _BASE03_PATH)
if _BASE03_SPEC is None or _BASE03_SPEC.loader is None:
    raise ImportError(f"Cannot load {_BASE03_PATH}")
base03 = importlib.util.module_from_spec(_BASE03_SPEC)
_BASE03_SPEC.loader.exec_module(base03)


DEFAULT_CONFIG = ROOT / "configs" / "088_00_fq2018_input_quality_correction.json"
SOURCE_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
OUT_ROOT = ROOT / "benchmark_results" / "088_00_fq2018_input_quality_correction"
CORRECTED_ROOT = OUT_ROOT / "corrected_input"
SOURCE_FQ = SOURCE_ROOT / "FQ"
SOURCE_WTH = SOURCE_FQ / "CNFQ1801.WTH"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_wth_line(line: str) -> tuple[list[str], int, float, float] | None:
    tokens = line.split()
    if len(tokens) < 5 or not re.fullmatch(r"20\d{5}", tokens[0]):
        return None
    try:
        date = int(tokens[0])
        tmax = float(tokens[2])
        tmin = float(tokens[3])
    except ValueError:
        return None
    return tokens, date, tmax, tmin


def make_corrected_input() -> dict[str, Any]:
    if not SOURCE_WTH.exists():
        raise FileNotFoundError(SOURCE_WTH)
    if CORRECTED_ROOT.exists():
        # This is an output-owned directory; refuse if it contains unexpected files.
        for p in CORRECTED_ROOT.rglob("*"):
            if p.is_file() and not str(p).startswith(str(CORRECTED_ROOT)):
                raise RuntimeError(f"unexpected correction path: {p}")
        shutil.rmtree(CORRECTED_ROOT)
    corrected_fq = CORRECTED_ROOT / "FQ"
    corrected_fq.mkdir(parents=True, exist_ok=True)
    for src in SOURCE_FQ.iterdir():
        if src.name == "CNFQ1801.WTH":
            continue
        if src.is_file():
            shutil.copy2(src, corrected_fq / src.name)

    original_lines = SOURCE_WTH.read_text(encoding="utf-8", errors="replace").splitlines()
    corrected_lines: list[str] = []
    corrections: list[dict[str, Any]] = []
    for line_no, line in enumerate(original_lines, start=1):
        parsed = parse_wth_line(line)
        if parsed is None:
            corrected_lines.append(line)
            continue
        tokens, date, tmax, tmin = parsed
        year = date // 1000
        doy = date % 1000
        if year == 2018 and 120 <= doy <= 300 and tmin < 0 and tmax > 15:
            old = tmin
            new = abs(tmin)
            # DSSAT's weather reader is fixed-column.  Preserve every column
            # boundary and replace only the five-character TMIN field
            # (columns 21--25 in these source rows); joining tokens would
            # shift RAIN and make DSSAT report a false precipitation error.
            fixed_line = line[:20] + f"{new:5.1f}" + line[25:]
            corrections.append(
                {
                    "line_no": line_no,
                    "date": date,
                    "doy": doy,
                    "tmax_c": tmax,
                    "old_tmin_c": old,
                    "new_tmin_c": new,
                    "reason": "summer TMIN sign inconsistency: TMAX>15 with negative TMIN",
                }
            )
            corrected_lines.append(fixed_line)
        else:
            corrected_lines.append(line)

    corrected_wth = corrected_fq / "CNFQ1801.WTH"
    corrected_wth.write_text("\n".join(corrected_lines) + "\n", encoding="utf-8")
    manifest = {
        "source_root": rel(SOURCE_ROOT),
        "source_weather": rel(SOURCE_WTH),
        "source_weather_sha256": sha256(SOURCE_WTH),
        "corrected_root": rel(CORRECTED_ROOT),
        "corrected_weather": rel(corrected_wth),
        "corrected_weather_sha256": sha256(corrected_wth),
        "rule": "2018 DOY 120-300, TMIN<0 and TMAX>15: replace TMIN with abs(TMIN)",
        "n_corrected": len(corrections),
        "corrections": corrections,
        "source_unchanged_sha256_after": sha256(SOURCE_WTH),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "088_00_correction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    import pandas as pd

    pd.DataFrame(corrections).to_csv(OUT_ROOT / "088_00_weather_corrections.csv", index=False, encoding="utf-8-sig")
    return manifest


def run_dry(config_path: Path) -> dict[str, Any]:
    manifest = make_corrected_input()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["task_id"] = "088_00"
    cfg["task_name"] = "fq2018_input_quality_correction"
    cfg["scope"] = {"validation_years": [2018]}
    # 051_02 reads this module-level profile map during its preflight/evaluation.
    base051.INPUT_PROFILES["originIC"] = CORRECTED_ROOT
    pf = base051.preflight(cfg)
    return {"mode": "dry_run", "preflight": pf, "correction": manifest}


def run_corrected(config_path: Path) -> dict[str, Any]:
    manifest = make_corrected_input()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["task_id"] = "088_00"
    cfg["task_name"] = "fq2018_input_quality_correction"
    cfg["scope"] = {"validation_years": [2018]}
    corrected_cfg = OUT_ROOT / "088_00_effective_config.json"
    corrected_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    base051.INPUT_PROFILES["originIC"] = CORRECTED_ROOT
    # One year, native DSSAT four-baseline run only. No PPO training/replay.
    result = base051.run(corrected_cfg, dry_run=False, years_override=[2018], run_id="corrected2018")
    result["correction_manifest"] = rel(OUT_ROOT / "088_00_correction_manifest.json")
    return result


def replay_ppo_corrected(config_path: Path, checkpoints: list[int]) -> dict[str, Any]:
    """Replay existing FQA PPO checkpoints against corrected FQ2018 weather.

    The original validation endpoint is intentionally not used as a replay
    oracle: it was produced with the corrupted weather and is expected to be
    zero.  The corrected native-DSSAT baseline run is the provenance anchor.
    """
    import numpy as np
    import pandas as pd
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    manifest = make_corrected_input()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ppo_root = ROOT / "benchmark_results" / "051_00_fqa_originIC_expanded_action_maskableppo"
    replay_root = OUT_ROOT / "ppo_replay_corrected2018"
    if replay_root.exists():
        suffix = 1
        while (OUT_ROOT / f"ppo_replay_corrected2018_rerun{suffix}").exists():
            suffix += 1
        replay_root = OUT_ROOT / f"ppo_replay_corrected2018_rerun{suffix}"
    replay_root.mkdir(parents=True, exist_ok=True)

    base051.INPUT_PROFILES["originIC"] = CORRECTED_ROOT
    base03.INPUT_PROFILES["originIC"] = CORRECTED_ROOT
    original_root = base03.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_values = {
        key: getattr(base03.engine, key)
        for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]
    }
    old_site_names = dict(base03.base03222.SITE_NAMES)
    rows: list[dict[str, Any]] = []
    try:
        base03.ppo_safe_rendering.MULTISITE_INPUT_ROOT = CORRECTED_ROOT
        base03.engine.TASK_ID = "051_00"
        base03.engine.TASK_NAME = "fqa_originIC_expanded_action_maskableppo"
        base03.engine.BASE_OUT = ppo_root
        base03.engine.BASE_DOC = ROOT / "docs" / "051_00_fqa_originIC_expanded_action_maskableppo_record.md"
        base03.engine.PROMPT = base03.PROMPT
        base03.engine.LOWIC_INPUT_ROOT = CORRECTED_ROOT
        base03.engine.STATION = base03.STATION
        base03.engine.SITES = [base03.STATION]
        base03.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg.get("actions", {}).get("irrigation_levels_mm", [0.0, 15.0, 30.0, 45.0])))
        base03.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg.get("actions", {}).get("nitrogen_levels_kg_ha", [0.0, 40.0, 80.0, 120.0])))
        base03.base03222.SITE_NAMES[base03.STATION] = base03.SITE
        base03.engine.patch_base_module(ppo_root, base03.engine.BASE_DOC, 100000, checkpoints)
        run_config = base03.base03222.load_config()
        selection = base03.base03222.build_selection(base03.base03222.load_split())
        env_config = base03.direct_ppo.build_env_config(run_config, selection)
        env_config["paths"]["output_root"] = rel(replay_root)

        inventory = pd.read_csv(ppo_root / "evaluation" / "051_00_training_checkpoint_inventory.csv", keep_default_na=False)
        for checkpoint in checkpoints:
            match = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(int(checkpoint))]
            if match.empty:
                raise RuntimeError(f"missing model inventory checkpoint {checkpoint}")
            model_path = ROOT / str(match.iloc[0]["model_path"])
            model = MaskablePPO.load(str(model_path), device="cpu")
            env = None
            try:
                env = base03.base03222.base.make_env(
                    run_config, env_config, base03.STATION, 2018, int(cfg.get("seed", 0)), f"FQA_2018_088_ppo_ckpt{checkpoint}", evaluation=True
                )
                obs, _info = env.reset()
                done, steps = False, 0
                while not done and steps < 260:
                    action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
                    obs, _reward, terminated, truncated, _info = env.step(action)
                    done = bool(terminated or truncated)
                    steps += 1
                if not done:
                    raise RuntimeError(f"checkpoint {checkpoint} did not finish in 260 steps")
                snapshot_tmp = base03.snapshot_from_env(env)
                target = replay_root / "snapshots" / "FQA" / "2018" / f"ckpt{checkpoint}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(snapshot_tmp, target)
                daily = base03.daily_from_snapshot(target, 2018, "rl_candidate")
                final_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
                metrics = base051.baseline.metrics_from_snapshot(target, final_yield)
                warnings = (target / "WARNING.OUT").read_text(encoding="utf-8", errors="replace")
                rows.append({
                    "station_code": "FQA", "site": "FQ", "year": 2018,
                    "scenario": "rl_candidate", "checkpoint_step": int(checkpoint),
                    "grain_yield_kg_ha": final_yield, "biomass_kg_ha": float(pd.to_numeric(daily["biomass_kg_ha"], errors="coerce").iloc[-1]),
                    "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                    "actual_nitrogen_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
                    "etcp_mm": float(metrics["etcp_mm"]), "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N_kg_kg": float(metrics["PFP_N_kg_kg"]) if np.isfinite(float(metrics["PFP_N_kg_kg"])) else np.nan,
                    "season_last_dap": int(pd.to_numeric(daily["dap"], errors="coerce").max()),
                    "slow_grain_filling_warning": "slowed grain filling" in warnings.lower(),
                    "precipitation_error": "precipitation data error" in warnings.lower(),
                    "snapshot_path": rel(target),
                })
            finally:
                if env is not None:
                    env.close()
    finally:
        base03.ppo_safe_rendering.MULTISITE_INPUT_ROOT = original_root
        base03.base03222.SITE_NAMES.clear()
        base03.base03222.SITE_NAMES.update(old_site_names)
        for key, value in old_values.items():
            setattr(base03.engine, key, value)

    ppo_df = pd.DataFrame(rows)
    corrected_baseline_root = ROOT / "benchmark_results" / "088_00_fq2018_input_quality_correction_run_corrected2018"
    baseline_summary = pd.read_csv(corrected_baseline_root / "evaluation" / "051_02_baseline_summary.csv")
    baseline_summary = baseline_summary.rename(columns={"scenario": "scenario"})
    baseline_summary["checkpoint_step"] = pd.NA
    baseline_summary["slow_grain_filling_warning"] = False
    baseline_summary["precipitation_error"] = False
    combined = pd.concat([baseline_summary, ppo_df], ignore_index=True, sort=False)
    ppo_df.to_csv(replay_root / "088_00_ppo_corrected2018_summary.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(replay_root / "088_00_corrected2018_four_baselines_plus_ppo.csv", index=False, encoding="utf-8-sig")
    result = {
        "task": "088_00_fq2018_input_quality_correction_ppo_replay",
        "corrected_input": rel(CORRECTED_ROOT), "checkpoints": checkpoints,
        "n_rows": len(ppo_df), "all_finished": bool(len(ppo_df) == len(checkpoints)),
        "ppo_summary_csv": rel(replay_root / "088_00_ppo_corrected2018_summary.csv"),
        "combined_csv": rel(replay_root / "088_00_corrected2018_four_baselines_plus_ppo.csv"),
        "correction_manifest": rel(OUT_ROOT / "088_00_correction_manifest.json"),
        "rows": rows,
        "provenance_note": "Original 2018 PPO validation endpoint was generated with corrupted weather and is not used as a replay oracle; corrected DSSAT snapshots are reported separately.",
    }
    (replay_root / "088_00_ppo_replay_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay-ppo", action="store_true", help="Replay existing PPO checkpoints on corrected FQ2018; no training")
    ap.add_argument("--checkpoints", type=str, default="25000,50000,75000,100000")
    args = ap.parse_args()
    cfg = args.config if args.config.is_absolute() else (ROOT / args.config).resolve()
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    if args.dry_run:
        result = run_dry(cfg)
    elif args.replay_ppo:
        result = replay_ppo_corrected(cfg, checkpoints)
    else:
        result = run_corrected(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
