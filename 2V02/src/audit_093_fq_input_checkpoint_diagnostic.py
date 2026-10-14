"""093_00: read-only FQ 2018 input comparison and checkpoint-collapse audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
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
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_table
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env


STATION = "FQA"
SITE = "FQ"
PPO_TASK = "051_00"
ACTION_I = [0.0, 15.0, 30.0, 45.0]
ACTION_N = [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entropy_bits(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = pd.Series(values).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def action_signature(daily: pd.DataFrame) -> str:
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nit = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    pairs = sorted({f"I{float(i):g}/N{float(n):g}" for i, n in zip(irr, nit) if float(i) > 1e-9 or float(n) > 1e-9})
    return "; ".join(pairs)


def daily_action_stats(daily: pd.DataFrame) -> dict[str, Any]:
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nit = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    labels = [f"I{float(i):g}/N{float(n):g}" for i, n in zip(irr, nit)]
    positive = (irr > 1e-9) | (nit > 1e-9)
    return {
        "action_signature": action_signature(daily),
        "positive_action_pair_count": int(len(set(np.asarray(labels, dtype=str)[positive.to_numpy()]))),
        "all_action_entropy_bits": entropy_bits(labels),
        "positive_action_entropy_bits": entropy_bits(list(np.asarray(labels, dtype=str)[positive.to_numpy()])),
        "total_irrigation_mm": float(irr.sum()),
        "total_n_kg_ha": float(nit.sum()),
        "positive_action_rows": int(positive.sum()),
        "positive_action_rows_after_dap1": int((positive & (dap > 1)).sum()),
        "off_grid_irrigation_rows": int((~np.isclose(irr.to_numpy()[:, None], np.asarray(ACTION_I)[None, :], atol=1e-6).any(axis=1)).sum()),
        "off_grid_nitrogen_rows": int((~np.isclose(nit.to_numpy()[:, None], np.asarray(ACTION_N)[None, :], atol=1e-6).any(axis=1)).sum()),
    }


def existing_checkpoint_audit(cfg: dict[str, Any], out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_path = ROOT / cfg["existing_validation_summary"]
    summary = pd.read_csv(summary_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for rec in summary.itertuples(index=False):
        daily_path = ROOT / str(rec.daily_csv_path)
        if not daily_path.exists():
            raise FileNotFoundError(daily_path)
        daily = pd.read_csv(daily_path)
        stats = daily_action_stats(daily)
        rows.append({
            "checkpoint_step": int(rec.checkpoint_step),
            "year": int(rec.year),
            "daily_path": rel(daily_path),
            "final_grnwt_saved": float(rec.final_grnwt),
            "PFP_N_saved": float(rec.PFP_N) if pd.notna(rec.PFP_N) else np.nan,
            **stats,
        })
    per_year = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    checkpoint_rows: list[dict[str, Any]] = []
    for step, group in per_year.groupby("checkpoint_step", sort=True):
        signature_counts = group["action_signature"].value_counts()
        checkpoint_rows.append({
            "checkpoint_step": int(step),
            "validation_years": int(group["year"].nunique()),
            "cross_year_distinct_signatures": int(group["action_signature"].nunique()),
            "minority_signature_year_count": int(signature_counts.iloc[-1]) if len(signature_counts) > 1 else 0,
            "union_positive_action_pair_count": int(len(set("; ".join(group["action_signature"]).split("; ")) - {""})),
            "mean_positive_action_entropy_bits": float(group["positive_action_entropy_bits"].mean()),
            "mean_all_action_entropy_bits": float(group["all_action_entropy_bits"].mean()),
            "mean_total_irrigation_mm": float(group["total_irrigation_mm"].mean()),
            "mean_total_n_kg_ha": float(group["total_n_kg_ha"].mean()),
            "mean_positive_action_rows_after_dap1": float(group["positive_action_rows_after_dap1"].mean()),
            "all_grid_pass": bool(group["off_grid_irrigation_rows"].sum() == 0 and group["off_grid_nitrogen_rows"].sum() == 0),
        })
    by_checkpoint = pd.DataFrame(checkpoint_rows)
    out_tables = out / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    per_year.to_csv(out_tables / "093_00_existing_checkpoint_year_actions.csv", index=False, encoding="utf-8-sig")
    by_checkpoint.to_csv(out_tables / "093_00_existing_checkpoint_collapse_summary.csv", index=False, encoding="utf-8-sig")
    return by_checkpoint, per_year, {
        "all_daily_files_exist": True,
        "all_grid_pass": bool(per_year["off_grid_irrigation_rows"].sum() == 0 and per_year["off_grid_nitrogen_rows"].sum() == 0),
        "any_checkpoint_cross_year_signature_gt1": bool((by_checkpoint["cross_year_distinct_signatures"] > 1).any()),
        "any_checkpoint_meaningful_cross_year_signature": bool(
            ((by_checkpoint["cross_year_distinct_signatures"] > 1) & (by_checkpoint["minority_signature_year_count"] >= 2)).any()
        ),
    }


def prepare_engine(cfg: dict[str, Any], out: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    old_values = {
        key: getattr(engine, key)
        for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]
    }
    engine.TASK_ID = PPO_TASK
    engine.TASK_NAME = "fqa_originIC_expanded_action_maskableppo"
    engine.BASE_OUT = ROOT / cfg["ppo_root"]
    engine.BASE_DOC = ROOT / "docs" / "051_00_fqa_originIC_expanded_action_maskableppo_record.md"
    engine.PROMPT = ROOT / "prompts" / "051_fqa_originIC_site_transfer_expanded_action_ppo_auto_static_level1_figures.md"
    engine.LOWIC_INPUT_ROOT = ROOT / cfg["original_input_root"]
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.BINARY_IRRIGATION_LEVELS = list(ACTION_I)
    engine.BINARY_NITROGEN_LEVELS = list(ACTION_N)
    base03222.SITE_NAMES[STATION] = SITE
    engine.patch_base_module(ROOT / cfg["ppo_root"], engine.BASE_DOC, 100000, [25000, 50000, 75000, 100000])
    config = base03222.load_config()
    selection = base03222.build_selection(base03222.load_split())
    env_config = direct_ppo.build_env_config(config, selection)
    env_config["paths"]["output_root"] = rel(out)
    return config, env_config, old_values


def restore_engine(old_values: dict[str, Any]) -> None:
    for key, value in old_values.items():
        setattr(engine, key, value)


def replay_one(cfg: dict[str, Any], out: Path, config: dict[str, Any], env_config: dict[str, Any], checkpoint: int, input_label: str, input_root: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    model_path = ROOT / cfg["ppo_root"] / "models" / STATION / f"FQA_half_split_stress_aware_maskableppo_seed0_ckpt{checkpoint}.zip"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_names = dict(base03222.SITE_NAMES)
    env = None
    start = time.time()
    try:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
        env = base03222.base.make_env(config, env_config, STATION, 2018, int(cfg["seed"]), f"FQA_2018_093_{input_label}_ckpt{checkpoint}", evaluation=True)
        model = MaskablePPO.load(str(model_path), device="cpu")
        obs, info = env.reset()
        records: list[dict[str, Any]] = []
        done = False
        steps = 0
        while not done and steps < 260:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ai = dict(getattr(env, "last_action_info", {}))
            records.append({
                "checkpoint_step": checkpoint,
                "input_label": input_label,
                "dap": int(round(float(ai.get("dap", steps + 1)))) if np.isfinite(float(ai.get("dap", steps + 1))) else steps + 1,
                "action_index": int(np.asarray(action).reshape(-1)[0]),
                "mask_valid_action_count": int(np.asarray(masks, dtype=bool).sum()),
                "safe_action_amir": float(ai.get("safe_action_amir", 0.0)),
                "safe_action_anfer": float(ai.get("safe_action_anfer", 0.0)),
                "season_cumulative_irrigation": float(ai.get("season_cumulative_irrigation", 0.0)),
                "season_cumulative_n": float(ai.get("season_cumulative_n", 0.0)),
                "reward": float(reward),
                "done": bool(terminated or truncated),
            })
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"incomplete PPO replay checkpoint={checkpoint}, input={input_label}")
        snapshot_tmp = snapshot_from_env(env)
        snapshot = out / "snapshots" / input_label / f"ckpt{checkpoint}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            raise FileExistsError(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
        actions = pd.DataFrame(records)
        action_path = out / "actions" / f"{input_label}_ckpt{checkpoint}_daily.csv"
        action_path.parent.mkdir(parents=True, exist_ok=True)
        actions.to_csv(action_path, index=False, encoding="utf-8-sig")
        plant = parse_table(snapshot / "PlantGro.OUT")
        if plant.empty or "GWAD" not in plant.columns:
            raise RuntimeError(f"missing PlantGro.OUT/GWAD for {snapshot}")
        final_yield = float(pd.to_numeric(plant["GWAD"], errors="coerce").dropna().iloc[-1])
        stats = daily_action_stats(actions.rename(columns={"safe_action_amir": "safe_action_amir", "safe_action_anfer": "safe_action_anfer"}))
        return {
            "checkpoint_step": checkpoint,
            "input_label": input_label,
            "input_root": rel(input_root),
            "input_root_sha256_marker": sha256(input_root / "FQ" / "CNFQ1801.WTH"),
            "model_path": rel(model_path),
            "steps": steps,
            "final_grain_yield_kg_ha": final_yield,
            "action_signature": stats["action_signature"],
            "total_irrigation_mm": stats["total_irrigation_mm"],
            "total_n_kg_ha": stats["total_n_kg_ha"],
            "positive_action_pair_count": stats["positive_action_pair_count"],
            "positive_action_rows_after_dap1": stats["positive_action_rows_after_dap1"],
            "off_grid_irrigation_rows": stats["off_grid_irrigation_rows"],
            "off_grid_nitrogen_rows": stats["off_grid_nitrogen_rows"],
            "action_path": rel(action_path),
            "snapshot_path": rel(snapshot),
            "elapsed_seconds": time.time() - start,
        }
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update(old_names)


def write_doc(cfg: dict[str, Any], out: Path, result: dict[str, Any]) -> None:
    collapse = pd.read_csv(out / "tables/093_00_existing_checkpoint_collapse_summary.csv")
    lines = [
        "# 093_00 FQ 输入链路与 checkpoint 塌缩诊断记录",
        "",
        "## 范围",
        "",
        "- 只读回放已有 FQ seed=0 PPO checkpoint；无训练、无 external-N、无天气预报。",
        "- 2018 原始天气与 088 修正天气各回放 25K/50K/75K/100K。",
        "",
        "## 既有 checkpoint 塌缩汇总",
        "",
        collapse.to_markdown(index=False),
        "",
        "## 2018 输入对照回放",
        "",
        pd.DataFrame(result["replays"])[["checkpoint_step", "input_label", "final_grain_yield_kg_ha", "total_irrigation_mm", "total_n_kg_ha", "action_signature", "snapshot_path"]].to_markdown(index=False),
        "",
        "## 判定边界",
        "",
        f"- corrected 2018 是否恢复产量：`{result['corrected_2018_restores_yield']}`。",
        f"- 是否存在跨年 signature>1 的已有 checkpoint：`{result['any_checkpoint_cross_year_signature_gt1']}`。",
        f"- 后续 checkpoint replay 是否允许：`{result['next_step_allowed']}`。",
        "- WP_ET 未由本诊断直接计算，不作推断。",
        "",
    ]
    (ROOT / "docs/093_00_fq_input_checkpoint_diagnostic_record.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/093_00_fq_input_checkpoint_diagnostic.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    required = [ROOT / cfg["ppo_root"], ROOT / cfg["original_input_root"], ROOT / cfg["corrected_input_root"], ROOT / cfg["existing_validation_summary"]]
    if any(not p.exists() for p in required):
        raise FileNotFoundError([str(p) for p in required if not p.exists()])
    if args.dry_run:
        print(json.dumps({"task_id": cfg["task_id"], "checkpoints": cfg["checkpoint_steps"], "target_year": cfg["target_year"], "replay_count": len(cfg["checkpoint_steps"]) * 2, "training_allowed": False, "external_n_allowed": False, "corrected_weather_exists": (ROOT / cfg["corrected_input_root"] / "FQ/CNFQ1801.WTH").exists()}, ensure_ascii=False, indent=2))
        return
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    collapse, per_year, collapse_gate = existing_checkpoint_audit(cfg, out)
    config, env_config, old_values = prepare_engine(cfg, out)
    replays: list[dict[str, Any]] = []
    started = time.time()
    try:
        for checkpoint in cfg["checkpoint_steps"]:
            replays.append(replay_one(cfg, out, config, env_config, int(checkpoint), "original2018", ROOT / cfg["original_input_root"]))
            replays.append(replay_one(cfg, out, config, env_config, int(checkpoint), "corrected2018", ROOT / cfg["corrected_input_root"]))
    finally:
        restore_engine(old_values)
    replay_df = pd.DataFrame(replays)
    replay_df.to_csv(out / "tables/093_00_2018_input_replay_summary.csv", index=False, encoding="utf-8-sig")
    grouped = replay_df.pivot(index="checkpoint_step", columns="input_label", values="final_grain_yield_kg_ha").reset_index()
    grouped["corrected_minus_original_yield"] = grouped["corrected2018"] - grouped["original2018"]
    grouped.to_csv(out / "tables/093_00_2018_input_yield_delta.csv", index=False, encoding="utf-8-sig")
    raw = replay_df[replay_df["input_label"].eq("original2018")]
    corrected = replay_df[replay_df["input_label"].eq("corrected2018")]
    corrected_restores = bool((corrected["final_grain_yield_kg_ha"] > raw["final_grain_yield_kg_ha"].to_numpy() + 100.0).any())
    result = {
        "task_id": cfg["task_id"],
        "status": "ok",
        "training_run": False,
        "external_n_included": False,
        "weather_forecast_included": False,
        "replay_count": len(replays),
        "replays": replays,
        **collapse_gate,
        "corrected_2018_restores_yield": corrected_restores,
        "elapsed_seconds": time.time() - started,
        "output_root": rel(out),
        "next_step_allowed": bool(collapse_gate["any_checkpoint_meaningful_cross_year_signature"] and corrected_restores),
    }
    (out / "093_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    (out / "093_00_manifest.json").write_text(json.dumps({**result, "input_sha256": {"original_wth": sha256(ROOT / cfg["original_input_root"] / "FQ/CNFQ1801.WTH"), "corrected_wth": sha256(ROOT / cfg["corrected_input_root"] / "FQ/CNFQ1801.WTH"), "ppo_summary": sha256(ROOT / cfg["existing_validation_summary"])}}, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    write_doc(cfg, out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
