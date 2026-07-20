#!/usr/bin/env python3
"""Frozen YC2014 MaskablePPO transfer to YC2008."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_extension_expert_baseline_018_03 as extension
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


OUT = ROOT / "benchmark_results" / "028_09_yc2014_frozen_maskableppo_yc2008_transfer"
SMOKE = ROOT / "benchmark_results" / "028_09_smoke_yc2008_seed0"
DOC = ROOT / "docs" / "2026-07-18_028_09_yc2014_frozen_maskableppo_yc2008_transfer.md"
SCALER = ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2" / "YC" / "readiness" / "observation_scaler.csv"
MODELS = {
    0: ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2" / "YC" / "seed0" / "checkpoint_000060.zip",
    1: ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2" / "YC" / "seed1" / "checkpoint_000060.zip",
    2: ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2" / "YC" / "seed2" / "checkpoint_000120.zip",
}
SPEC = siteppo.SiteSpec(
    "YC", "Yucheng", 2008, 1, siteppo.INPUT_PARENT / "YC", "CNYC0801.MZX",
    "CNYC0801.WTH", "YC99001200", "ZD0985", 22, 6,
    siteppo.OFFICIAL_HUANGHUAI_STAGES, siteppo.OFFICIAL_HUANGHUAI_STAGES, 106,
    1, "08153", "08153", "08170", 1, "CNYC0801",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_metrics(snapshot: Path, final_yield: float, irrigation: float, nitrogen: float) -> dict[str, Any]:
    return siteppo.strict_metrics_from_snapshot(snapshot, final_yield, irrigation, nitrogen)


def plant_end(snapshot: Path) -> tuple[float, float]:
    table = extension.parse_dssat_table(snapshot / "PlantGro.OUT")
    return float(table["GWAD"].dropna().iloc[-1]), float(table["CWAD"].dropna().iloc[-1])


def baseline_rows() -> pd.DataFrame:
    old_root = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_forward_screening_013_01" / "runs" / "YC"
    paths = {
        "null": old_root / "2008_null" / "pdi_tmp_snapshot",
        "recorded": old_root / "2008_recorded" / "pdi_tmp_snapshot",
        "dssat_auto": old_root / "2008_dssat_auto" / "pdi_tmp_snapshot",
        "official_extension_expert": ROOT / "benchmark_results" / "028_07_missing_official_expert_six_season_completion" / "runs_completed" / "YC2008" / "official_extension_expert" / "pdi_tmp_snapshot_eval",
    }
    expected = {"null": (0.0, 0.0), "recorded": (120.0, 303.0), "dssat_auto": (0.0, 0.0), "official_extension_expert": (228.75, 247.5)}
    rows = []
    for scenario, snapshot in paths.items():
        y, b = plant_end(snapshot)
        i, n = expected[scenario]
        metrics = snapshot_metrics(snapshot, y, i, n)
        rows.append({"site": "YC", "year": 2008, "scenario": scenario, "final_gwad": y, "final_cwad": b, "irrigation_total": i, "fertilizer_total": n, "snapshot": str(snapshot.relative_to(ROOT)).replace("\\", "/"), **metrics})
    return pd.DataFrame(rows)


def evaluate(seed: int, root: Path, baselines: pd.DataFrame, scaler: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = root / "runs" / "2008" / f"seed{seed}"
    linked = siteppo.make_stage_env(
        SPEC, run_dir, scaler,
        {"local_null_yield": float(baselines.loc[baselines.scenario == "null", "final_gwad"].iloc[0]), "local_feasibility_yield": float(baselines.loc[baselines.scenario == "official_extension_expert", "final_gwad"].iloc[0])},
        seed=1000 + seed, phase=f"frozen_transfer_seed{seed}",
    )
    model_path = MODELS[seed]
    before = sha256(model_path)
    actions: list[dict[str, Any]] = []
    try:
        model = MaskablePPO.load(model_path, device="cpu")
        obs, info = linked.reset()
        done = False
        while not done:
            mask = get_action_masks(linked)
            stage = int(linked.stage_index)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = linked.step(int(action))
            actions.append({"site": "YC", "year": 2008, "seed": seed, "stage_index": stage, **linked.stage_rows[-1]})
            done = bool(terminated or truncated)
        if linked.last_result is None:
            raise RuntimeError("YC2008 ended without final result")
        tmp = siteppo.snapshot_from_env(linked.raw_env)
        snapshot = run_dir / "pdi_tmp_snapshot_eval"
        shutil.copytree(tmp, snapshot)
        metrics = snapshot_metrics(snapshot, linked.last_result["final_yield"], linked.last_result["irrigation_total"], linked.last_result["nitrogen_total"])
        result = {"site": "YC", "year": 2008, "seed": seed, "source_model": str(model_path.relative_to(ROOT)).replace("\\", "/"), "model_sha256": before, "action_sequence": ",".join(str(x["action_index"]) for x in actions), "final_gwad": linked.last_result["final_yield"], "final_cwad": linked.last_result["final_biomass"], "irrigation_total": linked.last_result["irrigation_total"], "fertilizer_total": linked.last_result["nitrogen_total"], "invalid_action_attempts": linked.invalid_attempts, "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"), **metrics}
    finally:
        linked.close()
    if sha256(model_path) != before or len(actions) != 6:
        raise RuntimeError("Frozen-model hash or six-stage invariant failed")
    return result, actions


def add_rule(frame: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    ymax = float(baselines.final_gwad.max()); wpmax = float(baselines.WP_ET_kg_m3.max()); pmax = float(baselines.PFP_N_kg_kg.dropna().max())
    frame["four_baseline_max_yield"] = ymax; frame["four_baseline_max_WP_ET"] = wpmax; frame["four_baseline_max_PFP_N"] = pmax
    frame["yield_strict_win"] = frame.final_gwad > ymax
    frame["wp_et_strict_win"] = frame.WP_ET_kg_m3 > wpmax
    frame["pfp_n_strict_win"] = frame.PFP_N_kg_kg.fillna(-math.inf) > pmax
    frame["advisor_any_metric_win"] = frame[["yield_strict_win", "wp_et_strict_win", "pfp_n_strict_win"]].any(axis=1)
    frame["yield_gap_pct"] = (frame.final_gwad / ymax - 1) * 100
    frame["wp_et_gap_pct"] = (frame.WP_ET_kg_m3 / wpmax - 1) * 100
    frame["pfp_n_gap_pct"] = (frame.PFP_N_kg_kg / pmax - 1) * 100
    return frame


def run_smoke() -> None:
    if SMOKE.exists(): raise FileExistsError(SMOKE)
    SMOKE.mkdir(parents=True)
    b = baseline_rows(); s = pd.read_csv(SCALER)
    result, actions = evaluate(0, SMOKE, b, s)
    pd.DataFrame(actions).to_csv(SMOKE / "028_09_smoke_actions.csv", index=False, encoding="utf-8-sig")
    (SMOKE / "028_09_smoke_result.json").write_text(json.dumps({"status": "passed", "result": result, "training_steps": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "seed": 0, "yield": result["final_gwad"], "actions": result["action_sequence"]}, ensure_ascii=False))


def reuse_smoke() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads((SMOKE / "028_09_smoke_result.json").read_text(encoding="utf-8"))
    return payload["result"], pd.read_csv(SMOKE / "028_09_smoke_actions.csv").to_dict("records")


def main() -> None:
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    baselines = baseline_rows(); scaler = pd.read_csv(SCALER)
    baselines.to_csv(OUT / "028_09_reused_four_baselines.csv", index=False, encoding="utf-8-sig")
    rows=[]; actions=[]
    for seed in (0,1,2):
        result, acts = reuse_smoke() if seed == 0 else evaluate(seed, OUT, baselines, scaler)
        rows.append(result); actions.extend(acts)
        print(f"OK YC2008 seed{seed}: Y={result['final_gwad']:.3f} I={result['irrigation_total']} N={result['fertilizer_total']}")
    frame = add_rule(pd.DataFrame(rows), baselines)
    frame.to_csv(OUT / "028_09_yc2008_frozen_transfer_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(actions).to_csv(OUT / "028_09_yc2008_frozen_transfer_stage_actions.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "completed", "winner_count": int(frame.advisor_any_metric_win.sum()), "seed_count": 3, "zero_training_steps": True, "zero_invalid_actions": int(frame.invalid_action_attempts.sum()) == 0}
    (OUT / "028_09_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text("# 028_09 YC2014 固定 MaskablePPO 向 YC2008 迁移\n\n" + f"状态：`completed`；导师规则通过 {payload['winner_count']}/3 seed；训练步数 0。\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke-only", action="store_true"); args=parser.parse_args()
    run_smoke() if args.smoke_only else main()
