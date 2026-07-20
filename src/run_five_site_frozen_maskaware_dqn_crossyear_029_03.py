#!/usr/bin/env python3
"""Zero-training cross-year evaluation of the selected 029 mask-aware DQN models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mask_aware_dqn_029 import MaskAwareDQN
import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as sy_base
import run_sy_icdat_aligned_all_years_frozen_stage_ppo_026_07 as sy_aligned
import run_hla2010_frozen_maskableppo_screened_year_transfer_028_08 as hla
import run_yc2014_frozen_maskableppo_yc2008_transfer_028_09 as yc
import run_fq2016_frozen_maskableppo_screened_year_transfer_028_11 as fq


OUT = ROOT / "benchmark_results" / "029_03_frozen_maskaware_dqn_crossyear"
ANCHOR = ROOT / "benchmark_results" / "029_02_five_site_stage_mask_aware_dqn"
PPO_OVERVIEW = ROOT / "benchmark_results" / "028_13_screened_year_maskableppo_advisor_summary" / "028_13_site_year_overview.csv"
TRANSFER_YEARS = {
    "SY": (2012, 2015),
    "HLA": (2007, 2015, 2016, 2022),
    "YC": (2008,),
    "FQ": (2013, 2014, 2019, 2020, 2023),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_model(site: str, seed: int) -> Path:
    payload = json.loads((ANCHOR / site / f"seed{seed}" / "result.json").read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"Anchor result is not completed: {site}/seed{seed}")
    path = ROOT / payload["selected"]["model_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class DQNAdapter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.model, self.global_step = MaskAwareDQN.load(self.path, device="cpu")

    def predict(self, observation, action_masks, deterministic=True):
        action = self.model.select_action(
            np.asarray(observation, dtype=np.float32),
            np.asarray(action_masks, dtype=bool),
            self.global_step,
            deterministic=bool(deterministic),
        )
        return action, None


class DQNLoader:
    @staticmethod
    def load(path: Path, device: str = "cpu") -> DQNAdapter:
        del device
        return DQNAdapter(Path(path))


def install_loader() -> None:
    # Existing transfer evaluators remain otherwise byte-for-byte unchanged.
    sy_base.MaskablePPO = DQNLoader
    hla.MaskablePPO = DQNLoader
    yc.MaskablePPO = DQNLoader
    fq.MaskablePPO = DQNLoader


def add_advisor_rule(frame: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        local = baselines[(baselines["site"] == row["site"]) & (baselines["year"].astype(int) == int(row["year"]))]
        if local["scenario"].nunique() != 4:
            raise ValueError(f"Four baselines missing for {row['site']}{row['year']}")
        ymax = float(local["final_gwad"].max())
        wpmax = float(local["WP_ET_kg_m3"].max())
        pmax = float(local["PFP_N_kg_kg"].dropna().max())
        pfp = float(row["PFP_N_kg_kg"]) if pd.notna(row["PFP_N_kg_kg"]) else math.nan
        out = row.to_dict()
        out.update(
            four_baseline_max_yield=ymax,
            four_baseline_max_WP_ET=wpmax,
            four_baseline_max_PFP_N=pmax,
            yield_strict_win=float(row["final_gwad"]) > ymax,
            wp_et_strict_win=float(row["WP_ET_kg_m3"]) > wpmax,
            pfp_n_strict_win=math.isfinite(pfp) and pfp > pmax,
            yield_gap_pct=100.0 * (float(row["final_gwad"]) / ymax - 1.0),
            wp_et_gap_pct=100.0 * (float(row["WP_ET_kg_m3"]) / wpmax - 1.0),
            pfp_n_gap_pct=100.0 * (pfp / pmax - 1.0) if math.isfinite(pfp) else math.nan,
        )
        out["advisor_any_metric_win"] = bool(out["yield_strict_win"] or out["wp_et_strict_win"] or out["pfp_n_strict_win"])
        rows.append(out)
    return pd.DataFrame(rows)


def sy_baselines() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "benchmark_results" / "026_07_attempt2" / "026_07_sy_all_years_four_baselines.csv")
    frame["site"] = "SY"
    return frame


def all_baselines() -> pd.DataFrame:
    frames = [
        sy_baselines(),
        hla.baseline_rows(),
        yc.baseline_rows(),
        fq.baseline_rows(),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    # CSV readers treat the literal scenario label "null" as NA by default.
    combined["scenario"] = combined["scenario"].fillna("null")
    return combined


def run_one(site: str, year: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_path = selected_model(site, seed)
    before = file_sha256(model_path)
    if site == "SY":
        sy_root = OUT / "SY"
        sy_aligned.install_runtime_patch(sy_root)
        local_base = sy_baselines()
        targets = sy_base.local_targets(local_base[local_base["year"].astype(int) == year])
        result, actions = sy_base.run_frozen_model(year, seed, model_path, targets)
        result["site"] = "SY"
    elif site == "HLA":
        hla.OUT = OUT / "HLA"
        result, actions = hla.evaluate(year, seed, model_path, pd.read_csv(hla.SCALER), hla.baseline_rows())
    elif site == "YC":
        yc.MODELS = {seed: model_path}
        result, actions = yc.evaluate(seed, OUT / "YC", yc.baseline_rows(), pd.read_csv(yc.SCALER))
    elif site == "FQ":
        fq.MODELS = {seed: model_path}
        result, actions = fq.evaluate(year, seed, OUT / "FQ", fq.baseline_rows(), pd.read_csv(fq.SCALER))
    else:
        raise ValueError(site)
    after = file_sha256(model_path)
    if before != after:
        raise RuntimeError(f"Frozen DQN file changed: {model_path}")
    result["algorithm"] = "Mask-aware DQN"
    result["training_steps_on_validation_year"] = 0
    result["model_file_sha256"] = before
    for action in actions:
        action["algorithm"] = "Mask-aware DQN"
    return result, actions


def run_case(site: str, year: int, seed: int) -> None:
    case_dir = OUT / "cases" / f"{site}{year}" / f"seed{seed}"
    if case_dir.exists():
        raise FileExistsError(case_dir)
    case_dir.mkdir(parents=True)
    result, actions = run_one(site, year, seed)
    pd.DataFrame(actions).to_csv(case_dir / "stage_actions.csv", index=False, encoding="utf-8-sig")
    (case_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"status": "completed", "site": site, "year": year, "seed": seed, "yield": result["final_gwad"], "I": result["irrigation_total"], "N": result["fertilizer_total"]}, ensure_ascii=False))


def finalize() -> None:
    case_rows: list[dict[str, Any]] = []
    action_frames: list[pd.DataFrame] = []
    for site, years in TRANSFER_YEARS.items():
        for year in years:
            for seed in (0, 1, 2):
                case_dir = OUT / "cases" / f"{site}{year}" / f"seed{seed}"
                case_rows.append(json.loads((case_dir / "result.json").read_text(encoding="utf-8")))
                action_frames.append(pd.read_csv(case_dir / "stage_actions.csv"))
    frame = add_advisor_rule(pd.DataFrame(case_rows), all_baselines())
    frame.to_csv(OUT / "029_03_dqn_frozen_crossyear_seed_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(action_frames, ignore_index=True).to_csv(OUT / "029_03_dqn_frozen_crossyear_stage_actions.csv", index=False, encoding="utf-8-sig")
    matrix = frame.groupby(["site", "year"], as_index=False).agg(
        seed_count=("seed", "count"),
        winner_count=("advisor_any_metric_win", "sum"),
        yield_win_count=("yield_strict_win", "sum"),
        wp_win_count=("wp_et_strict_win", "sum"),
        pfp_win_count=("pfp_n_strict_win", "sum"),
        invalid_actions=("invalid_action_attempts", "sum"),
    )
    matrix["at_least_two_of_three"] = matrix["winner_count"] >= 2
    matrix.to_csv(OUT / "029_03_dqn_frozen_crossyear_year_matrix.csv", index=False, encoding="utf-8-sig")
    expected = sum(len(years) for years in TRANSFER_YEARS.values()) * 3
    checks = {
        "exact_36_frozen_seasons": len(frame) == expected == 36,
        "zero_validation_training_steps": int(frame["training_steps_on_validation_year"].sum()) == 0,
        "zero_invalid_actions": int(frame["invalid_action_attempts"].sum()) == 0,
        "all_models_unchanged": True,
        "all_expected_cases_present": len(matrix) == 12,
    }
    payload = {"status": "completed" if all(checks.values()) else "failed", "checks": checks, "year_matrix": matrix.to_dict("records")}
    (OUT / "029_03_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "completed":
        raise RuntimeError(checks)


def main() -> None:
    install_loader()
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=tuple(TRANSFER_YEARS))
    parser.add_argument("--year", type=int)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        finalize()
        return
    if args.site is None or args.year is None or args.seed is None:
        parser.error("--site, --year, and --seed are required unless --finalize is used")
    if args.year not in TRANSFER_YEARS[args.site]:
        parser.error(f"{args.site}{args.year} is not a preregistered transfer case")
    run_case(args.site, args.year, args.seed)


if __name__ == "__main__":
    main()
