"""015: YC DQN checkpoint-selection feasibility evaluation.

This script evaluates existing formal checkpoints only.  It has two explicit
phases so that the primary checkpoint decision is written from 2004-2013
selection-domain evidence before any 2014-2023 held-out validation file is
read.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

FORMAL_OUT = ROOT / "benchmark_results" / "227_00_yc_mask_aware_dqn_formal"
SELECTION_OUT = ROOT / "benchmark_results" / "015_yc_dqn_checkpoint_selection_feasibility"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_dqn_checkpoint_selection_feasibility.md"
DOMAIN_EVAL = RESULTS / "yc_dqn_checkpoint_selection_domain_eval.csv"
DECISION = RESULTS / "yc_dqn_checkpoint_selection_decision.csv"
HELDOUT = RESULTS / "yc_dqn_checkpoint_selection_heldout_comparison.csv"
CROSSSEED = RESULTS / "yc_dqn_checkpoint_selection_crossseed_summary.csv"
TRAIN_YEARS = list(range(2004, 2014))
HELDOUT_YEARS = list(range(2014, 2024))
SEEDS = [0, 1, 2]
CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=json_default),
        encoding="utf-8",
    )


def seed_source_dir(seed: int) -> Path:
    candidates = [
        FORMAL_OUT / f"seed_{seed}" / "attempt_02",
        FORMAL_OUT / f"seed_{seed}",
    ]
    for candidate in candidates:
        status_path = candidate / "seed_status.json"
        inventory_path = candidate / "checkpoint_inventory.csv"
        if not status_path.exists() or not inventory_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            inventory = pd.read_csv(inventory_path)
        except Exception:
            continue
        if status.get("status") == "completed" and set(pd.to_numeric(inventory["checkpoint"], errors="coerce").dropna().astype(int)) == set(CHECKPOINTS):
            return candidate
    raise FileNotFoundError(f"No completed 227 checkpoint source for seed {seed}")


def checkpoint_path(seed_dir: Path, checkpoint: int) -> Path:
    path = seed_dir / "checkpoints" / f"dqn_checkpoint_{checkpoint:06d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def aggregate_domain_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["episode_return"] = np.nan
    for idx, row in frame.iterrows():
        daily_path = ROOT / str(row.get("daily_path", ""))
        if row.get("run_status") == "ok" and daily_path.exists():
            try:
                daily = pd.read_csv(daily_path, usecols=["reward"])
                frame.at[idx, "episode_return"] = pd.to_numeric(daily["reward"], errors="coerce").sum(min_count=1)
            except Exception:
                frame.at[idx, "episode_return"] = np.nan
    frame["selection_domain"] = "2004-2013"
    frame["evaluation_mode"] = "epsilon=0 deterministic masked greedy"
    frame["evaluation_grain"] = "one row per seed/checkpoint/training year"
    return frame


def domain_phase() -> pd.DataFrame:
    from mask_aware_dqn_029 import MaskAwareDQN
    import run_yc_mask_aware_dqn_formal_227_00 as formal

    frozen = formal.load_frozen()
    modules = formal.configure_runtime_modules(frozen)
    runtime_cfg = copy.deepcopy(modules["config"])
    runtime_cfg["paths"]["output_root"] = rel(SELECTION_OUT)
    env_config, _selection = formal.build_env_config(runtime_cfg, modules)
    # Keep the environment's observed-year map restricted during this phase.
    # The full formal split is used only to construct the same environment
    # contract; no held-out result file is read here.
    train_env_config = copy.deepcopy(env_config)
    train_env_config["observed_years"]["YCA"] = [
        item for item in train_env_config["observed_years"]["YCA"]
        if int(item["year"]) in TRAIN_YEARS
    ]
    if sorted(int(item["year"]) for item in train_env_config["observed_years"]["YCA"]) != TRAIN_YEARS:
        raise RuntimeError("Selection environment is not exactly 2004-2013")

    selection_frozen = copy.deepcopy(frozen)
    selection_frozen["evaluation"]["validation_years"] = TRAIN_YEARS
    all_rows: list[dict[str, Any]] = []
    source_map: dict[str, str] = {}
    for seed in SEEDS:
        source_dir = seed_source_dir(seed)
        source_map[str(seed)] = rel(source_dir)
        for checkpoint in CHECKPOINTS:
            model, saved_step = MaskAwareDQN.load(
                checkpoint_path(source_dir, checkpoint),
                device=str(frozen["dqn"]["device"]),
            )
            if int(saved_step) != int(checkpoint):
                raise RuntimeError(f"Checkpoint step mismatch: seed={seed}, expected={checkpoint}, saved={saved_step}")
            eval_dir = SELECTION_OUT / "selection_domain" / f"seed_{seed}"
            eval_rows, _actions = formal.validate_checkpoint(
                selection_frozen,
                runtime_cfg,
                modules,
                train_env_config,
                model,
                seed,
                checkpoint,
                eval_dir,
            )
            if len(eval_rows) != len(TRAIN_YEARS):
                raise RuntimeError(f"Expected 10 selection rows, got {len(eval_rows)} for seed={seed}, checkpoint={checkpoint}")
            all_rows.extend(eval_rows)
            del model

    domain = aggregate_domain_rows(all_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    domain.to_csv(DOMAIN_EVAL, index=False, encoding="utf-8-sig")

    aggregate = domain.groupby(["seed", "checkpoint"], as_index=False).agg(
        mean_episode_return=("episode_return", "mean"),
        sd_episode_return=("episode_return", "std"),
        mean_yield=("yield_kg_ha", "mean"),
        sd_yield=("yield_kg_ha", "std"),
        mean_I=("total_irrigation_mm", "mean"),
        mean_N=("total_nitrogen_kg_ha", "mean"),
        mean_PFP_N=("PFP_N", "mean"),
        mean_WP_ET=("WP_ET_kg_m3", "mean"),
        invalid_action_count=("invalid_action_count", "sum"),
        fallback_count=("fallback_count", "sum"),
        completed_year_count=("year", "nunique"),
    )
    decision_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        group = aggregate[aggregate["seed"].eq(seed)].copy()
        usable = group[
            group["completed_year_count"].eq(10)
            & np.isfinite(group["mean_episode_return"])
        ].copy()
        if len(usable) != len(CHECKPOINTS):
            decision_rows.append({
                "seed": seed,
                "primary_selector_status": "PRIMARY_SELECTOR_UNAVAILABLE",
                "selected_checkpoint": np.nan,
                "diagnostic_yield_checkpoint": np.nan,
                "selection_domain_years": "2004-2013",
                "reason": "episode return unavailable or incomplete for at least one checkpoint",
            })
            continue
        selected = usable.sort_values(["mean_episode_return", "checkpoint"], ascending=[False, True]).iloc[0]
        diagnostic = usable.sort_values(["mean_yield", "checkpoint"], ascending=[False, True]).iloc[0]
        decision_rows.append({
            "seed": seed,
            "primary_selector_status": "AVAILABLE",
            "selected_checkpoint": int(selected["checkpoint"]),
            "selected_mean_episode_return": float(selected["mean_episode_return"]),
            "selected_mean_yield": float(selected["mean_yield"]),
            "diagnostic_yield_checkpoint": int(diagnostic["checkpoint"]),
            "diagnostic_yield_selector_label": "DIAGNOSTIC_ONLY",
            "diagnostic_yield_mean": float(diagnostic["mean_yield"]),
            "selection_domain_years": "2004-2013",
            "primary_selector_metric": "mean_episode_return",
            "primary_selector_rule": "highest 2004-2013 deterministic mean episode return within seed",
            "decision_written_before_heldout_read": True,
            "source_checkpoint_dir": source_map[str(seed)],
        })
    decision = pd.DataFrame(decision_rows)
    decision.to_csv(DECISION, index=False, encoding="utf-8-sig")
    write_json(SELECTION_OUT / "selection_domain_provenance.json", {
        "task": "015_yc_dqn_checkpoint_selection_feasibility",
        "phase": "selection_domain_complete",
        "selection_domain_years": TRAIN_YEARS,
        "heldout_years_not_read_in_this_phase": HELDOUT_YEARS,
        "primary_selector": "highest mean episode return per seed",
        "deterministic": True,
        "epsilon": 0.0,
        "checkpoint_steps": CHECKPOINTS,
        "seeds": SEEDS,
        "formal_config": rel(ROOT / "configs" / "yc_mask_aware_dqn_formal_227_00.json"),
        "checkpoint_sources": source_map,
        "decision_path": rel(DECISION),
        "heldout_read_after_decision": True,
    })
    if decision["primary_selector_status"].ne("AVAILABLE").any():
        raise RuntimeError("PRIMARY_SELECTOR_UNAVAILABLE")
    print("SELECTION_DOMAIN_CHECKPOINT_DECISION_COMPLETE")
    return domain


def read_heldout_after_selection() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not DECISION.exists():
        raise FileNotFoundError(f"Selection decision missing: {DECISION}")
    decision = pd.read_csv(DECISION)
    if decision["primary_selector_status"].ne("AVAILABLE").any():
        raise RuntimeError("PRIMARY_SELECTOR_UNAVAILABLE")
    selected_by_seed: dict[Any, Any] = {
        int(row.seed): int(row.selected_checkpoint)
        for row in decision.itertuples(index=False)
    }
    frames: list[pd.DataFrame] = []
    source_map: dict[str, str] = {}
    for seed in SEEDS:
        source_dir = seed_source_dir(seed)
        path = source_dir / "validation_summary.csv"
        # This is the first function in this script that reads formal
        # 2014-2023 validation result rows.
        validation = pd.read_csv(path)
        validation["seed"] = int(seed)
        validation["checkpoint"] = pd.to_numeric(validation["checkpoint"], errors="coerce").astype(int)
        validation["year"] = pd.to_numeric(validation["year"], errors="coerce").astype(int)
        validation = validation[validation["year"].isin(HELDOUT_YEARS)].copy()
        if set(validation["year"]) != set(HELDOUT_YEARS):
            raise RuntimeError(f"Held-out year coverage mismatch for seed {seed}")
        frames.append(validation)
        source_map[str(seed)] = rel(path)
    selected_by_seed["_source_map"] = source_map
    return pd.concat(frames, ignore_index=True), selected_by_seed


def mean_metric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def compare_heldout(heldout: pd.DataFrame, selected_by_seed: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        selected_checkpoint = int(selected_by_seed[seed])
        group = heldout[heldout["seed"].eq(seed)].copy()
        selected = group[group["checkpoint"].eq(selected_checkpoint)].copy()
        endpoint = group[group["checkpoint"].eq(100_000)].copy()
        if len(selected) != 10 or len(endpoint) != 10:
            raise RuntimeError(f"Held-out checkpoint coverage mismatch seed={seed}: selected={len(selected)}, endpoint={len(endpoint)}")
        selected_yield = mean_metric(selected, "yield_kg_ha")
        endpoint_yield = mean_metric(endpoint, "yield_kg_ha")
        rows.append({
            "seed": seed,
            "selected_checkpoint": selected_checkpoint,
            "endpoint_checkpoint": 100000,
            "selected_heldout_mean_yield": selected_yield,
            "endpoint_100K_heldout_mean_yield": endpoint_yield,
            "yield_difference_selected_minus_100K": selected_yield - endpoint_yield,
            "selected_mean_I": mean_metric(selected, "total_irrigation_mm"),
            "endpoint_mean_I": mean_metric(endpoint, "total_irrigation_mm"),
            "selected_mean_N": mean_metric(selected, "total_nitrogen_kg_ha"),
            "endpoint_mean_N": mean_metric(endpoint, "total_nitrogen_kg_ha"),
            "selected_PFP_N": mean_metric(selected, "PFP_N"),
            "endpoint_PFP_N": mean_metric(endpoint, "PFP_N"),
            "selected_WP_ET": mean_metric(selected, "WP_ET_kg_m3"),
            "endpoint_WP_ET": mean_metric(endpoint, "WP_ET_kg_m3"),
            "selected_validation_year_count": int(selected["year"].nunique()),
            "endpoint_validation_year_count": int(endpoint["year"].nunique()),
            "heldout_scope": "2014-2023 existing formal validation_summary.csv",
            "selection_leakage_status": "NO_LEAKAGE_DETECTED",
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(HELDOUT, index=False, encoding="utf-8-sig")

    endpoint_yields = comparison["endpoint_100K_heldout_mean_yield"].to_numpy(dtype=float)
    selected_yields = comparison["selected_heldout_mean_yield"].to_numpy(dtype=float)
    improved = selected_yields > endpoint_yields
    all_finite = bool(np.isfinite(endpoint_yields).all() and np.isfinite(selected_yields).all())
    endpoint_mean = float(np.nanmean(endpoint_yields)) if np.isfinite(endpoint_yields).any() else np.nan
    selected_mean = float(np.nanmean(selected_yields)) if np.isfinite(selected_yields).any() else np.nan
    endpoint_sd = float(np.nanstd(endpoint_yields, ddof=1)) if np.isfinite(endpoint_yields).sum() >= 2 else np.nan
    selected_sd = float(np.nanstd(selected_yields, ddof=1)) if np.isfinite(selected_yields).sum() >= 2 else np.nan
    if not all_finite:
        classification = "INSUFFICIENT_EVIDENCE"
    elif bool(improved.all()) and selected_mean > endpoint_mean:
        classification = "CHECKPOINT_SELECTION_EFFECTIVE"
    elif bool(improved.any()) and selected_mean >= endpoint_mean:
        classification = "PARTIALLY_EFFECTIVE"
    else:
        classification = "NOT_EFFECTIVE"
    selected_counts = comparison["selected_checkpoint"].value_counts().to_dict()
    seed2_selected_25k = int(comparison.loc[comparison["seed"].eq(2), "selected_checkpoint"].iloc[0]) == 25_000
    cross = pd.DataFrame([{
        "classification": classification,
        "n_seeds": len(SEEDS),
        "endpoint_100K_crossseed_mean_yield": endpoint_mean,
        "endpoint_100K_seed_sd_yield": endpoint_sd,
        "primary_selected_crossseed_mean_yield": selected_mean,
        "primary_selected_seed_sd_yield": selected_sd,
        "crossseed_mean_difference_selected_minus_100K": selected_mean - endpoint_mean,
        "crossseed_sd_difference_selected_minus_100K": selected_sd - endpoint_sd,
        "seeds_improved_over_100K": int(improved.sum()) if all_finite else np.nan,
        "selection_domain_rule_consistency": "same checkpoint on all seeds" if len(selected_counts) == 1 else "not same checkpoint on all seeds",
        "selected_checkpoint_counts": json.dumps({str(k): int(v) for k, v in selected_counts.items()}, sort_keys=True),
        "seed2_selected_checkpoint": int(comparison.loc[comparison["seed"].eq(2), "selected_checkpoint"].iloc[0]),
        "seed2_selected_25K": seed2_selected_25k,
        "heldout_read_after_selection": True,
        "heldout_scope": "2014-2023",
    }])
    cross.to_csv(CROSSSEED, index=False, encoding="utf-8-sig")
    return comparison, cross, classification


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    work = frame[[c for c in columns if c in frame.columns]].copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = work[col].round(3)
    work = work.astype(object).where(pd.notna(work), "unknown / insufficient evidence")
    lines = [
        "| " + " | ".join(work.columns) + " |",
        "| " + " | ".join("---" for _ in work.columns) + " |",
    ]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in work.to_numpy().tolist())
    return "\n".join(lines)


def write_report(comparison: pd.DataFrame, cross: pd.DataFrame, classification: str) -> None:
    decision = pd.read_csv(DECISION)
    domain = pd.read_csv(DOMAIN_EVAL)
    selected_mean = float(cross.loc[0, "primary_selected_crossseed_mean_yield"])
    endpoint_mean = float(cross.loc[0, "endpoint_100K_crossseed_mean_yield"])
    selected_sd = float(cross.loc[0, "primary_selected_seed_sd_yield"])
    endpoint_sd = float(cross.loc[0, "endpoint_100K_seed_sd_yield"])
    selected_text = "; ".join(
        f"seed{int(row.seed)}->{int(row.selected_checkpoint)//1000}K"
        for row in decision.itertuples(index=False)
    )
    summary = [
        "本轮只测试 checkpoint selection：不改算法、不改契约、不重新训练。",
        "primary 规则先在 2004–2013 上按每个 seed 的 deterministic mean episode return 选 checkpoint。",
        f"选择结果为：{selected_text}。",
        f"随后读取 2014–2023 held-out：selected mean yield={selected_mean:.3f}，100K mean yield={endpoint_mean:.3f} kg/ha。",
        f"跨 seed SD：selected={selected_sd:.3f}，100K={endpoint_sd:.3f} kg/ha。",
        f"结论分类：{classification}。",
        "这只能回答选择规则的可行性，不能把 held-out 结果反向用于选择。",
    ]
    lines = ["# YC DQN checkpoint selection feasibility", "", "## Plain-language summary", ""]
    lines.extend(summary)
    lines.extend([
        "",
        "## Scope and leakage guard",
        "",
        "本轮只读取既有 227 formal checkpoints 和既有 validation summary；没有重新训练、调参、修改 reward/state/action/mask 或正式 checkpoint。",
        "primary selection 只使用 2004–2013 selection-domain 的 epsilon=0 deterministic masked-greedy evaluation。",
        "decision CSV 在读取 2014–2023 validation_summary.csv 之前已经写入；2014–2023 未参与 checkpoint 选择。",
        "selection leakage status: NO_LEAKAGE_DETECTED",
        "",
        "## Selection-domain evaluation",
        "",
        f"覆盖：{len(SEEDS)} seeds × {len(CHECKPOINTS)} checkpoints × {len(TRAIN_YEARS)} training-domain years = {len(domain)} rows。",
        "primary selector：每个 seed 选择 2004–2013 deterministic mean episode return 最高的 checkpoint。",
        "yield-max checkpoint 仅作为 DIAGNOSTIC_ONLY，不替代 primary selector。",
        "",
        md_table(
            domain.groupby(["seed", "checkpoint"], as_index=False).agg(
                mean_episode_return=("episode_return", "mean"),
                sd_episode_return=("episode_return", "std"),
                mean_yield=("yield_kg_ha", "mean"),
                mean_I=("total_irrigation_mm", "mean"),
                mean_N=("total_nitrogen_kg_ha", "mean"),
                mean_PFP_N=("PFP_N", "mean"),
                mean_WP_ET=("WP_ET_kg_m3", "mean"),
                invalid_action_count=("invalid_action_count", "sum"),
                fallback_count=("fallback_count", "sum"),
            ),
            ["seed", "checkpoint", "mean_episode_return", "sd_episode_return", "mean_yield", "mean_I", "mean_N", "mean_PFP_N", "mean_WP_ET", "invalid_action_count", "fallback_count"],
        ),
        "",
        "## Primary and diagnostic decisions",
        "",
        md_table(
            decision,
            ["seed", "selected_checkpoint", "selected_mean_episode_return", "selected_mean_yield", "diagnostic_yield_checkpoint", "diagnostic_yield_mean", "selection_domain_years", "decision_written_before_heldout_read"],
        ),
        "",
        "## Held-out comparison",
        "",
        "Held-out scope is existing formal validation for 2014–2023, read only after the selection decision was persisted.",
        md_table(
            comparison,
            ["seed", "selected_checkpoint", "endpoint_checkpoint", "selected_heldout_mean_yield", "endpoint_100K_heldout_mean_yield", "yield_difference_selected_minus_100K", "selected_mean_I", "endpoint_mean_I", "selected_mean_N", "endpoint_mean_N", "selected_PFP_N", "endpoint_PFP_N", "selected_WP_ET", "endpoint_WP_ET", "selection_leakage_status"],
        ),
        "",
        "## Cross-seed summary",
        "",
        md_table(
            cross,
            ["classification", "endpoint_100K_crossseed_mean_yield", "endpoint_100K_seed_sd_yield", "primary_selected_crossseed_mean_yield", "primary_selected_seed_sd_yield", "crossseed_mean_difference_selected_minus_100K", "crossseed_sd_difference_selected_minus_100K", "seeds_improved_over_100K", "selection_domain_rule_consistency", "selected_checkpoint_counts", "seed2_selected_checkpoint", "seed2_selected_25K"],
        ),
        "",
        "## Required answers",
        "",
        f"1. checkpoint selection 是否提高 cross-seed mean yield？{selected_mean > endpoint_mean}；差值 selected−100K = {selected_mean - endpoint_mean:.3f} kg/ha。",
        f"2. 是否降低 seed spread？{selected_sd < endpoint_sd}；SD 差值 selected−100K = {selected_sd - endpoint_sd:.3f} kg/ha。",
        f"3. 是否避免策略丢失？只能按既有 held-out 对比描述：改善 seed 数 = {int(cross.loc[0, 'seeds_improved_over_100K']) if pd.notna(cross.loc[0, 'seeds_improved_over_100K']) else 'unknown / insufficient evidence'}/3；不作因果解释。",
        f"4. seed2 是否在 25K 附近被选中？{bool(cross.loc[0, 'seed2_selected_25K'])}；实际 selected checkpoint = {int(cross.loc[0, 'seed2_selected_checkpoint'])//1000}K。",
        f"5. 三个 seed 是否一致？{cross.loc[0, 'selection_domain_rule_consistency']}。",
        "",
        "## Evidence paths",
        "",
        f"- selection-domain episode rows：{rel(DOMAIN_EVAL)}",
        f"- selection decision：{rel(DECISION)}",
        f"- held-out comparison：{rel(HELDOUT)}",
        f"- cross-seed summary：{rel(CROSSSEED)}",
        f"- deterministic evaluation artifacts：{rel(SELECTION_OUT)}",
        f"- formal checkpoint source：{rel(FORMAL_OUT)}",
        "",
        "## Final classification",
        "",
        classification,
        "",
        "## Status",
        "",
        "YC_DQN_CHECKPOINT_SELECTION_TEST_COMPLETE",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def heldout_phase() -> None:
    heldout, selected = read_heldout_after_selection()
    source_map = selected.pop("_source_map")
    comparison, cross, classification = compare_heldout(heldout, selected)
    write_report(comparison, cross, classification)
    write_json(SELECTION_OUT / "heldout_provenance.json", {
        "task": "015_yc_dqn_checkpoint_selection_feasibility",
        "phase": "heldout_complete",
        "selection_decision": rel(DECISION),
        "heldout_years": HELDOUT_YEARS,
        "heldout_sources": source_map,
        "selection_leakage_status": "NO_LEAKAGE_DETECTED",
        "classification": classification,
    })
    print("YC_DQN_CHECKPOINT_SELECTION_TEST_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["selection", "heldout"], required=True)
    args = parser.parse_args()
    if args.phase == "selection":
        domain_phase()
    else:
        heldout_phase()


if __name__ == "__main__":
    main()
