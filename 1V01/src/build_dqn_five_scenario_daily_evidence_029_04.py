#!/usr/bin/env python3
"""Build DQN evidence in the exact table and eight-panel style used by 028_12 PPO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src import build_relaxed_success_five_scenario_daily_evidence_027_05 as old
import build_screened_year_representative_five_scenario_advisor_package_028_12 as ppo_package

OUT = ROOT / "benchmark_results/029_04_maskableppo_vs_maskaware_dqn_evidence/dqn_daily_package"
REP = ROOT / "benchmark_results/029_04_maskableppo_vs_maskaware_dqn_evidence/029_04_dqn_visualization_representatives.csv"
REPLAY = ROOT / "benchmark_results/029_04_dqn_daily_replay"
TRANSFER = ROOT / "benchmark_results/029_03_frozen_maskaware_dqn_crossyear"
ANCHOR = ROOT / "benchmark_results/029_02_five_site_stage_mask_aware_dqn"
STATION = {"HLA": "Hailun", "YC": "Yucheng", "FQ": "Fengqiu", "LC": "Luancheng", "SY": "Shenyang"}
REPLAY_CASES = {("SY", 2012), ("SY", 2014), ("SY", 2015), ("HLA", 2010), ("YC", 2014), ("FQ", 2016), ("LC", 2010)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_dir(site: str, year: int) -> Path:
    candidates = []
    for result in sorted((REPLAY / site).glob(f"{year}*/result.json")):
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("status") == "completed" and (result.parent / "snapshot/Summary.OUT").is_file():
            candidates.append(result.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"{site}{year}: expected exactly one completed replay, found {candidates}")
    return candidates[0]


def dqn_snapshot(site: str, year: int, seed: int) -> Path:
    if (site, year) in REPLAY_CASES:
        return replay_dir(site, year) / "snapshot"
    if site == "HLA":
        return TRANSFER / "HLA/runs" / str(year) / f"seed{seed}" / "pdi_tmp_snapshot_eval"
    if site == "YC":
        return TRANSFER / "YC/runs" / str(year) / f"seed{seed}" / "pdi_tmp_snapshot_eval"
    if site == "FQ":
        return TRANSFER / "FQ/runs" / str(year) / f"seed{seed}" / "pdi_tmp_snapshot_eval"
    raise ValueError((site, year))


def dqn_actions(site: str, year: int, seed: int, checkpoint: int) -> pd.DataFrame:
    if (site, year) in REPLAY_CASES:
        return pd.read_csv(replay_dir(site, year) / "stage_actions.csv")
    return pd.read_csv(TRANSFER / "cases" / f"{site}{year}" / f"seed{seed}" / "stage_actions.csv")


def model_path(site: str, seed: int, checkpoint: int, raw_source) -> Path:
    if pd.notna(raw_source) and str(raw_source).lower() != "nan":
        return ROOT / str(raw_source)
    payload = json.loads((ANCHOR / site / f"seed{seed}" / "result.json").read_text(encoding="utf-8"))
    path = ROOT / payload["selected"]["model_path"]
    if int(payload["selected"]["checkpoint"]) != checkpoint:
        raise RuntimeError("Anchor checkpoint provenance mismatch")
    return path


def plot_stage_actions(actions: pd.DataFrame, case: old.Case, fig_dir: Path) -> list[Path]:
    frame = actions.copy()
    renames = {}
    for source, target in (("executed_irrigation", "executed_irrigation_mm"), ("executed_nitrogen", "executed_nitrogen_kg_ha"), ("executed_executed_irrigation", "executed_irrigation_mm"), ("executed_executed_nitrogen", "executed_nitrogen_kg_ha")):
        if source in frame and target not in frame:
            renames[source] = target
    frame = frame.rename(columns=renames).sort_values("stage_index")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    for ax, column, title, ylabel, color in [
        (axes[0], "executed_irrigation_mm", "Frozen DQN irrigation decisions", "mm/event", "#3977A8"),
        (axes[1], "executed_nitrogen_kg_ha", "Frozen DQN nitrogen decisions", "kg/ha/event", "#18864B"),
    ]:
        values = pd.to_numeric(frame[column], errors="coerce")
        ax.vlines(frame.dap, 0, values, color=color, lw=2.4); ax.scatter(frame.dap, values, color=color, s=48, marker="D", zorder=3)
        for _, row in frame.iterrows():
            ax.annotate(f"a{int(row.action_index)}", (row.dap, row[column]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold"); ax.set_ylabel(ylabel); ax.grid(color="#E8E8E8", linewidth=0.7)
    axes[1].set_xlabel("DAP"); axes[1].set_xticks(frame.dap)
    fig.suptitle(f"{case.site}{case.year} Mask-aware DQN selected stage actions (seed {case.seed}, checkpoint {case.checkpoint})", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.01, "Frozen deterministic evidence replay; no training or checkpoint reselection.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    base = fig_dir / f"029_04_{case.site.lower()}{case.year}_dqn_selected_stage_actions"
    paths = [base.with_suffix(".png"), base.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight"); fig.savefig(paths[1], bbox_inches="tight"); plt.close(fig)
    return paths


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    reps = pd.read_csv(REP)
    registry = []; daily_all = []; summary_all = []; checks_all = []
    old.LABELS["rl_candidate"] = "Mask-aware DQN candidate"
    for row in reps.sort_values(["site", "year"]).to_dict("records"):
        site, year, seed, checkpoint = str(row["site"]), int(row["year"]), int(row["seed"]), int(row["checkpoint"])
        case_dir = OUT / site / str(year); fig_dir = case_dir / "figures"; fig_dir.mkdir(parents=True)
        snapshots = ppo_package.baseline_snapshots(site, year); snapshots["rl_candidate"] = dqn_snapshot(site, year, seed)
        missing = [str(path) for path in snapshots.values() if not (path / "Summary.OUT").is_file()]
        if missing:
            raise FileNotFoundError(f"{site}{year}: {missing}")
        source_model = model_path(site, seed, checkpoint, row.get("source_model"))
        case = old.Case(site, STATION[site], year, seed, checkpoint, source_model, snapshots, case_dir / "selection.json", "DQN representative selected only for visualization; main inference uses all three seeds.")
        daily, summary, checks = old.build_case(case, algorithm="Mask-aware DQN")
        actions = dqn_actions(site, year, seed, checkpoint)
        daily.to_csv(case_dir / "five_scenario_daily.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(case_dir / "five_scenario_summary.csv", index=False, encoding="utf-8-sig")
        actions.to_csv(case_dir / "stage_actions.csv", index=False, encoding="utf-8-sig")
        check_frame = pd.DataFrame(checks); check_frame.to_csv(case_dir / "daily_evidence_checks.csv", index=False, encoding="utf-8-sig")
        if not bool(check_frame.passed.all()):
            raise RuntimeError(f"{site}{year}: daily evidence checks failed")
        old.FIG = fig_dir
        figures = old.plot_endpoints(summary, case) + old.plot_daily(daily, case, algorithm="Mask-aware DQN") + plot_stage_actions(actions, case, fig_dir)
        manifest = {"site": site, "year": year, "seed": seed, "checkpoint": checkpoint, "model_path": str(source_model.relative_to(ROOT)).replace("\\", "/"), "model_sha256": sha256(source_model), "snapshot_path": str(snapshots["rl_candidate"].relative_to(ROOT)).replace("\\", "/"), "daily_sha256": sha256(case_dir / "five_scenario_daily.csv"), "summary_sha256": sha256(case_dir / "five_scenario_summary.csv"), "figures": [str(path.relative_to(ROOT)).replace("\\", "/") for path in figures], "training_calls_for_package": 0, "representative_only": True}
        (case_dir / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        registry.append(manifest); daily_all.append(daily); summary_all.append(summary); checks_all.append(check_frame)
        print(f"OK DQN package {site}{year} seed{seed}")
    pd.DataFrame(registry).to_csv(OUT / "029_04_dqn_representative_registry.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_all, ignore_index=True).to_csv(OUT / "029_04_all_dqn_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    pd.concat(summary_all, ignore_index=True).to_csv(OUT / "029_04_all_dqn_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(checks_all, ignore_index=True).to_csv(OUT / "029_04_all_dqn_daily_checks.csv", index=False, encoding="utf-8-sig")
    result = {"status": "completed", "cases": len(registry), "training_calls": 0, "figures": sum(len(x["figures"]) for x in registry), "all_checks_passed": True}
    (OUT / "029_04_daily_package_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

