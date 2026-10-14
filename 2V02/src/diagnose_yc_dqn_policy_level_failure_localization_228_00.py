"""Read-only YC DQN policy-level failure localization.

This diagnostic consumes only retained formal DQN/PPO CSVs and DQN checkpoint
files.  It deliberately does not construct a DSSAT environment or replay an
episode.  The formal daily logs do not retain the 22-dimensional observation
vector, so the Q-value tables keep the Q-dependent fields explicitly unknown
instead of fabricating states or re-running DSSAT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

FORMAL_ROOT = ROOT / "benchmark_results" / "227_00_yc_mask_aware_dqn_formal"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_dqn_policy_level_failure_localization.md"
SEEDS = [0, 1, 2]
CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]
YEARS = list(range(2014, 2024))
I_LEVELS = [0.0, 15.0, 30.0, 45.0]
N_LEVELS = [0.0, 40.0, 80.0, 120.0]
STRESS_THRESHOLD = 0.05
UNKNOWN = "unknown / insufficient evidence"
OBSERVATION_GAP = (
    "Formal validation daily logs retain observation_dim and selected-state scalars, "
    "but not the 22-dimensional observation vector; no Q-value is computed."
)


def seed_dir(seed: int) -> Path:
    if seed == 0:
        return FORMAL_ROOT / "seed_0" / "attempt_02"
    return FORMAL_ROOT / f"seed_{seed}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def num(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def action_label(index: int) -> str:
    index = int(index)
    return f"I{I_LEVELS[index // 4]:g}/N{N_LEVELS[index % 4]:g}"


def action_grid_text() -> str:
    return "; ".join(f"{i}={action_label(i)}" for i in range(16))


def parse_indices(value: Any) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(item) for item in str(value).split(",") if str(item).strip()]


def state_category(row: pd.Series) -> str:
    water = num(row.get("swfac"), 0.0) > STRESS_THRESHOLD
    nitrogen = num(row.get("nstres"), 0.0) > STRESS_THRESHOLD
    if water and nitrogen:
        return "dual-stress"
    if water:
        return "water-stress-only"
    if nitrogen:
        return "N-stress-only"
    return "no-stress"


def formal_daily_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        for checkpoint in CHECKPOINTS:
            root = seed_dir(seed) / "validation_daily" / f"checkpoint_{checkpoint:06d}"
            for year in YEARS:
                path = root / f"YCA_{year}_daily.csv"
                frame = read_csv(path)
                if frame.empty:
                    continue
                frame = frame.copy()
                frame["seed"] = seed
                frame["checkpoint"] = checkpoint
                frame["year"] = year
                frame["source_daily_path"] = path.relative_to(ROOT).as_posix()
                frame["source_row"] = np.arange(len(frame), dtype=int)
                frame["state_category"] = frame.apply(state_category, axis=1)
                frame["water_stress"] = pd.to_numeric(frame["swfac"], errors="coerce").gt(STRESS_THRESHOLD)
                frame["nitrogen_stress"] = pd.to_numeric(frame["nstres"], errors="coerce").gt(STRESS_THRESHOLD)
                frames.append(frame)
    if not frames:
        raise RuntimeError("No formal DQN validation daily logs were found")
    return pd.concat(frames, ignore_index=True)


def load_all_dqn_checkpoints() -> tuple[pd.DataFrame, dict[tuple[int, int], dict[str, Any]]]:
    """Load all 12 checkpoints without constructing an environment."""
    from mask_aware_dqn_029 import MaskAwareDQN

    rows: list[dict[str, Any]] = []
    models: dict[tuple[int, int], dict[str, Any]] = {}
    for seed in SEEDS:
        for checkpoint in CHECKPOINTS:
            path = seed_dir(seed) / "checkpoints" / f"dqn_checkpoint_{checkpoint:06d}.pt"
            row: dict[str, Any] = {
                "seed": seed,
                "checkpoint": checkpoint,
                "model_path": path.relative_to(ROOT).as_posix(),
                "model_load_status": UNKNOWN,
                "model_observation_dim": np.nan,
                "model_action_dim": np.nan,
                "network_architecture": UNKNOWN,
                "global_step": np.nan,
                "q_evaluation_status": UNKNOWN,
                "q_evaluation_note": OBSERVATION_GAP,
            }
            if not path.exists():
                row["model_load_status"] = "FAIL: checkpoint missing"
                rows.append(row)
                continue
            try:
                model, global_step = MaskAwareDQN.load(path, device="cpu")
                architecture = list(model.network_architecture)
                row.update(
                    {
                        "model_load_status": "PASS",
                        "model_observation_dim": int(model.observation_dim),
                        "model_action_dim": int(model.action_dim),
                        "network_architecture": json.dumps(architecture),
                        "global_step": int(global_step),
                    }
                )
                models[(seed, checkpoint)] = {
                    "model_load_status": "PASS",
                    "model_observation_dim": int(model.observation_dim),
                    "model_action_dim": int(model.action_dim),
                    "network_architecture": json.dumps(architecture),
                }
                del model
            except Exception as exc:
                row["model_load_status"] = f"FAIL: {type(exc).__name__}: {exc}"
            rows.append(row)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(RESULTS / "yc_dqn_checkpoint_model_load_inventory.csv", index=False, encoding="utf-8-sig")
    return inventory, models


def pfpn_definition_check() -> pd.DataFrame:
    rows = [
        {
            "metric": "N_100K",
            "exact_formula": "mean over validation years of sum(daily nitrogen_kg_ha action column)",
            "aggregation_order": "yearly agent-action sum first; mean across 2014-2023 second",
            "includes_agent_N": True,
            "includes_basal_or_fixed_N": False,
            "source_code": "src/run_yc_mask_aware_dqn_formal_227_00.py:637-639, 694-696",
            "notes": "The daily field is the executed DQN nitrogen action. The selected rendered treatment has FAMN=0; initial SNH4/SNO3 are soil initial conditions, not agent fertilizer N.",
        },
        {
            "metric": "PFP_N_100K",
            "exact_formula": "mean across validation rows of PFP_N; per positive-N year PFP_N is Summary.OUT YPNAM, with fallback final_yield / agent_action_N only if Summary.OUT parsing fails",
            "aggregation_order": "Summary.OUT YPNAM per year first; mean over non-null positive-N years second",
            "includes_agent_N": True,
            "includes_basal_or_fixed_N": False,
            "source_code": "src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py:497-510; src/run_yc_mask_aware_dqn_formal_227_00.py:640-657, 694-696",
            "notes": "YPNAM is matched to Summary.OUT NICM, which is checked against the daily agent-action N total. Zero-agent-N years have PFP_N missing, so mean(PFP_N) is not mean(yield)/mean(N).",
        },
        {
            "metric": "PFP_N denominator scope",
            "exact_formula": "Summary.OUT NICM / daily action nitrogen sum; fixed initial soil mineral N is not in this denominator",
            "aggregation_order": "per-year Summary.OUT/accounting match",
            "includes_agent_N": True,
            "includes_basal_or_fixed_N": False,
            "source_code": "benchmark_results/227_00_yc_mask_aware_dqn_formal/rendered_inputs/YCA/2017/seed0_eval_2017_ckpt100000/YCA_2017_seed0_eval_2017_ckpt100000.jinja2:28-35, 51-53",
            "notes": "The retained input shows initial SNH4/SNO3 soil pools and FAMN=0; no basal fertilizer is added by this formal adapter.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_pfpn_definition_check.csv", index=False, encoding="utf-8-sig")
    return out


def checkpoint_trajectory() -> pd.DataFrame:
    source = read_csv(RESULTS / "yc_dqn_formal_checkpoint_summary.csv")
    required = {"seed", "checkpoint", "mean_yield", "sd_yield", "mean_irrigation", "mean_nitrogen", "mean_PFP_N", "mean_WP_ET"}
    missing = required.difference(source.columns)
    if missing:
        raise RuntimeError(f"Formal checkpoint summary missing {sorted(missing)}")
    source["seed"] = pd.to_numeric(source["seed"], errors="coerce").astype(int)
    source["checkpoint"] = pd.to_numeric(source["checkpoint"], errors="coerce").astype(int)
    source = source.sort_values(["seed", "checkpoint"]).copy()
    source = source.rename(
        columns={
            "mean_irrigation": "mean_I",
            "mean_nitrogen": "mean_N",
            "mean_PFP_N": "mean_PFP_N",
            "mean_WP_ET": "mean_WP_ET",
        }
    )
    rows: list[dict[str, Any]] = []
    for seed, group in source.groupby("seed", sort=True):
        group = group.sort_values("checkpoint").reset_index(drop=True)
        previous_yield = np.nan
        previous_checkpoints: list[int] = []
        previous_yields: list[float] = []
        for item in group.to_dict("records"):
            current = num(item["mean_yield"])
            if previous_yields:
                best_index = int(np.argmax(previous_yields))
                best_checkpoint: Any = previous_checkpoints[best_index]
                best_yield = previous_yields[best_index]
                drop = current - best_yield
            else:
                best_checkpoint = UNKNOWN
                drop = np.nan
            rows.append(
                {
                    "seed": int(seed),
                    "checkpoint": int(item["checkpoint"]),
                    "mean_yield": current,
                    "sd_yield": num(item["sd_yield"]),
                    "mean_I": num(item["mean_I"]),
                    "mean_N": num(item["mean_N"]),
                    "mean_PFP_N": num(item["mean_PFP_N"]),
                    "mean_WP_ET": num(item["mean_WP_ET"]),
                    "yield_change_from_previous_checkpoint": current - previous_yield if np.isfinite(previous_yield) else np.nan,
                    "best_earlier_checkpoint": best_checkpoint,
                    "drop_from_earlier_best": drop,
                    "trajectory_note": "NO_EARLIER_CHECKPOINT" if not previous_yields else ("POST_PEAK_DECLINE" if drop < 0 else "NOT_BELOW_EARLIER_BEST"),
                }
            )
            previous_yield = current
            previous_checkpoints.append(int(item["checkpoint"]))
            previous_yields.append(current)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_checkpoint_trajectory_diagnostic.csv", index=False, encoding="utf-8-sig")
    return out


def mask_columns(valid_indices: list[int]) -> dict[str, Any]:
    valid = set(valid_indices)
    return {f"mask_{index}": bool(index in valid) for index in range(16)}


def q_columns_unknown() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix in ("Q", "masked_Q"):
        for index in range(16):
            result[f"{prefix}_action{index}"] = np.nan
    result.update(
        {
            "top1_action": UNKNOWN,
            "top1_Q": np.nan,
            "top2_action": UNKNOWN,
            "top2_Q": np.nan,
            "Q_margin": np.nan,
            "selected_action": UNKNOWN,
            "selected_action_index": np.nan,
            "best_Q_I_positive": np.nan,
            "best_Q_I_zero": np.nan,
            "delta_Q_water": np.nan,
            "best_Q_N_positive": np.nan,
            "best_Q_N_zero": np.nan,
            "delta_Q_N": np.nan,
            "best_Q_both": np.nan,
            "Q_noop": np.nan,
        }
    )
    return result


def fixed_state_qvalues(inventory: pd.DataFrame) -> pd.DataFrame:
    categories = ["DAP1 anchor", "no-stress", "water-stress-only", "N-stress-only", "dual-stress"]
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for checkpoint in CHECKPOINTS:
            model_meta = inventory[(inventory.seed == seed) & (inventory.checkpoint == checkpoint)].iloc[0].to_dict()
            for category in categories:
                row: dict[str, Any] = {
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "state_id": f"fixed_panel_{category.replace(' ', '_').replace('-', '_')}",
                    "state_category": category,
                    "state_panel_status": UNKNOWN,
                    "observation_vector_source": UNKNOWN,
                    "observation_dim": 22,
                    "model_load_status": model_meta["model_load_status"],
                    "q_evaluation_status": UNKNOWN,
                    "q_evaluation_note": "No reusable YC fixed-state 22D vector is retained. 224 saved state_hash only; re-running DSSAT to reconstruct it is prohibited by this prompt.",
                }
                row.update({f"mask_{index}": UNKNOWN for index in range(16)})
                row.update(q_columns_unknown())
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_fixed_state_qvalues.csv", index=False, encoding="utf-8-sig")
    return out


def validation_state_qvalues(daily: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in daily.to_dict("records"):
        seed = int(record["seed"])
        checkpoint = int(record["checkpoint"])
        meta = inventory[(inventory.seed == seed) & (inventory.checkpoint == checkpoint)].iloc[0].to_dict()
        valid = parse_indices(record.get("valid_action_indices", ""))
        row: dict[str, Any] = {
            "seed": seed,
            "checkpoint": checkpoint,
            "year": int(record["year"]),
            "source_row": int(record["source_row"]),
            "state_id": f"YCA_{int(record['year'])}_seed{seed}_ckpt{checkpoint}_row{int(record['source_row']):03d}",
            "dap": num(record.get("dap")),
            "state_category": record["state_category"],
            "swfac": num(record.get("swfac")),
            "nstres": num(record.get("nstres")),
            "observation_dim": num(record.get("observation_dim")),
            "observation_vector_available": False,
            "observation_source": UNKNOWN,
            "model_load_status": meta["model_load_status"],
            "q_evaluation_status": UNKNOWN,
            "q_evaluation_note": OBSERVATION_GAP,
            "selected_action_index": int(num(record.get("action_index"), -1)),
            "selected_action": action_label(int(num(record.get("action_index"), 0))),
            "executed_action_index": int(num(record.get("executed_action_index"), -1)),
            "daily_source_path": record["source_daily_path"],
        }
        row.update(mask_columns(valid))
        row.update(q_columns_unknown())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_validation_state_qvalues.csv", index=False, encoding="utf-8-sig")
    return out


def qranking_trajectory(inventory: pd.DataFrame) -> pd.DataFrame:
    categories = ["DAP1 anchor", "no-stress", "water-stress-only", "N-stress-only", "dual-stress"]
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for category in categories:
            row: dict[str, Any] = {
                "seed": seed,
                "state_category": category,
                "state_id": f"fixed_panel_{category.replace(' ', '_').replace('-', '_')}",
                "state_tracking_status": UNKNOWN,
                "pattern_class": UNKNOWN,
                "interpretation": "Cannot compare the same state across checkpoints without its retained 22D observation vector.",
            }
            for checkpoint in CHECKPOINTS:
                row[f"top1_action_{checkpoint // 1000}K"] = UNKNOWN
                row[f"top1_Q_{checkpoint // 1000}K"] = np.nan
                row[f"top2_action_{checkpoint // 1000}K"] = UNKNOWN
                row[f"Q_margin_{checkpoint // 1000}K"] = np.nan
                row[f"delta_Q_water_{checkpoint // 1000}K"] = np.nan
                row[f"delta_Q_N_{checkpoint // 1000}K"] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_qranking_trajectory.csv", index=False, encoding="utf-8-sig")
    return out


def action_frequency(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, checkpoint), group in daily.groupby(["seed", "checkpoint"], sort=True):
        actions = pd.to_numeric(group["action_index"], errors="coerce").astype(int)
        counts = actions.value_counts().reindex(range(16), fill_value=0).astype(int)
        total = int(len(actions))
        n_positive = [index for index in range(16) if N_LEVELS[index % 4] > 0]
        i_positive = [index for index in range(16) if I_LEVELS[index // 4] > 0]
        both_positive = [index for index in range(16) if I_LEVELS[index // 4] > 0 and N_LEVELS[index % 4] > 0]
        n_order = sorted(n_positive, key=lambda index: (-int(counts[index]), index))
        action_freq = {
            str(index): {"label": action_label(index), "count": int(counts[index]), "fraction": float(counts[index] / total) if total else np.nan}
            for index in range(16)
        }
        rows.append(
            {
                "seed": int(seed),
                "checkpoint": int(checkpoint),
                "total_decision_rows": total,
                "no_op_count": int(counts[0]),
                "no_op_frequency": float(counts[0] / total) if total else np.nan,
                "I_positive_count": int(counts[i_positive].sum()),
                "I_positive_frequency": float(counts[i_positive].sum() / total) if total else np.nan,
                "N_positive_count": int(counts[n_positive].sum()),
                "N_positive_frequency": float(counts[n_positive].sum() / total) if total else np.nan,
                "both_positive_count": int(counts[both_positive].sum()),
                "both_positive_frequency": float(counts[both_positive].sum() / total) if total else np.nan,
                "top_action_indices": ",".join(str(index) for index in sorted(range(16), key=lambda index: (-int(counts[index]), index))[:5]),
                "top_N_action_indices": ",".join(str(index) for index in n_order[:5]),
                "action_frequency_by_index": json.dumps(action_freq, ensure_ascii=False),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "yc_dqn_action_frequency_by_checkpoint.csv", index=False, encoding="utf-8-sig")
    return out


def ppo_alignment(dqn_trajectory: pd.DataFrame) -> pd.DataFrame:
    ppo_sources = {
        0: ROOT / "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/evaluation/055_00_checkpoint_validation_summary.csv",
        1: ROOT / "benchmark_results/222YCA_O1_yc_v1_original_seed1/evaluation/222YCA_O1_checkpoint_validation_summary.csv",
        2: ROOT / "benchmark_results/222YCA_O2_yc_v1_original_seed2/evaluation/222YCA_O2_checkpoint_validation_summary.csv",
    }
    ppo_rows: list[dict[str, Any]] = []
    for seed, path in ppo_sources.items():
        source = read_csv(path)
        if source.empty:
            continue
        source = source[
            pd.to_numeric(source["checkpoint_step"], errors="coerce").isin(CHECKPOINTS)
            & pd.to_numeric(source["year"], errors="coerce").isin(YEARS)
        ].copy()
        for checkpoint, group in source.groupby(source["checkpoint_step"].astype(int), sort=True):
            ppo_rows.append(
                {
                    "seed": int(seed),
                    "checkpoint": int(checkpoint),
                    "ppo_mean_yield": pd.to_numeric(group["final_grnwt"], errors="coerce").mean(),
                    "ppo_sd_yield": pd.to_numeric(group["final_grnwt"], errors="coerce").std(),
                    "ppo_mean_I": pd.to_numeric(group["total_irrigation"], errors="coerce").mean(),
                    "ppo_mean_N": pd.to_numeric(group["total_n"], errors="coerce").mean(),
                    "ppo_mean_PFP_N": pd.to_numeric(group["PFP_N"], errors="coerce").mean(),
                    "ppo_mean_WP_ET": UNKNOWN,
                    "ppo_source_path": path.relative_to(ROOT).as_posix(),
                }
            )
    ppo = pd.DataFrame(ppo_rows)
    dqn = dqn_trajectory.rename(
        columns={
            "mean_yield": "dqn_mean_yield",
            "sd_yield": "dqn_sd_yield",
            "mean_I": "dqn_mean_I",
            "mean_N": "dqn_mean_N",
            "mean_PFP_N": "dqn_mean_PFP_N",
            "mean_WP_ET": "dqn_mean_WP_ET",
        }
    )
    out = dqn.merge(ppo, on=["seed", "checkpoint"], how="left")
    out["comparison_scope"] = "same seed and checkpoint; both summaries aggregate validation years 2014-2023; descriptive only"
    out["trajectory_difference_note"] = "Compare checkpoint direction and resource/action mix; no causal equivalence claim."
    out.to_csv(RESULTS / "yc_dqn_vs_ppo_checkpoint_alignment.csv", index=False, encoding="utf-8-sig")
    return out


def table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}")
    view = view.fillna("")
    return "\n".join(
        [
            "| " + " | ".join(view.columns) + " |",
            "| " + " | ".join("---" for _ in view.columns) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy().tolist()],
        ]
    )


def make_report(
    inventory: pd.DataFrame,
    trajectory: pd.DataFrame,
    frequencies: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    load_pass = int(inventory["model_load_status"].eq("PASS").sum())
    trajectory_lines = table(
        trajectory,
        ["seed", "checkpoint", "mean_yield", "sd_yield", "mean_I", "mean_N", "mean_PFP_N", "mean_WP_ET", "yield_change_from_previous_checkpoint", "best_earlier_checkpoint", "drop_from_earlier_best"],
    )
    freq_lines = table(
        frequencies,
        ["seed", "checkpoint", "no_op_frequency", "I_positive_frequency", "N_positive_frequency", "both_positive_frequency", "top_action_indices", "top_N_action_indices"],
    )
    align_lines = table(
        alignment,
        ["seed", "checkpoint", "dqn_mean_yield", "ppo_mean_yield", "dqn_mean_I", "ppo_mean_I", "dqn_mean_N", "ppo_mean_N", "dqn_mean_PFP_N", "ppo_mean_PFP_N"],
    )
    lines = [
        "# YC DQN policy-level failure localization",
        "",
        "审计日期：2026-09-11。此报告只读取既有 formal DQN/PPO 结果与 12 个 DQN checkpoint；没有训练、调参、环境回放或 DSSAT 重跑。",
        "",
        "## Evidence boundary",
        "",
        f"- Formal DQN source：`benchmark_results/227_00_yc_mask_aware_dqn_formal/`；3 seeds × 4 checkpoints，模型加载核验 `{load_pass}/12`。",
        "- Retained daily logs contain `observation_dim=22`, stress scalars, masks and selected/executed actions, but not the 22-dimensional observation values.",
        "- No reusable YC fixed-state vector was found. The existing 224 mechanism check retained state hashes and PPO outputs, not state vectors.",
        f"- Therefore all Q-dependent fields are explicitly `{UNKNOWN}`; no zero-vector or reconstructed DSSAT state is used.",
        "",
        "## 1. PFP_N definition check",
        "",
        "- `N_100K` is the mean of yearly sums of executed DQN action N over 2014–2023. It is agent-action N only; the selected rendered treatment has `FAMN=0`, while initial `SNH4/SNO3` are soil initial conditions and are not counted as fertilizer N.",
        "- `PFP_N_100K` is the mean of per-year `PFP_N`. For positive-N years, the exact Summary.OUT `YPNAM` is used and matched to `NICM`; zero-agent-N years are missing, not zero.",
        "- Thus `mean(yield)/mean(N)` and `mean(PFP_N)` need not agree. Full formula evidence is in `results/yc_dqn_pfpn_definition_check.csv`.",
        "",
        "## 2. DQN checkpoint trajectory",
        "",
        trajectory_lines,
        "",
        "数字定位：seed 0 在 50K→75K 开始低于 earlier best，并在 75K→100K 继续下降，100K 比 50K 低 99.745 kg/ha；seed 1 在 25K→50K 有短暂回落（-67.553 kg/ha），但 100K 比 75K 回升 478.425 kg/ha，因此没有持续的 late degradation；seed 2 在 25K→50K 出现最大区间跳变（-3818.963 kg/ha），之后 50K–100K 处于较窄的 yield 平台。这里没有自定义 collapse threshold。",
        "",
        "seed2 的 N 行为从 25K 就与 seed0/1 分开：seed2 的 mean_N 为 192.0，而 seed0/1 均为 0；50K/75K seed2 回到 0，100K 为 88.0，而 seed0/1 仅为 20.0/16.0。",
        "",
        "## 3. Q-value ranking localization",
        "",
        "本轮无法定位 Q ranking failure。固定状态 panel 和真实 validation state 都缺失可直接输入 Q network 的 22D observation vector；因此无法证据化计算 `Q(action0..15)`、masked_Q、top1/top2、Q margin、`delta_Q_water` 或 `delta_Q_N`，也不能声称某个 stress state 出现了正负方向反转。",
        "",
        "已生成的 Q-value CSV 保留了每个真实 validation daily state 的 mask、stress category、selected/executed action 和模型加载状态；Q 列为空并标记 `unknown / insufficient evidence`，避免将 action choice 误写成 Q ranking 证据。",
        "",
        "## 4. Action behavior diagnostic, including seed2",
        "",
        freq_lines,
        "",
        "这些是已执行 action 的描述性频率，不是 Q-value 机制证据。seed2 的 N-positive/action-family 分离从 25K 已可见；不能据此预判 seed2 是坏 seed 或宣布 collapse。",
        "",
        "## 5. Matched Original PPO checkpoint alignment",
        "",
        align_lines,
        "",
        "PPO 的 `WP_ET` 在现有 matched checkpoint summary 中没有可直接对齐的 exact 字段，保留为 `unknown / insufficient evidence`。DQN 与 PPO 的 checkpoint 轨迹和资源行为不呈现可仅凭本表确认的同一种失败；由于 DQN 的 Q ranking 不可计算，本轮最终分类为 **D. insufficient evidence**，而不是 PPO-specific 或 shared task-level causal claim。",
        "",
        "## 6. Follow-up hypotheses only",
        "",
        "可作为下一层、另行预注册的候选假设：reward/credit assignment；state representation / partial observability；replay distribution；bootstrap/Q over- or under-estimation；function approximation/generalization；exploration path。",
        "",
        "These are follow-up hypotheses, not conclusions from this diagnostic.",
        "",
        "## Evidence index",
        "",
        "- `results/yc_dqn_pfpn_definition_check.csv`",
        "- `results/yc_dqn_checkpoint_trajectory_diagnostic.csv`",
        "- `results/yc_dqn_fixed_state_qvalues.csv`",
        "- `results/yc_dqn_validation_state_qvalues.csv`",
        "- `results/yc_dqn_qranking_trajectory.csv`",
        "- `results/yc_dqn_action_frequency_by_checkpoint.csv`",
        "- `results/yc_dqn_vs_ppo_checkpoint_alignment.csv`",
        "- Original PPO source summaries are the existing files recorded in the alignment CSV; no PPO model or environment was rerun.",
        "",
        "## Final status",
        "",
        "`POLICY_LEVEL_DIAGNOSTIC_COMPLETE`",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pfpn_definition_check()
    inventory, _models = load_all_dqn_checkpoints()
    daily = formal_daily_rows()
    trajectory = checkpoint_trajectory()
    fixed_state_qvalues(inventory)
    validation_state_qvalues(daily, inventory)
    qranking_trajectory(inventory)
    frequencies = action_frequency(daily)
    alignment = ppo_alignment(trajectory)
    make_report(inventory, trajectory, frequencies, alignment)
    print("POLICY_LEVEL_DIAGNOSTIC_COMPLETE")


if __name__ == "__main__":
    main()
