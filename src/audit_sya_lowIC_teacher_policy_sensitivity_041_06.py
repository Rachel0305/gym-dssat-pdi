from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TASK_ID = "041_06"
TASK_NAME = "sya_lowIC_teacher_policy_sensitivity_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

SOURCE_04102 = ROOT / "benchmark_results" / "041_02_sya_lowIC_layered_teacher_imitation_dataset"
SOURCE_04104 = ROOT / "benchmark_results" / "041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo"
TEACHER_SELECTED = SOURCE_04102 / "tables" / "041_02_selected_teacher_trajectories.csv"
BC_DATASET = SOURCE_04104 / "bc_dataset" / "041_03_bc_dataset.npz"
BC_META = SOURCE_04104 / "bc_dataset" / "041_03_bc_dataset_meta.csv"
BC_MODEL = SOURCE_04104 / "models" / "SYA" / "SYA_teacher_warmstart_maskableppo_seed0_bc_init.zip"
PPO100K_MODEL = SOURCE_04104 / "models" / "SYA" / "SYA_teacher_warmstart_maskableppo_seed0_ckpt100000.zip"
DAILY_DIR = SOURCE_04104 / "daily_outputs" / "SYA"
YEARS = list(range(2014, 2024))

IRRIGATION_LEVELS = [0.0, 30.0, 45.0]
NITROGEN_LEVELS = [0.0, 80.0, 120.0]
ACTION_GRID = [(i, n) for i in IRRIGATION_LEVELS for n in NITROGEN_LEVELS]


def ensure_dirs() -> None:
    for rel in ["tables", "figures"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def expanded_observation_names() -> list[str]:
    # Confirmed from env.unwrapped.observation_variables:
    # ['cumsumfert','dap','dtt','ep','grnwt','istage','nstres','rtdep',
    #  'srad','sw','swfac','tmax','topwt','totir','vstage','wtdep','xlai']
    # 25 dims means 'sw' expands to 9 layer-like values.
    return [
        "cumsumfert",
        "dap",
        "dtt",
        "ep",
        "grnwt",
        "istage",
        "nstres",
        "rtdep",
        "srad",
        "sw_1",
        "sw_2",
        "sw_3",
        "sw_4",
        "sw_5",
        "sw_6",
        "sw_7",
        "sw_8",
        "sw_9",
        "swfac",
        "tmax",
        "topwt",
        "totir",
        "vstage",
        "wtdep",
        "xlai",
    ]


def action_label(idx: int) -> str:
    if idx < 0 or idx >= len(ACTION_GRID):
        return f"a{idx}"
    i, n = ACTION_GRID[idx]
    return f"a{idx}(I{i:g},N{n:g})"


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    if len(df) > max_rows:
        lines.append(f"\n仅显示前 {max_rows} 行，共 {len(df)} 行。")
    return "\n".join(lines)


def load_teacher_and_bc() -> tuple[pd.DataFrame, np.lib.npyio.NpzFile, pd.DataFrame]:
    missing = [p for p in [TEACHER_SELECTED, BC_DATASET, BC_META, BC_MODEL, PPO100K_MODEL] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required 041_06 inputs: " + "; ".join(str(p) for p in missing))
    selected = pd.read_csv(TEACHER_SELECTED)
    data = np.load(BC_DATASET)
    meta = pd.read_csv(BC_META)
    return selected, data, meta


def summarize_teacher_trajectories(meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in meta.groupby("year"):
        group = group.sort_values("step")
        actions = pd.to_numeric(group["teacher_action_index"], errors="coerce").fillna(0).astype(int)
        irr = pd.to_numeric(group["teacher_irrigation_mm"], errors="coerce").fillna(0)
        nit = pd.to_numeric(group["teacher_nitrogen_kg_ha"], errors="coerce").fillna(0)
        nz = group[(irr > 0) | (nit > 0)].copy()
        event_days = [int(x) for x in pd.to_numeric(nz["step"], errors="coerce").fillna(-1).tolist()]
        event_actions = [action_label(int(x)) for x in pd.to_numeric(nz["teacher_action_index"], errors="coerce").fillna(0).tolist()]
        rows.append(
            {
                "year": int(year),
                "teacher_tier": str(group["teacher_tier"].iloc[0]),
                "candidate_id": str(group["candidate_id"].iloc[0]),
                "sample_days": int(len(group)),
                "nonzero_event_count": int(len(nz)),
                "irrigation_total": float(irr.sum()),
                "nitrogen_total": float(nit.sum()),
                "event_days": ";".join(map(str, event_days)),
                "event_actions": ";".join(event_actions),
                "action_sequence_hashlike": "-".join(map(str, actions.tolist())),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def teacher_pairwise_distance(meta: pd.DataFrame) -> pd.DataFrame:
    seqs: dict[int, np.ndarray] = {}
    for year, group in meta.groupby("year"):
        seqs[int(year)] = (
            group.sort_values("step")["teacher_action_index"].astype(int).to_numpy()
        )
    rows: list[dict[str, Any]] = []
    for a, b in combinations(sorted(seqs), 2):
        n = min(len(seqs[a]), len(seqs[b]))
        if n == 0:
            continue
        aa = seqs[a][:n]
        bb = seqs[b][:n]
        rows.append(
            {
                "year_a": a,
                "year_b": b,
                "compared_days": int(n),
                "hamming_rate_all_days": float(np.mean(aa != bb)),
                "different_nonzero_days": int(np.sum((aa != bb) & ((aa != 0) | (bb != 0)))),
            }
        )
    return pd.DataFrame(rows)


def observation_coverage(obs: np.ndarray) -> pd.DataFrame:
    names = expanded_observation_names()
    rows = []
    wanted = {
        "DAP": ["dap"],
        "TMAX": ["tmax"],
        "TMIN": ["tmin"],
        "SRAD": ["srad"],
        "RAIN": ["rain"],
        "soil_water": [n for n in names if n.startswith("sw_")],
        "SWFAC": ["swfac"],
        "NSTRES": ["nstres"],
        "cumulative_irrigation": ["totir"],
        "cumulative_nitrogen": ["cumsumfert"],
        "yield_state": ["grnwt"],
        "biomass_state": ["topwt"],
    }
    for concept, present_names in wanted.items():
        present = [n for n in present_names if n in names]
        indices = [names.index(n) for n in present]
        rows.append(
            {
                "concept": concept,
                "directly_in_observation": bool(indices),
                "observation_names": ";".join(present),
                "observation_indices": ";".join(map(str, indices)),
                "note": "" if indices else "not directly visible to policy in current 25-dim observation",
            }
        )
    range_rows = []
    for i, name in enumerate(names):
        values = obs[:, i]
        range_rows.append(
            {
                "obs_index": i,
                "obs_name": name,
                "min": float(np.nanmin(values)),
                "p10": float(np.nanpercentile(values, 10)),
                "mean": float(np.nanmean(values)),
                "p90": float(np.nanpercentile(values, 90)),
                "max": float(np.nanmax(values)),
                "std": float(np.nanstd(values)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(range_rows)


def load_model(model_path: Path) -> Any:
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(str(model_path), device="auto")


def masked_probs(model: Any, obs: np.ndarray, masks: np.ndarray, batch_size: int = 512) -> np.ndarray:
    probs_all: list[np.ndarray] = []
    for start in range(0, len(obs), batch_size):
        batch_obs = obs[start : start + batch_size]
        batch_masks = masks[start : start + batch_size].astype(bool)
        obs_tensor, _ = model.policy.obs_to_tensor(batch_obs)
        try:
            dist = model.policy.get_distribution(obs_tensor, action_masks=batch_masks)
        except TypeError:
            dist = model.policy.get_distribution(obs_tensor)
        distribution = getattr(dist, "distribution", dist)
        probs = distribution.probs.detach().cpu().numpy()
        probs = np.where(batch_masks, probs, 0.0)
        denom = probs.sum(axis=1, keepdims=True)
        probs = np.divide(probs, denom, out=np.zeros_like(probs), where=denom > 0)
        probs_all.append(probs)
    return np.vstack(probs_all)


def summarize_policy_on_teacher_states(model: Any, obs: np.ndarray, masks: np.ndarray, meta: pd.DataFrame, label: str) -> pd.DataFrame:
    probs = masked_probs(model, obs, masks)
    pred = np.argmax(probs, axis=1)
    out = meta[["year", "step", "dap", "teacher_action_index", "teacher_irrigation_mm", "teacher_nitrogen_kg_ha"]].copy()
    out["policy_label"] = label
    out["pred_action_index"] = pred
    out["pred_action_label"] = [action_label(int(x)) for x in pred]
    out["pred_irrigation_mm"] = [ACTION_GRID[int(x)][0] for x in pred]
    out["pred_nitrogen_kg_ha"] = [ACTION_GRID[int(x)][1] for x in pred]
    out["top_prob"] = probs.max(axis=1)
    out["nonzero_prob"] = probs[:, 1:].sum(axis=1)
    out["matches_teacher_action"] = pred == pd.to_numeric(out["teacher_action_index"], errors="coerce").fillna(-1).astype(int).to_numpy()
    return out


def policy_diversity_summary(policy_states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in policy_states.groupby("policy_label"):
        pred = group["pred_action_index"].astype(int)
        rows.append(
            {
                "policy_label": label,
                "sample_count": int(len(group)),
                "unique_pred_actions": int(pred.nunique()),
                "nonzero_pred_rate": float((pred != 0).mean()),
                "teacher_match_rate_all": float(group["matches_teacher_action"].mean()),
                "teacher_match_rate_on_teacher_nonzero": float(
                    group.loc[group["teacher_action_index"].astype(int) != 0, "matches_teacher_action"].mean()
                ),
                "mean_top_prob": float(group["top_prob"].mean()),
                "mean_nonzero_prob": float(group["nonzero_prob"].mean()),
                "mean_pred_irrigation": float(group["pred_irrigation_mm"].mean()),
                "mean_pred_nitrogen": float(group["pred_nitrogen_kg_ha"].mean()),
            }
        )
    return pd.DataFrame(rows)


def perturbation_groups(names: list[str]) -> dict[str, list[int]]:
    groups = {
        "dap": ["dap"],
        "srad": ["srad"],
        "tmax": ["tmax"],
        "swfac": ["swfac"],
        "nstres": ["nstres"],
        "soil_water_sw_all_layers": [n for n in names if n.startswith("sw_")],
        "cumulative_irrigation_totir": ["totir"],
        "cumulative_nitrogen_cumsumfert": ["cumsumfert"],
        "crop_biomass_topwt": ["topwt"],
        "grain_yield_state_grnwt": ["grnwt"],
    }
    return {k: [names.index(n) for n in v if n in names] for k, v in groups.items()}


def run_perturbation_sensitivity(model: Any, obs: np.ndarray, masks: np.ndarray, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = expanded_observation_names()
    base_probs = masked_probs(model, obs, masks)
    base_pred = np.argmax(base_probs, axis=1)
    groups = perturbation_groups(names)
    rows = []
    for group_name, indices in groups.items():
        if not indices:
            continue
        directions = [("low_p10", 10), ("high_p90", 90), ("high_max", 100)]
        for direction, q in directions:
            perturbed = obs.copy()
            for idx in indices:
                target = float(np.nanpercentile(obs[:, idx], q))
                perturbed[:, idx] = max(0.0, target)
            probs = masked_probs(model, perturbed, masks)
            pred = np.argmax(probs, axis=1)
            tvd = 0.5 * np.abs(probs - base_probs).sum(axis=1)
            base_arg_prob = base_probs[np.arange(len(base_pred)), base_pred]
            new_arg_prob = probs[np.arange(len(base_pred)), base_pred]
            rows.append(
                {
                    "policy_label": label,
                    "perturbation_group": group_name,
                    "direction": direction,
                    "obs_indices": ";".join(map(str, indices)),
                    "argmax_change_rate": float(np.mean(pred != base_pred)),
                    "mean_total_variation_distance": float(np.mean(tvd)),
                    "p95_total_variation_distance": float(np.percentile(tvd, 95)),
                    "mean_original_argmax_prob_change": float(np.mean(new_arg_prob - base_arg_prob)),
                    "mean_nonzero_prob_change": float(np.mean(probs[:, 1:].sum(axis=1) - base_probs[:, 1:].sum(axis=1))),
                }
            )
    dim_rows = []
    for idx, name in enumerate(names):
        if np.nanstd(obs[:, idx]) == 0:
            continue
        perturbed = obs.copy()
        perturbed[:, idx] = max(0.0, float(np.nanpercentile(obs[:, idx], 90)))
        probs = masked_probs(model, perturbed, masks)
        pred = np.argmax(probs, axis=1)
        tvd = 0.5 * np.abs(probs - base_probs).sum(axis=1)
        dim_rows.append(
            {
                "policy_label": label,
                "obs_index": idx,
                "obs_name": name,
                "argmax_change_rate_p90": float(np.mean(pred != base_pred)),
                "mean_total_variation_distance_p90": float(np.mean(tvd)),
                "std": float(np.nanstd(obs[:, idx])),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(dim_rows).sort_values(
        ["policy_label", "mean_total_variation_distance_p90"], ascending=[True, False]
    )


def write_figures(teacher_summary: pd.DataFrame, diversity: pd.DataFrame, sensitivity: pd.DataFrame) -> list[str]:
    figs: list[str] = []

    x = np.arange(len(teacher_summary))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, teacher_summary["irrigation_total"], width=0.4, label="teacher I")
    ax.bar(x + 0.2, teacher_summary["nitrogen_total"], width=0.4, label="teacher N")
    ax.set_xticks(x)
    ax.set_xticklabels(teacher_summary["year"].astype(str), rotation=45)
    ax.set_ylabel("Total amount")
    ax.set_title("041_06 teacher resource diversity across years")
    ax.legend()
    fig.tight_layout()
    p = OUT / "figures" / "041_06_teacher_resource_diversity.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    figs.append(p.relative_to(ROOT).as_posix())

    fig, ax = plt.subplots(figsize=(10, 4))
    sub = sensitivity.copy()
    sub["label"] = sub["policy_label"] + " | " + sub["perturbation_group"] + " | " + sub["direction"]
    top = sub.sort_values("mean_total_variation_distance", ascending=False).head(20)
    ax.barh(np.arange(len(top)), top["mean_total_variation_distance"])
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean total variation distance")
    ax.set_title("041_06 largest policy-probability sensitivity")
    fig.tight_layout()
    p = OUT / "figures" / "041_06_policy_sensitivity_top20.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    figs.append(p.relative_to(ROOT).as_posix())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(diversity["policy_label"], diversity["teacher_match_rate_on_teacher_nonzero"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Match rate on teacher nonzero states")
    ax.set_title("041_06 policy match to teacher key actions")
    fig.tight_layout()
    p = OUT / "figures" / "041_06_teacher_nonzero_match_rate.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    figs.append(p.relative_to(ROOT).as_posix())
    return figs


def write_record(result: dict[str, Any], coverage: pd.DataFrame, teacher_summary: pd.DataFrame, teacher_distance: pd.DataFrame, diversity: pd.DataFrame, sensitivity: pd.DataFrame, top_dims: pd.DataFrame) -> None:
    rain_visible = bool(coverage.loc[coverage["concept"].eq("RAIN"), "directly_in_observation"].iloc[0])
    tmin_visible = bool(coverage.loc[coverage["concept"].eq("TMIN"), "directly_in_observation"].iloc[0])
    lines = [
        "# 041_06 SYA lowIC teacher 轨迹差异与 PPO 策略输入敏感性审计记录",
        "",
        "## 结论",
        "",
        f"- 分支：`{result['branch']}`",
        "- 本任务没有训练，也没有重新跑 DSSAT 季节；只读取 041_04 的 BC dataset 与模型 checkpoint。",
        f"- 当前 observation 中 RAIN 直接可见：{rain_visible}；TMIN 直接可见：{tmin_visible}。",
        "- 因此不能说当前 PPO 直接根据当天降雨或 Tmin 决策；它最多通过 DSSAT 状态变量间接受历史天气影响。",
        "",
        "## observation 覆盖",
        "",
        md_table(coverage, 20),
        "",
        "## teacher 跨年份轨迹摘要",
        "",
        md_table(teacher_summary, 20),
        "",
        "## teacher 轨迹两两差异摘要",
        "",
        md_table(teacher_distance.describe(include='all').reset_index(), 20),
        "",
        "## BC init 与 PPO100K 在 teacher 状态上的动作多样性",
        "",
        md_table(diversity, 20),
        "",
        "## 输入扰动敏感性",
        "",
        md_table(sensitivity.sort_values(["policy_label", "mean_total_variation_distance"], ascending=[True, False]), 40),
        "",
        "## 单维度敏感性 Top",
        "",
        md_table(top_dims.head(40), 40),
        "",
        "## 输出文件",
        "",
    ]
    for key, val in result["outputs"].items():
        if isinstance(val, list):
            for item in val:
                lines.append(f"- {key}: `{item}`")
        else:
            lines.append(f"- {key}: `{val}`")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    selected, data, meta = load_teacher_and_bc()
    obs = np.asarray(data["obs"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=bool)
    names = expanded_observation_names()
    if obs.shape[1] != len(names):
        raise RuntimeError(f"Expected {len(names)} obs dims, got {obs.shape[1]}")

    teacher_summary = summarize_teacher_trajectories(meta)
    teacher_distance = teacher_pairwise_distance(meta)
    coverage, obs_ranges = observation_coverage(obs)

    bc_model = load_model(BC_MODEL)
    ppo_model = load_model(PPO100K_MODEL)
    bc_states = summarize_policy_on_teacher_states(bc_model, obs, masks, meta, "bc_init")
    ppo_states = summarize_policy_on_teacher_states(ppo_model, obs, masks, meta, "ppo100k")
    policy_states = pd.concat([bc_states, ppo_states], ignore_index=True)
    diversity = policy_diversity_summary(policy_states)

    bc_sens, bc_dims = run_perturbation_sensitivity(bc_model, obs, masks, "bc_init")
    ppo_sens, ppo_dims = run_perturbation_sensitivity(ppo_model, obs, masks, "ppo100k")
    sensitivity = pd.concat([bc_sens, ppo_sens], ignore_index=True)
    top_dims = pd.concat([bc_dims, ppo_dims], ignore_index=True).sort_values(
        ["policy_label", "mean_total_variation_distance_p90"], ascending=[True, False]
    )

    paths = {
        "teacher_summary": OUT / "tables" / "041_06_teacher_trajectory_summary.csv",
        "teacher_pairwise_distance": OUT / "tables" / "041_06_teacher_pairwise_action_distance.csv",
        "observation_coverage": OUT / "tables" / "041_06_observation_variable_coverage.csv",
        "observation_ranges": OUT / "tables" / "041_06_observation_dimension_ranges.csv",
        "policy_states": OUT / "tables" / "041_06_policy_predictions_on_teacher_states.csv",
        "policy_diversity": OUT / "tables" / "041_06_policy_diversity_summary.csv",
        "policy_sensitivity": OUT / "tables" / "041_06_policy_perturbation_sensitivity.csv",
        "top_sensitive_dimensions": OUT / "tables" / "041_06_top_sensitive_observation_dimensions.csv",
    }
    teacher_summary.to_csv(paths["teacher_summary"], index=False)
    teacher_distance.to_csv(paths["teacher_pairwise_distance"], index=False)
    coverage.to_csv(paths["observation_coverage"], index=False)
    obs_ranges.to_csv(paths["observation_ranges"], index=False)
    policy_states.to_csv(paths["policy_states"], index=False)
    diversity.to_csv(paths["policy_diversity"], index=False)
    sensitivity.to_csv(paths["policy_sensitivity"], index=False)
    top_dims.to_csv(paths["top_sensitive_dimensions"], index=False)

    figs = write_figures(teacher_summary, diversity, sensitivity)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "A_teacher_policy_sensitivity_audit_completed",
        "obs_shape": list(obs.shape),
        "teacher_years": sorted(selected["year"].astype(int).unique().tolist()),
        "models": {
            "bc_init": BC_MODEL.relative_to(ROOT).as_posix(),
            "ppo100k": PPO100K_MODEL.relative_to(ROOT).as_posix(),
        },
        "outputs": {k: v.relative_to(ROOT).as_posix() for k, v in paths.items()} | {
            "figures": figs,
            "record_md": DOC.relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "041_06_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, coverage, teacher_summary, teacher_distance, diversity, sensitivity, top_dims)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
