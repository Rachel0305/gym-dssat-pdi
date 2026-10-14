"""040_53: local policy-sensitivity audit for the 040_40 SYA lowIC PPO checkpoint.

This script does not train, does not alter DSSAT inputs, and does not rerun
counterfactual DSSAT seasons. It walks the real 040_40 deterministic policy
trajectory, then asks the frozen policy what it would do if selected resolved
observation dimensions were perturbed at the same decision state.
"""

from __future__ import annotations

import json
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

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as run04040
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


TASK = "040_53_sya_lowIC_04040_policy_sensitivity_audit"
OUT = ROOT / "benchmark_results" / TASK
TAB = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = "SYA"
YEARS = list(range(2014, 2024))
CHECKPOINT = 100_000
MODEL_PATH = (
    ROOT
    / "benchmark_results"
    / "040_40_sya_lowIC_ppo_yield_guardrail_v3"
    / "models"
    / "SYA"
    / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{CHECKPOINT}.zip"
)

RESOLUTION_CSV = (
    ROOT
    / "benchmark_results"
    / "040_47_sya_lowIC_04040_policy_observation_index_resolution"
    / "tables"
    / "040_47_name_to_index_resolution.csv"
)
RAW_OBS_CSV = (
    ROOT
    / "benchmark_results"
    / "040_47_sya_lowIC_04040_policy_observation_index_resolution"
    / "tables"
    / "040_47_raw_observation_by_step.csv"
)

AUDIT_DAPS = [1, 8, 31, 38, 43, 50, 61, 91]


def ensure_dirs() -> None:
    for p in [OUT, TAB, DOC.parent]:
        p.mkdir(parents=True, exist_ok=True)


def read_resolved_indices() -> dict[str, int]:
    df = pd.read_csv(RESOLUTION_CSV, keep_default_na=False)
    out: dict[str, int] = {}
    for row in df.itertuples(index=False):
        if str(row.status) == "unique_match" and str(row.resolved_index).strip() != "":
            out[str(row.name)] = int(float(row.resolved_index))
    required = {"dap", "nstres", "sw", "srad"}
    missing = sorted(required - set(out))
    if missing:
        raise RuntimeError(f"Missing required resolved observation indices from 040_47: {missing}")
    return out


def empirical_values(indices: dict[str, int]) -> dict[str, dict[str, float]]:
    if not RAW_OBS_CSV.exists():
        raise FileNotFoundError(RAW_OBS_CSV)
    raw = pd.read_csv(RAW_OBS_CSV, keep_default_na=False)
    values: dict[str, dict[str, float]] = {}
    for name, idx in indices.items():
        col = f"obs_{idx}"
        if col not in raw.columns:
            col = f"raw_{idx:02d}"
        if col not in raw.columns:
            continue
        series = pd.to_numeric(raw[col], errors="coerce").dropna()
        if series.empty:
            continue
        values[name] = {
            "p05": float(series.quantile(0.05)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return values


def build_env_config(config: dict[str, Any]) -> dict[str, Any]:
    split = base03222.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selection = base03222.build_selection(split)
    selection = selection[
        selection["station_code"].astype(str).eq(STATION)
        & selection["year"].isin(YEARS)
    ].copy()
    if selection.empty:
        raise RuntimeError("No SYA validation years found for 040_53.")
    return direct_ppo.build_env_config(config, selection)


def decode_action(env: Any, action: int) -> dict[str, float]:
    grid = getattr(env, "grid", None)
    if isinstance(grid, list) and 0 <= int(action) < len(grid):
        raw = dict(grid[int(action)])
        return {
            "action_irrigation_mm": float(raw.get("amir", 0.0)),
            "action_n_kg_ha": float(raw.get("anfer", 0.0)),
        }
    return {"action_irrigation_mm": np.nan, "action_n_kg_ha": np.nan}


def masked_probs(model: MaskablePPO, obs: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor, action_masks=np.asarray(mask, dtype=bool).reshape(1, -1))
        probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
        return probs.astype(float)
    except Exception:
        return None


def predict_action(model: MaskablePPO, obs: np.ndarray, mask: np.ndarray) -> tuple[int, np.ndarray | None]:
    action, _ = model.predict(obs, action_masks=np.asarray(mask, dtype=bool), deterministic=True)
    probs = masked_probs(model, obs, mask)
    return int(np.asarray(action).item()), probs


def perturbations(obs: np.ndarray, indices: dict[str, int], values: dict[str, dict[str, float]]) -> list[tuple[str, np.ndarray, str]]:
    base = np.asarray(obs, dtype=np.float32).reshape(-1)
    variants: list[tuple[str, np.ndarray, str]] = [("baseline", base.copy(), "no perturbation")]

    if "sw" in indices and "sw" in values:
        low = base.copy()
        low[indices["sw"]] = values["sw"]["p05"]
        variants.append(("low_sw", low, f"set sw index {indices['sw']} to empirical p05={values['sw']['p05']:.6g}"))
        high = base.copy()
        high[indices["sw"]] = values["sw"]["p95"]
        variants.append(("high_sw", high, f"set sw index {indices['sw']} to empirical p95={values['sw']['p95']:.6g}"))

    if "nstres" in indices:
        high_n = base.copy()
        high_n[indices["nstres"]] = max(0.30, float(base[indices["nstres"]]))
        variants.append(("high_nstres", high_n, f"set nstres index {indices['nstres']} to at least 0.30"))

    if "srad" in indices and "srad" in values:
        low_srad = base.copy()
        low_srad[indices["srad"]] = values["srad"]["p05"]
        variants.append(("low_srad", low_srad, f"set srad index {indices['srad']} to empirical p05={values['srad']['p05']:.6g}"))
        high_srad = base.copy()
        high_srad[indices["srad"]] = values["srad"]["p95"]
        variants.append(("high_srad", high_srad, f"set srad index {indices['srad']} to empirical p95={values['srad']['p95']:.6g}"))

    return variants


def latest_scalar(env: Any, obs: np.ndarray, info: dict[str, Any], key: str) -> float:
    latest = direct_ppo.latest_observation_dict(env, obs, info)
    return direct_ppo.scalar(latest.get(key, np.nan), np.nan)


def audit_year(model: MaskablePPO, config: dict[str, Any], env_config: dict[str, Any], year: int, indices: dict[str, int], values: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    env = run04040.make_env_with_yield_guardrail(config, env_config, STATION, int(year), int(config.get("seed", 0)), f"{STATION}_{year}_04053_policy_sensitivity", evaluation=True)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(config["runtime"]["max_steps"]):
            dap_value = latest_scalar(env, obs, info, "dap")
            dap = int(round(dap_value)) if np.isfinite(dap_value) and dap_value > 0 else step_count + 1
            mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
            base_action, base_probs = predict_action(model, np.asarray(obs), mask)
            if dap in AUDIT_DAPS:
                for scenario, obs_var, description in perturbations(np.asarray(obs), indices, values):
                    action, probs = predict_action(model, obs_var, mask)
                    decoded = decode_action(env, action)
                    base_decoded = decode_action(env, base_action)
                    row = {
                        "station_code": STATION,
                        "year": int(year),
                        "step_index": int(step_count),
                        "dap": int(dap),
                        "scenario": scenario,
                        "perturbation": description,
                        "baseline_action": int(base_action),
                        "perturbed_action": int(action),
                        "action_changed_vs_baseline": bool(action != base_action),
                        "baseline_irrigation_mm": base_decoded["action_irrigation_mm"],
                        "baseline_n_kg_ha": base_decoded["action_n_kg_ha"],
                        **decoded,
                        "valid_action_count": int(mask.sum()),
                        "mask_true_indices": ",".join(map(str, np.where(mask)[0].tolist())),
                    }
                    if probs is not None:
                        row["perturbed_action_prob"] = float(probs[action])
                        row["baseline_action_prob_under_perturbed_obs"] = float(probs[base_action])
                    if base_probs is not None:
                        row["baseline_action_prob_under_baseline_obs"] = float(base_probs[base_action])
                    for name, idx in indices.items():
                        row[f"obs_{name}_baseline"] = float(np.asarray(obs).reshape(-1)[idx])
                        row[f"obs_{name}_perturbed"] = float(obs_var.reshape(-1)[idx])
                    rows.append(row)
            obs, _reward, terminated, truncated, info = env.step(base_action)
            done = bool(terminated or truncated)
            step_count += 1
        return rows
    finally:
        env.close()


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    indices = read_resolved_indices()
    values = empirical_values(indices)
    config = run04040.load_config()
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = run04040.LOWIC_INPUT_ROOT
    try:
        env_config = build_env_config(config)
        model = MaskablePPO.load(str(MODEL_PATH), device="cpu")
        rows: list[dict[str, Any]] = []
        for year in YEARS:
            rows.extend(audit_year(model, config, env_config, int(year), indices, values))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root

    detail = pd.DataFrame(rows)
    detail_path = TAB / "040_53_policy_sensitivity_detail.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    if detail.empty:
        by_scenario = pd.DataFrame()
        by_dap = pd.DataFrame()
        branch = "no_audit_rows"
    else:
        nonbase = detail[detail["scenario"].ne("baseline")].copy()
        by_scenario = (
            nonbase.groupby("scenario", as_index=False)
            .agg(
                n_states=("action_changed_vs_baseline", "size"),
                n_changed=("action_changed_vs_baseline", "sum"),
                change_rate=("action_changed_vs_baseline", "mean"),
                n_irrigation_action=("action_irrigation_mm", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
                n_n_action=("action_n_kg_ha", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            )
            .sort_values("scenario")
        )
        by_dap = (
            nonbase.groupby(["dap", "scenario"], as_index=False)
            .agg(
                n_states=("action_changed_vs_baseline", "size"),
                n_changed=("action_changed_vs_baseline", "sum"),
                change_rate=("action_changed_vs_baseline", "mean"),
            )
            .sort_values(["dap", "scenario"])
        )
        overall_change = float(nonbase["action_changed_vs_baseline"].mean()) if len(nonbase) else 0.0
        if overall_change == 0.0:
            branch = "no_deterministic_action_sensitivity_detected"
        elif overall_change < 0.20:
            branch = "weak_deterministic_action_sensitivity_detected"
        else:
            branch = "deterministic_action_sensitivity_detected"

    by_scenario_path = TAB / "040_53_policy_sensitivity_by_scenario.csv"
    by_dap_path = TAB / "040_53_policy_sensitivity_by_dap.csv"
    by_scenario.to_csv(by_scenario_path, index=False, encoding="utf-8-sig")
    by_dap.to_csv(by_dap_path, index=False, encoding="utf-8-sig")

    result = {
        "task": TASK,
        "training_run": False,
        "dssat_counterfactual_run": False,
        "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "checkpoint": CHECKPOINT,
        "years": YEARS,
        "audit_daps": AUDIT_DAPS,
        "resolved_indices": indices,
        "branch": branch,
        "detail_csv": detail_path.relative_to(ROOT).as_posix(),
        "by_scenario_csv": by_scenario_path.relative_to(ROOT).as_posix(),
        "by_dap_csv": by_dap_path.relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_53_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {TASK}",
        "",
        "## 结论先说",
        "",
        f"- 分支：`{branch}`。",
        "- 本任务没有训练模型，也没有重跑 DSSAT 反事实季节；只是检查冻结 PPO 在同一 action mask 下对观测扰动的动作响应。",
        "",
        "## 固定对象",
        "",
        f"- 模型：`{MODEL_PATH.relative_to(ROOT).as_posix()}`",
        f"- 年份：`{YEARS[0]}–{YEARS[-1]}`",
        f"- 关键 DAP：`{AUDIT_DAPS}`",
        f"- 已解析扰动维度：`{indices}`",
        "",
        "## 按扰动类型汇总",
        "",
        md_table(by_scenario),
        "",
        "## 按 DAP 与扰动类型汇总",
        "",
        md_table(by_dap, max_rows=120),
        "",
        "## 输出文件",
        "",
        f"- 详细表：`{detail_path.relative_to(ROOT).as_posix()}`",
        f"- 扰动汇总：`{by_scenario_path.relative_to(ROOT).as_posix()}`",
        f"- DAP 汇总：`{by_dap_path.relative_to(ROOT).as_posix()}`",
        "",
        "## 边界",
        "",
        "- 这些扰动是局部 observation perturbation，不等同于真实 DSSAT 物理反事实。",
        "- 如果动作不变，只能说明当前 checkpoint 对这些已解析维度的 deterministic action 局部敏感性不足；不能说明 DSSAT 不重要。",
        "- 如果动作改变，仍需后续真实季节回放验证这些改变是否提高产量或资源效率。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
