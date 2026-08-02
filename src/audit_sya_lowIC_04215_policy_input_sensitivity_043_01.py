"""043_01: policy-input sensitivity audit for the frozen 042_15/042_12 PPO.

This script does not train, does not tune reward, and does not alter DSSAT
inputs.  It replays the frozen 042_11 ckpt25k binary-timing MaskablePPO policy
only to collect the *actual* policy observation vectors and action masks, then
asks whether local perturbations of already-observed variables change the
policy's masked action probabilities / deterministic argmax.

Why this exists:
    If the current best PPO is weakly sensitive to the existing weather/soil
    state variables, then adding perfect historical-weather forecast features is
    a defensible change in the decision information structure rather than a
    blind tuning trick.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as run04210


TASK = "043_01_sya_lowIC_04215_policy_input_sensitivity_audit"
OUT = ROOT / "benchmark_results" / TASK
TAB = OUT / "tables"
CFG = OUT / "configs"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = "SYA"
YEARS = list(range(2014, 2024))
AUDIT_DAPS = [1, 2, 31, 38, 51, 61, 91]
CHECKPOINT = 25_000
MODEL_PATH = (
    ROOT
    / "benchmark_results"
    / "042_11_sya_lowIC_binary_timing_training_length_curve"
    / "models"
    / STATION
    / f"{STATION}_half_split_stress_aware_maskableppo_seed0_ckpt{CHECKPOINT}.zip"
)

INVISIBLE_WEATHER_NAMES = [
    "rain",
    "tmin",
    "rain_past7",
    "rain_future7",
    "tmean_future7",
    "tmax_future7",
    "srad_future7",
]


def ensure_dirs() -> None:
    for p in [OUT, TAB, CFG, DOC.parent, PROMPT.parent]:
        p.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        dst = CFG / PROMPT.name
        try:
            shutil.copy2(PROMPT, dst)
        except PermissionError:
            # Docker-on-Windows bind mounts can reject copystat/utime even when
            # normal file writes are allowed.  The prompt backup only needs
            # content fidelity, not metadata fidelity.
            shutil.copyfile(PROMPT, dst)


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


def unwrap_chain(env: Any, max_depth: int = 30) -> list[Any]:
    out: list[Any] = []
    seen: set[int] = set()
    cur = env
    for _ in range(max_depth):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        out.append(cur)
        nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return out


def first_name_list(env: Any) -> tuple[list[str], str]:
    candidates = [
        "observation_variables",
        "observation_names",
        "obs_names",
        "state_names",
        "observation_keys",
        "obs_keys",
    ]
    for depth, obj in enumerate(unwrap_chain(env)):
        objects = [(f"env_chain[{depth}]", obj)]
        for attr in ["formator", "unwrapped"]:
            try:
                sub = getattr(obj, attr, None)
            except Exception:
                sub = None
            if sub is not None:
                objects.append((f"env_chain[{depth}].{attr}", sub))
        for label, target in objects:
            for name in candidates:
                try:
                    value = getattr(target, name, None)
                except Exception:
                    value = None
                if isinstance(value, (list, tuple)) and value:
                    return [str(x) for x in value], f"{label}.{name}"
    return [], "not_found"


def expand_observation_names(names: list[str], obs_size: int) -> tuple[list[str], str]:
    """Expand vector-valued DSSAT observation names to the actual policy vector.

    The underlying DSSAT wrapper can expose semantic names such as
    ``[..., "sw", "swfac", ...]`` while the policy receives ``sw`` expanded to
    multiple soil-layer dimensions.  In the current SYA observation this is the
    common 17 semantic names -> 25 numeric dimensions case, with ``sw`` occupying
    nine layers.  Without this expansion the sensitivity audit would falsely
    conclude that no named dimensions can be perturbed.
    """
    if len(names) == obs_size:
        return list(names), "names_match_vector_no_expansion"
    if "sw" in [str(x).lower() for x in names] and obs_size > len(names):
        sw_pos = [str(x).lower() for x in names].index("sw")
        extra_dims = obs_size - len(names)
        sw_layers = extra_dims + 1
        expanded = (
            list(names[:sw_pos])
            + [f"sw_{i}" for i in range(1, sw_layers + 1)]
            + list(names[sw_pos + 1 :])
        )
        if len(expanded) == obs_size:
            return expanded, f"expanded_sw_to_{sw_layers}_layers"
    fallback = list(names) + [""] * max(0, obs_size - len(names))
    return fallback[:obs_size], "name_vector_mismatch_unresolved"


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        return direct_ppo.scalar(value, default)
    except Exception:
        arr = np.asarray(value)
        if arr.size == 1:
            try:
                return float(arr.reshape(-1)[0])
            except Exception:
                return default
        return default


def latest(env: Any, obs: np.ndarray, info: dict[str, Any] | None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def build_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    split = base03222.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selection = base03222.build_selection(split)
    selection = selection[
        selection["station_code"].astype(str).eq(STATION)
        & selection["year"].isin(YEARS)
    ].copy()
    if selection.empty:
        raise RuntimeError("未找到 SYA 2014-2023 的 032_22/042_10 选择表记录。")
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, CFG / "043_01_resolved_env_config.yaml")
    selection.to_csv(CFG / "043_01_selection.csv", index=False, encoding="utf-8-sig")
    return env_config


def make_env(config: dict[str, Any], env_config: dict[str, Any], year: int):
    # 042_10's patch is essential: it replaces the base discrete wrapper with
    # the same binary-timing / late-irrigation-reserve-mask wrapper used by the
    # frozen 042_15 flow.
    return base03222.base.make_env(
        config,
        env_config,
        STATION,
        int(year),
        int(config.get("seed", 0)),
        f"{STATION}_{year}_04301_policy_sensitivity",
        evaluation=True,
    )


def decode_action(env: Any, action: int) -> dict[str, float | str]:
    grid = getattr(env, "grid", None)
    if isinstance(grid, list) and 0 <= int(action) < len(grid):
        raw = dict(grid[int(action)])
        return {
            "action_label": f"I{float(raw.get('amir', 0.0)):g}_N{float(raw.get('anfer', 0.0)):g}",
            "action_irrigation_mm": float(raw.get("amir", 0.0)),
            "action_n_kg_ha": float(raw.get("anfer", 0.0)),
        }
    return {"action_label": str(action), "action_irrigation_mm": np.nan, "action_n_kg_ha": np.nan}


def masked_probs(model: Any, obs: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    mask_arr = np.asarray(mask, dtype=bool).reshape(1, -1)
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
        try:
            dist = model.policy.get_distribution(obs_tensor, action_masks=mask_arr)
        except TypeError:
            dist = model.policy.get_distribution(obs_tensor)
            if hasattr(dist, "apply_masking"):
                dist.apply_masking(mask_arr)
        probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
        return probs.astype(float)
    except Exception:
        return None


def predict_action(model: Any, obs: np.ndarray, mask: np.ndarray) -> tuple[int, np.ndarray | None]:
    action, _ = model.predict(
        np.asarray(obs, dtype=np.float32),
        action_masks=np.asarray(mask, dtype=bool),
        deterministic=True,
    )
    return int(np.asarray(action).item()), masked_probs(model, obs, mask)


def collect_policy_states(model: Any, config: dict[str, Any], env_config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], str]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    rows: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    names: list[str] = []
    name_source = "not_found"

    for year in YEARS:
        env = make_env(config, env_config, year)
        try:
            obs, info = env.reset()
            if not names:
                raw_names, raw_name_source = first_name_list(env)
                obs_size_now = int(np.asarray(obs, dtype=np.float32).reshape(-1).size)
                names, expansion_note = expand_observation_names(raw_names, obs_size_now)
                name_source = f"{raw_name_source};{expansion_note}"
            done = False
            step_index = 0
            while not done and step_index < int(config["runtime"]["max_steps"]):
                obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                ctx = latest(env, obs_arr, info)
                dap_float = scalar(ctx.get("dap", np.nan), np.nan)
                dap = int(round(dap_float)) if np.isfinite(dap_float) and dap_float > 0 else step_index + 1
                mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                action, probs = predict_action(model, obs_arr, mask)
                if dap in AUDIT_DAPS:
                    decoded = decode_action(env, action)
                    rec_id = f"{year}_dap{dap}_step{step_index}"
                    state_records.append(
                        {
                            "state_id": rec_id,
                            "year": int(year),
                            "dap": int(dap),
                            "step_index": int(step_index),
                            "obs": obs_arr.copy(),
                            "mask": mask.copy(),
                            "baseline_action": int(action),
                            "baseline_probs": None if probs is None else probs.copy(),
                            "env_grid": getattr(env, "grid", None),
                        }
                    )
                    row = {
                        "state_id": rec_id,
                        "year": int(year),
                        "dap": int(dap),
                        "step_index": int(step_index),
                        "baseline_action": int(action),
                        **decoded,
                        "valid_action_count": int(mask.sum()),
                        "mask_true_indices": ",".join(map(str, np.where(mask)[0].tolist())),
                    }
                    name_to_idx = {str(name).lower(): idx for idx, name in enumerate(names)}
                    for key in ["tmax", "srad", "swfac", "nstres", "grnwt", "topwt", "totir", "dap"]:
                        idx = name_to_idx.get(key)
                        row[f"policy_obs_{key}"] = float(obs_arr[idx]) if idx is not None and idx < len(obs_arr) else np.nan
                    for key in ["rain", "tmin"]:
                        row[f"context_{key}"] = scalar(ctx.get(key, np.nan), np.nan)
                    rows.append(row)

                obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                step_index += 1
        finally:
            env.close()

    return pd.DataFrame(rows), state_records, names, name_source


def resolve_indices(names: list[str], obs_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx in range(obs_size):
        name = names[idx] if idx < len(names) else ""
        lname = str(name).lower()
        is_soil_water = lname == "sw" or lname.startswith("sw_")
        rows.append(
            {
                "index": idx,
                "name": name,
                "lower_name": lname,
                "is_soil_water": is_soil_water,
                "is_swfac": lname == "swfac",
                "is_nstres": lname == "nstres",
                "is_tmax": lname == "tmax",
                "is_srad": lname == "srad",
                "is_rain": lname == "rain",
                "is_tmin": lname == "tmin",
            }
        )
    if not names or len(names) != obs_size:
        # Keep the table explicit rather than pretending we know the semantic names.
        for row in rows:
            row.update(
                {
                    "is_soil_water": False,
                    "is_swfac": False,
                    "is_nstres": False,
                    "is_tmax": False,
                    "is_srad": False,
                    "is_rain": False,
                    "is_tmin": False,
                }
            )
    return pd.DataFrame(rows)


def empirical_quantiles(state_records: list[dict[str, Any]], indices: list[int]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not state_records:
        return out
    mat = np.vstack([np.asarray(x["obs"], dtype=float).reshape(-1) for x in state_records])
    for idx in indices:
        series = pd.Series(mat[:, idx]).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        out[int(idx)] = {
            "p05": float(series.quantile(0.05)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return out


def make_perturbations(obs: np.ndarray, var_df: pd.DataFrame, q: dict[int, dict[str, float]]) -> list[tuple[str, np.ndarray, str]]:
    base = np.asarray(obs, dtype=np.float32).reshape(-1)
    variants: list[tuple[str, np.ndarray, str]] = [("baseline", base.copy(), "no perturbation")]

    soil_idxs = var_df[var_df["is_soil_water"]]["index"].astype(int).tolist()
    if soil_idxs:
        dry = base.copy()
        wet = base.copy()
        for idx in soil_idxs:
            if idx in q:
                dry[idx] = q[idx]["p05"]
                wet[idx] = q[idx]["p95"]
        variants.append(("dry_soil", dry, "set soil-water dimensions to empirical p05 at audited states"))
        variants.append(("wet_soil", wet, "set soil-water dimensions to empirical p95 at audited states"))

    for _, row in var_df[var_df["is_swfac"]].iterrows():
        idx = int(row["index"])
        high = base.copy()
        high[idx] = max(float(high[idx]), 0.30)
        variants.append(("high_swfac", high, f"set swfac index {idx} to at least 0.30"))

    for _, row in var_df[var_df["is_nstres"]].iterrows():
        idx = int(row["index"])
        high = base.copy()
        high[idx] = max(float(high[idx]), 0.20)
        variants.append(("high_nstres", high, f"set nstres index {idx} to at least 0.20"))

    for _, row in var_df[var_df["is_tmax"]].iterrows():
        idx = int(row["index"])
        hot = base.copy()
        hot[idx] = float(hot[idx]) + 5.0
        variants.append(("hot_tmax", hot, f"add 5.0 to tmax index {idx}"))

    for _, row in var_df[var_df["is_srad"]].iterrows():
        idx = int(row["index"])
        low = base.copy()
        high = base.copy()
        low[idx] = float(low[idx]) * 0.80
        high[idx] = float(high[idx]) * 1.20
        variants.append(("low_srad", low, f"multiply srad index {idx} by 0.80"))
        variants.append(("high_srad", high, f"multiply srad index {idx} by 1.20"))

    return variants


def audit_sensitivity(model: Any, state_records: list[dict[str, Any]], var_df: pd.DataFrame) -> pd.DataFrame:
    if not state_records:
        return pd.DataFrame()
    all_idxs = var_df["index"].astype(int).tolist()
    q = empirical_quantiles(state_records, all_idxs)
    rows: list[dict[str, Any]] = []
    for rec in state_records:
        base_action = int(rec["baseline_action"])
        base_probs = rec["baseline_probs"]
        grid = rec.get("env_grid")
        for scenario, obs_variant, description in make_perturbations(rec["obs"], var_df, q):
            action, probs = predict_action(model, obs_variant, rec["mask"])
            decoded = decode_action(type("GridCarrier", (), {"grid": grid})(), action)
            base_decoded = decode_action(type("GridCarrier", (), {"grid": grid})(), base_action)
            row = {
                "state_id": rec["state_id"],
                "year": int(rec["year"]),
                "dap": int(rec["dap"]),
                "step_index": int(rec["step_index"]),
                "scenario": scenario,
                "perturbation": description,
                "baseline_action": base_action,
                "perturbed_action": int(action),
                "action_changed_vs_baseline": bool(action != base_action),
                "baseline_action_label": base_decoded["action_label"],
                "perturbed_action_label": decoded["action_label"],
                "baseline_irrigation_mm": base_decoded["action_irrigation_mm"],
                "baseline_n_kg_ha": base_decoded["action_n_kg_ha"],
                "perturbed_irrigation_mm": decoded["action_irrigation_mm"],
                "perturbed_n_kg_ha": decoded["action_n_kg_ha"],
                "valid_action_count": int(np.asarray(rec["mask"], dtype=bool).sum()),
            }
            if probs is not None:
                row["perturbed_argmax_prob"] = float(probs[action])
                row["baseline_action_prob_under_perturbed_obs"] = float(probs[base_action])
            if base_probs is not None:
                row["baseline_argmax_prob"] = float(base_probs[base_action])
                if probs is not None and len(probs) == len(base_probs):
                    row["prob_l1_distance_vs_baseline"] = float(np.abs(probs - base_probs).sum())
                    eps = 1e-12
                    row["prob_kl_baseline_to_perturbed"] = float(
                        np.sum((base_probs + eps) * np.log((base_probs + eps) / (probs + eps)))
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame(), "no_audit_rows"
    nonbase = detail[detail["scenario"].ne("baseline")].copy()
    by_scenario = (
        nonbase.groupby("scenario", as_index=False)
        .agg(
            n_states=("action_changed_vs_baseline", "size"),
            n_changed=("action_changed_vs_baseline", "sum"),
            change_rate=("action_changed_vs_baseline", "mean"),
            mean_prob_l1=("prob_l1_distance_vs_baseline", "mean"),
            mean_prob_kl=("prob_kl_baseline_to_perturbed", "mean"),
            n_irrigation_action=("perturbed_irrigation_mm", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            n_n_action=("perturbed_n_kg_ha", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
        .sort_values("scenario")
    )
    by_dap = (
        nonbase.groupby(["dap", "scenario"], as_index=False)
        .agg(
            n_states=("action_changed_vs_baseline", "size"),
            n_changed=("action_changed_vs_baseline", "sum"),
            change_rate=("action_changed_vs_baseline", "mean"),
            mean_prob_l1=("prob_l1_distance_vs_baseline", "mean"),
        )
        .sort_values(["dap", "scenario"])
    )
    overall_change = float(nonbase["action_changed_vs_baseline"].mean()) if len(nonbase) else 0.0
    mean_l1 = float(pd.to_numeric(nonbase.get("prob_l1_distance_vs_baseline", pd.Series(dtype=float)), errors="coerce").mean())
    if overall_change >= 0.20:
        branch = "A_deterministic_action_sensitivity_detected"
    elif overall_change > 0.0 or (np.isfinite(mean_l1) and mean_l1 >= 0.05):
        branch = "B_probability_or_partial_action_sensitivity_detected"
    else:
        branch = "C_weak_or_no_policy_input_sensitivity_detected"
    return by_scenario, by_dap, branch


def preflight() -> dict[str, Any]:
    ensure_dirs()
    config = run04210.load_config()
    run04210.patch_base_module(OUT, DOC, 100_000, [CHECKPOINT])
    split = base03222.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    split_sub = split[split["station_code"].astype(str).eq(STATION)]
    result = {
        "task": TASK,
        "mode": "dry_run_or_preflight",
        "station": STATION,
        "years": YEARS,
        "audit_daps": AUDIT_DAPS,
        "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "model_exists": MODEL_PATH.exists(),
        "prompt_path": PROMPT.relative_to(ROOT).as_posix(),
        "prompt_exists": PROMPT.exists(),
        "lowIC_input_root": run04210.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": run04210.LOWIC_INPUT_ROOT.exists(),
        "binary_irrigation_levels": run04210.BINARY_IRRIGATION_LEVELS,
        "binary_nitrogen_levels": run04210.BINARY_NITROGEN_LEVELS,
        "split_years_available": split_sub["year"].astype(int).tolist(),
        "config_reward": config.get("reward", {}),
        "config_action_safety": config.get("action_safety", {}),
        "next_step_allowed": bool(MODEL_PATH.exists() and PROMPT.exists() and run04210.LOWIC_INPUT_ROOT.exists()),
    }
    (OUT / "043_01_preflight.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def write_record(result: dict[str, Any], context: pd.DataFrame, var_df: pd.DataFrame, invisible: pd.DataFrame, by_scenario: pd.DataFrame, by_dap: pd.DataFrame) -> None:
    lines = [
        f"# {TASK}",
        "",
        "## 结论先说",
        "",
        f"- 分支：`{result['branch']}`。",
        "- 本任务没有训练、没有调参、没有修改 reward、没有修改 DSSAT 输入。",
        "- 审计对象是 042_15 冻结口径中的 042_11 ckpt25k binary-timing MaskablePPO。",
        "- 这一步检查的是：冻结 PPO 在同一 action mask 下，对已有 observation 变量的局部扰动是否改变动作概率或确定性动作。",
        "",
        "## 为什么这一步能支撑“完美天气预报”的论文叙事",
        "",
        "- 如果当前 observation 中没有 rain/tmin/未来7天降雨等信息，PPO 不可能直接对这些变量作出响应。",
        "- 如果已有的 tmax/srad/soil/stress 扰动也只带来弱响应，说明现有输入结构更容易学成阶段/预算模板。",
        "- 因此，后续加入历史天气构造的完美天气预报窗口，可被表述为改变决策信息结构，而不是事后调参。",
        "",
        "## 固定对象",
        "",
        f"- 模型：`{result['model_path']}`",
        f"- 输入：`{result['input_root']}`",
        f"- 年份：`{YEARS[0]}–{YEARS[-1]}`",
        f"- DAP：`{AUDIT_DAPS}`",
        "",
        "## 当前 policy observation 中不可直接响应的天气变量",
        "",
        md_table(invisible),
        "",
        "## observation 变量清单",
        "",
        md_table(var_df, max_rows=80),
        "",
        "## 抽样状态上下文",
        "",
        md_table(context, max_rows=80),
        "",
        "## 按扰动类型汇总",
        "",
        md_table(by_scenario, max_rows=80),
        "",
        "## 按 DAP 和扰动类型汇总",
        "",
        md_table(by_dap, max_rows=120),
        "",
        "## 输出文件",
        "",
        f"- 详细表：`{result['detail_csv']}`",
        f"- 扰动汇总：`{result['by_scenario_csv']}`",
        f"- DAP 汇总：`{result['by_dap_csv']}`",
        f"- observation 变量：`{result['observation_variables_csv']}`",
        f"- 不可见天气变量：`{result['unobservable_weather_csv']}`",
        "",
        "## 边界",
        "",
        "- 这是 policy 局部敏感性审计，不等同于 DSSAT 物理反事实。",
        "- 若某扰动导致 action 改变，只说明冻结 PPO 对该输入维度有局部响应；是否提高产量或效率需要后续季节回放验证。",
        "- 若某扰动不改变 action，也不能说明该变量农学上不重要，只能说明当前 checkpoint 没有把它强烈用于动作选择。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit() -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    pre = preflight()
    if not pre["next_step_allowed"]:
        raise RuntimeError(f"043_01 preflight failed: {json.dumps(pre, ensure_ascii=False)}")

    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = run04210.LOWIC_INPUT_ROOT
    try:
        config = run04210.load_config()
        run04210.patch_base_module(OUT, DOC, 100_000, [CHECKPOINT])
        env_config = build_eval_config(config)
        model = MaskablePPO.load(str(MODEL_PATH), device="cpu")
        context_df, state_records, names, name_source = collect_policy_states(model, config, env_config)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root

    if not state_records:
        raise RuntimeError("未抽取到任何 policy state；请检查 AUDIT_DAPS 或环境 max_steps。")
    obs_size = int(np.asarray(state_records[0]["obs"]).reshape(-1).size)
    var_df = resolve_indices(names, obs_size)
    var_df["name_source"] = name_source
    detail = audit_sensitivity(model, state_records, var_df)
    by_scenario, by_dap, branch = summarize(detail)

    invisible = pd.DataFrame(
        [
            {
                "variable": name,
                "in_policy_observation": bool((var_df["lower_name"].astype(str) == name.lower()).any()),
                "interpretation": "current PPO can directly use it" if bool((var_df["lower_name"].astype(str) == name.lower()).any()) else "not directly visible to current 042_15 policy",
            }
            for name in INVISIBLE_WEATHER_NAMES
        ]
    )

    context_path = TAB / "043_01_sampled_policy_state_context.csv"
    var_path = TAB / "043_01_policy_observation_variables.csv"
    invisible_path = TAB / "043_01_unobservable_weather_variables.csv"
    detail_path = TAB / "043_01_policy_input_sensitivity_detail.csv"
    by_scenario_path = TAB / "043_01_policy_input_sensitivity_by_scenario.csv"
    by_dap_path = TAB / "043_01_policy_input_sensitivity_by_dap.csv"

    context_df.to_csv(context_path, index=False, encoding="utf-8-sig")
    var_df.to_csv(var_path, index=False, encoding="utf-8-sig")
    invisible.to_csv(invisible_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    by_scenario.to_csv(by_scenario_path, index=False, encoding="utf-8-sig")
    by_dap.to_csv(by_dap_path, index=False, encoding="utf-8-sig")

    result = {
        "task": TASK,
        "branch": branch,
        "training_run": False,
        "reward_changed": False,
        "dssat_input_changed": False,
        "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "input_root": run04210.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "checkpoint": CHECKPOINT,
        "years": YEARS,
        "audit_daps": AUDIT_DAPS,
        "observation_name_source": name_source,
        "observation_names_match_vector": bool(len(names) == obs_size),
        "detail_csv": detail_path.relative_to(ROOT).as_posix(),
        "by_scenario_csv": by_scenario_path.relative_to(ROOT).as_posix(),
        "by_dap_csv": by_dap_path.relative_to(ROOT).as_posix(),
        "observation_variables_csv": var_path.relative_to(ROOT).as_posix(),
        "unobservable_weather_csv": invisible_path.relative_to(ROOT).as_posix(),
        "sampled_context_csv": context_path.relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "043_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, context_df, var_df, invisible, by_scenario, by_dap)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only check paths/config; do not replay DSSAT/env.")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(preflight(), indent=2, ensure_ascii=False))
        return
    print(json.dumps(run_audit(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
