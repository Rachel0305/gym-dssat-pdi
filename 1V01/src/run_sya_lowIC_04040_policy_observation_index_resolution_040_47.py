"""040_47: SYA lowIC 040_40 checkpoint100k policy observation index resolution.

Follow-up to 040_46, which ended unresolved
(`observation_names_unresolved_run_runtime_probe`): the raw observation
vector has 25 dims, but `env_chain[0].observation_variables` only names 17
of them, so it was unclear
  (a) which raw index corresponds to which named variable,
  (b) what the extra ~8 dims are (wrapper-appended features, not raw
      gym_dssat_pdi state), and
  (c) whether any channel that *is* named actually varies across years at
      matched timesteps, i.e. whether the frozen policy can even see
      weather/year-distinguishing signal.

Scope discipline (per project convention):
  - Does NOT train.
  - Does NOT change reward, action mask, or action safety.
  - Does NOT re-run 040_40; only replays the frozen checkpoint100000 policy.
  - Read-only diagnostic. Any conclusion here should gate, not replace,
    the mask-geometry ablation this task was designed to unblock.

Known interface assumptions (verify on first run against your actual repo):
  - `ppo_evaluate.latest_observation_dict(env, obs, info)` and
    `ppo_evaluate.scalar(x, default)` exist and behave as used elsewhere in
    run_all_year_direct_action_safe_ppo.py.
  - Some layer in the env wrapper chain (env -> env.env -> env.env.env ...)
    exposes `.observation_variables` (a list[str]); 040_46 found this at
    `env_chain[0]`, but this script does not hard-code the index and instead
    searches the whole chain, recording which layer it found it at.
  - Checkpoint zip lives at
    benchmark_results/040_40_sya_lowIC_ppo_yield_guardrail_v3/models/SYA/
    SYA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip
    (per the base03222.model_path naming convention). Adjust
    MODEL_ZIP_OVERRIDE below if your actual path differs.
"""

from __future__ import annotations

import json
import shutil
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

import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as base04040
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
from ppo_evaluate import latest_observation_dict, scalar  # noqa: F401 (scalar kept for parity with other scripts)

TASK = "040_47_sya_lowIC_04040_policy_observation_index_resolution"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = base04040.STATION
SITE = "SY"
SEED = base03222.SEED
CHECKPOINT_STEP = 100_000
VALIDATION_YEARS = list(range(2014, 2024))
VALUE_MATCH_TOL = 1e-6

# Set this if the checkpoint is not where the naming convention predicts.
MODEL_ZIP_OVERRIDE: Path | None = None


def ensure_dirs() -> None:
    for rel in ["tables", "logs", "configs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def model_zip_path() -> Path:
    if MODEL_ZIP_OVERRIDE is not None:
        return MODEL_ZIP_OVERRIDE
    return (
        base04040.BASE_OUT
        / "models"
        / STATION
        / f"{STATION}_half_split_stress_aware_maskableppo_seed{SEED}_ckpt{CHECKPOINT_STEP}.zip"
    )


def build_env_chain(env: Any) -> list[Any]:
    chain = [env]
    seen = {id(env)}
    current = env
    while hasattr(current, "env"):
        nxt = current.env
        if id(nxt) in seen:
            break
        chain.append(nxt)
        seen.add(id(nxt))
        current = nxt
    return chain


def resolve_names(env: Any) -> tuple[list[Any], list[str], int]:
    chain = build_env_chain(env)
    names: list[str] = []
    source_layer = -1
    for idx, layer in enumerate(chain):
        candidate = getattr(layer, "observation_variables", None)
        if candidate:
            names = [str(n) for n in candidate]
            source_layer = idx
            break
    return chain, names, source_layer


def candidate_indices_for_snapshot(raw_obs: np.ndarray, named: dict[str, Any], names: list[str]) -> dict[str, set[int]]:
    """For one snapshot: which raw indices could hold each named value.

    Ambiguous on a single snapshot (ties happen, e.g. multiple zero
    channels). Intersecting this set across many snapshots (different
    years/DAPs) collapses it to the true index whenever the variable is not
    constantly degenerate.
    """
    out: dict[str, set[int]] = {}
    for name in names:
        if name not in named:
            continue
        try:
            target = float(named[name])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(target):
            continue
        out[name] = {i for i, v in enumerate(raw_obs) if np.isfinite(v) and abs(float(v) - target) <= VALUE_MATCH_TOL}
    return out


def rollout_one_year(model: Any, env: Any, station: str, year: int) -> tuple[list[dict[str, Any]], list[Any]]:
    obs, info = env.reset()
    chain, names, source_layer = resolve_names(env)
    rows: list[dict[str, Any]] = []
    step_idx = 0
    done = False
    while not done:
        named = latest_observation_dict(env, obs, info)
        raw = np.asarray(obs, dtype=float).flatten()
        row: dict[str, Any] = {
            "station_code": station,
            "year": int(year),
            "step_index": step_idx,
            "observation_size": int(raw.size),
            "observation_names_count": len(names),
            "observation_names_source_layer_index": source_layer,
            "observation_names_source_layer_class": type(chain[source_layer]).__name__ if source_layer >= 0 else None,
            "named_dap": named.get("dap"),
        }
        for i, v in enumerate(raw):
            row[f"raw_{i:02d}"] = float(v)
        for name in names:
            row[f"named_{name}"] = named.get(name)
        rows.append(row)

        action_masks = env.action_masks() if hasattr(env, "action_masks") else None
        if action_masks is not None:
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_idx += 1
    return rows, names


def resolve_index_name_map(all_rows: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    records = []
    for name in names:
        col = f"named_{name}"
        if col not in all_rows.columns:
            continue
        candidate_sets: list[set[int]] = []
        for _, r in all_rows.iterrows():
            raw = np.array([r[f"raw_{i:02d}"] for i in range(r["observation_size"])], dtype=float)
            named_val = r[col]
            if named_val is None or (isinstance(named_val, float) and not np.isfinite(named_val)):
                continue
            try:
                target = float(named_val)
            except (TypeError, ValueError):
                continue
            candidate_sets.append({i for i, v in enumerate(raw) if np.isfinite(v) and abs(v - target) <= VALUE_MATCH_TOL})
        if not candidate_sets:
            records.append({"name": name, "resolved_index": None, "status": "no_finite_values_observed"})
            continue
        intersection = set.intersection(*candidate_sets) if candidate_sets else set()
        if len(intersection) == 1:
            records.append({"name": name, "resolved_index": next(iter(intersection)), "status": "unique_match"})
        elif len(intersection) == 0:
            records.append({"name": name, "resolved_index": None, "status": "no_consistent_match_check_scaling_or_derived_var"})
        else:
            records.append({"name": name, "resolved_index": None, "status": f"ambiguous_{len(intersection)}_candidates:{sorted(intersection)}"})
    return pd.DataFrame(records)


def cross_year_variability(all_rows: pd.DataFrame, size: int) -> pd.DataFrame:
    """For each raw index, at matched step_index (proxy for matched DAP),
    does the value vary across years, or is it constant/degenerate?"""
    records = []
    for i in range(size):
        col = f"raw_{i:02d}"
        if col not in all_rows.columns:
            continue
        by_step = all_rows.groupby("step_index")[col].agg(["std", "mean", "count"])
        max_std_step = by_step["std"].idxmax() if not by_step.empty else None
        records.append(
            {
                "raw_index": i,
                "n_steps_with_multiple_years": int((by_step["count"] > 1).sum()),
                "max_cross_year_std_at_any_step": float(by_step["std"].max()) if not by_step.empty else np.nan,
                "step_index_of_max_std": int(max_std_step) if max_std_step is not None and np.isfinite(by_step["std"].max()) else None,
                "overall_std_across_all_rows": float(all_rows[col].std()),
                "carries_cross_year_signal": bool(by_step["std"].max() > VALUE_MATCH_TOL) if not by_step.empty else False,
            }
        )
    return pd.DataFrame(records)


def dap_index_sanity_check(all_rows: pd.DataFrame, resolved_map: pd.DataFrame) -> dict[str, Any]:
    dap_row = resolved_map[resolved_map["name"].eq("dap")]
    if dap_row.empty or pd.isna(dap_row.iloc[0]["resolved_index"]):
        return {"dap_index_resolved": False, "note": "dap could not be uniquely resolved to a raw index; see resolve_index_name_map status"}
    idx = int(dap_row.iloc[0]["resolved_index"])
    col = f"raw_{idx:02d}"
    checks = []
    for (station, year), g in all_rows.groupby(["station_code", "year"]):
        g = g.sort_values("step_index")
        diffs = g[col].diff().dropna()
        checks.append(bool((diffs.round(6) == 1.0).all()) if len(diffs) else False)
    return {
        "dap_index_resolved": True,
        "resolved_dap_raw_index": idx,
        "all_years_increment_by_one_each_step": bool(all(checks)) if checks else None,
        "n_years_checked": len(checks),
    }


def write_record(all_rows: pd.DataFrame, names: list[str], resolved_map: pd.DataFrame, variability: pd.DataFrame, dap_check: dict[str, Any]) -> None:
    named_channels_with_signal = 0
    for name in names:
        r = resolved_map[resolved_map["name"].eq(name)]
        if r.empty or pd.isna(r.iloc[0]["resolved_index"]):
            continue
        idx = int(r.iloc[0]["resolved_index"])
        v = variability[variability["raw_index"].eq(idx)]
        if not v.empty and bool(v.iloc[0]["carries_cross_year_signal"]):
            named_channels_with_signal += 1
    unnamed_with_signal = variability[~variability["raw_index"].isin(
        [int(x) for x in resolved_map["resolved_index"].dropna().tolist()]
    ) & variability["carries_cross_year_signal"]]

    lines = [
        f"# {TASK} 记录",
        "",
        "## 任务目的",
        "",
        "解决 040_46 遗留的 `observation_names_unresolved_run_runtime_probe`：",
        "确认 25 维原始观测向量里，哪些下标对应哪个命名变量、未命名的尾部维度是什么、",
        "以及哪些通道在跨年匹配步数上真的会变化（而不是同一份天气无关的常数）。",
        "",
        "## 边界",
        "",
        "- 不训练，不改 reward/mask/action，不重跑 040_40。",
        "- 只回放冻结的 040_40 checkpoint100000 策略。",
        "",
        "## 结论先说",
        "",
        f"- observation 总维度：`{all_rows['observation_size'].iloc[0] if not all_rows.empty else 'NA'}`；",
        f"  命名变量数：`{len(names)}`；未命名维度数：",
        f"  `{(all_rows['observation_size'].iloc[0] - len(names)) if not all_rows.empty else 'NA'}`。",
        f"- 命名变量里能唯一解析到具体下标的数量：`{int(resolved_map['resolved_index'].notna().sum())}/{len(names)}`。",
        f"- 命名且已解析下标的变量中，跨年在匹配步数上有真实变化（非退化常数）的数量："
        f"`{named_channels_with_signal}`。",
        f"- 未命名维度中仍检测到跨年信号的数量：`{len(unnamed_with_signal)}`"
        + (f"（下标：{sorted(unnamed_with_signal['raw_index'].tolist())}）" if len(unnamed_with_signal) else ""),
        f"- DAP 下标核验：`{dap_check}`",
        "",
        "## 变量名 -> 原始下标 解析表",
        "",
        resolved_map.to_markdown(index=False) if not resolved_map.empty else "无记录。",
        "",
        "## 每个原始下标的跨年变异性",
        "",
        variability.to_markdown(index=False) if not variability.empty else "无记录。",
        "",
        "## 解释边界",
        "",
        "- 本任务只做只读诊断，不对 mask 几何或 reward 做任何调整。",
        "- 若某个已解析下标 `carries_cross_year_signal=False`，只能说明该变量在本checkpoint的",
        "  验证年份匹配步数上退化为常数，不能反推 policy 训练全程都看不到该信号。",
        "- 若未命名维度里发现有跨年信号的通道，需要回到 wrapper 源码确认它具体是什么工程特征",
        "  （例如 safety_state 剩余额度、mask 标志位等），本表不猜测具体含义。",
        "- 这个结果决定的是任务2（observation 疑点）是否已解决，不直接决定要不要做 mask 几何消融；",
        "  mask 几何消融是否值得做，取决于本表是否排除了“看不见天气”这个假设。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_env_config_for_station(config: dict[str, Any]) -> dict[str, Any]:
    """Mirror base03222.main()'s selection -> env_config construction, but
    read-only and scoped to STATION, so this script never mutates
    base03222's module-level state (SITES/OUT/etc.) the way patch_base_module
    does for actual training runs.
    """
    split = pd.read_csv(base03222.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    split = split[split["station_code"].eq(STATION)].sort_values("year").reset_index(drop=True)
    pool = pd.read_csv(base03222.POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selection = pool.merge(split[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selection["selected_for_train"] = selection["split"].eq("train")
    selection["selected_for_eval"] = True
    selection["selection_reason"] = "040_47_observation_index_resolution_readonly"
    return base03222.direct_ppo.build_env_config(config, selection)


def main() -> None:
    ensure_dirs()
    from sb3_contrib import MaskablePPO

    zip_path = model_zip_path()
    if not zip_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {zip_path}. Set MODEL_ZIP_OVERRIDE if the path differs.")
    model = MaskablePPO.load(str(zip_path))

    config = base04040.load_config()
    env_config = build_env_config_for_station(config)

    all_rows: list[dict[str, Any]] = []
    names_ref: list[str] = []
    for year in VALIDATION_YEARS:
        env = base04040.make_env_with_yield_guardrail(
            config, env_config, STATION, year, SEED, f"{STATION}_{year}_040_47_observation_audit", evaluation=True
        )
        rows, names = rollout_one_year(model, env, STATION, year)
        if not names_ref:
            names_ref = names
        all_rows.extend(rows)
        if hasattr(env, "close"):
            env.close()

    all_rows_df = pd.DataFrame(all_rows)
    all_rows_df.to_csv(OUT / "tables" / "040_47_raw_observation_by_step.csv", index=False, encoding="utf-8-sig")

    resolved_map = resolve_index_name_map(all_rows_df, names_ref)
    resolved_map.to_csv(OUT / "tables" / "040_47_name_to_index_resolution.csv", index=False, encoding="utf-8-sig")

    size = int(all_rows_df["observation_size"].iloc[0]) if not all_rows_df.empty else 0
    variability = cross_year_variability(all_rows_df, size)
    variability.to_csv(OUT / "tables" / "040_47_cross_year_index_variability.csv", index=False, encoding="utf-8-sig")

    dap_check = dap_index_sanity_check(all_rows_df, resolved_map)

    write_record(all_rows_df, names_ref, resolved_map, variability, dap_check)

    result = {
        "task": TASK,
        "checkpoint": zip_path.relative_to(ROOT).as_posix() if zip_path.is_relative_to(ROOT) else str(zip_path),
        "validation_years": VALIDATION_YEARS,
        "observation_size": size,
        "named_count": len(names_ref),
        "dap_index_check": dap_check,
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_47_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
