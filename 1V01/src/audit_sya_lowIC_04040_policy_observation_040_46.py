"""040_46: audit the exact policy observation used by 040_40 MaskablePPO.

This task does not train a model and does not modify reward/action constraints.
It creates the same SYA lowIC environment as 040_40, resets it, and records:
- policy observation shape and values;
- resolved observation-variable names, where discoverable;
- whether explicit DAP is present in the policy vector;
- possible time-proxy variables;
- action-mask size and valid-action count.
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

TASK = "040_46_sya_lowIC_04040_policy_observation_audit"
OUT = ROOT / "benchmark_results" / TASK
TAB = OUT / "tables"
CFG = OUT / "configs"
DOC = ROOT / "docs" / f"{TASK}_record.md"
STATION = "SYA"
AUDIT_YEARS = [2005, 2014]
TIME_PROXY_NAMES = {
    "dap", "doy", "date", "istage", "vstage", "dtt", "cumdtt",
    "topwt", "grnwt", "xlai", "rtdep", "wtdep", "totir",
}


def ensure_dirs() -> None:
    for p in [OUT, TAB, CFG, DOC.parent]:
        p.mkdir(parents=True, exist_ok=True)


def unwrap_chain(env: Any, max_depth: int = 20) -> list[Any]:
    chain: list[Any] = []
    seen: set[int] = set()
    cur = env
    for _ in range(max_depth):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        chain.append(cur)
        nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return chain


def first_name_list(env: Any) -> tuple[list[str], str]:
    candidates = [
        "observation_variables", "observation_names", "obs_names",
        "state_names", "observation_keys", "obs_keys",
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


def scalarize(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.size == 1:
        try:
            return float(arr.reshape(-1)[0])
        except Exception:
            return str(value)
    return str(value)


def inspect_year(config: dict[str, Any], env_config: dict[str, Any], year: int) -> dict[str, Any]:
    env = run04040.make_env_with_yield_guardrail(
        config, env_config, STATION, int(year), int(config.get("seed", 0)),
        f"{STATION}_{year}_04046_observation_audit", evaluation=True,
    )
    try:
        obs, info = env.reset()
        obs_arr = np.asarray(obs).reshape(-1)
        names, names_source = first_name_list(env)
        latest = direct_ppo.latest_observation_dict(env, obs, info)

        rows: list[dict[str, Any]] = []
        if len(names) == len(obs_arr):
            for idx, (name, value) in enumerate(zip(names, obs_arr)):
                rows.append({
                    "year": int(year), "index": idx, "name": name,
                    "value_at_reset": scalarize(value),
                    "is_explicit_dap": str(name).lower() == "dap",
                    "is_time_proxy": str(name).lower() in TIME_PROXY_NAMES,
                })
        else:
            for idx, value in enumerate(obs_arr):
                rows.append({
                    "year": int(year), "index": idx, "name": "",
                    "value_at_reset": scalarize(value),
                    "is_explicit_dap": False, "is_time_proxy": False,
                })

        mask = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
        result = {
            "station_code": STATION,
            "year": int(year),
            "observation_shape": list(np.asarray(obs).shape),
            "observation_size": int(obs_arr.size),
            "observation_names_count": int(len(names)),
            "observation_names_source": names_source,
            "names_match_vector": bool(len(names) == len(obs_arr)),
            "explicit_dap_in_named_policy_observation": bool(
                len(names) == len(obs_arr) and any(str(x).lower() == "dap" for x in names)
            ),
            "dap_index": next((i for i, x in enumerate(names) if str(x).lower() == "dap"), None),
            "latest_observation_dict_has_dap": "dap" in latest,
            "latest_observation_dict_dap": scalarize(latest.get("dap", np.nan)),
            "named_time_proxy_variables": [x for x in names if str(x).lower() in TIME_PROXY_NAMES],
            "action_mask_size": int(mask.size),
            "valid_action_count_at_reset": int(mask.sum()),
            "observation_space_repr": repr(env.observation_space),
            "action_space_repr": repr(env.action_space),
        }
        return {"summary": result, "rows": rows}
    finally:
        env.close()


def main() -> None:
    ensure_dirs()
    config = run04040.load_config()
    split = base03222.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selection = base03222.build_selection(split)
    selected = selection[
        selection["station_code"].astype(str).eq(STATION)
        & selection["year"].isin(AUDIT_YEARS)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No {STATION} audit years found: {AUDIT_YEARS}")

    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = run04040.LOWIC_INPUT_ROOT
    try:
        env_config = direct_ppo.build_env_config(config, selected)
        direct_ppo.write_yaml(env_config, CFG / "040_46_resolved_env_config.yaml")
        all_summaries: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        for year in AUDIT_YEARS:
            inspected = inspect_year(config, env_config, year)
            all_summaries.append(inspected["summary"])
            all_rows.extend(inspected["rows"])
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root

    summary_df = pd.DataFrame(all_summaries)
    obs_df = pd.DataFrame(all_rows)
    summary_path = TAB / "040_46_policy_observation_summary.csv"
    obs_path = TAB / "040_46_policy_observation_variables.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    obs_df.to_csv(obs_path, index=False, encoding="utf-8-sig")

    conclusive = bool(
        len(summary_df)
        and summary_df["names_match_vector"].all()
        and summary_df["explicit_dap_in_named_policy_observation"].all()
    )
    if conclusive:
        conclusion = "explicit_dap_present_in_policy_observation"
    elif len(summary_df) and summary_df["names_match_vector"].all():
        conclusion = "explicit_dap_absent_from_named_policy_observation"
    else:
        conclusion = "observation_names_unresolved_run_runtime_probe"

    result = {
        "task": TASK,
        "training_run": False,
        "reward_changed": False,
        "constraints_changed": False,
        "station": STATION,
        "years": AUDIT_YEARS,
        "conclusion": conclusion,
        "summary_csv": summary_path.relative_to(ROOT).as_posix(),
        "variables_csv": obs_path.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_46_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# {TASK}", "", "## 结论", "", f"- `{conclusion}`", "",
        "## 边界", "", "- 不训练。", "- 不修改 reward。", "- 不修改动作、安全层或 action mask。",
        "- 只检查 040_40 MaskablePPO 实际接收的 observation。", "",
        "## 汇总", "", summary_df.to_markdown(index=False), "",
        "## 输出", "", f"- `{summary_path.relative_to(ROOT).as_posix()}`",
        f"- `{obs_path.relative_to(ROOT).as_posix()}`", "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
