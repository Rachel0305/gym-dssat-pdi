"""162: read-only response audit for the 161 SYA weather-augmentation pilot.

This audit never trains and never edits the 160 weather scenario bank.  It
replays the frozen 161 checkpoint on the six SYA 2005 weather variants, then
compares the resulting action trajectories by DAP.  The comparison is a
paired-episode replay, not a same-state counterfactual: weather is allowed to
change the DSSAT state before later actions are compared.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_161_sya_weather_aug_pilot_smoke2k as pilot


TASK = "162_sya_weather_aug_response_audit"
RUN = ROOT / "benchmark_results" / "161_sya_weather_aug_pilot_smoke2k"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
CHECKPOINT = 2000
SEED = 0
EXPECTED_VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
IRRIGATION_GRID = {0.0, 15.0, 30.0, 45.0}
NITROGEN_GRID = {0.0, 40.0, 80.0, 120.0}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def action_columns(frame: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("safe_action_amir", "safe_action_anfer"),
        ("action_amir", "action_anfer"),
    ]
    for irrigation, nitrogen in candidates:
        if irrigation in frame.columns and nitrogen in frame.columns:
            return irrigation, nitrogen
    raise KeyError(f"No action columns found in {list(frame.columns)}")


def action_signature(frame: pd.DataFrame, irrigation: str, nitrogen: str) -> str:
    dap = pd.to_numeric(frame["dap"], errors="coerce")
    irr = pd.to_numeric(frame[irrigation], errors="coerce").fillna(0.0)
    n = pd.to_numeric(frame[nitrogen], errors="coerce").fillna(0.0)
    rows = []
    for d, i, fert in zip(dap, irr, n):
        if i > 1e-9 or fert > 1e-9:
            rows.append(f"DAP{int(round(d))} I{i:g}/N{fert:g}")
    return "; ".join(rows)


def patch_runtime(out: Path, doc: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Patch the same runtime contract used by the 161 runner."""

    old_engine = {
        key: copy.deepcopy(getattr(engine, key))
        for key in [
            "TASK_ID",
            "TASK_NAME",
            "BASE_OUT",
            "BASE_DOC",
            "PROMPT",
            "LOWIC_INPUT_ROOT",
            "STATION",
            "SITES",
            "BINARY_IRRIGATION_LEVELS",
            "BINARY_NITROGEN_LEVELS",
        ]
    }
    old_base = {
        "SITES": list(base03222.SITES),
        "SITE_NAMES": dict(base03222.SITE_NAMES),
        "OUT": base03222.OUT,
        "DOC": base03222.DOC,
        "PROMPT": base03222.PROMPT,
        "TOTAL_TIMESTEPS": base03222.TOTAL_TIMESTEPS,
        "CHECKPOINT_STEPS": list(base03222.CHECKPOINT_STEPS),
        "load_config": base03222.load_config,
    }
    old_wrapper = base03222.base.StressAwareDiscreteWrapper
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT

    engine.TASK_ID = "162"
    engine.TASK_NAME = TASK
    engine.BASE_OUT = out
    engine.BASE_DOC = doc
    engine.PROMPT = pilot.PROMPT
    engine.LOWIC_INPUT_ROOT = pilot.target_root(cfg)
    engine.STATION = "SYA"
    engine.SITES = ["SYA"]
    engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
    engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
    engine.patch_base_module(out, doc, CHECKPOINT, [CHECKPOINT])
    base03222.SITES = ["SYA"]
    base03222.SITE_NAMES.clear()
    base03222.SITE_NAMES.update({"SYA": "SY"})
    base03222.OUT = out
    base03222.DOC = doc
    base03222.PROMPT = pilot.PROMPT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = pilot.target_root(cfg)

    old = {
        "engine": old_engine,
        "base": old_base,
        "wrapper": old_wrapper,
        "input_root": old_input_root,
    }
    return old


def restore_runtime(old: dict[str, Any]) -> None:
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["input_root"]
    base03222.base.StressAwareDiscreteWrapper = old["wrapper"]
    for key, value in old["base"].items():
        if key == "SITE_NAMES":
            base03222.SITE_NAMES.clear()
            base03222.SITE_NAMES.update(value)
        else:
            setattr(base03222, key, value)
    for key, value in old["engine"].items():
        setattr(engine, key, value)


def run_audit() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    config_path = RUN / "configs" / "161_sya_weather_aug_pilot_smoke2k.json"
    inventory_path = RUN / "evaluation" / "042_10_training_checkpoint_inventory.csv"
    split_path = RUN / "configs" / "161_training_validation_split.csv"
    cfg = pilot.load_config(config_path)
    inventory = read_frame(inventory_path)
    split = read_frame(split_path)
    train = split[split["split"].eq("train")].copy()
    train = train.sort_values(["source_year", "year"]).reset_index(drop=True)
    if train["weather_variant"].tolist() != EXPECTED_VARIANTS:
        raise RuntimeError(f"Unexpected training variants: {train['weather_variant'].tolist()}")
    model_rows = inventory[inventory["checkpoint_step"].astype(int).eq(CHECKPOINT)]
    if len(model_rows) != 1:
        raise RuntimeError(f"Expected one checkpoint row, found {len(model_rows)}")
    model_row = model_rows.iloc[0].copy()
    if not Path(ROOT / str(model_row["model_path"])).exists():
        raise FileNotFoundError(ROOT / str(model_row["model_path"]))

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "daily_outputs" / "SYA").mkdir(parents=True)
    shutil.copy2(config_path, OUT / "configs" / config_path.name)
    shutil.copy2(split_path, OUT / "configs" / split_path.name)
    (OUT / "configs" / "162_audit_contract.json").write_text(
        json.dumps(
            {
                "task": TASK,
                "source_task": "161_sya_weather_aug_pilot_smoke2k",
                "checkpoint_step": CHECKPOINT,
                "seed": SEED,
                "variants": EXPECTED_VARIANTS,
                "training_executed": False,
                "same_state_counterfactual": False,
                "comparison": "paired_episode_replay_by_DAP",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    old = patch_runtime(OUT, DOC, cfg)
    evaluation_rows: list[dict[str, Any]] = []
    try:
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, train)
        for _, scenario in train.iterrows():
            pseudo_year = int(scenario["year"])
            row = model_row.copy()
            row["station_code"] = "SYA"
            result = base03222.evaluate_checkpoint(runtime_config, env_config, row, pseudo_year)
            result.update(
                {
                    "source_year": int(scenario["source_year"]),
                    "pseudo_year": pseudo_year,
                    "weather_variant": str(scenario["weather_variant"]),
                    "weather_file": str(scenario["weather_file"]),
                }
            )
            evaluation_rows.append(result)
    finally:
        restore_runtime(old)

    metrics = pd.DataFrame(evaluation_rows).sort_values("pseudo_year").reset_index(drop=True)
    metrics.to_csv(OUT / "162_scenario_episode_metrics.csv", index=False, encoding="utf-8-sig")

    daily_by_variant: dict[str, pd.DataFrame] = {}
    for _, record in metrics.iterrows():
        variant = str(record["weather_variant"])
        daily_path = OUT / "daily_outputs" / "SYA" / f"SYA_{int(record['pseudo_year'])}_seed{SEED}_ckpt{CHECKPOINT}_daily.csv"
        daily = read_frame(daily_path)
        irrigation, nitrogen = action_columns(daily)
        daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce").astype(int)
        daily[irrigation] = pd.to_numeric(daily[irrigation], errors="coerce").fillna(0.0)
        daily[nitrogen] = pd.to_numeric(daily[nitrogen], errors="coerce").fillna(0.0)
        daily_by_variant[variant] = daily

    baseline = daily_by_variant["original"]
    base_i, base_n = action_columns(baseline)
    action_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant, daily in daily_by_variant.items():
        irr, fert = action_columns(daily)
        pair = baseline[["dap", base_i, base_n]].rename(columns={base_i: "baseline_irrigation", base_n: "baseline_n"}).merge(
            daily[["dap", irr, fert]].rename(columns={irr: "variant_irrigation", fert: "variant_n"}),
            on="dap",
            how="inner",
        )
        pair["changed"] = (
            ~np.isclose(pair["baseline_irrigation"], pair["variant_irrigation"])
            | ~np.isclose(pair["baseline_n"], pair["variant_n"])
        )
        pair["changed_after_dap1"] = pair["changed"] & pair["dap"].gt(1)
        pair["weather_variant"] = variant
        action_rows.extend(pair.to_dict(orient="records"))
        legal = bool(
            daily[irr].round(6).isin(IRRIGATION_GRID).all()
            and daily[fert].round(6).isin(NITROGEN_GRID).all()
        )
        summary_rows.append(
            {
                "checkpoint_step": CHECKPOINT,
                "weather_variant": variant,
                "source_year": int(metrics.loc[metrics["weather_variant"].eq(variant), "source_year"].iloc[0]),
                "pseudo_year": int(metrics.loc[metrics["weather_variant"].eq(variant), "pseudo_year"].iloc[0]),
                "matched_decision_states": int(len(pair)),
                "action_changed_rows": int(pair["changed"].sum()),
                "action_change_rate": float(pair["changed"].mean()) if len(pair) else np.nan,
                "action_changed_rows_after_dap1": int(pair["changed_after_dap1"].sum()),
                "action_change_rate_after_dap1": float(pair.loc[pair["dap"].gt(1), "changed"].mean()) if int((pair["dap"] > 1).sum()) else np.nan,
                "all_actions_on_grid": legal,
                "unique_nonzero_action_pairs": int(
                    daily.loc[(daily[irr] > 1e-9) | (daily[fert] > 1e-9), [irr, fert]].drop_duplicates().shape[0]
                ),
                "action_sequence": action_signature(daily, irr, fert),
            }
        )

    action_detail = pd.DataFrame(action_rows)
    action_summary = pd.DataFrame(summary_rows)
    action_detail.to_csv(OUT / "162_action_pair_detail.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(OUT / "162_action_change_summary.csv", index=False, encoding="utf-8-sig")

    non_original = action_summary[action_summary["weather_variant"].ne("original")]
    response_passed = bool(
        not non_original.empty
        and bool(non_original["all_actions_on_grid"].all())
        and bool(non_original["action_change_rate_after_dap1"].fillna(0.0).max() > 0.01)
    )
    summary = {
        "status": "completed_read_only_paired_episode_replay",
        "task": TASK,
        "source_task": "161_sya_weather_aug_pilot_smoke2k",
        "model": str(model_row["model_path"]),
        "checkpoint_step": CHECKPOINT,
        "training_executed": False,
        "same_state_counterfactual": False,
        "comparison": "paired_episode_replay_by_DAP",
        "scenario_count": int(len(action_summary)),
        "decision_state_count_total": int(len(action_detail)),
        "response_threshold": "max(non_original action_change_rate_after_dap1) > 1% and all actions legal",
        "response_passed": response_passed,
        "outputs": {
            "episode_metrics": rel(OUT / "162_scenario_episode_metrics.csv"),
            "action_detail": rel(OUT / "162_action_pair_detail.csv"),
            "action_summary": rel(OUT / "162_action_change_summary.csv"),
        },
    }
    (OUT / "162_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {TASK}",
        "",
        "- 只读加载 161 的 2K checkpoint；不训练、不修改 160 天气库。",
        "- 在 SYA 2005 的六个天气情景上分别运行 DSSAT episode。",
        "- 动作变化率是按 DAP 对齐的 paired-episode replay，不是同状态天气反事实。",
        "- 只有在非 original 情景的 DAP>1 动作变化率最高超过 1%，且动作均在合法网格内时，才通过本轮响应门槛。",
        "",
        "## 情景汇总",
        "",
        action_summary.to_markdown(index=False),
        "",
        "## 结果边界",
        "",
        "- 本审计判断天气情景是否伴随管理行为变化，不判断 PPO 是否优于任何基线。",
        "- 由于不同天气会改变 DSSAT 状态，这里的动作差异不能单独解释为纯粹的同状态因果效应。",
        f"- response_passed: `{response_passed}`",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, default=CHECKPOINT)
    args = parser.parse_args()
    if int(args.checkpoint) != CHECKPOINT:
        raise ValueError("This low-cost audit is fixed to the 161 checkpoint 2000")
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
