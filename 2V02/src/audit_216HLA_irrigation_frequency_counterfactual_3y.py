"""Small counterfactual audit for the 216HLA frequent-small-irrigation policy.

Question:
    Does the 200K PPO policy's frequent 15 mm irrigation pattern matter when
    total irrigation, total nitrogen, weather, soil, cultivar, and evaluation
    code are held fixed?

Design:
    - PPO: deterministic replay of the saved 216HLA checkpoint-200000.
    - Fewer-event control: the same 240 mm irrigation total is consolidated
      from the observed PPO 16 x 15 mm pattern into 8 x 30 mm events.  The
      event dates are pairwise midpoints of the PPO dates, so the season
      window is kept approximately the same.
    - Nitrogen remains 3 x 80 kg N/ha at the PPO dates (DAP 0, 8, 15).
    - Years are one dry, one normal, and one wet original validation year.

This is an evaluation-only audit.  It does not resume training and writes to
a fresh output directory.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
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
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as static_baseline
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
from ppo_action_safety import normalize_action
from run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04 import ProjectProfitDiscreteWrapper
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env

from audit_216HLA_ckpt200k_validation import (
    CHECKPOINT,
    CONFIG_PATH,
    MODEL_PATH,
    PROMPT,
    SITE,
    STATION,
    TASK_ID,
    TASK_NAME,
    action_signature,
    load_054_helpers,
    restore_modules,
    setup_modules,
)


FIG054 = load_054_helpers()

AUDIT_RUN_NAME = "216HLA_irrigation_frequency_counterfactual_3y_ckpt200k_v2"
OUT = ROOT / "benchmark_results" / AUDIT_RUN_NAME
RECORD = OUT / "216HLA_irrigation_frequency_counterfactual_3y_record.md"

# Selected from the original 2014--2023 validation weather by seasonal rainfall:
# 2015 is dry, 2019 is normal, and 2018 is wet.
YEARS = [2015, 2019, 2018]
YEAR_LABELS = {2015: "dry", 2019: "normal", 2018: "wet"}
MAX_STEPS = 260

# The saved PPO replay shows 16 x 15 mm irrigation at these DAPs in every
# validation year.  Pairwise midpoint dates create a controlled 8 x 30 mm
# alternative without changing the approximate irrigation season window.
PPO_IRRIGATION_DAPS = [0, 9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100, 107]
FEWER_IRRIGATION_DAPS = [5, 20, 34, 48, 62, 76, 90, 104]
PPO_NITROGEN_DAPS = [0, 8, 15]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def zero_action(env: Any) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})


def fewer_event_schedule() -> dict[int, dict[str, float]]:
    schedule: dict[int, dict[str, float]] = {}
    for dap in FEWER_IRRIGATION_DAPS:
        schedule[int(dap)] = {"amir": 30.0, "anfer": 0.0}
    for dap in PPO_NITROGEN_DAPS:
        schedule.setdefault(int(dap), {"amir": 0.0, "anfer": 0.0})
        schedule[int(dap)]["anfer"] = 80.0
    return schedule


def replace_static_schedule_for_treatment(text: str, schedule: dict[int, dict[str, float]], planting_date: str, year: int) -> str:
    """Replace static rows while retaining DSSAT treatment IDs.

    In DSSAT's application tables, the first column is the treatment ID, not
    an event counter.  The existing generic helper is intentionally not
    modified here because this audit needs a single, explicit treatment-1
    contract for the validation replay.
    """
    irrigation_dates = []
    fertilizer_dates = []
    for dap, action in sorted(schedule.items()):
        date = static_baseline.yyddd_from_dap(planting_date, int(dap))
        if float(action.get("amir", 0.0)) > 1e-9:
            irrigation_dates.append((date, float(action["amir"])))
        if float(action.get("anfer", 0.0)) > 1e-9:
            fertilizer_dates.append((date, float(action["anfer"])))
    if not irrigation_dates:
        irrigation_dates = [(static_baseline.yyddd_from_dap(planting_date, 1), 0.0)]
    if not fertilizer_dates:
        fertilizer_dates = [(static_baseline.yyddd_from_dap(planting_date, 1), 0.0)]

    irrigation_rows_by_treatment: dict[int, list[str]] = {}
    for treatment in (1, 2, 3):
        irrigation_rows_by_treatment[treatment] = [
            f"{treatment:2d} {date} IR001 {amount:5.1f}" for date, amount in irrigation_dates
        ]
    fertilizer_rows = [
        f" 1 {date} FE005 AP002     5 {amount:6.1f}   -99   -99   -99   -99   -99 {year}"
        for date, amount in fertilizer_dates
    ]

    lines = text.splitlines()
    output: list[str] = []
    mode: str | None = None
    current_treatment = 1
    waiting_for_irrigation_treatment = False
    inserted_i = False
    inserted_f = False
    for line in lines:
        stripped = line.strip()
        if line.startswith("@I  EFIR"):
            waiting_for_irrigation_treatment = True
            mode = None
            output.append(line)
            continue
        if waiting_for_irrigation_treatment and stripped and not stripped.startswith("@") and not stripped.startswith("*"):
            match = re.match(r"^\s*(\d+)\s+", line)
            if match:
                current_treatment = int(match.group(1))
            waiting_for_irrigation_treatment = False
            output.append(line)
            continue
        if line.startswith("@I IDATE"):
            output.append(line)
            output.extend(irrigation_rows_by_treatment.get(current_treatment, irrigation_rows_by_treatment[1]))
            mode = "I"
            inserted_i = True
            continue
        if line.startswith("@F FDATE"):
            output.append(line)
            output.extend(fertilizer_rows)
            mode = "F"
            inserted_f = True
            continue
        if mode is not None:
            if stripped.startswith("@") or stripped.startswith("*"):
                mode = None
            elif stripped == "":
                continue
            elif re.match(r"^\s*\d+\s+\d{5}\b", line):
                continue
        output.append(line)
    if not inserted_i or not inserted_f:
        raise ValueError("渲染模板中缺少 DSSAT 静态灌溉或施肥表")
    return "\n".join(output) + "\n"


def save_snapshot(env: Any, target: Path) -> None:
    temporary = snapshot_from_env(env)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(temporary, target)


def metrics_for_snapshot(snapshot: Path, year: int, policy: str) -> dict[str, Any]:
    daily = FIG054.daily_from_snapshot(snapshot, year, policy)
    final_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
    metrics = FIG054.baseline.metrics_from_snapshot(snapshot, final_yield)
    irrigation = pd.to_numeric(daily["irrigation_executed_mm"], errors="coerce").fillna(0.0)
    nitrogen = pd.to_numeric(daily["nitrogen_executed_kg_ha"], errors="coerce").fillna(0.0)
    rain = pd.to_numeric(daily.get("rain", pd.Series(dtype=float)), errors="coerce")
    event_rows = daily[irrigation.gt(0) | nitrogen.gt(0)].copy()
    return {
        "station": STATION,
        "site": SITE,
        "year": int(year),
        "year_type": YEAR_LABELS[int(year)],
        "policy": policy,
        "grain_yield_kg_ha": final_yield,
        "actual_irrigation_mm_daily_sum": float(irrigation.sum()),
        "actual_nitrogen_kg_ha_daily_sum": float(nitrogen.sum()),
        "summary_irrigation_total": float(metrics["actual_irrigation_mm"]),
        "summary_nitrogen_total": float(metrics["actual_nitrogen_kg_ha"]),
        "etcp_mm": float(metrics["etcp_mm"]),
        "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
        "PFP_N_kg_kg": None if pd.isna(metrics["PFP_N_kg_kg"]) else float(metrics["PFP_N_kg_kg"]),
        "simple_profit": final_yield - 1.1 * float(irrigation.sum()) - 1.58 * float(nitrogen.sum()),
        "season_rain_mm_from_snapshot": float(rain.sum(min_count=1)) if len(rain) else None,
        "positive_action_rows": int(len(event_rows)),
        "irrigation_event_count": int(irrigation.gt(0).sum()),
        "nitrogen_event_count": int(nitrogen.gt(0).sum()),
        "action_signature": action_signature(daily),
        "snapshot": rel(snapshot),
    }


def replay_ppo_year(cfg: dict[str, Any], model: Any, year: int) -> dict[str, Any]:
    old = setup_modules(cfg)
    env = None
    try:
        config = engine.load_config()
        selection = base03222.build_selection(base03222.load_split())
        env_config = direct_ppo.build_env_config(config, selection)
        env_config["paths"]["output_root"] = rel(OUT)
        env = base03222.base.make_env(
            config,
            env_config,
            STATION,
            int(year),
            int(cfg.get("seed", 0)),
            f"HLA_{year}_216HLA_frequency_ppo200k",
            evaluation=True,
        )
        from sb3_contrib.common.maskable.utils import get_action_masks

        obs, _info = env.reset()
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"PPO HLA{year} replay did not finish within {MAX_STEPS} steps")
        target = OUT / "snapshots" / STATION / str(year) / "ppo_200k"
        save_snapshot(env, target)
        row = metrics_for_snapshot(target, year, "ppo_200k")
        row["replay_steps"] = int(steps)
        return row
    finally:
        if env is not None:
            env.close()
        restore_modules(old)
        gc.collect()


def replay_fewer_event_year(cfg: dict[str, Any], year: int) -> dict[str, Any]:
    old = setup_modules(cfg)
    env = None
    try:
        config = engine.load_config()
        selection = base03222.build_selection(base03222.load_split())
        env_config = direct_ppo.build_env_config(config, selection)
        env_config["paths"]["output_root"] = rel(OUT)
        static_baseline.ensure_project_on_path()
        import gym
        from sb3_wrapper import GymDssatWrapper

        year_info = direct_ppo.find_year(env_config, STATION, int(year))
        env_args = ppo_safe_rendering.build_env_args(
            station=STATION,
            year=int(year),
            planting_date=year_info["planting_date"],
            seed=int(cfg.get("seed", 0)),
            config=env_config,
            run_tag=f"HLA_{year}_216HLA_frequency_8x30",
            evaluation=True,
            mode=env_config.get("runtime", {}).get("mode", "all"),
        )
        template = Path(env_args["fileX_template_path"])
        template.write_text(
            replace_static_schedule_for_treatment(
                template.read_text(encoding="utf-8", errors="replace"),
                fewer_event_schedule(),
                direct_ppo.find_year(env_config, STATION, int(year))["planting_date"],
                int(year),
            ),
            encoding="utf-8",
        )
        env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
        obs, _info = env.reset()
        done = False
        steps = 0
        action = zero_action(env)
        while not done and steps < MAX_STEPS:
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"Static HLA{year} replay did not finish within {MAX_STEPS} steps")
        target = OUT / "snapshots" / STATION / str(year) / "fewer_8x30"
        save_snapshot(env, target)
        row = metrics_for_snapshot(target, year, "fewer_8x30")
        row["replay_steps"] = int(steps)
        return row
    finally:
        if env is not None:
            env.close()
        restore_modules(old)
        gc.collect()


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "logs").mkdir()
    shutil.copy2(CONFIG_PATH, OUT / "configs" / CONFIG_PATH.name)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)

    from sb3_contrib import MaskablePPO

    old = setup_modules(cfg)
    try:
        model = MaskablePPO.load(str(MODEL_PATH), device="cpu")
    finally:
        restore_modules(old)

    rows: list[dict[str, Any]] = []
    for year in YEARS:
        print(f"[216HLA frequency audit] PPO {year}", flush=True)
        rows.append(replay_ppo_year(cfg, model, year))
        print(f"[216HLA frequency audit] fewer events {year}", flush=True)
        rows.append(replay_fewer_event_year(cfg, year))
        pd.DataFrame(rows).to_csv(OUT / "logs" / "216HLA_frequency_counterfactual_rows.csv", index=False, encoding="utf-8-sig")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "logs" / "216HLA_frequency_counterfactual_rows.csv", index=False, encoding="utf-8-sig")
    ppo = frame[frame["policy"].eq("ppo_200k")].set_index("year")
    fewer = frame[frame["policy"].eq("fewer_8x30")].set_index("year")
    pairs = []
    for year in YEARS:
        p = ppo.loc[year]
        f = fewer.loc[year]
        pairs.append(
            {
                "year": int(year),
                "year_type": YEAR_LABELS[int(year)],
                "yield_fewer_minus_ppo_kg_ha": float(f["grain_yield_kg_ha"] - p["grain_yield_kg_ha"]),
                "WP_ET_fewer_minus_ppo_kg_m3": float(f["WP_ET_kg_m3"] - p["WP_ET_kg_m3"]),
                "simple_profit_fewer_minus_ppo": float(f["simple_profit"] - p["simple_profit"]),
                "irrigation_total_difference_mm": float(f["actual_irrigation_mm_daily_sum"] - p["actual_irrigation_mm_daily_sum"]),
                "nitrogen_total_difference_kg_ha": float(f["actual_nitrogen_kg_ha_daily_sum"] - p["actual_nitrogen_kg_ha_daily_sum"]),
                "irrigation_event_count_ppo": int(p["irrigation_event_count"]),
                "irrigation_event_count_fewer": int(f["irrigation_event_count"]),
            }
        )
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(OUT / "logs" / "216HLA_frequency_counterfactual_pairs.csv", index=False, encoding="utf-8-sig")

    result = {
        "task": AUDIT_RUN_NAME,
        "status": "ok",
        "question": "Whether frequent small irrigation is necessary under the saved PPO policy's same total water and nitrogen budget.",
        "model_path": rel(MODEL_PATH),
        "checkpoint_step": CHECKPOINT,
        "validation_weather": f"original_lowIC_HLA_2014_2023_{len(YEARS)}year_subset",
        "years": [{"year": int(y), "type": YEAR_LABELS[y]} for y in YEARS],
        "held_fixed": ["weather", "soil", "cultivar", "nitrogen schedule", "total nitrogen", "total irrigation", "reward metric", "DSSAT evaluator"],
        "policies": {
            "ppo_200k": {"description": "deterministic replay", "irrigation_daps": PPO_IRRIGATION_DAPS, "irrigation_mm_each": 15.0, "nitrogen_daps": PPO_NITROGEN_DAPS, "nitrogen_kg_ha_each": 80.0},
            "fewer_8x30": {"description": "pairwise-midpoint consolidation", "irrigation_daps": FEWER_IRRIGATION_DAPS, "irrigation_mm_each": 30.0, "nitrogen_daps": PPO_NITROGEN_DAPS, "nitrogen_kg_ha_each": 80.0},
        },
        "means_by_policy": {
            policy: {key: float(pd.to_numeric(group[key], errors="coerce").mean()) for key in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "simple_profit", "actual_irrigation_mm_daily_sum", "actual_nitrogen_kg_ha_daily_sum", "irrigation_event_count", "nitrogen_event_count"]}
            for policy, group in frame.groupby("policy")
        },
        "paired_differences": pairs,
        "mean_fewer_minus_ppo_yield_kg_ha": float(pair_frame["yield_fewer_minus_ppo_kg_ha"].mean()),
        "mean_fewer_minus_ppo_WP_ET_kg_m3": float(pair_frame["WP_ET_fewer_minus_ppo_kg_m3"].mean()),
        "same_total_irrigation_pass": bool(np.allclose(pair_frame["irrigation_total_difference_mm"], 0.0, atol=1e-6)),
        "same_total_nitrogen_pass": bool(np.allclose(pair_frame["nitrogen_total_difference_kg_ha"], 0.0, atol=1e-6)),
        "code_sha256": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
        "model_sha256": sha256(MODEL_PATH),
        "output_root": rel(OUT),
        "interpretation_limit": f"Preliminary {len(YEARS)}-year counterfactual only; it tests frequency at fixed totals and does not establish a universal agronomic optimum.",
    }
    (OUT / "216HLA_frequency_counterfactual_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    record = [
        f"# {AUDIT_RUN_NAME}",
        "",
        "- Evaluation-only replay; no training was resumed.",
        "- The same original weather, soil, cultivar, nitrogen schedule, total irrigation (240 mm), and total nitrogen (240 kg N/ha) were used in both policies.",
        "- PPO policy: 16 irrigation events of 15 mm; control: 8 irrigation events of 30 mm at pairwise-midpoint DAPs.",
        f"- Years: {', '.join(f'{year} ({YEAR_LABELS[year]})' for year in YEARS)}; selected from the original HLA validation years by seasonal rainfall for the 3-year smoke or retained as the complete 10-year validation set.",
        "",
        "## Mean paired result",
        "",
        json.dumps({key: result[key] for key in ["mean_fewer_minus_ppo_yield_kg_ha", "mean_fewer_minus_ppo_WP_ET_kg_m3", "same_total_irrigation_pass", "same_total_nitrogen_pass"]}, indent=2, ensure_ascii=False),
        "",
        "## Interpretation",
        "",
        "A negative yield or WP_ET difference for the fewer-event control supports the usefulness of frequent smaller applications under this fixed HLA setup. A near-zero difference means the apparent frequency preference is not necessary for the tested years. The result remains a local, preliminary counterfactual rather than a universal irrigation recommendation.",
        "",
        f"- Detailed rows: `{rel(OUT / 'logs' / '216HLA_frequency_counterfactual_rows.csv')}`.",
        f"- Paired differences: `{rel(OUT / 'logs' / '216HLA_frequency_counterfactual_pairs.csv')}`.",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
