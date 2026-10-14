"""YC 218YCA-5K action counterfactual audit.

This is a replay-only experiment.  It starts from the saved 218YCA checkpoint
5K action plans on the original 2014--2023 weather and reruns DSSAT under
three action transformations:

* original: the saved PPO action plan;
* sparse_pack: preserve seasonal I/N totals, pack irrigation into the minimum
  number of available irrigation actions, and spread those events;
* same_total_spread14d: preserve every irrigation dose and total I/N, but
  spread irrigation events at approximately 14-day intervals;
* drop_15mm: remove only 15-mm irrigation events, leaving all other actions.

No model weights are updated.  The purpose is to test whether the current
periodic irrigation pattern is necessary for the simulated outcome.
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
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k as runner
import run_all_year_direct_action_safe_ppo as direct_ppo
from plot_218YCA_yca_lowIC_five_scenario_daily_ckpt5000 import FIG055, rel
from run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04 import ProjectProfitDiscreteWrapper
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env


TASK_ID = "218YCA"
CHECKPOINT = 5_000
STATION = "YCA"
SITE = "YC"
YEARS = list(range(2014, 2024))
VARIANTS = ["original", "sparse_pack", "same_total_spread14d", "drop_15mm"]

PPO_ROOT = ROOT / "benchmark_results" / "218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k"
REPLAY_ROOT = PPO_ROOT / "replay_2014_2023"
MODEL_DAILY_ROOT = REPLAY_ROOT / "daily_outputs" / STATION
CONFIG_PATH = PPO_ROOT / "configs" / "218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k.json"
MANIFEST_PATH = PPO_ROOT / "configs" / "218YCA_selected_manifest.csv"

DEFAULT_OUT = ROOT / "benchmark_results" / "218YCA_yca_lowIC_action_counterfactual_5k"
MODEL_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "yca_lowIC_weather_scenario_bank_217_v1_fixed_width"

I_LEVELS = [0.0, 15.0, 30.0, 45.0]
N_LEVELS = [0.0, 40.0, 80.0, 120.0]


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def load_contract() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def source_daily_path(year: int) -> Path:
    return MODEL_DAILY_ROOT / f"YCA_{year}_seed0_ckpt{CHECKPOINT}_daily.csv"


def load_original_plan(year: int) -> dict[int, tuple[float, float]]:
    path = source_daily_path(year)
    if not path.exists():
        raise FileNotFoundError(path)
    daily = pd.read_csv(path, keep_default_na=False)
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce").fillna(0).astype(int)
    daily["safe_action_amir"] = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    daily["safe_action_anfer"] = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
    plan: dict[int, tuple[float, float]] = {}
    for _, row in daily.iterrows():
        dap = int(row["dap"])
        current_i, current_n = plan.get(dap, (0.0, 0.0))
        # The saved evaluator can emit more than one decision row on DAP1.
        # Preserve a non-zero action instead of letting a following no-op
        # overwrite it.
        plan[dap] = (
            max(current_i, float(row["safe_action_amir"])),
            max(current_n, float(row["safe_action_anfer"])),
        )
    return plan


def irrigation_events(plan: dict[int, tuple[float, float]]) -> list[tuple[int, float]]:
    return sorted(
        [(dap, values[0]) for dap, values in plan.items() if values[0] > 0.0],
        key=lambda item: item[0],
    )


def select_spread_dates(original_daps: list[int], count: int, minimum_gap: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [original_daps[0]]
    first = int(original_daps[0])
    last = max(int(original_daps[-1]), first + minimum_gap * (count - 1))
    values = np.rint(np.linspace(first, last, count)).astype(int).tolist()
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] + minimum_gap)
    return values


def pack_total(total: float) -> list[float]:
    full = int(round(total // max(I_LEVELS)))
    remainder = round(total - full * max(I_LEVELS), 6)
    doses = [max(I_LEVELS)] * full
    if remainder > 0:
        if remainder not in I_LEVELS:
            raise ValueError(f"Cannot pack total irrigation {total} with grid {I_LEVELS}")
        doses.append(remainder)
    return doses


def transform_plan(plan: dict[int, tuple[float, float]], variant: str, minimum_gap: int) -> dict[int, tuple[float, float]]:
    transformed = dict(plan)
    original_i = irrigation_events(plan)
    original_daps = [dap for dap, _amount in original_i]
    original_amounts = [amount for _dap, amount in original_i]
    n_by_dap = {dap: values[1] for dap, values in plan.items() if values[1] > 0.0}

    if variant == "original":
        return transformed

    # Remove all original irrigation first; N remains unchanged unless a new
    # irrigation date happens to share a day with an N event.
    for dap in list(transformed):
        transformed[dap] = (0.0, transformed[dap][1])

    if variant == "drop_15mm":
        for dap, amount in original_i:
            if amount != 15.0:
                transformed[dap] = (amount, transformed[dap][1])
        return transformed

    if variant == "sparse_pack":
        doses = pack_total(sum(original_amounts))
        dates = select_spread_dates(original_daps, len(doses), minimum_gap)
        for dap, amount in zip(dates, doses):
            n = n_by_dap.get(dap, transformed.get(dap, (0.0, 0.0))[1])
            transformed[dap] = (amount, n)
        return transformed

    if variant == "same_total_spread14d":
        dates = select_spread_dates(original_daps, len(original_amounts), 14)
        for dap, amount in zip(dates, original_amounts):
            n = n_by_dap.get(dap, transformed.get(dap, (0.0, 0.0))[1])
            transformed[dap] = (amount, n)
        return transformed

    raise ValueError(f"Unknown action counterfactual variant: {variant}")


def build_env() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = load_contract()
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    split = runner.base.make_split(manifest)
    config = runner.base03222.load_config()
    config["reward"] = dict(runner.prior.simple_profit_engine_config()["reward"])
    config["seed"] = int(contract["seed"])
    env_config = direct_ppo.build_env_config(config, split)
    return contract, config, env_config


def action_index_map(env: Any) -> dict[tuple[float, float], int]:
    return {
        (round(float(raw["amir"]), 6), round(float(raw["anfer"]), 6)): index
        for index, raw in enumerate(env.grid)
    }


def replay_plan(
    year: int,
    variant: str,
    plan: dict[int, tuple[float, float]],
    out_root: Path,
    contract: dict[str, Any],
    config: dict[str, Any],
    env_config: dict[str, Any],
) -> tuple[Path, pd.DataFrame]:
    target = out_root / "snapshots" / STATION / str(year) / variant
    if target.exists():
        raise FileExistsError(f"Refusing to reuse counterfactual target: {rel(target)}")

    env_config = dict(env_config)
    env_config["paths"] = dict(env_config.get("paths", {}))
    env_config["paths"]["output_root"] = rel(out_root)
    env = None
    rows: list[dict[str, Any]] = []
    try:
        env = runner.base03222.base.make_env(
            config,
            env_config,
            STATION,
            year,
            int(contract["seed"]),
            f"YCA_{year}_218YCA_{variant}_counterfactual",
            evaluation=True,
        )
        obs, info = env.reset()
        index_map = action_index_map(env)
        done = False
        steps = 0
        seen_daps: set[int] = set()
        while not done and steps < 260:
            current = getattr(env, "last_obs_dict", {})
            dap_raw = scalar(current.get("dap", steps + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else steps + 1
            if dap in seen_daps:
                amount_i, amount_n = 0.0, 0.0
            else:
                amount_i, amount_n = plan.get(dap, (0.0, 0.0))
            seen_daps.add(dap)
            key = (round(float(amount_i), 6), round(float(amount_n), 6))
            if key not in index_map:
                raise ValueError(f"Action {key} is not in the 16-action grid for {variant}, YCA{year}, DAP{dap}")
            action = index_map[key]
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            action_info = dict(getattr(env, "last_action_info", {}))
            rows.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": year,
                    "variant": variant,
                    "dap": dap,
                    "requested_irrigation_mm": amount_i,
                    "requested_nitrogen_kg_ha": amount_n,
                    "executed_irrigation_mm": action_info.get("safe_action_amir", np.nan),
                    "executed_nitrogen_kg_ha": action_info.get("safe_action_anfer", np.nan),
                    "discrete_action_index": action_info.get("discrete_action_index", np.nan),
                    "mask_forced_noop": action_info.get("mask_forced_noop", False),
                    "action_safety_triggered": action_info.get("action_safety_triggered", ""),
                    "reward": reward,
                }
            )
            steps += 1
        if not done:
            raise RuntimeError(f"Counterfactual replay did not finish for {variant}, YCA{year}")
        snapshot_tmp = snapshot_from_env(env)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot_tmp, target)
        return target, pd.DataFrame(rows)
    finally:
        if env is not None:
            env.close()


def evaluate_snapshot(snapshot: Path, year: int, variant: str, actions: pd.DataFrame) -> dict[str, Any]:
    daily = FIG055.daily_from_snapshot(snapshot, year, variant)
    final_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
    metrics = FIG055.baseline.metrics_from_snapshot(snapshot, final_yield)
    irrigation = float(metrics["actual_irrigation_mm"])
    nitrogen = float(metrics["actual_nitrogen_kg_ha"])
    action_rows = actions[actions["executed_irrigation_mm"].astype(float) > 0.0]
    n_rows = actions[actions["executed_nitrogen_kg_ha"].astype(float) > 0.0]
    pfp_n = float(metrics["PFP_N_kg_kg"]) if nitrogen > 0 and pd.notna(metrics["PFP_N_kg_kg"]) else np.nan
    return {
        "station_code": STATION,
        "site": SITE,
        "year": year,
        "variant": variant,
        "grain_yield_kg_ha": final_yield,
        "actual_irrigation_mm": irrigation,
        "actual_nitrogen_kg_ha": nitrogen,
        "simple_profit": final_yield - 1.1 * irrigation - 1.58 * nitrogen,
        "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
        "PFP_N_kg_kg": pfp_n,
        "max_wspd": float(pd.to_numeric(daily["water_stress_index_wspd"], errors="coerce").max()),
        "max_nstd": float(pd.to_numeric(daily["nitrogen_stress_index_nstd"], errors="coerce").max()),
        "water_stress_days_gt_0p05": int((pd.to_numeric(daily["water_stress_index_wspd"], errors="coerce") > 0.05).sum()),
        "nitrogen_stress_days_gt_0p05": int((pd.to_numeric(daily["nitrogen_stress_index_nstd"], errors="coerce") > 0.05).sum()),
        "irrigation_event_count": int(len(action_rows)),
        "n_event_count": int(len(n_rows)),
        "mask_forced_noop_count": int(actions["mask_forced_noop"].astype(str).str.lower().eq("true").sum()),
        "safety_trigger_count": int(actions["action_safety_triggered"].astype(str).str.strip().ne("").sum()),
        "snapshot_path": rel(snapshot),
    }


def run(years: list[int], out_root: Path) -> dict[str, Any]:
    missing = [str(path.relative_to(ROOT)).replace("\\", "/") for path in [CONFIG_PATH, MANIFEST_PATH] if not path.exists()]
    for year in years:
        if not source_daily_path(year).exists():
            missing.append(rel(source_daily_path(year)))
    if missing:
        raise FileNotFoundError("Missing counterfactual inputs: " + "; ".join(missing))
    if out_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {rel(out_root)}")

    for directory in [out_root / "snapshots", out_root / "tables", out_root / "configs"]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, out_root / "configs" / CONFIG_PATH.name)

    contract, config, env_config = build_env()
    old_wrapper = runner.base03222.base.StressAwareDiscreteWrapper
    old_renderer_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    runner.base03222.base.StressAwareDiscreteWrapper = ProjectProfitDiscreteWrapper
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = MODEL_INPUT_ROOT

    result_rows: list[dict[str, Any]] = []
    action_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    plan_rows: list[dict[str, Any]] = []
    try:
        for year in years:
            original = load_original_plan(year)
            minimum_gap = int(config["action_safety"]["min_days_between_irrigation"])
            for variant in VARIANTS:
                plan = transform_plan(original, variant, minimum_gap)
                for dap, (amount_i, amount_n) in sorted(plan.items()):
                    if amount_i > 0.0 or amount_n > 0.0:
                        original_i, original_n = original.get(dap, (0.0, 0.0))
                        plan_rows.append(
                            {
                                "year": year,
                                "variant": variant,
                                "dap": dap,
                                "irrigation_mm": amount_i,
                                "nitrogen_kg_ha": amount_n,
                                "original_irrigation_mm": original_i,
                                "original_nitrogen_kg_ha": original_n,
                            }
                        )
                print(f"[218YCA counterfactual] variant={variant} year={year}", flush=True)
                snapshot, actions = replay_plan(year, variant, plan, out_root, contract, config, env_config)
                result_rows.append(evaluate_snapshot(snapshot, year, variant, actions))
                actions.to_csv(out_root / "tables" / f"actions_{variant}_{year}.csv", index=False, encoding="utf-8-sig")
                action_rows.append(actions)
                daily = FIG055.daily_from_snapshot(snapshot, year, variant)
                daily_rows.append(daily)
    finally:
        runner.base03222.base.StressAwareDiscreteWrapper = old_wrapper
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_renderer_root

    summary = pd.DataFrame(result_rows).sort_values(["year", "variant"])
    actions_all = pd.concat(action_rows, ignore_index=True)
    daily_all = pd.concat(daily_rows, ignore_index=True)
    plan_df = pd.DataFrame(plan_rows).sort_values(["year", "variant", "dap"])
    summary.to_csv(out_root / "tables" / "218YCA_action_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    actions_all.to_csv(out_root / "tables" / "218YCA_action_counterfactual_step_actions.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(out_root / "tables" / "218YCA_action_counterfactual_daily.csv", index=False, encoding="utf-8-sig")
    plan_df.to_csv(out_root / "tables" / "218YCA_action_counterfactual_plans.csv", index=False, encoding="utf-8-sig")

    grouped = summary.groupby("variant", as_index=False).agg(
        mean_yield_kg_ha=("grain_yield_kg_ha", "mean"),
        mean_irrigation_mm=("actual_irrigation_mm", "mean"),
        mean_n_kg_ha=("actual_nitrogen_kg_ha", "mean"),
        mean_simple_profit=("simple_profit", "mean"),
        mean_wp_et=("WP_ET_kg_m3", "mean"),
        mean_pfp_n=("PFP_N_kg_kg", "mean"),
        mean_water_stress_days=("water_stress_days_gt_0p05", "mean"),
        mean_irrigation_events=("irrigation_event_count", "mean"),
    )
    grouped.to_csv(out_root / "tables" / "218YCA_action_counterfactual_variant_summary.csv", index=False, encoding="utf-8-sig")

    result = {
        "task": "218YCA_action_counterfactual_5k",
        "checkpoint": CHECKPOINT,
        "years": years,
        "variants": VARIANTS,
        "output_root": rel(out_root),
        "source_action_plans": rel(MODEL_DAILY_ROOT),
        "contract": {
            "no_training": True,
            "same_original_weather": True,
            "same_16_action_grid": True,
            "variants": {
                "original": "saved PPO action plan",
                "sparse_pack": "preserve seasonal I/N totals; pack irrigation into the minimum grid actions and spread events",
                "same_total_spread14d": "preserve irrigation doses and total I/N; spread irrigation events at approximately 14-day intervals",
                "drop_15mm": "remove only 15-mm irrigation events; preserve all other original actions",
            },
            "not_a_policy_retraining_result": True,
        },
        "summary_csv": rel(out_root / "tables" / "218YCA_action_counterfactual_summary.csv"),
        "variant_summary_csv": rel(out_root / "tables" / "218YCA_action_counterfactual_variant_summary.csv"),
        "plan_csv": rel(out_root / "tables" / "218YCA_action_counterfactual_plans.csv"),
        "daily_csv": rel(out_root / "tables" / "218YCA_action_counterfactual_daily.csv"),
    }
    (out_root / "218YCA_action_counterfactual_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "218YCA_action_counterfactual_record.md").write_text(
        "\n".join(
            [
                "# 218YCA-5K YC 动作反事实实验记录",
                "",
                "## 目的",
                "",
                "检验当前 PPO 的周期性灌溉动作是否对产量和收益有必要。实验只回放已保存的 218YCA checkpoint 5K，不更新模型参数。",
                "",
                "## 处理组",
                "",
                "- `original`：保存的 PPO 原始动作计划。",
                "- `sparse_pack`：保持每年总灌溉量和总施氮量不变，把灌溉量尽量压缩到最少的合法动作数，并拉开动作日期。",
                "- `same_total_spread14d`：保持每次灌溉剂量和季节总投入不变，把灌溉日期拉开到约 14 天间隔。",
                "- `drop_15mm`：只删除 15 mm 灌溉动作，其他动作保持原样，用来判断 15 mm 小灌是否有必要。",
                "",
                "## 固定条件",
                "",
                "- 原始 2014--2023 YC 天气；低 IC 输入；16 个联合动作；原有动作掩码和安全约束。",
                "- 不重新训练，不改变奖励函数，不改变土壤、管理基线或 DSSAT 参数。",
                "- 每个反事实动作都重新运行 DSSAT，而不是用产量公式后处理。",
                "",
                "## 解释边界",
                "",
                "- 这是动作时序/动作剂量的必要性检验，不是对 PPO 重新训练后的新策略评价。",
                "- 如果 `sparse_pack` 或 `drop_15mm` 的产量和收益几乎不下降，说明原策略的重复灌溉存在冗余；如果明显下降，说明这些动作可能有实际作用。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()] or YEARS
    out_root = DEFAULT_OUT.with_name(DEFAULT_OUT.name + args.suffix.strip())
    print(json.dumps(run(years, out_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
