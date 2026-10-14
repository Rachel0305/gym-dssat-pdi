"""215HLA: reproduce the established HLA five-scenario daily figures.

The four non-PPO scenarios come from the frozen 054 lowIC baseline package.
Only the PPO candidate is replaced by the 215HLA checkpoint-2000 replay on
the original 2014--2023 validation weather.

This script does not train PPO. It replays ten validation episodes, captures
DSSAT snapshots, and uses the same 054_03 five-scenario figure layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def load_054_figure_module():
    source = ROOT / "src" / "054_hla_lowIC_site_transfer" / "run_054_03_hla_lowIC_five_scenario_figures.py"
    spec = importlib.util.spec_from_file_location("hla_054_figure_helpers", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load existing HLA figure helpers: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIG054 = load_054_figure_module()
import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
from run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04 import ProjectProfitDiscreteWrapper
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env


TASK_ID = "215HLA"
TASK_NAME = "hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo"
CHECKPOINT = 2_000
STATION = "HLA"
SITE = "HLA"
YEARS = list(range(2014, 2024))

PPO_ROOT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_smoke2k"
BASELINE_ROOT = ROOT / "benchmark_results" / "054_02_hla_lowIC_four_baselines_static_level1"
AUTO_ROOT = ROOT / "benchmark_results" / "054_01_hla_lowIC_external_auto_n_rule_nstd050_minimal"
OUT_ROOT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_five_scenario_figures_ckpt{CHECKPOINT}"
RECORD = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_five_scenario_figures_ckpt{CHECKPOINT}_record.md"
PROMPT = ROOT / "prompts" / "2026-09-03_hla_simple_profit_weather_augmentation.md"

SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto_external_n": "DSSAT auto + external N",
    "official_extension_expert": "Official expert",
    "rl_candidate": "PPO (215HLA-2K)",
}
COLORS = {
    "null": "#555555",
    "recorded_farmer_template": "#C44E52",
    "dssat_auto_external_n": "#D8A305",
    "official_extension_expert": "#7E63B6",
    "rl_candidate": "#2A9D55",
}
STYLES = {
    "null": "-",
    "recorded_farmer_template": "--",
    "dssat_auto_external_n": "-.",
    "official_extension_expert": ":",
    "rl_candidate": "-",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_inputs() -> list[str]:
    required = [
        PPO_ROOT / "evaluation" / "215HLA_validation_summary_by_station_checkpoint.csv",
        PPO_ROOT / "evaluation" / "215HLA_training_checkpoint_inventory.csv",
        BASELINE_ROOT / "evaluation" / "054_02_baseline_summary.csv",
        AUTO_ROOT / "evaluation" / "054_01_external_auto_n_summary.csv",
    ]
    for year in YEARS:
        required.extend(
            [
                BASELINE_ROOT / "snapshots" / STATION / str(year) / "null" / "Summary.OUT",
                BASELINE_ROOT / "snapshots" / STATION / str(year) / "recorded_farmer_template" / "Summary.OUT",
                BASELINE_ROOT / "snapshots" / STATION / str(year) / "official_extension_expert" / "Summary.OUT",
                AUTO_ROOT / "snapshots" / STATION / str(year) / "dssat_auto_irrigation_external_n_rule" / "Summary.OUT",
            ]
        )
    return [rel(path) for path in required if not path.exists()]


def replay_ppo_snapshot(year: int, out: Path) -> Path:
    inventory_path = FIG054.resolve_training_inventory(
        PPO_ROOT, CHECKPOINT, preferred_prefix=TASK_ID
    )
    eval_path = FIG054.resolve_validation_summary(
        PPO_ROOT, CHECKPOINT, YEARS, preferred_prefix=TASK_ID
    )
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    source_eval = pd.read_csv(eval_path, keep_default_na=False)
    model_rows = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(CHECKPOINT)]
    if model_rows.empty:
        raise RuntimeError(f"No 215HLA model inventory row for checkpoint {CHECKPOINT}")
    model_path = ROOT / str(model_rows.iloc[0]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    expected = source_eval[
        pd.to_numeric(source_eval["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
        & pd.to_numeric(source_eval["year"], errors="coerce").eq(year)
    ]
    if len(expected) != 1:
        raise RuntimeError(f"Expected one 215HLA validation row for HLA{year}")

    target = out / "snapshots" / STATION / str(year) / "rl_candidate"
    if target.exists():
        raise FileExistsError(f"Refusing to reuse an existing replay target: {rel(target)}")

    old_engine = {
        key: getattr(engine, key)
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
        "SEED": base03222.SEED,
        "OUT": base03222.OUT,
        "DOC": base03222.DOC,
        "PROMPT": base03222.PROMPT,
        "TOTAL_TIMESTEPS": base03222.TOTAL_TIMESTEPS,
        "CHECKPOINT_STEPS": list(base03222.CHECKPOINT_STEPS),
        "load_split": base03222.load_split,
        "build_selection": base03222.build_selection,
    }
    old_wrapper = base03222.base.StressAwareDiscreteWrapper
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    env = None
    try:
        engine.TASK_ID = TASK_ID
        engine.TASK_NAME = TASK_NAME
        engine.BASE_OUT = out
        engine.BASE_DOC = RECORD
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = FIG054.INPUT_PROFILES["lowIC"]
        engine.STATION = STATION
        engine.SITES = [STATION]
        engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]

        base03222.OUT = out
        base03222.DOC = RECORD
        base03222.PROMPT = PROMPT
        base03222.SITES = [STATION]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update({STATION: SITE})
        base03222.SEED = 0
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = FIG054.INPUT_PROFILES["lowIC"]

        # Keep the same PPO action/mask contract as 215HLA. The reward is not
        # used to choose actions during deterministic replay, but using the
        # 215 wrapper preserves the actual action-safety implementation.
        engine.patch_base_module(out, RECORD, 100_000, [CHECKPOINT])
        base03222.base.StressAwareDiscreteWrapper = ProjectProfitDiscreteWrapper
        config = engine.load_config()
        selection = base03222.build_selection(base03222.load_split())
        env_config = direct_ppo.build_env_config(config, selection)
        env_config["paths"]["output_root"] = rel(out)

        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        model = MaskablePPO.load(str(model_path), device="cpu")
        env = base03222.base.make_env(
            config,
            env_config,
            STATION,
            year,
            0,
            f"HLA_{year}_215HLA_replay",
            evaluation=True,
        )
        obs, _info = env.reset()
        done = False
        steps = 0
        while not done and steps < 260:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"215HLA replay did not finish within 260 steps for HLA{year}")

        snapshot_tmp = snapshot_from_env(env)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot_tmp, target)
        daily = FIG054.daily_from_snapshot(target, year, "rl_candidate")
        replay_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        saved_yield = float(pd.to_numeric(expected.iloc[0]["final_grnwt"], errors="coerce"))
        if not np.isclose(replay_yield, saved_yield, rtol=0.0, atol=1.0):
            raise RuntimeError(
                f"215HLA replay endpoint mismatch for HLA{year}: "
                f"saved={saved_yield:.4f}, replay={replay_yield:.4f}"
            )
        return target
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        base03222.base.StressAwareDiscreteWrapper = old_wrapper
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update(old_base["SITE_NAMES"])
        for key, value in old_engine.items():
            setattr(engine, key, value)
        for key, value in old_base.items():
            if key != "SITE_NAMES":
                setattr(base03222, key, value)


def plot_year(daily: pd.DataFrame, year: int, fig_dir: Path) -> Path:
    """The established 054_03 4x2 layout, with a 215HLA-specific title."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    axes[0, 0].bar(weather["dap"], weather["rainfall_mm"], color="#80A9C7", alpha=0.85, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(weather["dap"], weather["tmax_c"], color="#C23B32", lw=1.25, label="Tmax")
    temp.plot(weather["dap"], weather["tmin_c"], color="#666666", lw=1.1, ls="--", label="Tmin")
    axes[0, 0].set_title("Weather")
    axes[0, 0].set_ylabel("Rain (mm)")
    temp.set_ylabel("Temperature (C)")

    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        color, style, label = COLORS[scenario], STYLES[scenario], LABELS[scenario]
        axes[0, 1].plot(sub["dap"], sub["soil_water_mm"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], sub["water_stress_index_wspd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], sub["nitrogen_stress_index_nstd"], color=color, ls=style, lw=1.35, label=label)
        positive_i = sub[sub["irrigation_executed_mm"].gt(0)]
        positive_n = sub[sub["nitrogen_executed_kg_ha"].gt(0)]
        if not positive_i.empty:
            axes[2, 0].stem(
                positive_i["dap"], positive_i["irrigation_executed_mm"],
                linefmt=color, markerfmt="o", basefmt=" ", label=label,
            )
        if not positive_n.empty:
            axes[2, 1].stem(
                positive_n["dap"], positive_n["nitrogen_executed_kg_ha"],
                linefmt=color, markerfmt="o", basefmt=" ", label=label,
            )
        axes[3, 0].plot(sub["dap"], sub["grain_yield_kg_ha"], color=color, ls=style, lw=1.45, label=f"{label} grain")
        axes[3, 0].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=style, lw=0.85, alpha=0.4)
        axes[3, 1].plot(sub["dap"], sub["unified_cumulative_reward"], color=color, ls=style, lw=1.35, label=label)

    titles = [
        (axes[0, 1], "Soil water", "SWTD (mm)"),
        (axes[1, 0], "Water stress index", "WSPD"),
        (axes[1, 1], "Nitrogen stress index", "NSTD"),
        (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"),
        (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
        (axes[3, 1], "Unified cumulative reward", "common units"),
    ]
    for axis, title, ylabel in titles:
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    for axis in [axes[0, 1], axes[2, 0], axes[2, 1], axes[3, 1]]:
        axis.legend(fontsize=7, ncol=2)
    axes[3, 0].set_xlabel("DAP")
    axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"HLA{year} five-scenario daily process | PPO: 215HLA checkpoint {CHECKPOINT}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = fig_dir / f"215HLA_hla{year}_five_scenario_daily_ckpt{CHECKPOINT}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run(years: list[int], dry_run: bool = False) -> dict[str, Any]:
    missing = check_inputs()
    result: dict[str, Any] = {
        "task": f"{TASK_ID}_{TASK_NAME}_five_scenario_figures",
        "ppo_checkpoint": CHECKPOINT,
        "years": years,
        "ppo_root": rel(PPO_ROOT),
        "frozen_baseline_root": rel(BASELINE_ROOT),
        "frozen_auto_root": rel(AUTO_ROOT),
        "output_root": rel(OUT_ROOT),
        "missing_inputs": missing,
        "five_scenarios": SCENARIOS,
        "figure_contract": {
            "core_conclusion": "Compare the 215HLA 2K simple-profit PPO process with the same four frozen HLA baseline scenarios on original 2014-2023 weather.",
            "archetype": "quantitative grid",
            "panel_logic": "weather; soil water; water stress; nitrogen stress; irrigation; nitrogen; grain/biomass; unified cumulative score",
            "backend": "Python/matplotlib",
            "export": "300 dpi PNG, matching the established 054_03 daily-process layout",
        },
    }
    if dry_run:
        return result
    if missing:
        raise FileNotFoundError("Missing required five-scenario figure inputs: " + "; ".join(missing))
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure output root: {rel(OUT_ROOT)}")

    fig_dir = OUT_ROOT / "figures"
    tab_dir = OUT_ROOT / "tables"
    cfg_dir = OUT_ROOT / "configs"
    for directory in [fig_dir, tab_dir, cfg_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "configs" / "215HLA_hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo.json", cfg_dir / "215HLA_config.json")
    if PROMPT.exists():
        shutil.copy2(PROMPT, cfg_dir / PROMPT.name)

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    figures: list[str] = []
    for year in years:
        ppo_snapshot = replay_ppo_snapshot(year, OUT_ROOT)
        snapshot_map = FIG054.build_snapshot_map(BASELINE_ROOT, AUTO_ROOT, ppo_snapshot, year)
        year_frames = [
            FIG054.daily_from_snapshot(snapshot, year, scenario)
            for scenario, snapshot in snapshot_map.items()
        ]
        merged = FIG054.add_unified_reward(pd.concat(year_frames, ignore_index=True))
        figure_path = plot_year(merged, year, fig_dir)
        figures.append(rel(figure_path))
        frames.append(merged)
        for scenario, sub in merged.groupby("scenario"):
            last = sub.iloc[-1]
            metrics = FIG054.baseline.metrics_from_snapshot(
                snapshot_map[scenario], float(last["grain_yield_kg_ha"])
            )
            n_total = float(metrics["actual_nitrogen_kg_ha"])
            pfp_n = float(metrics["PFP_N_kg_kg"]) if n_total > 0 and pd.notna(metrics["PFP_N_kg_kg"]) else np.nan
            summaries.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": year,
                    "scenario": scenario,
                    "grain_yield_kg_ha": float(last["grain_yield_kg_ha"]),
                    "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                    "actual_nitrogen_kg_ha": n_total,
                    "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N_kg_kg": pfp_n,
                    "max_wspd": float(pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce").max()),
                    "max_nstd": float(pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce").max()),
                    "unified_cumulative_reward": float(last["unified_cumulative_reward"]),
                    "snapshot_path": rel(snapshot_map[scenario]),
                }
            )

    all_daily = pd.concat(frames, ignore_index=True)
    actions = all_daily[
        all_daily["irrigation_executed_mm"].gt(0)
        | all_daily["nitrogen_executed_kg_ha"].gt(0)
    ].copy()
    summary = pd.DataFrame(summaries)
    daily_path = tab_dir / "215HLA_five_scenario_daily.csv"
    action_path = tab_dir / "215HLA_five_scenario_management_events.csv"
    summary_path = tab_dir / "215HLA_five_scenario_season_summary.csv"
    all_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    actions.to_csv(action_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    result.update(
        {
            "figures": figures,
            "daily_csv": rel(daily_path),
            "management_events_csv": rel(action_path),
            "season_summary_csv": rel(summary_path),
            "replay_validation_rows": len(years),
            "note": "215HLA is a 2K smoke checkpoint; this figure package is exploratory and does not freeze or rank the policy.",
        }
    )
    (OUT_ROOT / "215HLA_five_scenario_figure_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    RECORD.write_text(
        "\n".join(
            [
                "# 215HLA HL 逐年五情景日过程图记录",
                "",
                "## 口径",
                "",
                "- 四个非 PPO 情景沿用冻结的 054 lowIC 基线；PPO 使用 215HLA checkpoint 2000。",
                "- PPO 在原始 2014--2023 HLA 天气上逐年回放；没有重新训练。",
                "- 图的 4×2 面板、DAP 横轴、情景顺序和配色沿用 054_03 五情景日过程图。",
                "- 最后一面板是 054_03 的统一累计得分，用于过程对照，不等同于 215HLA 训练时的简单奖励定义。",
                "- 215HLA 仅完成 2K smoke，因此本图不是正式冻结结果，也不据此宣称 PPO 优于基线。",
                "",
                "## 文件",
                "",
                f"- 图：`{rel(fig_dir)}`。",
                f"- 日过程源数据：`{rel(daily_path)}`。",
                f"- 管理动作源数据：`{rel(action_path)}`。",
                f"- 季节汇总：`{rel(summary_path)}`。",
                f"- 机器可读清单：`{rel(OUT_ROOT / '215HLA_five_scenario_figure_manifest.json')}`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    years = [int(value.strip()) for value in args.years.split(",") if value.strip()] or YEARS
    result = run(years, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
