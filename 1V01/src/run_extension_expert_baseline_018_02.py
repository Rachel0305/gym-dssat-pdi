from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
from run_hla_official_reward_restart_smoke import (
    install_official_reward_module,
    parse_events as parse_hla_events,
    prepare_case_at as prepare_hla_case_at,
)

import run_hla_unified_dqn_long_train_015_09 as hla_base
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_base


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_02"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_02_extension_expert_baseline_hla2010_yc2014_record.md"
SCHEDULE_PATH = OUT_DIR / "extension_expert_schedule.csv"

SCENARIO = "extension_expert_fixed_dap"
MAX_SINGLE_IRRIGATION_MM = 50.0


def midpoint(low: float, high: float) -> float:
    return (float(low) + float(high)) / 2.0


def kg_mu_to_kg_ha(value: float) -> float:
    return float(value) * 15.0


def fang_mu_to_mm(value: float) -> float:
    return float(value) * 1.5


def build_schedule() -> pd.DataFrame:
    """Manually encode the maize recommendations read from the extension PDF.

    Units in the source article:
    - N: kg/mu
    - irrigation: fang/mu

    Converted units:
    - N kg/ha = kg/mu * 15
    - irrigation mm = fang/mu * 1.5
    """
    rows: list[dict[str, Any]] = []

    # Table 1: Northeast and Great-Wall spring maize region.
    hla_stages = [
        ("sowing_base", "播种期/基肥", 0, (7.0, 8.0), (20.0, 30.0)),
        ("small_bell", "小喇叭口期", 30, (2.0, 2.5), (25.0, 35.0)),
        ("large_bell", "大喇叭口期", 50, (4.0, 5.0), (35.0, 40.0)),
        ("tasseling_silking", "抽雄散粉期", 65, (1.5, 2.0), (35.0, 40.0)),
        ("early_grain_filling", "灌浆初期", 85, (2.0, 2.5), (25.0, 30.0)),
        ("late_milk", "乳熟末期", 110, (1.5, 2.0), (15.0, 25.0)),
    ]
    for code, stage, dap, n_range, i_range in hla_stages:
        rows.append(
            {
                "region": "northeast_greatwall_spring_maize",
                "source_table": "表1 东北及长城沿线春玉米区",
                "station": "HLA",
                "year": 2010,
                "stage_code": code,
                "stage_cn": stage,
                "dap": dap,
                "n_kg_mu_low": n_range[0],
                "n_kg_mu_high": n_range[1],
                "n_kg_ha_mid": kg_mu_to_kg_ha(midpoint(*n_range)),
                "irrigation_fang_mu_low": i_range[0],
                "irrigation_fang_mu_high": i_range[1],
                "irrigation_mm_mid": fang_mu_to_mm(midpoint(*i_range)),
            }
        )

    # Table 3: North China Huang-Huai and Fen-Wei plain summer maize region.
    yc_stages = [
        ("emergence_water", "出苗水", 7, (5.0, 6.0), (10.0, 20.0)),
        ("small_bell", "小喇叭口期", 30, (2.5, 3.0), (20.0, 30.0)),
        ("large_bell", "大喇叭口期", 45, (3.5, 4.0), (30.0, 35.0)),
        ("tasseling_silking", "抽雄散粉期", 60, (2.0, 2.5), (30.0, 35.0)),
        ("grain_filling", "灌浆期", 80, (2.0, 2.5), (25.0, 30.0)),
        ("milk_stage", "乳熟期", 100, (0.0, 0.0), (15.0, 25.0)),
    ]
    for code, stage, dap, n_range, i_range in yc_stages:
        rows.append(
            {
                "region": "huanghuai_fenwei_summer_maize",
                "source_table": "表3 华北黄淮和汾渭平原夏玉米区",
                "station": "YC",
                "year": 2014,
                "stage_code": code,
                "stage_cn": stage,
                "dap": dap,
                "n_kg_mu_low": n_range[0],
                "n_kg_mu_high": n_range[1],
                "n_kg_ha_mid": kg_mu_to_kg_ha(midpoint(*n_range)),
                "irrigation_fang_mu_low": i_range[0],
                "irrigation_fang_mu_high": i_range[1],
                "irrigation_mm_mid": fang_mu_to_mm(midpoint(*i_range)),
            }
        )

    schedule = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(SCHEDULE_PATH, index=False, encoding="utf-8-sig")
    return schedule


def split_irrigation_events(schedule: pd.DataFrame) -> dict[int, dict[str, float]]:
    actions: dict[int, dict[str, float]] = {}
    for _, row in schedule.sort_values("dap").iterrows():
        dap = int(row["dap"])
        irrigation = float(row["irrigation_mm_mid"])
        nitrogen = float(row["n_kg_ha_mid"])

        actions.setdefault(dap, {"amir": 0.0, "anfer": 0.0})
        actions[dap]["anfer"] += nitrogen

        remaining = irrigation
        offset = 0
        while remaining > 1e-9:
            amount = min(MAX_SINGLE_IRRIGATION_MM, remaining)
            action_dap = dap + offset
            actions.setdefault(action_dap, {"amir": 0.0, "anfer": 0.0})
            actions[action_dap]["amir"] += amount
            remaining -= amount
            offset += 1
    return actions


def make_hla_env_args(year: int, run_dir: Path) -> dict[str, Any]:
    hla_base.configure_shared_settings()
    install_official_reward_module()
    prepare_hla_case_at(year, run_dir)
    return json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))


def make_yc_env_args(run_dir: Path) -> dict[str, Any]:
    yc_base.OUT_DIR = OUT_DIR / "yc2014_source"
    yc_base.SEED = 0
    yc_base.WATER_COST = 1.0
    yc_base.NITROGEN_COST = 5.0
    yc_base.IRRIGATION_BUDGET = 9999.0
    yc_base.NITROGEN_BUDGET = 9999.0
    yc_base.DAILY_IRRIGATION_CAP = 9999.0
    yc_base.DAILY_NITROGEN_CAP = 9999.0
    yc_base.MIN_INTERVAL_DAYS = 0
    source_run_dir = yc_base.prepare_case_for_scenario("dqn_extension_expert_fixed_dap")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(source_run_dir, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env_args["log_saving_path"] = str(run_dir / "extension_expert_fixed_dap.log")
    input_dir = run_dir / "input"
    filex = next(input_dir.glob("*.MZX"))
    env_args["fileX_template_path"] = str(filex)
    env_args["auxiliary_file_paths"] = [
        str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}
    ]
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return env_args


def run_fixed_schedule(station: str, year: int, env_args: dict[str, Any], schedule: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    actions = split_irrigation_events(schedule)
    env = yc_base.make_raw_env(env_args)
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    snapshot_dir = run_dir / "pdi_tmp_snapshot_eval"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    try:
        obs, info = env.reset()
        for step in range(380):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", step)) or 0))
            real = actions.get(dap_before, {"amir": 0.0, "anfer": 0.0})
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "station": station,
                    "year": year,
                    "scenario": SCENARIO,
                    "step": step,
                    "dap_action": dap_before,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "rain": scalar(latest.get("rain")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm_action": float(real.get("amir", 0.0)),
                    "fertilizer_kg_ha_action": float(real.get("anfer", 0.0)),
                    "raw_reward": repr(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if real.get("amir", 0.0) > 0 or real.get("anfer", 0.0) > 0:
                action_rows.append(
                    {
                        "station": station,
                        "year": year,
                        "scenario": SCENARIO,
                        "dap": dap_before,
                        "irrigation_mm_action": float(real.get("amir", 0.0)),
                        "fertilizer_kg_ha_action": float(real.get("anfer", 0.0)),
                    }
                )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot_dir, dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    event_summary = parse_hla_events(snapshot_dir / "MgmtEvent.OUT")
    events = pd.DataFrame(action_rows)
    if (snapshot_dir / "PlantGro.OUT").exists():
        plantgro = parse_dssat_table(snapshot_dir / "PlantGro.OUT")
    else:
        plantgro = pd.DataFrame()
    final_gwad = float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan
    final_cwad = float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan
    summary = {
        "station": station,
        "year": year,
        "scenario": SCENARIO,
        "final_gwad": final_gwad,
        "final_cwad": final_cwad,
        "rain_total": float(pd.to_numeric(daily.get("rain", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not daily.empty else 0.0,
        "action_irrigation_total": float(daily["irrigation_mm_action"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha_action"].sum()) if not daily.empty else 0.0,
        "event_irrigation_total": float(event_summary.get("irrigation_total_mgmtevent", np.nan)),
        "event_fertilizer_total": float(event_summary.get("fertilizer_total_mgmtevent", np.nan)),
        "max_water_stress": float(pd.to_numeric(daily["swfac"], errors="coerce").max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(pd.to_numeric(daily["nstres"], errors="coerce").max()) if not daily.empty else np.nan,
        "final_dap": float(pd.to_numeric(daily["dap"], errors="coerce").dropna().iloc[-1]) if not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    return daily, summary, events


def load_existing_baselines() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    hla_paths = [
        PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_to_2015_dqn_transfer_eval_015_17" / "hla2010_to_2015_transfer_eval_summary.csv",
        PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_final_dqn_four_scenario_015_16" / "hla_2010_2015_four_scenario_final_dqn_all_summary.csv",
    ]
    for path in hla_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            year = int(r.get("requested_year", r.get("year", 0)) or 0)
            if year != 2010:
                continue
            rows.append(
                {
                    "station": "HLA",
                    "year": year,
                    "scenario": str(r.get("scenario", r.get("label", ""))),
                    "label": str(r.get("label", r.get("scenario", ""))),
                    "final_gwad": float(r.get("final_gwad", r.get("final_grain_kg_ha", np.nan))),
                    "final_cwad": float(r.get("final_cwad", r.get("final_biomass_kg_ha", np.nan))),
                    "irrigation_total": float(r.get("irrigation_total", r.get("irrigation_total_mm", np.nan))),
                    "fertilizer_total": float(r.get("fertilizer_total", r.get("fertilizer_total_kg_ha", np.nan))),
                    "source": str(path.relative_to(PROJECT_ROOT)),
                }
            )

    yc_paths = [
        PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_checkpoint_refresh_016_08" / "yc2014_checkpoint_refresh_summary.csv",
        PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_formal_four_scenario_016_10" / "yc2014_formal_four_scenario_summary.csv",
        PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_dqn_015_10" / "seed1" / "dqn_baseline_relative_9action" / "015_10_yc2014_baseline_relative_summary.csv",
        PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_dqn_015_10" / "seed0" / "dqn_baseline_relative_9action" / "015_10_yc2014_baseline_relative_summary.csv",
    ]
    for path in yc_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "station": "YC",
                    "year": 2014,
                    "scenario": str(r.get("scenario", path.parent.name)),
                    "label": str(r.get("scenario", path.parent.name)),
                    "final_gwad": float(r.get("final_grain_kg_ha", r.get("final_gwad", np.nan))),
                    "final_cwad": float(r.get("final_biomass_kg_ha", r.get("final_cwad", np.nan))),
                    "irrigation_total": float(r.get("irrigation_total", r.get("action_irrigation_total", np.nan))),
                    "fertilizer_total": float(r.get("fertilizer_total", r.get("action_fertilizer_total", np.nan))),
                    "source": str(path.relative_to(PROJECT_ROOT)),
                }
            )

    return pd.DataFrame(rows)


def plot_summary(combined: pd.DataFrame, out_path: Path) -> None:
    if combined.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title, ylabel in [
        (axes[0], "final_gwad", "Grain yield", "kg/ha"),
        (axes[1], "irrigation_total", "Irrigation", "mm"),
        (axes[2], "fertilizer_total", "Nitrogen", "kg/ha"),
    ]:
        pivot = combined.pivot_table(index=["station", "year", "label"], values=col, aggfunc="first").reset_index()
        pivot["x_label"] = pivot["station"] + "\n" + pivot["label"].str.slice(0, 22)
        ax.bar(np.arange(len(pivot)), pivot[col], color="#8FAADC", edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(np.arange(len(pivot)))
        ax.set_xticklabels(pivot["x_label"], rotation=65, ha="right", fontsize=7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("018_02 extension expert baseline smoke: HLA2010 and YC2014", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_record(extension_summary: pd.DataFrame, combined: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        view = df.copy()
        for c in view.columns:
            if pd.api.types.is_numeric_dtype(view[c]):
                view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        headers = list(view.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
        return "\n".join(lines)

    lines = [
        "# 018_02 官方农技推广 extension expert baseline 小试记录",
        "",
        "## 本轮原则",
        "",
        "- 不修改 DQN 奖励函数。",
        "- 不重新训练 DQN。",
        "- 不改动原始输入文件。",
        "- 只新增 `extension_expert_fixed_dap` 管理情景。",
        "- HLA2010 使用推文表1，YC2014 使用推文表3。",
        "- 本轮采用固定 DAP 映射，不使用实测生育期，也不使用 DSSAT 事后生育期对齐。",
        "",
        "## Extension expert 结果",
        "",
        md_table(extension_summary),
        "",
        "## 与现有结果合并后的初步对照",
        "",
        md_table(combined[["station", "year", "label", "final_gwad", "final_cwad", "irrigation_total", "fertilizer_total", "source"]]),
        "",
        "## 解释",
        "",
        "本轮只检验新增官方农技推广 expert baseline 对现有叙事的冲击。若 extension expert 产量更高且资源投入也更高，应解释为“高投入推广方案”；DQN 若接近该产量但用水氮更少，则可作为节水节氮策略优势。",
        "",
        "## 输出文件",
        "",
        f"- schedule: `{SCHEDULE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- summary: `{(OUT_DIR / '018_02_extension_expert_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- combined: `{(OUT_DIR / '018_02_extension_expert_combined_comparison.csv').relative_to(PROJECT_ROOT)}`",
        f"- figure: `{(OUT_DIR / 'figures' / '018_02_extension_expert_summary.png').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    schedule = build_schedule()

    results: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    hla_run = OUT_DIR / "HLA2010" / SCENARIO
    hla_env_args = make_hla_env_args(2010, hla_run)
    hla_schedule = schedule[(schedule["station"].eq("HLA")) & (schedule["year"].eq(2010))]
    hla_daily, hla_summary, hla_events = run_fixed_schedule("HLA", 2010, hla_env_args, hla_schedule, hla_run)
    results.append(hla_summary)
    daily_frames.append(hla_daily)
    event_frames.append(hla_events)

    yc_run = OUT_DIR / "YC2014" / SCENARIO
    yc_env_args = make_yc_env_args(yc_run)
    yc_schedule = schedule[(schedule["station"].eq("YC")) & (schedule["year"].eq(2014))]
    yc_daily, yc_summary, yc_events = run_fixed_schedule("YC", 2014, yc_env_args, yc_schedule, yc_run)
    results.append(yc_summary)
    daily_frames.append(yc_daily)
    event_frames.append(yc_events)

    extension_summary = pd.DataFrame(results)
    extension_summary["label"] = "Extension expert fixed DAP"
    extension_summary["irrigation_total"] = extension_summary["event_irrigation_total"].fillna(extension_summary["action_irrigation_total"])
    extension_summary["fertilizer_total"] = extension_summary["event_fertilizer_total"].fillna(extension_summary["action_fertilizer_total"])
    extension_summary["source"] = "018_02 extension expert fixed DAP"

    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_events = pd.concat(event_frames, ignore_index=True)
    existing = load_existing_baselines()
    combined = pd.concat(
        [
            existing,
            extension_summary[["station", "year", "scenario", "label", "final_gwad", "final_cwad", "irrigation_total", "fertilizer_total", "source"]],
        ],
        ignore_index=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_daily.to_csv(OUT_DIR / "018_02_extension_expert_daily.csv", index=False, encoding="utf-8-sig")
    all_events.to_csv(OUT_DIR / "018_02_extension_expert_events.csv", index=False, encoding="utf-8-sig")
    extension_summary.to_csv(OUT_DIR / "018_02_extension_expert_summary.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUT_DIR / "018_02_extension_expert_combined_comparison.csv", index=False, encoding="utf-8-sig")
    plot_summary(combined, OUT_DIR / "figures" / "018_02_extension_expert_summary.png")
    write_record(extension_summary, combined)
    print(extension_summary.to_string(index=False))


if __name__ == "__main__":
    main()

