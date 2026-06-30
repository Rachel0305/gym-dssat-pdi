from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_dqn_headroom_probe_015_07"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_015_07_yc2014_dqn_headroom_probe_record.md"

WATER_COST = 1.0
N_COST = 5.0
RECORDED_EXPERT_GWAD = 9418.0

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


IRRIGATION_PATTERNS: dict[str, dict[int, float]] = {
    "I_seed0_like": {2: 15, 10: 30, 44: 15, 53: 15, 60: 15, 67: 15, 74: 15},
    "I_seed1_like": {45: 30, 52: 30, 59: 30, 66: 30},
    "I_expert_window_discrete": {22: 30, 29: 30, 36: 30, 43: 30},
    "I_auto_window_plus": {35: 30, 42: 30, 76: 30, 83: 30},
}

N_PATTERNS: dict[str, dict[int, float]] = {
    "N_seed0_250": {2: 50, 10: 50, 87: 50, 94: 50, 101: 50},
    "N_seed1_300": {24: 50, 31: 50, 38: 50, 45: 100, 52: 50},
    "N_expert_window_300": {2: 100, 43: 100, 50: 100},
    "N_front_300": {2: 100, 10: 100, 18: 100},
    "N_mid_300": {31: 100, 43: 100, 55: 100},
    "N_split_300": {2: 50, 24: 50, 45: 100, 66: 50, 87: 50},
}


def configure_yc_module() -> None:
    yc.OUT_DIR = OUT_DIR
    yc.DOC_PATH = DOC_PATH
    yc.SEED = 0
    yc.WATER_COST = WATER_COST
    yc.NITROGEN_COST = N_COST
    yc.IRRIGATION_BUDGET = 120.0
    yc.NITROGEN_BUDGET = 300.0
    yc.DAILY_IRRIGATION_CAP = 30.0
    yc.DAILY_NITROGEN_CAP = 100.0
    yc.MIN_INTERVAL_DAYS = 7
    yc.FREE_DAILY_WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}


def merge_schedule(i_pattern: dict[int, float], n_pattern: dict[int, float]) -> dict[int, dict[str, float]]:
    schedule: dict[int, dict[str, float]] = {}
    for dap, val in i_pattern.items():
        schedule.setdefault(int(dap), {"amir": 0.0, "anfer": 0.0})["amir"] += float(val)
    for dap, val in n_pattern.items():
        schedule.setdefault(int(dap), {"amir": 0.0, "anfer": 0.0})["anfer"] += float(val)
    return dict(sorted(schedule.items()))


def validate_schedule(schedule: dict[int, dict[str, float]]) -> tuple[bool, str]:
    last_op: int | None = None
    total_i = 0.0
    total_n = 0.0
    for dap, action in sorted(schedule.items()):
        i = float(action.get("amir", 0.0))
        n = float(action.get("anfer", 0.0))
        if i > 30.0 + 1e-8 or n > 100.0 + 1e-8:
            return False, f"single cap exceeded at DAP {dap}"
        if i > 0 or n > 0:
            if last_op is not None and dap - last_op < 7:
                return False, f"interval <7 days at DAP {dap}"
            last_op = dap
        total_i += i
        total_n += n
    if total_i > 120.0 + 1e-8 or total_n > 300.0 + 1e-8:
        return False, "season budget exceeded"
    return True, "ok"


def prepare_env(schedule_id: str) -> tuple[Any, Path]:
    scenario = f"dqn_headroom_{schedule_id}"
    run_dir = yc.prepare_case_for_scenario(scenario)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = yc.make_raw_env(env_args)
    return env, run_dir


def run_schedule(schedule_id: str, i_name: str, n_name: str, schedule: dict[int, dict[str, float]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    env, run_dir = prepare_env(schedule_id)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(260):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(scalar(latest_before.get("dap", 0.0)) or 0))
            real_action = schedule.get(dap_before, {"amir": 0.0, "anfer": 0.0})
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, real_action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "schedule_id": schedule_id,
                    "i_pattern": i_name,
                    "n_pattern": n_name,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": float(real_action.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(real_action.get("anfer", 0.0)),
                    "reward_proxy_step": max(0.0, scalar(latest.get("grnwt", 0.0)) or 0.0) - WATER_COST * float(real_action.get("amir", 0.0)) - N_COST * float(real_action.get("anfer", 0.0)),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        snapshot = run_dir / "pdi_tmp_snapshot_eval"
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    plantgro_path = run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT"
    plantgro = parse_dssat_table(plantgro_path)
    final_gwad = float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns else np.nan
    final_cwad = float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns else np.nan
    total_i = sum(float(v.get("amir", 0.0)) for v in schedule.values())
    total_n = sum(float(v.get("anfer", 0.0)) for v in schedule.values())
    summary = {
        "schedule_id": schedule_id,
        "i_pattern": i_name,
        "n_pattern": n_name,
        "irrigation_events": json.dumps({k: v.get("amir", 0.0) for k, v in schedule.items() if v.get("amir", 0.0) > 0}, ensure_ascii=False),
        "fertilizer_events": json.dumps({k: v.get("anfer", 0.0) for k, v in schedule.items() if v.get("anfer", 0.0) > 0}, ensure_ascii=False),
        "total_irrigation": total_i,
        "total_n": total_n,
        "final_gwad": final_gwad,
        "final_cwad": final_cwad,
        "max_swfac": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nstres": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "reward_proxy": final_gwad - WATER_COST * total_i - N_COST * total_n,
        "beats_recorded_by_50": bool(final_gwad > RECORDED_EXPERT_GWAD + 50.0),
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }
    return daily, summary


def plot_candidates(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.26})
    colors = np.where(summary["final_gwad"] > RECORDED_EXPERT_GWAD + 50.0, "#238B45", "#6B7280")
    axes[0].scatter(summary["total_n"], summary["final_gwad"], c=colors, s=55, edgecolor="#333333", linewidth=0.5)
    axes[0].axhline(RECORDED_EXPERT_GWAD, color="#B2182B", ls="--", lw=1.5, label="Recorded expert")
    axes[0].set_xlabel("Total N (kg/ha)")
    axes[0].set_ylabel("GWAD (kg/ha)")
    axes[0].set_title("Yield headroom under fixed I≤120/N≤300")
    axes[0].legend()

    axes[1].scatter(summary["total_n"], summary["reward_proxy"], c=colors, s=55, edgecolor="#333333", linewidth=0.5)
    axes[1].set_xlabel("Total N (kg/ha)")
    axes[1].set_ylabel("Reward proxy")
    axes[1].set_title("Economic proxy headroom")
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_record(summary: pd.DataFrame, daily_path: Path, summary_path: Path, top_path: Path, fig_path: Path) -> None:
    best_yield = summary.sort_values("final_gwad", ascending=False).iloc[0]
    best_proxy = summary.sort_values("reward_proxy", ascending=False).iloc[0]
    n_beating = int(summary["beats_recorded_by_50"].sum())
    lines = [
        "# 015_07 YC2014 DQN headroom 诊断记录",
        "",
        "## 目的",
        "",
        "本轮不训练 DQN，只用同一 PDI/gym-DSSAT 环境前向模拟一小批候选水氮调度，判断在当前 I≤120 mm、N≤300 kg/ha、单次 I≤30 mm、单次 N≤100 kg/ha、最小间隔 7 天的约束内，是否存在明确超过 recorded expert 9418 kg/ha 的产量空间。",
        "",
        "## 结果",
        "",
        f"- 候选数量：{len(summary)}",
        f"- 超过 recorded expert + 50 kg/ha 的候选数量：{n_beating}",
        f"- 最高产量候选：{best_yield['schedule_id']}，GWAD={best_yield['final_gwad']:.1f} kg/ha，I={best_yield['total_irrigation']:.1f} mm，N={best_yield['total_n']:.1f} kg/ha，proxy={best_yield['reward_proxy']:.1f}",
        f"- 最高 proxy 候选：{best_proxy['schedule_id']}，GWAD={best_proxy['final_gwad']:.1f} kg/ha，I={best_proxy['total_irrigation']:.1f} mm，N={best_proxy['total_n']:.1f} kg/ha，proxy={best_proxy['reward_proxy']:.1f}",
        "",
        "## 决策解释",
        "",
        "如果本轮没有找到超过 9468 kg/ha 的候选，则不建议在 YC2014 上盲目加长 DQN 训练来追求更高产；当前可汇报结论应是 DQN 少氮追平专家，而不是产量超越专家。如果找到明确候选，则下一轮再围绕该候选的时点和动作空间做 DQN 长训练。",
        "",
        "## 输出文件",
        "",
        f"- 全部日值：`{daily_path.relative_to(PROJECT_ROOT)}`",
        f"- 候选汇总：`{summary_path.relative_to(PROJECT_ROOT)}`",
        f"- top 候选：`{top_path.relative_to(PROJECT_ROOT)}`",
        f"- 图：`{fig_path.relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_yc_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    all_daily = []
    summaries = []
    idx = 0
    for i_name, i_pattern in IRRIGATION_PATTERNS.items():
        for n_name, n_pattern in N_PATTERNS.items():
            schedule = merge_schedule(i_pattern, n_pattern)
            ok, note = validate_schedule(schedule)
            if not ok:
                print(f"[skip] {i_name}+{n_name}: {note}", flush=True)
                continue
            idx += 1
            schedule_id = f"s{idx:02d}_{i_name}_{n_name}"
            print(f"[run] {schedule_id}", flush=True)
            daily, summary = run_schedule(schedule_id, i_name, n_name, schedule)
            summary["note"] = note
            all_daily.append(daily)
            summaries.append(summary)

    daily_df = pd.concat(all_daily, ignore_index=True, sort=False) if all_daily else pd.DataFrame()
    summary_df = pd.DataFrame(summaries).sort_values(["final_gwad", "reward_proxy"], ascending=[False, False]).reset_index(drop=True)
    top_df = pd.concat(
        [
            summary_df.sort_values("final_gwad", ascending=False).head(10).assign(rank_type="top_yield"),
            summary_df.sort_values("reward_proxy", ascending=False).head(10).assign(rank_type="top_proxy"),
        ],
        ignore_index=True,
    )

    daily_path = OUT_DIR / "015_07_yc2014_dqn_headroom_daily.csv"
    summary_path = OUT_DIR / "015_07_yc2014_dqn_headroom_candidates.csv"
    top_path = OUT_DIR / "015_07_yc2014_dqn_headroom_top_schedules.csv"
    fig_path = fig_dir / "yc2014_headroom_yield_vs_input.png"
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")
    plot_candidates(summary_df, fig_path)
    write_record(summary_df, daily_path, summary_path, top_path, fig_path)
    print(summary_df[["schedule_id", "total_irrigation", "total_n", "final_gwad", "final_cwad", "max_swfac", "max_nstres", "reward_proxy", "beats_recorded_by_50"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
