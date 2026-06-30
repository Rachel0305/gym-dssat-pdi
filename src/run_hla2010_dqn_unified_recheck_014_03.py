from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import install_official_reward_module, parse_events, prepare_case_at
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_unified_recheck_014_03"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_014_03_hla2010_dqn_unified_recheck_record.md"

YEAR = 2010
WATER_COST = 1.0
NITROGEN_COST = 5.0

WINDOWS = {
    "free_daily": {
        "irrigation": [(1, 120)],
        "nitrogen": [(1, 120)],
    },
    "agronomic_window": {
        "irrigation": [(35, 65)],
        "nitrogen": [(1, 10), (35, 55)],
    },
}


def log_line(path: Path, message: str) -> None:
    line = f"{pd.Timestamp.now().isoformat()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def make_env(env_args: dict[str, Any], windows: dict[str, list[tuple[int, int]]]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    linked = yc_dqn.YCDiscreteBudgetedWrapper(
        yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw)),
        irrigation_windows=windows["irrigation"],
        nitrogen_windows=windows["nitrogen"],
    )
    return yc_dqn.EconomicRewardWrapper(linked)


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    values: list[float] = []
    if not path.exists():
        return values
    import re

    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            match = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if match:
                values.append(float(match.group(1)))
    return values


def train_and_eval(window_name: str, timesteps: int, seed: int) -> Path:
    install_official_reward_module()
    from stable_baselines3 import DQN

    run_name = f"{window_name}_seed{seed}_{timesteps}steps"
    run_dir = OUT_DIR / str(YEAR) / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    prepare_case_at(YEAR, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    debug_log = run_dir / "014_03_debug.log"
    snapshot = run_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    log_line(debug_log, f"train:start window={window_name} timesteps={timesteps} seed={seed}")
    env = make_env(env_args, WINDOWS[window_name])
    try:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=min(100, max(10, timesteps // 10)),
            batch_size=32,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        )
        log_line(debug_log, "learn:start")
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        log_line(debug_log, "learn:done")
        model_dir = run_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir / "dqn_hla2010_unified_recheck"))
    finally:
        env.close()
        log_line(debug_log, "train_env:closed")

    rows: list[dict[str, Any]] = []
    eval_env = make_env(env_args, WINDOWS[window_name])
    try:
        obs, info = eval_env.reset()
        for step in range(260):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "year": YEAR,
                    "window": window_name,
                    "seed": seed,
                    "timesteps": timesteps,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "action_index": int(np.asarray(action).item()),
                    "raw_amir": eval_env.last_raw_real_action.get("amir", np.nan),
                    "raw_anfer": eval_env.last_raw_real_action.get("anfer", np.nan),
                    "safe_amir": eval_env.last_safe_real_action.get("amir", np.nan),
                    "safe_anfer": eval_env.last_safe_real_action.get("anfer", np.nan),
                    "used_irrigation": eval_env.used_irrigation,
                    "used_nitrogen": eval_env.used_nitrogen,
                    "reward": float(reward),
                    "delta_grnwt": eval_env.last_reward_components.get("delta_grnwt", np.nan),
                    "water_cost_term": eval_env.last_reward_components.get("water_cost_term", np.nan),
                    "nitrogen_cost_term": eval_env.last_reward_components.get("nitrogen_cost_term", np.nan),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": bool(terminated or truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        eval_env.close()
        log_line(debug_log, "eval_env:closed")

    daily = pd.DataFrame(rows)
    daily.to_csv(run_dir / "dqn_eval_daily.csv", index=False, encoding="utf-8-sig")
    event_summary = parse_events(snapshot / "MgmtEvent.OUT")
    harvest_values = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "year": YEAR,
        "algorithm": "DQN",
        "window": window_name,
        "seed": seed,
        "timesteps": timesteps,
        "reward": "delta_grnwt - 1.0*irrigation - 5.0*nitrogen",
        "irrigation_budget": yc_dqn.IRRIGATION_BUDGET,
        "nitrogen_budget": yc_dqn.NITROGEN_BUDGET,
        "daily_irrigation_cap": yc_dqn.DAILY_IRRIGATION_CAP,
        "daily_nitrogen_cap": yc_dqn.DAILY_NITROGEN_CAP,
        "windows": WINDOWS[window_name],
        "daily_final_grnwt": float(daily["grnwt"].dropna().iloc[-1]) if not daily.empty else None,
        "daily_final_topwt": float(daily["topwt"].dropna().iloc[-1]) if not daily.empty else None,
        "action_irrigation_total": float(daily["safe_amir"].sum()) if not daily.empty else None,
        "action_nitrogen_total": float(daily["safe_anfer"].sum()) if not daily.empty else None,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else None,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else None,
        "eval_reward_total": float(daily["reward"].sum()) if not daily.empty else None,
        "harvest_yield_values_from_mgmtevent": harvest_values,
        **event_summary,
        "run_dir": str(run_dir),
    }
    (run_dir / "event_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log_line(debug_log, "train_and_eval:done")
    return run_dir


def collect_summary() -> pd.DataFrame:
    rows = []
    for path in sorted((OUT_DIR / str(YEAR)).glob("*/*event_summary.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            rows.append({"path": str(path), "error": repr(exc)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT_DIR / "014_03_hla2010_dqn_unified_recheck_summary.csv", index=False, encoding="utf-8-sig")
    return df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无结果。_"
    keep = [
        "window",
        "seed",
        "timesteps",
        "daily_final_grnwt",
        "daily_final_topwt",
        "action_irrigation_total",
        "action_nitrogen_total",
        "max_water_stress",
        "max_nitrogen_stress",
        "irrigation_total_mgmtevent",
        "fertilizer_total_mgmtevent",
    ]
    show = df[[c for c in keep if c in df.columns]].copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            show[col] = show[col].astype(str)
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_record(summary: pd.DataFrame) -> None:
    lines = [
        "# 014_03 HLA2010 DQN 统一流程复核记录",
        "",
        "## 目的",
        "",
        "用当前 YC/FQ linked DQN 方法复核 HLA2010，先 smoke test，确认动作链路、奖励、日志和输出正常。",
        "",
        "## 方法",
        "",
        "- 输入年份：HLA2010",
        "- 初始条件：IC=1，沿用 HLA 新品种参数输入。",
        "- linked 管理：脚本通过 `prepare_case_at` 插入 Jinja 占位符，并切换为 PDI/gym-DSSAT 可接收动作的管理方式。",
        "- DQN 动作：0 不操作；1 灌溉30mm；2 施氮100kg/ha；3 灌溉30mm+施氮100kg/ha。",
        "- 预算：I120/N300，单次 I30/N100，最小操作间隔7天。",
        "- 奖励：`delta_grnwt - 1.0*irrigation - 5.0*nitrogen`。",
        "",
        "## 当前结果",
        "",
        dataframe_to_markdown(summary),
        "",
        "## 文件",
        "",
        f"- 汇总：`{(OUT_DIR / '014_03_hla2010_dqn_unified_recheck_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 输出目录：`{OUT_DIR.relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows", nargs="+", choices=sorted(WINDOWS.keys()), default=["free_daily", "agronomic_window"])
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.collect_only:
        for window in args.windows:
            train_and_eval(window, timesteps=args.timesteps, seed=args.seed)
    summary = collect_summary()
    write_record(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
