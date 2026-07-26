from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline_034
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


TASK_ID = "034_04"
OUT = ROOT / "benchmark_results" / "034_04_linked_free_timing_ppo_dqn_train_smoke"
DOC = ROOT / "docs" / "034_04_linked_free_timing_ppo_dqn_train_smoke_record.md"
PROMPT = ROOT / "prompts" / "034_04_linked_free_timing_ppo_dqn_train_smoke.md"
CONFIG_SRC = ROOT / "benchmark_results" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun" / "configs" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
SOURCE_SPLIT = ROOT / "benchmark_results" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun" / "configs" / "033_04_available_weather_half_split_years.csv"


def ensure_dirs() -> None:
    for rel in [
        "configs",
        "evaluation",
        "logs/FQA",
        "models/FQA",
        "daily_outputs/FQA",
        "snapshots/FQA/2014",
        "tensorboard/FQA",
        "rendered_inputs",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env: Any, obs: Any | None = None, info: dict | None = None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def overview_modes(snapshot: Path) -> str:
    path = snapshot / "OVERVIEW.OUT"
    if not path.exists():
        return ""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "MANAGEMENT OPT" in line:
            lines.append(line.strip())
    return " | ".join(lines[:3])


def load_config_and_env() -> tuple[dict[str, Any], dict[str, Any]]:
    config = direct_ppo.load_yaml(CONFIG_SRC)
    config["seed"] = 0
    config["total_timesteps"] = 5000
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["smoke_station"] = "FQA"
    config["runtime"]["smoke_year"] = 2014
    config["runtime"]["max_steps"] = 260
    direct_ppo.write_yaml(config, OUT / "configs" / "034_04_train_config.yaml")
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    split = pd.read_csv(SOURCE_SPLIT, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selection = batch_ppo.build_selection(split)
    selection.to_csv(OUT / "configs" / "034_04_selection_source.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    base.OUT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "034_04_resolved_env_config.yaml")
    return config, env_config


def model_path(algorithm: str) -> Path:
    return OUT / "models" / "FQA" / f"{algorithm.lower()}_stress_aware_seed0.zip"


def train_one(config: dict[str, Any], env_config: dict[str, Any], algorithm: str) -> pd.DataFrame:
    base.OUT = OUT
    row = base.train_one(config, env_config, algorithm)
    row["task"] = TASK_ID
    row["linked_management_expected"] = True
    return row


def masked_greedy_action(model: Any, obs: Any, mask: np.ndarray) -> int:
    return base.masked_greedy_action(model, obs, mask)


def evaluate_with_snapshot(config: dict[str, Any], env_config: dict[str, Any], train_row: pd.Series) -> pd.DataFrame:
    algorithm = str(train_row["algorithm"])
    if str(train_row["run_status"]) != "ok":
        return pd.DataFrame([{"algorithm": algorithm, "run_status": "failed", "notes": train_row.get("notes", "")}])
    station = "FQA"
    year = 2014
    seed = 0
    mpath = ROOT / str(train_row["model_path"])
    if algorithm == "MaskablePPO":
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        model = MaskablePPO.load(str(mpath), device="cpu")
    else:
        from stable_baselines3 import DQN

        model = DQN.load(str(mpath), device="cpu")
        get_action_masks = None
    weather = direct_ppo.weather_for_daily(config)
    env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_{TASK_ID}_{algorithm}_eval", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            if algorithm == "MaskablePPO":
                mask = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            else:
                action = masked_greedy_action(model, obs, env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "year": year,
                    "seed": seed,
                    "algorithm": algorithm,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        daily_path = OUT / "daily_outputs" / station / f"{year}_{algorithm.lower()}_linked_eval_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]) if len(daily) else np.nan
        snapshot = OUT / "snapshots" / station / str(year) / f"{algorithm.lower()}_linked_eval"
        tmp = siteppo.snapshot_from_env(env)
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(tmp, snapshot)
        metrics = baseline_034.metrics_from_snapshot(snapshot, final_y)
        summary = base.summarize_daily(algorithm, daily, daily_path, mpath)
        summary.update(
            {
                "task": TASK_ID,
                "station_code": station,
                "year": year,
                "seed": seed,
                "summary_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                "summary_n_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
                "safe_summary_i_match": abs(float(summary["total_irrigation"]) - float(metrics["actual_irrigation_mm"])) < 1e-6,
                "safe_summary_n_match": abs(float(summary["total_n"]) - float(metrics["actual_nitrogen_kg_ha"])) < 1e-6,
                "overview_management_opt": overview_modes(snapshot),
                "overview_is_linked": "IRRIG   :L" in overview_modes(snapshot) and "FERT :L" in overview_modes(snapshot),
                "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        summary["interface_pass"] = bool(
            str(summary.get("run_status")) == "ok"
            and bool(summary["overview_is_linked"])
            and bool(summary["safe_summary_i_match"])
            and bool(summary["safe_summary_n_match"])
        )
        return pd.DataFrame([summary])
    finally:
        env.close()


def write_record(config: dict[str, Any], train: pd.DataFrame, eval_df: pd.DataFrame, failures: pd.DataFrame, elapsed: float) -> None:
    pass_count = int(eval_df["interface_pass"].sum()) if not eval_df.empty and "interface_pass" in eval_df else 0
    total = int(len(eval_df))
    lines = [
        "# 034_04 linked 修复后自由时序 PPO/DQN 训练 smoke 记录",
        "",
        "## 结论先说",
        "",
        f"- 接口闭环通过：{pass_count}/{total}。",
        "- 本任务只验证修复后训练与回放链路，不评价最终农学优劣。",
        f"- 站点年份：FQA2014；seed=0；训练步数={int(config['total_timesteps'])}。",
        "",
        "## 训练 summary",
        "",
        md_table(train, 20),
        "",
        "## 评估 summary",
        "",
        md_table(eval_df[[
            "algorithm",
            "run_status",
            "episode_length",
            "final_grnwt",
            "total_irrigation",
            "total_n",
            "summary_irrigation_mm",
            "summary_n_kg_ha",
            "safe_summary_i_match",
            "safe_summary_n_match",
            "overview_is_linked",
            "interface_pass",
            "action_sequence",
            "daily_csv_path",
            "snapshot_path",
        ]] if not eval_df.empty else eval_df, 20),
        "",
        "## 失败记录",
        "",
        md_table(failures, 20),
        "",
        "## 边界",
        "",
        "- 5K 是 smoke，不是最终训练长度。",
        "- 通过本任务只表示 linked action 已进入 DSSAT，可开始重新训练；不代表 PPO/DQN 已优于四情景。",
        "- 033_04 旧结果仍应标注为 external action 未落地条件下的历史结果，不应用于正式比较。",
        "",
        f"耗时：{elapsed:.1f} 秒",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config = load_config_and_env()
    failures: list[dict[str, Any]] = []
    train_rows: list[pd.DataFrame] = []
    for alg in ["MaskablePPO", "DQN"]:
        try:
            train_rows.append(train_one(config, env_config, alg))
        except Exception:
            failures.append({"phase": "train", "algorithm": alg, "traceback": traceback.format_exc()[-5000:]})
    train = pd.concat(train_rows, ignore_index=True, sort=False) if train_rows else pd.DataFrame()
    train.to_csv(OUT / "evaluation" / "034_04_training_summary.csv", index=False, encoding="utf-8-sig")
    eval_rows: list[pd.DataFrame] = []
    for _, row in train.iterrows():
        try:
            eval_rows.append(evaluate_with_snapshot(config, env_config, row))
        except Exception:
            failures.append({"phase": "eval", "algorithm": str(row.get("algorithm", "")), "traceback": traceback.format_exc()[-5000:]})
    eval_df = pd.concat(eval_rows, ignore_index=True, sort=False) if eval_rows else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    eval_df.to_csv(OUT / "evaluation" / "034_04_eval_summary.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / "034_04_failures.csv", index=False, encoding="utf-8-sig")
    write_record(config, train, eval_df, failures_df, time.time() - start)
    result = {
        "task": TASK_ID,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "interface_pass": int(eval_df["interface_pass"].sum()) if not eval_df.empty and "interface_pass" in eval_df else 0,
        "total": int(len(eval_df)),
        "failures": int(len(failures_df)),
    }
    (OUT / "034_04_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not eval_df.empty:
        cols = [
            "algorithm",
            "final_grnwt",
            "total_irrigation",
            "total_n",
            "summary_irrigation_mm",
            "summary_n_kg_ha",
            "interface_pass",
            "action_sequence",
        ]
        print(eval_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
