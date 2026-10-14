from __future__ import annotations

import json
import sys
from pathlib import Path

import gym
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
from ppo_evaluate import latest_observation_dict, scalar
from ppo_safe_rendering import build_env_args
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import parse_summary_out, snapshot_from_env
from sb3_wrapper import GymDssatWrapper


OUT = ROOT / "benchmark_results" / "034_00_external_action_channel_audit"
DOC = ROOT / "docs" / "034_00_external_action_channel_audit_record.md"


def build_single_env_config() -> tuple[dict, dict]:
    cfg = direct_ppo.load_yaml(ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml")
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["max_steps"] = 260
    split = pd.read_csv(ROOT / "benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/configs/033_04_available_weather_half_split_years.csv", keep_default_na=False)
    pool = pd.read_csv(ROOT / "Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/all_year_weather_scenario_pool.csv", keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selected = pool.merge(split[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selected = selected[(selected["station_code"].eq("FQA")) & (selected["year"].eq(2005))].copy()
    selected["selected_for_train"] = False
    selected["selected_for_eval"] = True
    env_config = direct_ppo.build_env_config(cfg, selected)
    env_config["paths"]["output_root"] = cfg["paths"]["output_root"]
    env_config["runtime"]["max_steps"] = 260
    return cfg, env_config


def make_env(env_config: dict, tag: str):
    year_info = direct_ppo.find_year(env_config, "FQA", 2005)
    env_args = build_env_args("FQA", 2005, year_info["planting_date"], 0, env_config, tag, True, "all")
    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return GymDssatWrapper(raw)


def run_case(env_config: dict, case_name: str, action_mode: str) -> tuple[pd.DataFrame, dict]:
    env = make_env(env_config, f"034_00_action_channel_{case_name}")
    rows = []
    try:
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < 140:
            latest_pre = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap", step + 1))
            dap = int(round(float(dap_raw))) if np.isfinite(float(dap_raw)) and float(dap_raw) > 0 else step + 1
            if action_mode == "raw_once" and dap == 1:
                action = np.array([50.0, 200.0], dtype=np.float32)
            elif action_mode == "raw_repeated" and dap in {1, 10, 20, 30}:
                action = np.array([50.0, 200.0], dtype=np.float32)
            else:
                action = np.array([0.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "case": case_name,
                    "dap": dap,
                    "action_amir_sent": float(action[0]),
                    "action_anfer_sent": float(action[1]),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": float(reward),
                    "done": done,
                }
            )
            step += 1
        daily = pd.DataFrame(rows)
        snapshot = snapshot_from_env(env)
        summary_rows = parse_summary_out(snapshot / "Summary.OUT")
        selected = {}
        if summary_rows:
            selected = summary_rows[-1]
        result = {
            "case": case_name,
            "final_grnwt_daily": float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]),
            "max_nstres_daily": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "max_swfac_daily": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "summary_HWAM_last": selected.get("HWAM"),
            "summary_IRCM_last": selected.get("IRCM"),
            "summary_NICM_last": selected.get("NICM"),
            "summary_ETCP_last": selected.get("ETCP"),
        }
        return daily, result
    finally:
        env.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation").mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    _, env_config = build_single_env_config()
    all_daily = []
    results = []
    for case, mode in [("zero", "zero"), ("raw_once_dap1", "raw_once"), ("raw_repeated_dap1_10_20_30", "raw_repeated")]:
        daily, result = run_case(env_config, case, mode)
        all_daily.append(daily)
        results.append(result)
    daily_df = pd.concat(all_daily, ignore_index=True)
    result_df = pd.DataFrame(results)
    daily_df.to_csv(OUT / "evaluation" / "034_00_external_action_channel_daily.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(OUT / "evaluation" / "034_00_external_action_channel_summary.csv", index=False, encoding="utf-8-sig")
    def md_table(df: pd.DataFrame) -> str:
        work = df.copy()
        for col in work.select_dtypes(include=["number"]).columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
        work = work.astype(object).where(pd.notna(work), "")
        header = "| " + " | ".join(map(str, work.columns)) + " |"
        sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
        return "\n".join([header, sep, *rows])

    lines = [
        "# 034_00 external action channel audit record",
        "",
        "目的：检查通过 GymDssatWrapper 向 DSSAT 发送外部灌溉/施氮动作时，作物轨迹与 Summary.OUT 是否发生响应。",
        "",
        "测试对象：FQA2005，multisite_new_cultivar_inputs_013，IC=1 渲染链。",
        "",
        "测试组：zero、DAP1 raw_once(50/200)、DAP1/10/20/30 raw_repeated(50/200)。",
        "",
        md_table(result_df),
        "",
        "判读：若三组作物产量和 Summary.OUT 的 IRCM/NICM 基本一致，则说明当前外部动作通道没有按预期改变 DSSAT 管理结果，不能直接用 RL 安全层累计动作冒充 DSSAT 实际管理。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(result_df), "out": str(OUT), "record": str(DOC)}, ensure_ascii=False, indent=2))
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
