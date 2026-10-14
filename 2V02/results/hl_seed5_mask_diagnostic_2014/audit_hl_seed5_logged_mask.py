from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TASK = ROOT / "benchmark_results" / "8hl05_hl_seed5_8seed"
CKPT = TASK / "models" / "HLA" / "HLA_half_split_stress_aware_maskableppo_seed5_ckpt100000.zip"
CONFIG = TASK / "configs" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
DAILY_2014 = TASK / "daily_outputs" / "HLA" / "HLA_2014_seed5_ckpt100000_daily.csv"
MASK_SOURCE = ROOT / "src" / "run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py"
SAFETY_SOURCE = ROOT / "src" / "ppo_action_safety.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    required = [CKPT, CONFIG, DAILY_2014, MASK_SOURCE, SAFETY_SOURCE]
    missing = [rel(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen inputs missing: {missing}")

    # Import the exact production wrapper class, without constructing DSSAT.
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from ppo_action_safety import ActionSafetyState, update_action_safety_state
    import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as mask_module
    import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine_04210
    import run_all_year_direct_action_safe_ppo as direct_ppo

    # 054 HLA delegates to 042_10, and sets these globals from hl_seed5.json
    # before calling 042_10.load_config(). That loader delegates safety to
    # 040_36.load_config() while retaining the HLA 4x4 grid.
    seed_config = json.loads((TASK / "configs" / "hl_seed5.json").read_text(encoding="utf-8"))
    engine_04210.BINARY_IRRIGATION_LEVELS = list(map(float, seed_config["actions"]["irrigation_levels_mm"]))
    engine_04210.BINARY_NITROGEN_LEVELS = list(map(float, seed_config["actions"]["nitrogen_levels_kg_ha"]))
    config = engine_04210.load_config()
    base_template = direct_ppo.load_yaml(CONFIG)
    min_interval = int(config["action_safety"]["min_days_between_fertilization"])

    class MinimalBaseEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self.action_space = spaces.Discrete(16)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
            self.metadata = {}

    wrapper = mask_module.LateIrrigationReserveMaskWrapper(MinimalBaseEnv(), config)
    grid = list(wrapper.grid)
    daily = pd.read_csv(DAILY_2014, keep_default_na=False)
    state = ActionSafetyState()
    trace: list[dict[str, Any]] = []

    for row_idx, row in daily.iterrows():
        dap = int(round(float(row["dap"])))
        last_n_before = state.last_fertilization_dap
        wrapper.safety_state = ActionSafetyState(**state.__dict__)
        wrapper.last_obs_dict = {"dap": dap}
        mask = np.asarray(wrapper.action_masks(), dtype=bool)
        requested_idx = int(row["requested_discrete_action_index"])
        recorded_idx = int(row["discrete_action_index"])
        requested_action = grid[requested_idx]
        recorded_n = float(row["safe_action_anfer"])
        expected_gap = None if last_n_before is None else dap - int(last_n_before)
        trace.append(
            {
                "row_index_zero_based": int(row_idx),
                "year": int(row["year"]),
                "date": str(row["date"]),
                "dap_pre_action": dap,
                "last_fertilization_dap_pre_action_reconstructed": last_n_before,
                "days_since_last_fertilization": expected_gap,
                "min_days_between_fertilization": min_interval,
                "recorded_requested_action_index": requested_idx,
                "recorded_requested_action_amir": float(requested_action["amir"]),
                "recorded_requested_action_anfer": float(requested_action["anfer"]),
                "pre_action_mask_bits_index_0_to_15": "".join("1" if bit else "0" for bit in mask),
                "pre_action_mask_valid_count_reconstructed": int(mask.sum()),
                "pre_action_mask_all_16_bits_json": json.dumps(mask.astype(int).tolist()),
                "requested_action_mask_bit_reconstructed": bool(mask[requested_idx]),
                "recorded_action_index": recorded_idx,
                "recorded_mask_forced_noop": parse_bool(row["mask_forced_noop"]),
                "recorded_mask_valid_action_count_post_step": int(row["mask_valid_action_count"]),
                "recorded_action_safety_triggered": str(row["action_safety_triggered"]),
                "recorded_safe_action_amir": float(row["safe_action_amir"]),
                "recorded_safe_action_anfer": recorded_n,
                "expected_mask_vs_record_disagreement": bool(
                    (not bool(mask[requested_idx]))
                    and not parse_bool(row["mask_forced_noop"])
                    and recorded_idx == requested_idx
                ),
                "last_fertilization_dap_post_action_reconstructed": dap if recorded_n > 1e-9 else last_n_before,
            }
        )
        update_action_safety_state(
            state,
            {"amir": float(row["safe_action_amir"]), "anfer": recorded_n},
            dap,
        )
        if row_idx + 1 < len(daily):
            next_dap = int(round(float(daily.iloc[row_idx + 1]["dap"])))
            wrapper.safety_state = ActionSafetyState(**state.__dict__)
            wrapper.last_obs_dict = {"dap": next_dap}
            post_mask = np.asarray(wrapper.action_masks(), dtype=bool)
            trace[-1].update(
                {
                    "next_logged_dap_after_this_action": next_dap,
                    "post_step_mask_bits_reconstructed_for_next_row": "".join("1" if bit else "0" for bit in post_mask),
                    "post_step_mask_valid_count_reconstructed_for_next_row": int(post_mask.sum()),
                    "post_step_mask_count_matches_recorded_count": int(post_mask.sum()) == int(row["mask_valid_action_count"]),
                    "current_pre_mask_count_matches_previous_row_post_count": (
                        int(mask.sum()) == int(daily.iloc[row_idx - 1]["mask_valid_action_count"])
                        if row_idx > 0
                        else None
                    ),
                }
            )
        else:
            trace[-1].update(
                {
                    "next_logged_dap_after_this_action": None,
                    "post_step_mask_bits_reconstructed_for_next_row": "",
                    "post_step_mask_valid_count_reconstructed_for_next_row": None,
                    "post_step_mask_count_matches_recorded_count": None,
                    "current_pre_mask_count_matches_previous_row_post_count": (
                        int(mask.sum()) == int(daily.iloc[row_idx - 1]["mask_valid_action_count"])
                        if row_idx > 0
                        else None
                    ),
                }
            )

    trace_df = pd.DataFrame(trace)
    trace_path = OUT / "trace_reconstruction_v2.csv"
    metadata_path = OUT / "metadata_v2.json"
    findings_path = OUT / "findings_v2.md"
    for output_path in (trace_path, metadata_path, findings_path):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite: {rel(output_path)}")

    year_csvs = sorted((TASK / "daily_outputs" / "HLA").glob("HLA_*_seed5_ckpt100000_daily.csv"))
    boundary_rows: list[dict[str, Any]] = []
    annual_reset_rows: list[dict[str, Any]] = []
    prior_year = None
    prior_dap = None
    for path in year_csvs:
        frame = pd.read_csv(path, usecols=["year", "dap"])
        if frame.empty:
            continue
        year = int(frame.iloc[0]["year"])
        first_dap = int(frame.iloc[0]["dap"])
        last_dap = int(frame.iloc[-1]["dap"])
        if prior_year is not None and year != prior_year and first_dap < int(prior_dap):
            boundary_rows.append(
                {
                    "previous_year": prior_year,
                    "previous_year_last_dap": int(prior_dap),
                    "next_year": year,
                    "next_year_first_dap": first_dap,
                }
            )
        full_frame = pd.read_csv(path)
        daps = pd.to_numeric(full_frame["dap"], errors="coerce").to_numpy()
        for idx in np.flatnonzero(daps[1:] < daps[:-1]):
            cum_i = pd.to_numeric(full_frame["season_cumulative_irrigation"], errors="coerce")
            cum_n = pd.to_numeric(full_frame["season_cumulative_n"], errors="coerce")
            done = parse_bool(full_frame.iloc[int(idx + 1)]["done"])
            annual_reset_rows.append(
                {
                    "year": year,
                    "row_after_reset_zero_based": int(idx + 1),
                    "dap_before": int(daps[idx]),
                    "dap_after": int(daps[idx + 1]),
                    "cumulative_irrigation_before": float(cum_i.iloc[int(idx)]),
                    "cumulative_irrigation_after": float(cum_i.iloc[int(idx + 1)]),
                    "cumulative_n_before": float(cum_n.iloc[int(idx)]),
                    "cumulative_n_after": float(cum_n.iloc[int(idx + 1)]),
                    "done_after_reset_row": done,
                }
            )
        prior_year, prior_dap = year, last_dap

    try:
        import gym_dssat_pdi  # type: ignore[import-not-found]  # noqa: F401

        backend_importable = True
        backend_error = ""
    except Exception as exc:
        backend_importable = False
        backend_error = f"{type(exc).__name__}: {exc}"

    metadata = {
        "task": "HL seed5 2014 decision-time action-mask diagnostic",
        "runtime_replay_status": "NOT_EXECUTED_ENV_BACKEND_UNAVAILABLE",
        "static_reconstruction_status": "COMPLETED_FROM_SAVED_DAILY_LOG",
        "runtime_backend_importable_from_host": backend_importable,
        "runtime_backend_probe_error": backend_error,
        "dssat_runtime_started": False,
        "training_or_checkpoint_update": False,
        "config_or_source_modified": False,
        "wgen_generated": False,
        "checkpoint_path": rel(CKPT),
        "checkpoint_sha256": sha256(CKPT),
        "frozen_seed_config_path": rel(TASK / "configs" / "hl_seed5.json"),
        "frozen_seed_config_sha256": sha256(TASK / "configs" / "hl_seed5.json"),
        "engine_config_snapshot_path": rel(CONFIG),
        "engine_config_sha256": sha256(CONFIG),
        "source_mask_path": rel(MASK_SOURCE),
        "source_mask_sha256": sha256(MASK_SOURCE),
        "effective_config_loader": "src/054_hla_lowIC_site_transfer/run_054_00_hla_lowIC_expanded_action_maskableppo.py sets action levels from hl_seed5.json; src/run_sya_lowIC_binary_timing_maskableppo_042_10.py::load_config delegates safety to 040_36.load_config",
        "effective_action_safety": config["action_safety"],
        "base_template_action_safety": base_template["action_safety"],
        "base_template_is_effective_hla_config": config["action_safety"] == base_template["action_safety"],
        "effective_config_source_hashes": {
            rel(ROOT / "src" / "054_hla_lowIC_site_transfer" / "run_054_00_hla_lowIC_expanded_action_maskableppo.py"): sha256(ROOT / "src" / "054_hla_lowIC_site_transfer" / "run_054_00_hla_lowIC_expanded_action_maskableppo.py"),
            rel(ROOT / "src" / "run_sya_lowIC_binary_timing_maskableppo_042_10.py"): sha256(ROOT / "src" / "run_sya_lowIC_binary_timing_maskableppo_042_10.py"),
            rel(ROOT / "src" / "run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py"): sha256(ROOT / "src" / "run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py"),
            rel(ROOT / "src" / "run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py"): sha256(ROOT / "src" / "run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py"),
            rel(ROOT / "src" / "run_sya_lowIC_ppo_i240_staged_reserve_040_26.py"): sha256(ROOT / "src" / "run_sya_lowIC_ppo_i240_staged_reserve_040_26.py"),
            rel(ROOT / "src" / "run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py"): sha256(ROOT / "src" / "run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py"),
        },
        "source_safety_state_path": rel(SAFETY_SOURCE),
        "source_safety_state_sha256": sha256(SAFETY_SOURCE),
        "existing_2014_daily_csv": rel(DAILY_2014),
        "existing_2014_daily_csv_sha256": sha256(DAILY_2014),
        "effective_min_days_between_fertilization_from_saved_config": min_interval,
        "action_grid": grid,
        "daily_rows_reconstructed": len(trace_df),
        "mask_vs_record_disagreement_rows": int(trace_df["expected_mask_vs_record_disagreement"].sum()),
        "post_step_mask_count_disagreement_rows": int((~trace_df["post_step_mask_count_matches_recorded_count"].fillna(True)).sum()),
        "pre_mask_vs_previous_post_count_disagreement_rows": int((~trace_df["current_pre_mask_count_matches_previous_row_post_count"].fillna(True)).sum()),
        "dap8_positive_n_rows": int(
            ((pd.to_numeric(trace_df["dap_pre_action"], errors="coerce") == 8)
             & (pd.to_numeric(trace_df["recorded_safe_action_anfer"], errors="coerce") > 0)).sum()
        ),
        "within_year_dap_resets_seed5_all_2014_2023": annual_reset_rows,
        "year_boundary_dap_resets_seed5_all_2014_2023": boundary_rows,
        "new_trace_path": rel(trace_path),
        "new_findings_path": rel(findings_path),
    }

    dap8 = trace_df[
        (pd.to_numeric(trace_df["dap_pre_action"], errors="coerce") == 8)
        & (pd.to_numeric(trace_df["recorded_safe_action_anfer"], errors="coerce") > 0)
    ]
    first_dap8 = dap8.iloc[0].to_dict() if not dap8.empty else None
    findings = [
        "# HL seed5 2014 action-mask 只读诊断",
        "",
        "## 执行状态",
        "",
        "- 动态 DSSAT runtime replay：未执行。宿主 Python 无法导入 `gym_dssat_pdi`；按冻结边界未启动/修改 Docker，也未安装依赖。",
        "- 已执行离线日志重建：使用当前正式 `LateIrrigationReserveMaskWrapper` 类、seed5 保存的 032_00 配置快照，以及原始行序的 2014 逐日 CSV。此结果不是运行时重放。",
        "- 未训练、未更改配置/源码/旧结果、未生成 WGEN 天气。",
        "",
        "## 检查到的差异",
        "",
        f"- 保存的最短施氮间隔为 `{min_interval}` 天。",
        f"- 2014 日志行数：`{len(trace_df)}`；按上一行安全施氮动作重建的决策前 mask 与记录不一致行数：`{metadata['mask_vs_record_disagreement_rows']}`。",
        f"- 用 HLA runner 实际 loader 得到季节灌溉上限 `{config['action_safety']['season_irrigation_soft_limit']}` mm；输出里的基础 YAML 上限为 `{base_template['action_safety']['season_irrigation_soft_limit']}` mm，因此初版 trace 与 findings 仅作存档，已被本版 supersede。",
        f"- 记录的 post-step mask count 与按下一行 DAP 重建的差异行数：`{metadata['post_step_mask_count_disagreement_rows']}`；当前行 pre-mask count 与上一行日志 post-mask count 的差异行数：`{metadata['pre_mask_vs_previous_post_count_disagreement_rows']}`。",
    ]
    if first_dap8:
        findings.extend(
            [
                f"- DAP 8 正施氮记录：决策前上次施氮 DAP=`{first_dap8['last_fertilization_dap_pre_action_reconstructed']}`，间隔=`{first_dap8['days_since_last_fertilization']}` 天，日志请求 action index=`{first_dap8['recorded_requested_action_index']}`，该动作在按当前代码重建的 mask 中为 `{first_dap8['requested_action_mask_bit_reconstructed']}`；日志记 `mask_forced_noop={first_dap8['recorded_mask_forced_noop']}`、实际 N=`{first_dap8['recorded_safe_action_anfer']}`。",
                "- 仅凭现有逐日日志重建时，DAP 8 的正施氮请求与 7 天规则不一致；重放期间记录的 post-step mask count 也用于核对下一决策状态。仍需在可用 DSSAT runtime 中同时捕获评估器 DAP、wrapper 内部 DAP/state 与 `predict` 前的真实 mask，才能定位状态/记录/运行时代码差异。",
            ]
        )
    else:
        findings.append("- 未找到 DAP 8 正施氮记录，无法执行目标状态对照。")
    findings.extend(
        [
            "",
            "## DAP 重置口径",
            "",
            f"- seed5 每个年度 CSV 内发现的 DAP 回退数：`{len(annual_reset_rows)}`；每次回退的累计灌溉、累计 N 和 done flag 见 metadata。",
            f"- 跨年度 CSV 连接处检测到的 DAP 回退数：`{len(boundary_rows)}`。年界处的 DAP 重置不代表同一年度 episode 内回退。",
            "",
            "## 产物与复核",
            "",
            f"- v2 逐行 mask 重建：`{trace_path.relative_to(ROOT).as_posix()}`",
            f"- checkpoint/config/source 哈希与环境探测：`{metadata_path.relative_to(ROOT).as_posix()}`",
            f"- 可重复执行脚本：`{Path(__file__).relative_to(ROOT).as_posix()}`",
            f"- 初版 `trace_or_reconstruction.csv` 和 `findings.md` 使用了未派生的基础 YAML；请以本版 v2 有效 loader 重建为准。",
            "- 没有确认运行时 mask 生效；离线重建仅显示按现行代码和保存配置计算应出现的行为。",
            "",
        ]
    )

    trace_df.to_csv(trace_path, index=False, encoding="utf-8-sig")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    findings_path.write_text("\n".join(findings), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
