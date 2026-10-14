from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TASK = ROOT / "benchmark_results" / "8hl05_hl_seed5_8seed"
SEED_CONFIG = TASK / "configs" / "hl_seed5.json"
CKPT = TASK / "models" / "HLA" / "HLA_half_split_stress_aware_maskableppo_seed5_ckpt100000.zip"
OLD_DAILY = TASK / "daily_outputs" / "HLA" / "HLA_2014_seed5_ckpt100000_daily.csv"
EXPECTED_CKPT_SHA256 = "a126d5685edc2eb44b18086acb70c7ae44fc715825496fc8035fd3b91e369ab9"
EXPECTED_SEED_CONFIG_SHA256 = "d9748ac511896c0f38b633750845dfb28ec2bfd73f2974baf7e69dc9b3144f40"
EXPECTED_DAILY_SHA256 = "4c04e7a1fe13bfacbf9d08fae315ecda006c66b36a9708f1e0a406a9ec3e3d32"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar(value: Any) -> float | None:
    try:
        arr = np.asarray(value).reshape(-1)
        if not len(arr):
            return None
        val = float(arr[0])
        return val if np.isfinite(val) else None
    except Exception:
        return None


def dap_value(mapping: Any) -> float | None:
    return scalar(mapping.get("dap")) if isinstance(mapping, dict) else None


def get_history_dap(env: Any) -> tuple[float | None, int | None]:
    raw = getattr(env.unwrapped, "history", {})
    if not isinstance(raw, dict):
        return None, None
    observations = raw.get("observation", [])
    if not isinstance(observations, (list, tuple)) or not observations:
        return None, 0
    last = observations[-1]
    return dap_value(last), len(observations)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def compare_to_saved(trace: pd.DataFrame, old: pd.DataFrame) -> dict[str, Any]:
    columns = {
        "evaluator_dap_pre": "dap",
        "requested_action_index": "requested_discrete_action_index",
        "wrapper_action_index": "discrete_action_index",
        "safe_action_amir": "safe_action_amir",
        "safe_action_anfer": "safe_action_anfer",
        "cumulative_irrigation_post": "season_cumulative_irrigation",
        "cumulative_n_post": "season_cumulative_n",
        "done": "done",
        "date_from_dap": "date",
    }
    report: dict[str, Any] = {
        "runtime_rows": int(len(trace)),
        "saved_rows": int(len(old)),
        "compared_rows": int(min(len(trace), len(old))),
        "field_mismatch_counts": {},
        "first_mismatch": None,
    }
    first: dict[str, Any] | None = None
    for runtime_col, saved_col in columns.items():
        if runtime_col not in trace or saved_col not in old:
            report["field_mismatch_counts"][runtime_col] = "column_missing"
            continue
        mismatches: list[int] = []
        for i in range(min(len(trace), len(old))):
            a = trace.iloc[i][runtime_col]
            b = old.iloc[i][saved_col]
            if runtime_col in {"safe_action_amir", "safe_action_anfer", "cumulative_irrigation_post", "cumulative_n_post", "evaluator_dap_pre"}:
                av, bv = scalar(a), scalar(b)
                equal = (av is None and bv is None) or (av is not None and bv is not None and np.isclose(av, bv, atol=1e-6))
            elif runtime_col == "done":
                equal = str(a).strip().lower() == str(b).strip().lower()
            else:
                equal = str(a) == str(b)
            if not equal:
                mismatches.append(i)
                if first is None:
                    first = {
                        "row_index_zero_based": i,
                        "runtime_column": runtime_col,
                        "saved_column": saved_col,
                        "runtime_value": a,
                        "saved_value": b,
                    }
        report["field_mismatch_counts"][runtime_col] = len(mismatches)
    if len(trace) != len(old) and first is None:
        first = {
            "row_index_zero_based": min(len(trace), len(old)),
            "runtime_column": "row_count",
            "saved_column": "row_count",
            "runtime_value": len(trace),
            "saved_value": len(old),
        }
    report["first_mismatch"] = first
    report["all_compared_fields_match"] = (
        len(trace) == len(old)
        and all(isinstance(v, int) and v == 0 for v in report["field_mismatch_counts"].values())
    )
    return report


def findings_text(metadata: dict[str, Any], trace: pd.DataFrame, comparison: dict[str, Any] | None) -> str:
    lines = [
        "# HL seed5 2014 runtime trace 只读诊断",
        "",
        f"- Runtime replay 状态：`{metadata.get('status')}`。",
        f"- 记录 step 数：`{len(trace)}`；自然 episode done：`{metadata.get('episode_done')}`；达到配置 max_steps：`{metadata.get('reached_step_cap')}`。",
        f"- 现有逐日 action sequence 全字段复现：`{comparison.get('all_compared_fields_match') if comparison else '未比较'}`。",
    ]
    if comparison:
        lines.append(f"- 与旧日志首个差异：`{json.dumps(comparison.get('first_mismatch'), ensure_ascii=False, default=str)}`。")
    if not trace.empty:
        row = trace.iloc[0]
        lines.extend([
            "",
            "## DAP / mask 关键核对",
            "",
            f"- 评估器 DAP 与 wrapper `_dap()` 全步一致：`{bool((trace['evaluator_dap_pre'] == trace['wrapper_dap_pre']).all())}`。",
            f"- wrapper DAP 与 underlying history DAP 全步一致：`{bool((trace['wrapper_dap_pre'] == trace['history_dap_pre']).all())}`。",
        ])
        resets = trace[trace["post_evaluator_dap"].notna() & trace["evaluator_dap_pre"].notna() & (trace["post_evaluator_dap"] < trace["evaluator_dap_pre"])]
        if len(resets):
            lines.append(f"- Episode 内观察到 DAP 回退：`{resets[['row_index_zero_based','evaluator_dap_pre','post_evaluator_dap','terminated','truncated']].to_dict(orient='records')}`。")
        dap8 = trace[trace["evaluator_dap_pre"].round().eq(8)]
        if len(dap8):
            lines.append(f"- DAP8 step mask 是否允许请求 action：`{dap8[['row_index_zero_based','requested_action_index','requested_action_allowed_by_mask','pre_mask_bits_0_to_15','last_fertilization_dap_pre','days_since_last_fertilization_pre','wrapper_action_index','mask_forced_noop','safe_action_anfer','post_evaluator_dap']].to_dict(orient='records')}`。")
        dap7 = trace[trace["evaluator_dap_pre"].round().eq(7)]
        if len(dap7):
            lines.append(f"- DAP7 step 后的 wrapper post-mask count/bits：`{dap7[['row_index_zero_based','post_mask_valid_count','post_mask_bits_0_to_15','last_fertilization_dap_post']].to_dict(orient='records')}`。")
        lines.extend([
            "",
            "## 证据边界",
            "",
            "- 这是现有 HLA seed5 100K checkpoint 在历史 `.WTH` 2014 输入上的单次 CPU runtime replay，不包含训练、更新 checkpoint 或随机天气。",
            "- 仅用该 replay 和旧逐日日志定位动作/状态/日志口径；本报告不自动修复配置或代码，不推断其他 seed。",
        ])
    else:
        lines.extend(["", "## 执行错误", "", f"```text\n{metadata.get('error', 'no trace rows')}\n```", ""])
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reserved = [OUT / "runtime_trace.csv", OUT / "metadata.json", OUT / "findings.md"]
    collisions = [p.name for p in reserved if p.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite diagnostic outputs: {collisions}")

    trace_rows: list[dict[str, Any]] = []
    env = None
    metadata: dict[str, Any] = {
        "task": "HL seed5 2014 runtime trace diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREFLIGHT",
        "repo_root": str(ROOT),
        "container_expected_project_root": "/workspace",
        "python": sys.version,
        "platform": platform.platform(),
        "device": "cpu",
        "training_or_checkpoint_update": False,
        "source_config_weather_action_reward_modified": False,
        "existing_task_outputs_overwritten": False,
        "wgen_generated": False,
    }
    trace = pd.DataFrame()
    comparison: dict[str, Any] | None = None
    try:
        for path in [SEED_CONFIG, CKPT, OLD_DAILY]:
            if not path.is_file():
                raise FileNotFoundError(path)
        hashes = {"seed_config": sha256(SEED_CONFIG), "checkpoint": sha256(CKPT), "saved_daily_csv": sha256(OLD_DAILY)}
        metadata["input_hashes"] = hashes
        if hashes["seed_config"] != EXPECTED_SEED_CONFIG_SHA256:
            raise RuntimeError(f"seed config hash changed: {hashes['seed_config']}")
        if hashes["checkpoint"] != EXPECTED_CKPT_SHA256:
            raise RuntimeError(f"checkpoint hash changed: {hashes['checkpoint']}")
        if hashes["saved_daily_csv"] != EXPECTED_DAILY_SHA256:
            raise RuntimeError(f"saved daily CSV hash changed: {hashes['saved_daily_csv']}")

        frozen = json.loads(SEED_CONFIG.read_text(encoding="utf-8"))
        src = ROOT / "src"
        for p in [src, src / "054_hla_lowIC_site_transfer"]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        adapter = importlib.import_module("run_seeded_site_transfer")
        runner = adapter.import_runner("HLA")
        engine = runner.engine
        engine.TASK_ID = str(frozen["task_id"])
        engine.TASK_NAME = str(frozen["task_name"])
        engine.BASE_OUT = TASK
        engine.BASE_DOC = ROOT / "docs" / "8hl05_hl_seed5_8seed_record.md"
        engine.PROMPT = runner.PROMPT
        engine.LOWIC_INPUT_ROOT = runner.INPUT_PROFILES[str(frozen["input_profile"])]
        engine.STATION = "HLA"
        engine.SITES = ["HLA"]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, frozen["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, frozen["actions"]["nitrogen_levels_kg_ha"]))
        engine.base03222.SEED = int(frozen["seed"])
        engine.base03222.SITE_NAMES["HLA"] = "HLA"

        engine.patch_base_module(OUT, OUT / "diagnostic_only.md", int(frozen["training"]["total_timesteps"]), [100000])
        import ppo_safe_rendering
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
        config = engine.load_config()
        config["paths"]["output_root"] = OUT.relative_to(ROOT).as_posix()
        split = engine.base03222.load_split()
        selection = engine.base03222.build_selection(split)
        env_config = engine.base03222.direct_ppo.build_env_config(config, selection)

        selected = engine.base03222.direct_ppo.find_year(env_config, "HLA", 2014)
        wth_path = Path(selected["weather_file"])
        if not wth_path.is_absolute():
            wth_path = ROOT / wth_path
        mask_cfg = config["action_safety"]
        if int(mask_cfg["min_days_between_fertilization"]) != 7:
            raise RuntimeError(f"effective min N interval is not the frozen 7 days: {mask_cfg['min_days_between_fertilization']}")
        grid = engine.base03222.base.action_grid(config)
        if len(grid) != 16:
            raise RuntimeError(f"expected frozen 16-action grid, found {len(grid)}")
        if not wth_path.is_file():
            raise FileNotFoundError(f"2014 historical weather file missing: {wth_path}")

        metadata.update({
            "status": "PREFLIGHT_PASSED",
            "station": "HLA",
            "seed": int(frozen["seed"]),
            "year": 2014,
            "checkpoint_path": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
            "effective_action_safety": mask_cfg,
            "action_grid": grid,
            "weather_file": str(wth_path.relative_to(ROOT)).replace("\\", "/"),
            "weather_file_sha256": sha256(wth_path),
            "planting_date": selected["planting_date"],
            "max_steps": int(config["runtime"]["max_steps"]),
            "effective_config_paths": config["paths"],
            "sources_sha256": {
                "seeded_runner": sha256(src / "hl_fq_8seed_cross_site" / "run_seeded_site_transfer.py"),
                "hla_runner": sha256(src / "054_hla_lowIC_site_transfer" / "run_054_00_hla_lowIC_expanded_action_maskableppo.py"),
                "engine_042_10": sha256(src / "run_sya_lowIC_binary_timing_maskableppo_042_10.py"),
                "mask_040_36": sha256(src / "run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py"),
                "safety": sha256(src / "ppo_action_safety.py"),
                "daily_runner": sha256(src / "run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py"),
            },
        })

        import torch
        torch.set_num_threads(1)
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        model = MaskablePPO.load(str(CKPT), device="cpu")
        metadata["status"] = "RUNNING"
        env = engine.base03222.base.make_env(
            config,
            env_config,
            "HLA",
            2014,
            int(frozen["seed"]),
            "HLA_2014_seed5_ckpt100000_runtime_trace_20260929_01",
            evaluation=True,
        )
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(selected["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = engine.base03222.base.latest_observation_dict(env, obs, info)
            eval_dap = engine.base03222.direct_ppo.scalar(latest.get("dap", step_count + 1))
            eval_dap = int(round(eval_dap)) if eval_dap is not None and eval_dap > 0 else step_count + 1
            wrapper_obs = dict(getattr(env, "last_obs_dict", {}))
            wrapper_dap = int(env._dap())
            hist_dap, history_rows = get_history_dap(env)
            pre_mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
            if len(pre_mask) != 16:
                raise RuntimeError(f"unexpected runtime mask length {len(pre_mask)} at step {step_count}")
            safety_pre = env.safety_state
            last_i_pre = safety_pre.last_irrigation_dap
            last_n_pre = safety_pre.last_fertilization_dap
            cum_i_pre = float(safety_pre.cumulative_irrigation)
            cum_n_pre = float(safety_pre.cumulative_n)
            action, _ = model.predict(obs, action_masks=pre_mask, deterministic=True)
            requested_idx = int(np.asarray(action).reshape(-1)[0])
            if requested_idx < 0 or requested_idx >= len(grid):
                raise RuntimeError(f"model returned out-of-grid action index {requested_idx}")
            requested_action = grid[requested_idx]
            step_date = planting + pd.Timedelta(days=max(eval_dap - 1, 0))
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            post_latest = engine.base03222.base.latest_observation_dict(env, obs, info)
            post_eval_dap = scalar(post_latest.get("dap"))
            post_wrapper_dap = int(env._dap())
            post_hist_dap, post_history_rows = get_history_dap(env)
            post_mask = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
            action_info = dict(env.last_action_info)
            safety_post = env.safety_state
            trace_rows.append({
                "row_index_zero_based": step_count,
                "episode_step_one_based": step_count + 1,
                "date_from_dap": step_date.strftime("%Y-%m-%d"),
                "evaluator_dap_pre": eval_dap,
                "wrapper_dap_pre": wrapper_dap,
                "wrapper_last_obs_dap_pre": scalar(wrapper_obs.get("dap")),
                "history_dap_pre": hist_dap,
                "history_rows_pre": history_rows,
                "last_irrigation_dap_pre": last_i_pre,
                "last_fertilization_dap_pre": last_n_pre,
                "days_since_last_fertilization_pre": None if last_n_pre is None else eval_dap - int(last_n_pre),
                "cumulative_irrigation_pre": cum_i_pre,
                "cumulative_n_pre": cum_n_pre,
                "pre_mask_bits_0_to_15": "".join("1" if x else "0" for x in pre_mask),
                "pre_mask_json": json.dumps(pre_mask.astype(int).tolist()),
                "pre_mask_valid_count": int(pre_mask.sum()),
                "requested_action_index": requested_idx,
                "requested_action_amir": float(requested_action["amir"]),
                "requested_action_anfer": float(requested_action["anfer"]),
                "requested_action_allowed_by_mask": bool(pre_mask[requested_idx]),
                "wrapper_action_index": action_info.get("discrete_action_index"),
                "mask_forced_noop": action_info.get("mask_forced_noop"),
                "raw_action_amir": action_info.get("raw_action_amir"),
                "raw_action_anfer": action_info.get("raw_action_anfer"),
                "safe_action_amir": action_info.get("safe_action_amir"),
                "safe_action_anfer": action_info.get("safe_action_anfer"),
                "action_safety_triggered": action_info.get("action_safety_triggered"),
                "cumulative_irrigation_post": action_info.get("season_cumulative_irrigation"),
                "cumulative_n_post": action_info.get("season_cumulative_n"),
                "post_mask_valid_count": action_info.get("mask_valid_action_count"),
                "post_mask_bits_0_to_15": "".join("1" if x else "0" for x in post_mask),
                "last_fertilization_dap_post": safety_post.last_fertilization_dap,
                "post_evaluator_dap": post_eval_dap,
                "post_wrapper_dap": post_wrapper_dap,
                "post_wrapper_last_obs_dap": scalar(getattr(env, "last_obs_dict", {}).get("dap")),
                "post_history_dap": post_hist_dap,
                "post_history_rows": post_history_rows,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "done": done,
            })
            step_count += 1
            if step_count % 25 == 0:
                print(f"trace_progress steps={step_count} pre_dap={eval_dap} post_dap={post_eval_dap} done={done}", flush=True)

        trace = pd.DataFrame(trace_rows)
        old = pd.read_csv(OLD_DAILY, keep_default_na=False)
        comparison = compare_to_saved(trace, old)
        natural_done = bool(len(trace) and trace.iloc[-1]["done"])
        metadata.update({
            "status": "COMPLETED" if natural_done else "STOPPED_AT_CONFIGURED_MAX_STEPS",
            "episode_done": natural_done,
            "reached_step_cap": bool(not natural_done and step_count >= int(config["runtime"]["max_steps"])),
            "steps": int(step_count),
            "saved_daily_comparison": comparison,
            "within_episode_dap_reset_steps": trace.loc[trace["post_evaluator_dap"].notna() & (trace["post_evaluator_dap"] < trace["evaluator_dap_pre"]), "row_index_zero_based"].astype(int).tolist() if not trace.empty else [],
            "runtime_files_written_only_to": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        })
    except Exception as exc:
        metadata["status"] = "FAILED"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback"] = traceback.format_exc()
        trace = pd.DataFrame(trace_rows)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                metadata["env_close_error"] = traceback.format_exc()

    trace.to_csv(OUT / "runtime_trace.csv", index=False, encoding="utf-8-sig")
    if comparison is not None:
        metadata["saved_daily_comparison"] = comparison
    write_json(OUT / "metadata.json", metadata)
    (OUT / "findings.md").write_text(findings_text(metadata, trace, comparison), encoding="utf-8")
    print(json.dumps({"status": metadata["status"], "steps": len(trace), "comparison": comparison, "output": str(OUT.relative_to(ROOT)).replace("\\", "/")}, ensure_ascii=False, default=str), flush=True)
    return 0 if metadata["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
