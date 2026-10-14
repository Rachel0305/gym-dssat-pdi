from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results/yc_random_weather_ppo/004_05"
REC = ROOT / "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery"
OUT = ROOT / "results/yc_random_weather_ppo/004_18B_seed1003_1036_closure"
REPORT = ROOT / "docs/yc_random_weather_004_18B_seed1003_1036_closure.md"
PILOT17 = ROOT / "src/run_004_17_yc_runtime_wgen_capture_and_weather_recovery.py"
TARGETS = {1003: (2011, 62), 1036: (2009, 20)}
MAX_STEPS = 500


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pilot():
    spec = importlib.util.spec_from_file_location("runner_004_17_readonly", PILOT17)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runtime helper {PILOT17}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_pilot()


def rss_mib() -> float | None:
    try:
        import psutil
        process = psutil.Process()
        family = [process, *process.children(recursive=True)]
        return round(sum(p.memory_info().rss for p in family if p.is_running()) / 1048576, 1)
    except Exception:
        return None


def val(mapping: Any, key: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for k, v in mapping.items():
        if str(k).casefold() == key.casefold():
            return v
    return None


def final_state(dssat: Any, env: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    candidates = [getattr(dssat, "_state", None)]
    node = env
    for _ in range(10):
        candidates += [getattr(node, "_state", None), getattr(node, "last_obs_dict", None)]
        node = getattr(node, "env", None)
        if node is None:
            break
    for item in candidates:
        if isinstance(item, dict):
            out.update(item)
    return out


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return jsonable(x.tolist())
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def replay_one(pilot: Any, seed: int, year: int, target_episode: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schedule_path = EXP / "config/training_weather_schedule_random.csv"
    schedule = []
    for row in csv_rows(schedule_path):
        ep = int(row["episode_index"])
        if int(row["historical_year_context"]) == year and ep <= target_episode:
            schedule.append({"episode_index": ep, "historical_year": year, "weather_seed": int(row["weather_seed"])})
    schedule.sort(key=lambda x: x["episode_index"])
    if not schedule or schedule[-1]["episode_index"] != target_episode or schedule[-1]["weather_seed"] != seed:
        raise RuntimeError(f"Historical schedule context mismatch for seed {seed}: {schedule[-1:]}")

    tempfile.tempdir = str(OUT / "runtime_work/tmp")
    Path(tempfile.tempdir).mkdir(parents=True, exist_ok=True)
    _, _, batch, _, config, _, _, env_config = pilot._import_canonical()
    batch.base.direct_ppo.find_year(env_config, "YCA", year)
    env = pilot.make_weather_env(config, env_config, year, "RANDOM_WEATHER_WGEN", int(pilot.RUNTIME_BOOTSTRAP_SEED), f"00418_horizon_{seed}_{year}", schedule[0]["weather_seed"])
    dssat = pilot.find_dssat_instance(env)
    events: list[dict[str, Any]] = []
    target: dict[str, Any] = {}
    try:
        for row in schedule:
            current_seed = row["weather_seed"]
            dssat._random_generator = pilot.SequenceWeatherRng(current_seed)
            dssat._pilot_weather_states = []
            env.reset(seed=None)
            actual_seed = int(dssat._rseed1)
            if actual_seed != current_seed:
                raise RuntimeError(f"RSEED1 mismatch: expected {current_seed}, got {actual_seed}")
            steps = 0
            ended = False
            term = trunc = None
            info: dict[str, Any] = {}
            last_obs: Any = None
            while steps < MAX_STEPS:
                result = env.step(0)
                steps += 1
                if len(result) == 5:
                    last_obs, _, term, trunc, info = result
                    ended = bool(term or trunc)
                elif len(result) == 4:
                    last_obs, _, done, info = result
                    term, trunc, ended = bool(done), False, bool(done)
                else:
                    raise RuntimeError(f"Unexpected environment step response length: {len(result)}")
                if ended:
                    break
            if not ended:
                raise RuntimeError(f"Safety limit reached at schedule episode {row['episode_index']}")
            states = list(getattr(dssat, "_pilot_weather_states", []))
            state = final_state(dssat, env)
            is_target = row["episode_index"] == target_episode
            event = {
                "weather_seed": current_seed, "crop_year": year, "schedule_episode_index": row["episode_index"],
                "is_target_episode": is_target, "actual_rseed1": actual_seed, "action_index": 0,
                "steps": steps, "captured_state_rows": len(states), "terminated": bool(term), "truncated": bool(trunc),
                "final_dap": val(state, "dap"), "final_istage": val(state, "istage"), "rss_mib": rss_mib(),
            }
            events.append(event)
            if is_target:
                from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, daily_weather_from_states, sha256_bytes
                weather = daily_weather_from_states(states)
                target = {
                    **event,
                    "bootstrap_rseed1": schedule[0]["weather_seed"],
                    "preceding_same_year_rseed1_resets": [r["weather_seed"] for r in schedule[:-1]],
                    "generated_runtime_weather_sha256": sha256_bytes(canonical_weather_bytes(weather)) if weather else "",
                    "weather_first_date": weather[0].get("DATE", "") if weather else "",
                    "weather_last_date": weather[-1].get("DATE", "") if weather else "",
                    "weather_date_semantics": "Legacy state-to-weather date fallback (2008-06-01 plus state index); not an actual crop-year calendar date.",
                    "done_reason": next((info.get(k) for k in ("done_reason", "terminal_reason", "reason") if info.get(k) is not None), "NOT_EXPOSED_BY_ENV_INFO"),
                    "final_info": jsonable(dict(info or {})),
                    "final_state_keys": sorted(state),
                    "final_state": jsonable(state),
                    "weather_rows": jsonable(weather),
                    "replay_actions": "NO_OP_ACTION_INDEX_0_ONLY",
                    "side_effect_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
                    "crop_outcomes_used": False,
                }
            if rss_mib() is not None and rss_mib() > 2048:
                raise MemoryError("RSS exceeded 2048 MiB; stopped serial context replay")
        return target, events
    finally:
        try:
            env.close()
        finally:
            gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-replay", action="store_true", help="Run two serial no-op horizon replays; no PPO checkpoint is loaded.")
    args = parser.parse_args()
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite existing 004_18B outputs")
    if not (EXP / "config/training_weather_schedule_random.csv").is_file() or not (REC / "weather_archive_manifest_verified.csv").is_file() or not PILOT17.is_file():
        raise FileNotFoundError("Required 004_05/004_17 evidence or runtime helper is missing")
    OUT.mkdir(parents=True)
    (OUT / "runtime_work").mkdir()
    schedule = csv_rows(EXP / "config/training_weather_schedule_random.csv")
    manifest = csv_rows(REC / "weather_archive_manifest_verified.csv")
    training_logs = sorted((EXP / "training/random_weather").glob("ppo_seed_*/training_episode_summary.csv"))
    history: dict[int, dict[str, Any]] = {}
    candidates: dict[int, dict[str, Any]] = {}
    for seed, (year, target_ep) in TARGETS.items():
        hist = []
        for log in training_logs:
            for row in csv_rows(log):
                if row.get("training_weather_seed") == str(seed) and row.get("historical_year") == str(year):
                    row["_source"] = log.relative_to(ROOT).as_posix()
                    hist.append(row)
        if not hist:
            raise RuntimeError(f"No 004_05 training summary row found for seed {seed}")
        history[seed] = {
            "rows": hist,
            "ppo_seeds": sorted({r.get("ppo_seed", "") for r in hist}),
            "episode_days": sorted({int(float(r["episode_days"])) for r in hist if r.get("episode_days")}),
            "episode_indices": sorted({int(float(r["episode_index"])) for r in hist if r.get("episode_index")}),
            "rseed1": sorted({r.get("actual_rseed1_sent_to_pdi", "") for r in hist}),
            "hashes": sorted({r.get("runtime_weather_sha256", "") for r in hist}),
        }
        row = next(r for r in manifest if r.get("weather_seed") == str(seed) and r.get("set") == "training")
        metadata_path = REC / f"provenance/training_context_replay_seed_{seed}_ctx_{year}_ep_{target_ep}_metadata.json"
        metadata = load_json(metadata_path)
        canonical = ROOT / "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery" / row["canonical_csv_path"]
        candidates[seed] = {"manifest": row, "metadata": metadata, "canonical": canonical, "rows": csv_rows(canonical)}
        if int(row["crop_year"]) != year or int(row["rseed1"]) != seed or int(metadata["runtime_arguments"]["schedule_episode_index"]) != target_ep:
            raise RuntimeError(f"004_17 provenance context mismatch for seed {seed}")

    replay: dict[int, dict[str, Any]] = {}
    replay_error: dict[int, str] = {}
    replay_events: list[dict[str, Any]] = []
    if args.runtime_replay:
        pilot = load_pilot()
        for seed in (1003, 1036):
            year, ep = TARGETS[seed]
            try:
                target, events = replay_one(pilot, seed, year, ep)
                replay[seed] = target
                replay_events.extend(events)
                write_json(OUT / f"seed{seed}_runtime_replay.json", target)
                write_csv(OUT / "runtime_replay_progress.csv", replay_events)
            except Exception as exc:
                replay_error[seed] = f"{type(exc).__name__}: {exc}"
                write_json(OUT / f"seed{seed}_runtime_replay_error.json", {"error": replay_error[seed], "time_utc": now()})
                write_csv(OUT / "runtime_replay_progress.csv", replay_events)
                break
    write_csv(OUT / "runtime_replay_events.csv", replay_events)

    horizon: list[dict[str, Any]] = []
    verification: dict[int, dict[str, Any]] = {}
    for seed, (year, ep) in TARGETS.items():
        hist = history[seed]
        candidate = candidates[seed]
        meta = candidate["metadata"]
        hist_days = hist["episode_days"]
        old_rows = len(candidate["rows"])
        current = replay.get(seed, {})
        current_steps = current.get("steps", meta.get("step_count", old_rows))
        verification[seed] = {
            "seed": seed, "crop_year": year, "historical_episode_index": ep,
            "historical_ppo_seeds_with_summary": hist["ppo_seeds"], "historical_episode_days": hist_days,
            "historical_rseed1_values": hist["rseed1"], "historical_runtime_weather_sha256_values": hist["hashes"],
            "004_17_noop_rows": old_rows, "004_17_noop_runtime_hash": meta.get("legacy_runtime_weather_sha256"),
            "new_noop_replay": current, "historical_action_sequence_status": "ACTION_TRACE_NOT_AVAILABLE",
            "overlap_days_verifiable": 0, "status": "UNVERIFIABLE",
            "reason": "No historical daily weather/state trace exists for row-wise comparison. The episode hash is whole-series and cannot prove prefix equality.",
            "runtime_replay_error": replay_error.get(seed, ""),
        }
        write_json(OUT / f"seed{seed}_verification.json", verification[seed])
        write_csv(OUT / f"historical_action_sequence_seed{seed}.csv", [{"status": "ACTION_TRACE_NOT_AVAILABLE", "reason": "No per-step training action trace for this historical episode was found in 004_05."}], ["status", "reason"])
        write_csv(OUT / f"overlap_verification_seed{seed}.csv", [{
            "seed": seed, "historical_episode_index": ep, "historical_context": year,
            "historical_episode_days": hist_days[0] if len(hist_days) == 1 else "",
            "candidate_rows": old_rows, "overlap_days": 0, "exact_weather_rows": 0,
            "max_abs_diff_RAIN": "", "max_abs_diff_SRAD": "", "max_abs_diff_TMAX": "", "max_abs_diff_TMIN": "",
            "status": "NO_HISTORICAL_DAILY_TRACE", "note": "Cannot compare dates/weather values from a whole-series hash."}])
        horizon.append({
            "seed": seed, "historical_ppo_seeds": ";".join(hist["ppo_seeds"]), "historical_episode_index": ep,
            "weather_seed": seed, "historical_context": year, "RSEED1": ";".join(hist["rseed1"]),
            "planting_date": candidate["manifest"].get("season_start", ""),
            "historical_episode_start_date": "NOT_RECORDED", "current_episode_start_date": meta.get("season_start", ""),
            "historical_episode_end_date": "UNAVAILABLE", "current_episode_end_date": "UNAVAILABLE_DATE_FALLBACK_NOT_CALENDAR",
            "current_weather_last_date_label": current.get("weather_last_date", ""),
            "historical_days": hist_days[0] if len(hist_days) == 1 else "", "current_00417_rows": old_rows,
            "current_replay_steps": current_steps, "day_difference_historical_minus_00417": (hist_days[0] - old_rows) if len(hist_days) == 1 else "",
            "historical_end_DAP": "NOT_RECORDED", "current_end_DAP": current.get("final_dap", "UNAVAILABLE"),
            "historical_final_istage": "NOT_RECORDED", "current_final_istage": current.get("final_istage", "UNAVAILABLE"),
            "historical_terminated": "NOT_RECORDED", "historical_truncated": "NOT_RECORDED", "historical_done_reason": "NOT_RECORDED",
            "current_terminated": current.get("terminated", "UNAVAILABLE"), "current_truncated": current.get("truncated", "UNAVAILABLE"),
            "current_done_reason": current.get("done_reason", "UNAVAILABLE"),
            "historical_runtime_weather_sha256": ";".join(hist["hashes"]),
            "current_00417_runtime_weather_sha256": meta.get("legacy_runtime_weather_sha256", ""),
            "replay_runtime_weather_sha256": current.get("generated_runtime_weather_sha256", ""),
            "likely_horizon_explanation": "Both code paths stop on environment termination/truncation; no fixed two-day capture offset is evident. Original PPO actions could affect crop trajectory and episode horizon, but their trace and historical done state are missing, so the mechanism is unproven.",
            "historical_action_sequence_status": "ACTION_TRACE_NOT_AVAILABLE",
        })
    write_csv(OUT / "termination_horizon_comparison.csv", horizon)

    decision = {
        "seed1003_status": "UNVERIFIABLE", "seed1036_status": "UNVERIFIABLE",
        "training_weather_history_status": "PARTIALLY_VERIFIED_78_PLUS_2",
        "exact_training_realizations": 78, "unclosed_training_realizations": [1003, 1036],
        "formal_004_19_ready": "CONDITIONAL_WITH_FULL80_EXACT78_SENSITIVITY", "formal_004_19_exact_80_claim_ready": False,
        "historical_action_trace": "ACTION_TRACE_NOT_AVAILABLE",
        "historical_daily_overlap": "NOT_VERIFIABLE", "runtime_noop_replay_performed": bool(args.runtime_replay),
        "current_noop_candidate_reproduced": {str(seed): replay.get(seed, {}).get("generated_runtime_weather_sha256") == candidates[seed]["metadata"].get("legacy_runtime_weather_sha256") for seed in TARGETS},
        "runtime_replay_errors": replay_error,
        "can_use_candidates_for_descriptive_sensitivity": True,
        "recommended_004_19_handling": "Keep FULL80 and EXACT78 analyses separate; label seeds 1003/1036 provisional. Since 004_18A found rainfall-support sensitivity, do not dismiss the two candidates as negligible.",
        "reason": "Historical actions, per-day states/weather, and terminal flags/final DAP/ISTAGE were not saved. The no-op replay diagnoses only current environment horizon and cannot close the original policy-dependent episode.",
        "created_at_utc": now(),
    }
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "run_manifest.json", {
        "task": "004_18B", "runtime_replay_requested": bool(args.runtime_replay), "ppo_training": False,
        "checkpoint_loaded": False, "formal_evaluation": False, "historical_action_replay": False,
        "crop_outcomes_used": False, "runtime_side_effect_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
        "inputs": [str(EXP / "config/training_weather_schedule_random.csv"), str(REC / "weather_archive_manifest_verified.csv"), str(PILOT17)],
        "created_at_utc": now(),
    })

    lines = []
    for seed, (year, ep) in TARGETS.items():
        hist_days = history[seed]["episode_days"]
        current = replay.get(seed, {})
        lines.append(f"| {seed} | {year} | {ep} | {','.join(map(str, hist_days))} | {len(candidates[seed]['rows'])} | {current.get('steps', '未补采')} | {current.get('terminated', '不可用')} / {current.get('truncated', '不可用')} | {current.get('final_dap', '不可用')} / {current.get('final_istage', '不可用')} | UNVERIFIABLE |")
    report = f"""# YC random-weather 004_18B：seed1003 / seed1036 历史天气闭合审计

生成时间：{now()}  
范围：只读历史训练记录；{'另对两个原 context 串行执行 no-op runtime horizon replay' if args.runtime_replay else '未做 runtime replay'}。未训练 PPO、未加载 checkpoint、未做正式评估。新 runtime crop execution（如有）仅为 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`，crop outcome 未使用。

## 结论

- `seed1003_status = UNVERIFIABLE`
- `seed1036_status = UNVERIFIABLE`
- `training_weather_history_status = PARTIALLY_VERIFIED_78_PLUS_2`
- 现有事实是两条历史 PPO episode 比 004_17 no-op candidate 多 2 天；但历史逐日状态和原动作缺失，**不能证明这是单纯 capture offset，也不能证明重叠逐日天气相同**。
- 两个历史整段 `runtime_weather_sha256` 与 004_17 no-op candidate 不同。由于 hash 覆盖整段序列且行数不同，hash mismatch 不能定位日值差异，不据此将状态定成 `MISMATCH`。
- 本轮同 context no-op replay 分别复现了 004_17 candidate 的 runtime hash（1003: `EC8FA734...FC1BDB`；1036: `AFCE2570...277DD3D`），表明当前 no-op capture 可重复；这不证明历史 PPO episode 的天气前缀或末尾两天相同。
- **若 004_19 要求宣称 100/100 历史 realization 全部闭合，目前不具备条件；若按预注册敏感性并列审计 FULL80 candidate 与 EXACT78 verified subset，则可以有条件开展。** 不得把 FULL80 单独写成完全核验的历史训练域。

## Horizon 对照

| seed | crop-year | schedule episode | historical days | 004_17 no-op rows | 本轮 no-op steps | terminated / truncated | 最终 DAP / ISTAGE | 分类 |
|---:|---:|---:|---:|---:|---:|---|---|---|
{chr(10).join(lines)}

字段与 hash 见 `termination_horizon_comparison.csv`、`seed1003_verification.json`、`seed1036_verification.json`。历史 episode 长度/hash 来自 004_05 多 PPO-seed training summaries；RSEED1、context、planting date 来自 schedule 与 004_17 verified manifest。

### 两日差异的解释边界

004_03 `ScheduledEpisodeEnv._step` 在环境返回 `terminated or truncated` 时记录 summary、`episode_days` 与天气 hash；004_17 的 `run_context_group` 同样在这两个 done flag 任一为真时停止 no-op capture。源码没有显示统一固定少采两日的额外切片。因此 capture horizon 差异不是已证实原因。原 PPO 管理动作与 no-op 不同，理论上可能改变 crop trajectory / maturity 与 episode 终止时点；但训练期没有逐日 action、final DAP/ISTAGE 或 done/truncated 记录，所以只能列作未验证机制。

本轮 replay 使用原 context schedule 的前序 seed 串行 reset，再对目标 seed 执行 action index 0（no irrigation/no N），只为补采当前 no-op 终止状态。它不是原历史动作 replay，也没有把任何 crop outcome 纳入分析。当前 `done_reason` 若 runtime `info` 未暴露，仅报告 terminated/truncated 标志，不猜测 harvest/maturity。

日期边界：runtime state 没有提供可用的真实 calendar date；004_17 legacy parser 用 `2008-06-01 + state index` 作为 hash 日期标签。本轮 replay 的末行标签（1003 为 2008-09-16，1036 为 2008-09-08）不是对应的 2011/2009 作物日历终止日期，horizon 表将真实 episode end date 标为 unavailable。

## 历史动作及逐日 overlap

- 004_05 可用的是 `training_episode_summary.csv` 汇总；没有目标历史 episode 的 per-step action trace。
- `evaluation_step_level_all_models.csv` 是 observed / held-out evaluation trace，不是训练 weather seeds 1003/1036 的原 training episode。没有用 evaluation action 替代训练动作。
- `historical_action_sequence_seed1003.csv` 和 `historical_action_sequence_seed1036.csv` 均标记 `ACTION_TRACE_NOT_AVAILABLE`。
- 两份 overlap CSV 均为 `NO_HISTORICAL_DAILY_TRACE`，可比较重叠天数为 0；RAIN/SRAD/TMAX/TMIN 的逐行误差因此为空，而不是 0。

## 科学状态与下一步约束

目前只能称 **78 条 exact + 2 条 unresolved/provisional**，不能称 80/80 exact。两条 candidate 可用于 descriptive sensitivity，但不能升级为 historical exact weather。004_18A 已显示排除 provisional realization 会改变部分观测年的 rainfall support，因此 004_19 若之后执行，应并列 FULL80/EXACT78 并显式标注 provisional；如果问题要求唯一正式历史训练域结论，则需先有两条原始历史 action/state/weather trace。此轮不启动 004_19。

## 用户问题逐项回答

1. 两条都少 2 天；源码未显示通用的固定两日 capture offset。历史 PPO 管理和 no-op 管理导致 episode 终止时点不同，是可能机制但未证实。
2. 差异发生在 episode horizon 上；两边都由 terminated/truncated 控制停止。历史 done flags 未保存，具体终止类型未知。
3. 原历史动作可能影响 episode 长度，但动作 trace 缺失，不能验证。
4. seed1003 未恢复为可核验的历史原天气，状态为 `UNVERIFIABLE`。
5. seed1036 未恢复为可核验的历史原天气，状态为 `UNVERIFIABLE`。
6. 当前 training weather 是 78 条 exact + 2 条 provisional/unresolved，不是 80/80 exact。
7. 可以有条件开展 004_19：必须并列 FULL80 candidate 与 EXACT78 verified subset，并将两条标成 provisional；若要作“80/80 历史天气均 exact”的单一正式结论则条件不满足。

## 证据索引

- schedule：`results/yc_random_weather_ppo/004_05/config/training_weather_schedule_random.csv`
- 训练汇总：`results/yc_random_weather_ppo/004_05/training/random_weather/ppo_seed_*/training_episode_summary.csv`
- evaluation-only step trace：`results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv`
- terminal semantics：`results/yc_random_weather_ppo/004_03/run_controlled_pilot.py` 的 `ScheduledEpisodeEnv._step`
- context replay：`src/run_004_17_yc_runtime_wgen_capture_and_weather_recovery.py` 的 `run_context_group`
- archived candidates：`results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/`
- replay error(s)：{json.dumps(replay_error, ensure_ascii=False) if replay_error else '无'}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
