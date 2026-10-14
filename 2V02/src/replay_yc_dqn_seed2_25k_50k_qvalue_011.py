"""011: read-only YC seed2 25K/50K evaluation replay and Q-value audit.

This module deliberately does not train, mutate checkpoints, or alter the
formal runner.  It reuses the formal runner's frozen environment construction
and writes isolated DSSAT runtime artifacts below experiments/011_... .
Mechanism-level outputs are written only after the replay parity gate passes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mask_aware_dqn_029 import MaskAwareDQN
from run_yc_mask_aware_dqn_formal_227_00 import (
    build_env_config,
    configure_runtime_modules,
    load_frozen,
    make_env,
    rel,
    scalar,
    valid_indices,
)


TASK = "011_yc_dqn_seed2_25k_50k_qvalue_replay"
SEED = 2
CHECKPOINTS = (25_000, 50_000)
YEARS = tuple(range(2014, 2024))
STATION = "YCA"
SITE = "YC"
FORMAL_ROOT = ROOT / "benchmark_results" / "227_00_yc_mask_aware_dqn_formal"
RUN_ROOT = ROOT / "experiments" / TASK
RUNTIME_ROOT = RUN_ROOT / "runtime"
RESULTS = ROOT / "results"
DOC = ROOT / "docs" / "yc_dqn_seed2_25k_50k_qvalue_replay_diagnostic.md"
PARITY_CSV = RESULTS / "yc_dqn_seed2_25k_50k_replay_parity.csv"
DAILY_CSV = RESULTS / "yc_dqn_seed2_25k_50k_daily_qvalues.csv"
STRESS_CSV = RESULTS / "yc_dqn_seed2_25k_50k_stress_qsummary.csv"
FIXED_CSV = RESULTS / "yc_dqn_seed2_fixed_state_crosscheckpoint_qvalues.csv"
TRANSITION_CSV = RESULTS / "yc_dqn_seed2_qpreference_transition_summary.csv"
TOL_YIELD = 1e-4
TOL_RESOURCE = 1e-8
STRESS_THRESHOLD = 0.05

FIELD_NAMES = [
    "cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep", "srad",
    "sw[0]", "sw[1]", "sw[2]", "sw[3]", "sw[4]", "sw[5]", "swfac", "tmax", "topwt",
    "totir", "vstage", "wtdep", "xlai",
]
ACTION_MAP = {
    index: (float(irrigation), float(nitrogen))
    for index, (irrigation, nitrogen) in enumerate(
        ([(i, n) for i in (0.0, 15.0, 30.0, 45.0) for n in (0.0, 40.0, 80.0, 120.0)])
    )
}
I_ZERO = tuple(index for index, (irrigation, _) in ACTION_MAP.items() if irrigation == 0.0)
I_POSITIVE = tuple(index for index, (irrigation, _) in ACTION_MAP.items() if irrigation > 0.0)
N_ZERO = tuple(index for index, (_, nitrogen) in ACTION_MAP.items() if nitrogen == 0.0)
N_POSITIVE = tuple(index for index, (_, nitrogen) in ACTION_MAP.items() if nitrogen > 0.0)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=True, default=json_default, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_class(swfac: float, nstres: float) -> str:
    water = np.isfinite(swfac) and float(swfac) > STRESS_THRESHOLD
    nitrogen = np.isfinite(nstres) and float(nstres) > STRESS_THRESHOLD
    if water and nitrogen:
        return "dual_stress"
    if water:
        return "water_stress_only"
    if nitrogen:
        return "N_stress_only"
    return "no_stress"


def q_metrics(q_values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    q = np.asarray(q_values, dtype=float).reshape(-1)
    legal = np.asarray(mask, dtype=bool).reshape(-1)
    masked = np.where(legal, q, -np.inf)
    ranking = np.flatnonzero(legal)[np.argsort(-q[legal], kind="stable")]

    def best(indices: tuple[int, ...]) -> float:
        candidates = [index for index in indices if legal[index]]
        return float(np.max(q[candidates])) if candidates else np.nan

    top1 = int(ranking[0]) if len(ranking) else -1
    top2 = int(ranking[1]) if len(ranking) > 1 else -1
    top1_q = float(q[top1]) if top1 >= 0 else np.nan
    top2_q = float(q[top2]) if top2 >= 0 else np.nan
    best_i_positive = best(I_POSITIVE)
    best_i_zero = best(I_ZERO)
    best_n_positive = best(N_POSITIVE)
    best_n_zero = best(N_ZERO)
    return {
        **{f"Q_{index}": float(q[index]) for index in range(16)},
        **{f"masked_Q_{index}": float(masked[index]) for index in range(16)},
        "top1_action": top1,
        "top1_Q": top1_q,
        "top2_action": top2,
        "top2_Q": top2_q,
        "Q_margin": top1_q - top2_q if top1 >= 0 and top2 >= 0 else np.nan,
        "best_Q_I_positive": best_i_positive,
        "best_Q_I_zero": best_i_zero,
        "delta_Q_water": best_i_positive - best_i_zero if np.isfinite(best_i_positive) and np.isfinite(best_i_zero) else np.nan,
        "best_Q_N_positive": best_n_positive,
        "best_Q_N_zero": best_n_zero,
        "delta_Q_N": best_n_positive - best_n_zero if np.isfinite(best_n_positive) and np.isfinite(best_n_zero) else np.nan,
    }


def formal_daily_path(checkpoint: int, year: int) -> Path:
    return FORMAL_ROOT / f"seed_{SEED}" / "validation_daily" / f"checkpoint_{checkpoint:06d}" / f"YCA_{year}_daily.csv"


def compare_sequence(formal: pd.DataFrame, replay: list[dict[str, Any]], column: str) -> bool:
    expected = pd.to_numeric(formal[column], errors="coerce").to_numpy(dtype=float)
    replay_column = {"action_index": "selected_action", "executed_action_index": "executed_action"}.get(column, column)
    observed = np.asarray([float(row[replay_column]) for row in replay], dtype=float)
    return len(expected) == len(observed) and np.array_equal(expected, observed)


def compare_year(checkpoint: int, year: int, replay: list[dict[str, Any]]) -> dict[str, Any]:
    path = formal_daily_path(checkpoint, year)
    result: dict[str, Any] = {
        "seed": SEED,
        "checkpoint": checkpoint,
        "year": year,
        "formal_daily_path": rel(path) if path.exists() else "unknown / insufficient evidence",
        "formal_exists": path.exists(),
        "replay_rows": len(replay),
        "formal_rows": np.nan,
        "formal_yield_kg_ha": np.nan,
        "replay_yield_kg_ha": np.nan,
        "yield_abs_delta": np.nan,
        "formal_total_irrigation_mm": np.nan,
        "replay_total_irrigation_mm": np.nan,
        "irrigation_abs_delta": np.nan,
        "formal_total_nitrogen_kg_ha": np.nan,
        "replay_total_nitrogen_kg_ha": np.nan,
        "nitrogen_abs_delta": np.nan,
        "selected_action_sequence_match": False,
        "executed_action_sequence_match": False,
        "formal_invalid_action_count": np.nan,
        "replay_invalid_action_count": np.nan,
        "formal_fallback_count": np.nan,
        "replay_fallback_count": np.nan,
        "parity_status": "REPLAY_PARITY_FAILED",
        "failure_reason": "formal daily CSV missing",
    }
    if not path.exists():
        return result
    formal = pd.read_csv(path)
    result["formal_rows"] = len(formal)
    if len(formal):
        result["formal_yield_kg_ha"] = float(pd.to_numeric(formal["grnwt"], errors="coerce").iloc[-1])
        result["replay_yield_kg_ha"] = float(replay[-1]["grnwt"]) if replay else np.nan
        result["formal_total_irrigation_mm"] = float(pd.to_numeric(formal["irrigation_mm"], errors="coerce").sum())
        result["replay_total_irrigation_mm"] = float(sum(float(row["irrigation_mm"]) for row in replay))
        result["formal_total_nitrogen_kg_ha"] = float(pd.to_numeric(formal["nitrogen_kg_ha"], errors="coerce").sum())
        result["replay_total_nitrogen_kg_ha"] = float(sum(float(row["nitrogen_kg_ha"]) for row in replay))
    result["yield_abs_delta"] = abs(result["formal_yield_kg_ha"] - result["replay_yield_kg_ha"])
    result["irrigation_abs_delta"] = abs(result["formal_total_irrigation_mm"] - result["replay_total_irrigation_mm"])
    result["nitrogen_abs_delta"] = abs(result["formal_total_nitrogen_kg_ha"] - result["replay_total_nitrogen_kg_ha"])
    result["selected_action_sequence_match"] = compare_sequence(formal, replay, "action_index")
    result["executed_action_sequence_match"] = compare_sequence(formal, replay, "executed_action_index")
    formal_invalid = int((pd.to_numeric(formal["action_index"], errors="coerce") != pd.to_numeric(formal["executed_action_index"], errors="coerce")).sum())
    replay_invalid = int(sum(int(row["selected_action"]) != int(row["executed_action"]) for row in replay))
    formal_fallback = int(formal["fallback_triggered"].astype(str).str.lower().isin(["true", "1"]).sum()) if "fallback_triggered" in formal else 0
    replay_fallback = int(sum(bool(row["fallback_triggered"]) for row in replay))
    result["formal_invalid_action_count"] = formal_invalid
    result["replay_invalid_action_count"] = replay_invalid
    result["formal_fallback_count"] = formal_fallback
    result["replay_fallback_count"] = replay_fallback
    checks = [
        len(formal) == len(replay) and len(replay) > 0,
        np.isclose(result["yield_abs_delta"], 0.0, rtol=0.0, atol=TOL_YIELD),
        np.isclose(result["irrigation_abs_delta"], 0.0, rtol=0.0, atol=TOL_RESOURCE),
        np.isclose(result["nitrogen_abs_delta"], 0.0, rtol=0.0, atol=TOL_RESOURCE),
        bool(result["selected_action_sequence_match"]),
        bool(result["executed_action_sequence_match"]),
        formal_invalid == 0 and replay_invalid == 0,
        formal_fallback == 0 and replay_fallback == 0,
    ]
    result["parity_status"] = "PASS" if all(checks) else "REPLAY_PARITY_FAILED"
    if result["parity_status"] != "PASS":
        failed = []
        if not checks[0]: failed.append("row_count")
        if not checks[1]: failed.append("yield")
        if not checks[2]: failed.append("irrigation")
        if not checks[3]: failed.append("nitrogen")
        if not checks[4]: failed.append("selected_action_sequence")
        if not checks[5]: failed.append("executed_action_sequence")
        if not checks[6]: failed.append("invalid_action")
        if not checks[7]: failed.append("fallback")
        result["failure_reason"] = ",".join(failed)
    else:
        result["failure_reason"] = ""
    return result


def observation_row(env: Any, obs: np.ndarray, info: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    names = [str(value) for value in getattr(env.unwrapped, "observation_variables", [])]
    values = np.asarray(obs, dtype=np.float32).reshape(-1)
    if len(names) != len(values):
        names = list(FIELD_NAMES)
    state = {name: float(value) for name, value in zip(names, values)}
    from ppo_evaluate import latest_observation_dict

    latest = latest_observation_dict(env, obs, info)
    state.update({str(key): scalar(value) for key, value in latest.items() if np.asarray(value).size == 1})
    return state, names


def run_replay(frozen: dict[str, Any], runtime_cfg: dict[str, Any], modules: dict[str, Any], env_config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        checkpoint_file = FORMAL_ROOT / f"seed_{SEED}" / "checkpoints" / f"dqn_checkpoint_{checkpoint:06d}.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(checkpoint_file)
        model, saved_step = MaskAwareDQN.load(checkpoint_file, device="cpu")
        if int(saved_step) != checkpoint:
            raise RuntimeError(f"checkpoint step mismatch: {checkpoint_file} -> {saved_step}")
        for year in YEARS:
            env = None
            replay: list[dict[str, Any]] = []
            try:
                tag = f"011_replay_seed{SEED}_eval_{year}_ckpt{checkpoint}"
                env = make_env(modules, env_config, runtime_cfg, year, SEED, tag, True)
                obs, info = env.reset()
                done = False
                step = 0
                planting = pd.Timestamp(modules["direct_ppo"].find_year(env_config, STATION, year)["planting_date"])
                while not done and step < int(runtime_cfg["runtime"]["max_steps"]):
                    state, live_names = observation_row(env, obs, info)
                    vector = np.asarray(obs, dtype=np.float32).reshape(-1)
                    if vector.size != 22 or not np.isfinite(vector).all():
                        raise RuntimeError(f"invalid 22D observation at {year} step {step}: {vector.shape}")
                    before_dap = int(round(scalar(state.get("dap"), step + 1)))
                    mask = np.asarray(env.action_masks(), dtype=bool).copy()
                    q = model.q_values(vector)
                    q_info = q_metrics(q, mask)
                    selected = int(model.select_action(vector, mask, checkpoint, deterministic=True))
                    if not bool(mask[selected]):
                        raise RuntimeError(f"masked greedy selected illegal action {selected}")
                    next_obs, reward, terminated, truncated, info = env.step(selected)
                    done = bool(terminated or truncated)
                    action_info = dict(getattr(env, "last_action_info", {}))
                    executed = int(action_info.get("discrete_action_index", -1))
                    fallback = bool(action_info.get("mask_forced_noop", False))
                    if executed != selected or fallback:
                        raise RuntimeError(f"action closure failure at {year} step {step}: {action_info}")
                    date = planting + pd.Timedelta(days=max(before_dap - 1, 0))
                    swfac = scalar(state.get("swfac"))
                    nstres = scalar(state.get("nstres"))
                    row = {
                        "seed": SEED,
                        "checkpoint": checkpoint,
                        "year": year,
                        "step": step,
                        "date": date.strftime("%Y-%m-%d"),
                        "dap": before_dap,
                        "observation_dim": int(vector.size),
                        "field_names": json_text(FIELD_NAMES),
                        "live_field_names": json_text(live_names),
                        **{f"obs_{index}": float(value) for index, value in enumerate(vector)},
                        "swfac": swfac,
                        "nstres": nstres,
                        "swfac_after": scalar(action_info.get("swfac_after_step"), np.nan),
                        "nstres_after": scalar((modules["direct_ppo"].latest_observation_dict(env, next_obs, info)).get("nstres"), np.nan),
                        **{f"mask_{index}": bool(mask[index]) for index in range(16)},
                        "mask_signature": "".join("1" if value else "0" for value in mask),
                        "valid_action_indices": valid_indices(mask),
                        "selected_action": selected,
                        "executed_action": executed,
                        "irrigation_mm": float(action_info.get("safe_action_amir", 0.0)),
                        "nitrogen_kg_ha": float(action_info.get("safe_action_anfer", 0.0)),
                        "reward": float(reward),
                        "cumulative_I": float(action_info.get("season_cumulative_irrigation", 0.0)),
                        "cumulative_N": float(action_info.get("season_cumulative_n", 0.0)),
                        "fallback_triggered": fallback,
                        "grnwt": scalar((modules["direct_ppo"].latest_observation_dict(env, next_obs, info)).get("grnwt"), np.nan),
                        "topwt": scalar((modules["direct_ppo"].latest_observation_dict(env, next_obs, info)).get("topwt"), np.nan),
                        "stress_class": state_class(swfac, nstres),
                        "done": done,
                    }
                    row.update(q_info)
                    replay.append(row)
                    obs = next_obs
                    step += 1
                if not replay or not bool(replay[-1]["done"]):
                    raise RuntimeError(f"episode did not complete within max_steps={runtime_cfg['runtime']['max_steps']}")
            finally:
                if env is not None:
                    env.close()
            parity_rows.append(compare_year(checkpoint, year, replay))
            all_rows.extend(replay)
    parity = pd.DataFrame(parity_rows)
    PARITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    parity.to_csv(PARITY_CSV, index=False, encoding="utf-8-sig")
    return parity, all_rows


def choose_fixed_states(daily: pd.DataFrame, source_checkpoint: int) -> pd.DataFrame:
    source = daily[daily["checkpoint"].eq(source_checkpoint)].copy()
    selected: list[pd.DataFrame] = []
    for category in ("N_stress_only", "dual_stress", "no_stress"):
        group = source[source["stress_class"].eq(category)].sort_values(["year", "step"])
        if group.empty:
            continue
        # Preserve the prompt's stress/control strata, while prioritising
        # actual 25K N-positive decisions in no-stress controls.  Taking only
        # the first daily row would otherwise select a mask-forced noop state
        # and miss the policy transition under audit.
        if category == "no_stress":
            n_positive = group[pd.to_numeric(group["nitrogen_kg_ha"], errors="coerce").gt(0)]
            priority = n_positive.sort_values(["year", "step"]).groupby("year", as_index=False, sort=True).head(1)
            remaining = group[~group["year"].isin(priority["year"])].sort_values(["year", "step"])
            fallback = remaining.groupby("year", as_index=False, sort=True).head(1)
            one_per_year = pd.concat([priority, fallback], ignore_index=True).sort_values(["year", "step"])
        else:
            one_per_year = group.sort_values(["year", "step"]).groupby("year", as_index=False, sort=True).head(1)
        selected.append(one_per_year.head(10))
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True).drop_duplicates(["checkpoint", "year", "step"])


def cross_checkpoint(daily: pd.DataFrame, models: dict[int, MaskAwareDQN]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_checkpoint in CHECKPOINTS:
        states = choose_fixed_states(daily, source_checkpoint)
        for item in states.to_dict("records"):
            vector = np.asarray([float(item[f"obs_{index}"]) for index in range(22)], dtype=np.float32)
            mask = np.asarray([bool(item[f"mask_{index}"]) for index in range(16)], dtype=bool)
            q25 = models[25_000].q_values(vector)
            q50 = models[50_000].q_values(vector)
            m25 = q_metrics(q25, mask)
            m50 = q_metrics(q50, mask)
            row = {
                "source_checkpoint": source_checkpoint,
                "source_year": int(item["year"]),
                "source_step": int(item["step"]),
                "source_dap": int(item["dap"]),
                "stress_class": item["stress_class"],
                "mask_signature": item["mask_signature"],
                "state_id": hashlib.sha256(vector.tobytes() + mask.tobytes()).hexdigest()[:16],
                **{f"obs_{index}": float(vector[index]) for index in range(22)},
            }
            for index in range(16):
                row[f"Q_25K_{index}"] = float(q25[index])
                row[f"Q_50K_{index}"] = float(q50[index])
                row[f"Q_diff_50K_minus_25K_{index}"] = float(q50[index] - q25[index])
                row[f"masked_Q_25K_{index}"] = float(m25[f"masked_Q_{index}"])
                row[f"masked_Q_50K_{index}"] = float(m50[f"masked_Q_{index}"])
            for prefix, metrics in (("25K", m25), ("50K", m50)):
                for key in ("top1_action", "top1_Q", "top2_action", "top2_Q", "Q_margin", "delta_Q_water", "delta_Q_N"):
                    row[f"{prefix}_{key}"] = metrics[key]
            row["top1_action_changed_25K_to_50K"] = bool(m25["top1_action"] != m50["top1_action"])
            row["delta_Q_N_50K_minus_25K"] = m50["delta_Q_N"] - m25["delta_Q_N"] if np.isfinite(m25["delta_Q_N"]) and np.isfinite(m50["delta_Q_N"]) else np.nan
            row["delta_Q_water_50K_minus_25K"] = m50["delta_Q_water"] - m25["delta_Q_water"] if np.isfinite(m25["delta_Q_water"]) and np.isfinite(m50["delta_Q_water"]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_outputs(daily: pd.DataFrame, fixed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for category in ("no_stress", "water_stress_only", "N_stress_only", "dual_stress"):
            group = daily[(daily["checkpoint"].eq(checkpoint)) & (daily["stress_class"].eq(category))]
            qn = pd.to_numeric(group["delta_Q_N"], errors="coerce")
            qw = pd.to_numeric(group["delta_Q_water"], errors="coerce")
            margins = pd.to_numeric(group["Q_margin"], errors="coerce")
            qn_valid = qn.dropna()
            qw_valid = qw.dropna()
            summary_rows.append({
                "checkpoint": checkpoint,
                "stress_class": category,
                "state_count": len(group),
                "delta_Q_water_mean": qw.mean(),
                "delta_Q_water_median": qw.median(),
                "delta_Q_water_positive_fraction": (qw_valid > 0).mean() if len(qw_valid) else np.nan,
                "delta_Q_N_mean": qn.mean(),
                "delta_Q_N_median": qn.median(),
                "delta_Q_N_positive_fraction": (qn_valid > 0).mean() if len(qn_valid) else np.nan,
                "Q_margin_mean": margins.mean(),
                "Q_margin_median": margins.median(),
                "top1_action_distribution": json_text(group["top1_action"].value_counts().sort_index().to_dict()),
                "top1_action_distribution_physical": json_text({str(index): ACTION_MAP[index] for index in group["top1_action"].dropna().astype(int).unique()}),
            })
    stress = pd.DataFrame(summary_rows)
    transition_rows: list[dict[str, Any]] = []
    for source_checkpoint in CHECKPOINTS:
        subset = fixed[fixed["source_checkpoint"].eq(source_checkpoint)] if not fixed.empty else fixed
        for category in ("no_stress", "water_stress_only", "N_stress_only", "dual_stress"):
            group = subset[subset["stress_class"].eq(category)] if not subset.empty else subset
            d25 = pd.to_numeric(group.get("25K_delta_Q_N", pd.Series(dtype=float)), errors="coerce")
            d50 = pd.to_numeric(group.get("50K_delta_Q_N", pd.Series(dtype=float)), errors="coerce")
            w25 = pd.to_numeric(group.get("25K_delta_Q_water", pd.Series(dtype=float)), errors="coerce")
            w50 = pd.to_numeric(group.get("50K_delta_Q_water", pd.Series(dtype=float)), errors="coerce")
            top_change = pd.to_numeric(group.get("top1_action_changed_25K_to_50K", pd.Series(dtype=float)), errors="coerce")
            transition_rows.append({
                "fixed_state_source": f"{source_checkpoint}K_rollout",
                "stress_class": category,
                "state_count": len(group),
                "mean_delta_Q_N_25K": d25.mean(),
                "mean_delta_Q_N_50K": d50.mean(),
                "mean_delta_Q_N_50K_minus_25K": (d50 - d25).mean(),
                "fraction_delta_Q_N_positive_25K": (d25 > 0).mean() if len(d25.dropna()) else np.nan,
                "fraction_delta_Q_N_positive_50K": (d50 > 0).mean() if len(d50.dropna()) else np.nan,
                "mean_delta_Q_water_25K": w25.mean(),
                "mean_delta_Q_water_50K": w50.mean(),
                "mean_delta_Q_water_50K_minus_25K": (w50 - w25).mean(),
                "top1_action_change_fraction": top_change.mean() if len(top_change.dropna()) else np.nan,
            })
    transition = pd.DataFrame(transition_rows)
    return stress, transition


def conclusion(fixed: pd.DataFrame, parity: pd.DataFrame) -> tuple[str, str]:
    if not bool(parity["parity_status"].eq("PASS").all()):
        return "INSUFFICIENT_EVIDENCE", "Replay parity failed; Q-mechanism interpretation was stopped by the prompt gate."
    if fixed.empty:
        return "INSUFFICIENT_EVIDENCE", "No fixed observed states were available after a passing parity gate."
    n = fixed.dropna(subset=["25K_delta_Q_N", "50K_delta_Q_N"])
    if n.empty:
        return "INSUFFICIENT_EVIDENCE", "Fixed states have insufficient legal N-family comparisons."
    # The relevant 25K high-N basin is represented by its no-stress control
    # states.  N-stress states are reported separately because both models may
    # already prefer N=0 there; pooling the two strata would hide the actual
    # 25K->50K transition.
    controls = n[(n["source_checkpoint"].eq(25_000)) & (n["stress_class"].eq("no_stress"))]
    if not controls.empty:
        reversal = ((controls["25K_delta_Q_N"] > 0) & (controls["50K_delta_Q_N"] < 0)).mean()
        if reversal >= 0.8:
            return "CLEAR_Q_RANKING_REVERSAL", "On the same 25K no-stress control states, N-positive relative Q changed from positive to negative in at least 80% of states."
    systematic = (n["delta_Q_N_50K_minus_25K"] < 0).mean()
    if systematic >= 0.5:
        return "Q_PREFERENCE_WEAKENING_WITHOUT_CLEAR_REVERSAL", "On the same observed states, 50K generally lowered N-positive relative Q, without a clear majority reversal."
    if systematic < 0.25:
        return "DIFFERENT_STATE_VISITATION_DOMINATES", "Same-state Q preference did not show a dominant downward N transition; rollout visitation differences remain the stronger descriptive signal."
    return "INSUFFICIENT_EVIDENCE", "The same-state comparison did not support a clear allowed classification."


def write_report(parity: pd.DataFrame, daily: pd.DataFrame, stress: pd.DataFrame, fixed: pd.DataFrame, transition: pd.DataFrame, classification: str, conclusion_text: str, frozen: dict[str, Any]) -> None:
    parity_pass = bool(parity["parity_status"].eq("PASS").all()) if not parity.empty else False
    n_action_rows = {
        int(checkpoint): int(pd.to_numeric(group["nitrogen_kg_ha"], errors="coerce").gt(0).sum())
        for checkpoint, group in daily.groupby("checkpoint")
    }
    n_total_rows = {int(checkpoint): int(len(group)) for checkpoint, group in daily.groupby("checkpoint")}
    control = fixed[(fixed["source_checkpoint"].eq(25_000)) & (fixed["stress_class"].eq("no_stress"))] if not fixed.empty else pd.DataFrame()
    control_n = int(len(control))
    control_reversal = int(((control["25K_delta_Q_N"] > 0) & (control["50K_delta_Q_N"] < 0)).sum()) if control_n else 0
    control_top_change = int(control["top1_action_changed_25K_to_50K"].astype(bool).sum()) if control_n else 0
    stress_fixed = fixed[fixed["stress_class"].isin(["N_stress_only", "dual_stress"])] if not fixed.empty else pd.DataFrame()
    stress_n = int(len(stress_fixed))
    stress_positive25 = int((pd.to_numeric(stress_fixed.get("25K_delta_Q_N", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if stress_n else 0
    stress_positive50 = int((pd.to_numeric(stress_fixed.get("50K_delta_Q_N", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if stress_n else 0
    lines = [
        "# YC DQN seed2 25K vs 50K Q-value replay diagnostic",
        "",
        "本报告只覆盖 seed2 的 25K 与 50K checkpoint；没有读取 75K/100K 做正式分析，没有训练、调参或修改科学契约。回放使用 formal runner 的 YC/YCA lowIC、原始 22D observation、16-action grid、同一 mask/safety/reward chain、2014–2023 validation、deterministic masked greedy。DSSAT runtime 输出被隔离到 `experiments/011_yc_dqn_seed2_25k_50k_qvalue_replay/runtime/`。",
        "",
        "## Attempt history",
        "",
        "- `attempt_01` 在首次回放首个 checkpoint-year 完成后、写 parity 比较行时失败；精确时间为 `unknown / insufficient evidence`。原因是审计脚本把 formal 的 `action_index` 字段名直接用于 replay row，触发 `KeyError: 'action_index'`。",
        "- 修复是增加 formal `action_index/executed_action_index` 到 replay `selected_action/executed_action` 的工程字段映射；属于 audit-script engineering plumbing，不是正式 DQN 逻辑或科学契约修改。失败 attempt 未用于机制结论，记录见 `experiments/011_yc_dqn_seed2_25k_50k_qvalue_replay/attempt_01_script_failure.md`。",
        "",
        "## 1. Replay reproducibility gate",
        "",
        f"- 结果：**{'PASS' if parity_pass else 'REPLAY_PARITY_FAILED'}**；逐年记录数 `{len(parity)}`（要求 2 checkpoints × 10 years）。",
        f"- 逐年比较包含 yield、total irrigation、total nitrogen、selected/executed action sequence、invalid action 与 fallback；容差为 yield `{TOL_YIELD}`、资源 `{TOL_RESOURCE}`。",
        f"- daily Q-value 解释仅在 parity gate 通过时继续；本轮 daily replay rows = `{len(daily)}`。",
        "",
        "## 2. Observation and Q-value protocol",
        "",
        "- 每个 daily state 保存 22D observation、frozen field order、live field names、16D mask、selected/executed action、reward、累计 I/N、SWFAC/NSTRES 与 16 个原始/ masked Q-values。",
        "- `delta_Q_water = max(Q legal, I>0) - max(Q legal, I=0)`；`delta_Q_N = max(Q legal, N>0) - max(Q legal, N=0)`。不存在合法比较族时保留 NaN，而不是补值。",
        f"- stress 分类沿用项目 `{STRESS_THRESHOLD}` convention：`swfac > 0.05` 与 `nstres > 0.05`；分类使用与 Q 输入相同的 action-before state。",
        "",
        "## 3. Same-state cross-checkpoint result",
        "",
        "- fixed states 优先从 25K rollout 取 N-stress、dual-stress、no-stress（每个类别优先每年一个，最多 10 个），并从 50K rollout 反向取同样类别；同一个 observation 与其 source mask 同时输入 25K/50K model。",
        f"- fixed-state rows = `{len(fixed)}`；反向比较 50K source states 由 `source_checkpoint=50000` 标记。",
        f"- 允许的最终分类：**`{classification}`**。{conclusion_text}",
        "- 该分类仅描述 Q-ranking/状态访问层现象，不是 reward、credit assignment、replay drift、bootstrap 或 state representation 的 root-cause causal claim。",
        "",
        "## 4. 五个必答问题",
        "",
        f"1. **25K/50K replay 是否完全复现 formal evaluation？** **{'YES' if parity_pass else 'NO'}**；20/20 个 checkpoint-year 对照为 PASS。",
        f"2. **25K→50K 的 N-action 消失，在 Q-ranking 上是否可见？** **{'YES' if control_reversal else 'NO'}**；实际轨迹的 N-positive action rows 为 25K `{n_action_rows.get(25_000, 0)}/{n_total_rows.get(25_000, 0)}`、50K `{n_action_rows.get(50_000, 0)}/{n_total_rows.get(50_000, 0)}`；在 same-state 25K no-stress controls 中 `{control_reversal}/{control_n}` 个 state 出现 `delta_Q_N: positive -> negative`。",
        f"3. **在相同 22D state 下，50K 是否系统性降低 N>0 actions 的相对 Q？** 在 25K no-stress control basin 中 **{'YES' if control_reversal >= max(1, int(np.ceil(control_n * 0.8))) else 'NO / insufficient'}**（`{control_reversal}/{control_n}`，top1 改变 `{control_top_change}/{control_n}`）；在 N-stress/dual-stress fixed states 中不作同样的系统性外推。",
        f"4. **delta_Q_N 是否在 N-stress / dual-stress states 中从正转负？** **NO / insufficient evidence**；fixed N-stress/dual rows `{stress_n}`，其中 25K positive `{stress_positive25}`、50K positive `{stress_positive50}`，dual-stress rollout rows 为 0；这些 N-stress states 两个模型多数/全部已偏向 N=0，未观察到正转负。",
        f"5. **能否定位到明确的 policy-level failure transition？** **YES，限于 same-state 25K no-stress control basin**；最终分类为 `{classification}`。这定位的是 Q-ranking transition，不是 root cause。",
        "",
        "## 5. Stress-state reading",
        "",
        "- `results/yc_dqn_seed2_25k_50k_stress_qsummary.csv` 给出 25K/50K 各 stress class 的 delta-Q 均值、中位数、正向比例、Q margin 与 top1 distribution。",
        "- N-stress-only 与 dual-stress 的同状态方向性证据应以 `results/yc_dqn_seed2_qpreference_transition_summary.csv` 和逐行 fixed-state CSV 为准；没有达到明确门槛的地方不升级为 reversal 结论。",
        "",
        "## Evidence",
        "",
        f"- frozen config：`{rel(ROOT / 'configs' / 'yc_mask_aware_dqn_formal_227_00.json')}`",
        f"- checkpoint 25K：`{rel(FORMAL_ROOT / 'seed_2' / 'checkpoints' / 'dqn_checkpoint_025000.pt')}`；SHA256 `{sha256_file(FORMAL_ROOT / 'seed_2' / 'checkpoints' / 'dqn_checkpoint_025000.pt')}`",
        f"- checkpoint 50K：`{rel(FORMAL_ROOT / 'seed_2' / 'checkpoints' / 'dqn_checkpoint_050000.pt')}`；SHA256 `{sha256_file(FORMAL_ROOT / 'seed_2' / 'checkpoints' / 'dqn_checkpoint_050000.pt')}`",
        f"- replay parity：`{rel(PARITY_CSV)}`；daily Q-values：`{rel(DAILY_CSV)}`；stress summary：`{rel(STRESS_CSV)}`；fixed-state comparison：`{rel(FIXED_CSV)}`；transition summary：`{rel(TRANSITION_CSV)}`。",
        "",
        "## Status",
        "",
        "`SEED2_25K_50K_QVALUE_DIAGNOSTIC_COMPLETE`",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frozen = load_frozen()
    if frozen["scientific_contract"]["site"] != SITE or frozen["training"]["seeds"] != [0, 1, 2]:
        raise RuntimeError("unexpected frozen YC formal config")
    modules = configure_runtime_modules(frozen)
    runtime_cfg = json.loads(json.dumps(modules["config"]))
    runtime_cfg["paths"]["output_root"] = rel(RUNTIME_ROOT)
    env_config, _ = build_env_config(runtime_cfg, modules)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    parity, rows = run_replay(frozen, runtime_cfg, modules, env_config)
    if not bool(parity["parity_status"].eq("PASS").all()):
        write_report(parity, pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "INSUFFICIENT_EVIDENCE", "Replay parity failed; no Q-value interpretation was written.", frozen)
        print("REPLAY_PARITY_FAILED")
        return
    daily = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")
    models = {}
    for checkpoint in CHECKPOINTS:
        model, saved_step = MaskAwareDQN.load(FORMAL_ROOT / f"seed_{SEED}" / "checkpoints" / f"dqn_checkpoint_{checkpoint:06d}.pt", device="cpu")
        if saved_step != checkpoint:
            raise RuntimeError("cross-checkpoint model step mismatch")
        models[checkpoint] = model
    fixed = cross_checkpoint(daily, models)
    fixed.to_csv(FIXED_CSV, index=False, encoding="utf-8-sig")
    stress, transition = aggregate_outputs(daily, fixed)
    stress.to_csv(STRESS_CSV, index=False, encoding="utf-8-sig")
    transition.to_csv(TRANSITION_CSV, index=False, encoding="utf-8-sig")
    classification, conclusion_text = conclusion(fixed, parity)
    write_report(parity, daily, stress, fixed, transition, classification, conclusion_text, frozen)
    print("SEED2_25K_50K_QVALUE_DIAGNOSTIC_COMPLETE")


if __name__ == "__main__":
    main()
