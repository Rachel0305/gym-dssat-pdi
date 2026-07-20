from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

import run_extension_expert_baseline_018_03 as extension
import run_hla_five_scenario_completion_020_11 as hla
import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as metric_source
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from stage_based_dqn_core_022 import (
    ACTION_TABLE_9,
    EXECUTABLE_STAGE_DAPS,
    FEASIBILITY_BONUS,
    IRRIGATION_BUDGET,
    NITROGEN_BUDGET,
    execute_stage_action,
    valid_action_indices,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmark_results" / "027_01"
OUT = DEFAULT_OUT
BASELINE_ROOT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_five_scenario_nstep_020_11"
BASELINE_SUMMARY = BASELINE_ROOT / "020_11_hla_five_scenario_summary.csv"
SCENARIOS = ("null", "recorded_farmer", "dssat_auto", "extension_expert")
LABELS = [
    "cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep", "srad",
    "sw_layer_1", "sw_layer_2", "sw_layer_3", "sw_layer_4",
    "sw_layer_5", "sw_layer_6", "sw_layer_7", "sw_layer_8",
    "swfac", "tmax", "topwt", "totir", "vstage", "wtdep", "xlai",
]
OBS_DIM = 24
STD_EPSILON = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(hla.YEAR_INPUTS[2010].iterdir()) if path.is_file()}


def baseline_rows() -> pd.DataFrame:
    source = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    rows = source[(source["year"].eq(2010)) & source["scenario"].isin(SCENARIOS)].copy()
    if set(rows["scenario"]) != set(SCENARIOS) or len(rows) != 4:
        raise ValueError("HLA2010 four-baseline set is incomplete")
    output = []
    for row in rows.to_dict("records"):
        replay = ROOT / str(row["source_file"])
        snapshot = replay.parent / "pdi_tmp_snapshot_eval"
        if not replay.exists() or not (snapshot / "Summary.OUT").exists():
            raise FileNotFoundError(f"Missing baseline evidence for {row['scenario']}: {replay}")
        metrics = metric_source.metrics_from_snapshot(
            snapshot,
            float(row["final_grain_kg_ha"]),
            float(row["irrigation_executed_total_mm"]),
            float(row["nitrogen_executed_total_kg_ha"]),
        )
        output.append({
            "site": "HLA", "year": 2010, "scenario": row["scenario"],
            "final_gwad": float(row["final_grain_kg_ha"]),
            "final_cwad": float(row["final_biomass_kg_ha"]),
            "irrigation_total": float(row["irrigation_executed_total_mm"]),
            "fertilizer_total": float(row["nitrogen_executed_total_kg_ha"]),
            "source_file": str(row["source_file"]), **metrics,
        })
    return pd.DataFrame(output).sort_values("scenario").reset_index(drop=True)


def scenario_schedule(scenario: str) -> pd.DataFrame:
    if scenario == "recorded_farmer":
        return hla.recorded_schedule()
    if scenario == "extension_expert":
        return hla.official_schedule()
    return pd.DataFrame(columns=["dap", "n_kg_ha_mid", "irrigation_mm_mid"])


def no_op(env: Any) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})


def real_action(env: Any, irrigation: float, nitrogen: float) -> np.ndarray:
    return normalize_action(
        env.formator.action_names,
        env.formator.action_space_dict,
        {"amir": float(irrigation), "anfer": float(nitrogen)},
    )


def collect_scenario(scenario: str, expected: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hla.OUT = OUT / "runtime_scaler"
    env_args, run_dir, copy_audit = hla.prepare_fixed_case(2010, scenario)
    schedule = scenario_schedule(scenario)
    scheduled = extension.split_irrigation_events(schedule) if not schedule.empty else {}
    env = extension.make_raw_env(env_args)
    stage_rows: list[dict[str, Any]] = []
    fired: set[int] = set()
    action_i = 0.0
    action_n = 0.0
    snapshot = run_dir / "pdi_tmp_snapshot_eval"
    try:
        obs, info = env.reset()
        for step in range(420):
            state = latest_observation_dict(env, obs, info)
            dap = int(round(float(scalar(state.get("dap"), step))))
            if dap in EXECUTABLE_STAGE_DAPS and not any(row["dap"] == dap for row in stage_rows):
                vector = np.asarray(obs, dtype=np.float64).reshape(-1)
                if vector.size != OBS_DIM or not np.isfinite(vector).all():
                    raise ValueError(f"{scenario} DAP{dap}: invalid observation shape/value {vector.shape}")
                stage_rows.append({
                    "scenario": scenario, "dap": dap,
                    **{label: float(value) for label, value in zip(LABELS, vector)},
                })
            if dap in scheduled and dap not in fired:
                request = scheduled[dap]
                fired.add(dap)
            else:
                request = {"amir": 0.0, "anfer": 0.0}
            action_i += float(request.get("amir", 0.0))
            action_n += float(request.get("anfer", 0.0))
            obs, _, terminated, truncated, info = env.step(
                real_action(env, request.get("amir", 0.0), request.get("anfer", 0.0))
            )
            if terminated or truncated:
                final = latest_observation_dict(env, obs, info)
                break
        else:
            raise RuntimeError(f"{scenario} did not terminate within 420 daily steps")
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
    if [row["dap"] for row in stage_rows] != list(EXECUTABLE_STAGE_DAPS):
        raise ValueError(f"{scenario}: missing stage states; got {[row['dap'] for row in stage_rows]}")
    final_yield = float(scalar(final.get("grnwt"), np.nan))
    final_biomass = float(scalar(final.get("topwt"), np.nan))
    metrics = metric_source.metrics_from_snapshot(
        snapshot,
        final_yield,
        float(expected["irrigation_total"]),
        float(expected["fertilizer_total"]),
    )
    # For native auto, actual management is read from Summary.OUT rather than external no-op requests.
    if scenario == "dssat_auto":
        action_i = float(metrics["summary_irrigation_total"])
        action_n = float(metrics["summary_nitrogen_total"])
    return stage_rows, {
        "scenario": scenario, "final_gwad": final_yield, "final_cwad": final_biomass,
        "irrigation_total": action_i, "fertilizer_total": action_n,
        "stage_state_count": len(stage_rows), "copied_file_count": len(copy_audit),
        "copied_hashes_match": bool(all(row["hash_match"] for row in copy_audit)),
        "run_dir": str(run_dir.relative_to(ROOT)), **metrics,
    }


def fit_scaler(states: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = states[LABELS].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    near_constant = std < STD_EPSILON
    scale = std.copy()
    scale[near_constant] = 1.0
    normalized = (values - mean) / scale
    variable = ~near_constant
    reconstructed = normalized * scale + mean
    validation = {
        "state_count": int(len(states)), "dimension": int(values.shape[1]),
        "near_constant_count": int(near_constant.sum()),
        "max_abs_normalized_mean_nonconstant": float(np.max(np.abs(normalized[:, variable].mean(axis=0)))) if variable.any() else 0.0,
        "max_abs_normalized_std_minus_one_nonconstant": float(np.max(np.abs(normalized[:, variable].std(axis=0) - 1.0))) if variable.any() else 0.0,
        "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - values))),
    }
    table = pd.DataFrame({
        "observation_index": np.arange(OBS_DIM), "observation_label": LABELS,
        "mean": mean, "scale_std_or_one": scale, "near_constant": near_constant,
    })
    return table, validation


class HLAStageEnv027(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, raw_env: Any, scaler: pd.DataFrame, null_yield: float, gate_yield: float) -> None:
        super().__init__()
        scaler = scaler.sort_values("observation_index")
        if scaler["observation_index"].tolist() != list(range(OBS_DIM)):
            raise ValueError("HLA scaler is not ordered 0..23")
        self.raw_env = raw_env
        self.mean = scaler["mean"].to_numpy(dtype=np.float32)
        self.scale = scaler["scale_std_or_one"].to_numpy(dtype=np.float32)
        self.null_yield = float(null_yield)
        self.gate_yield = float(gate_yield)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.obs = None
        self.state: dict[str, Any] = {}
        self.stage_index = 0
        self.used_i = self.used_n = 0.0
        self.invalid_attempts = 0
        self.stage_rows: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None

    def _dap(self) -> int:
        return int(round(float(scalar(self.state.get("dap"), -999))))

    def _raw(self, irrigation: float, nitrogen: float) -> np.ndarray:
        return real_action(self.raw_env, irrigation, nitrogen)

    def _scaled(self) -> np.ndarray:
        vector = np.asarray(self.obs, dtype=np.float32).reshape(-1)
        if vector.size != OBS_DIM or not np.isfinite(vector).all():
            raise ValueError(f"Invalid HLA observation {vector.shape}")
        return ((vector - self.mean) / self.scale).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        dap = EXECUTABLE_STAGE_DAPS[self.stage_index]
        remaining_i = IRRIGATION_BUDGET - self.used_i
        remaining_n = NITROGEN_BUDGET - self.used_n
        mask = np.zeros(9, dtype=bool)
        for index in valid_action_indices(dap):
            request = ACTION_TABLE_9[index]
            if request["amir"] <= remaining_i + 1e-9 and request["anfer"] <= remaining_n + 1e-9:
                mask[index] = True
        if not mask[0]:
            raise RuntimeError("No-op must remain valid")
        return mask

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.obs, info = self.raw_env.reset()
        self.state = latest_observation_dict(self.raw_env, self.obs, info)
        while self._dap() == 0:
            self.obs, _, terminated, truncated, info = self.raw_env.step(self._raw(0.0, 0.0))
            if terminated or truncated:
                raise RuntimeError("Season ended during DAP0 initialization")
            self.state = latest_observation_dict(self.raw_env, self.obs, info)
        if self._dap() != EXECUTABLE_STAGE_DAPS[0]:
            raise RuntimeError(f"Expected DAP1, got DAP{self._dap()}")
        self.stage_index = 0
        self.used_i = self.used_n = 0.0
        self.invalid_attempts = 0
        self.stage_rows = []
        self.last_result = None
        return self._scaled(), {"dap": self._dap(), "action_mask": self.action_masks().copy()}

    def step(self, action_index: int):
        action_index = int(action_index)
        mask = self.action_masks()
        if action_index < 0 or action_index >= 9 or not mask[action_index]:
            self.invalid_attempts += 1
            raise ValueError(f"Invalid/masked action {action_index} at DAP{self._dap()}")
        stage_dap = EXECUTABLE_STAGE_DAPS[self.stage_index]
        if self._dap() != stage_dap:
            raise RuntimeError(f"Expected DAP{stage_dap}, got DAP{self._dap()}")
        action = execute_stage_action(action_index, stage_dap, self.used_i, self.used_n)
        self.used_i += action.executed_irrigation
        self.used_n += action.executed_nitrogen
        resource_reward = -(action.executed_irrigation + 5.0 * action.executed_nitrogen) / 1000.0
        row = {"stage_index": self.stage_index, "dap": stage_dap, **asdict(action), "resource_reward": resource_reward}
        self.obs, _, terminated, truncated, info = self.raw_env.step(
            self._raw(action.executed_irrigation, action.executed_nitrogen)
        )
        self.state = latest_observation_dict(self.raw_env, self.obs, info)
        next_stage = EXECUTABLE_STAGE_DAPS[self.stage_index + 1] if self.stage_index + 1 < len(EXECUTABLE_STAGE_DAPS) else None
        while not (terminated or truncated) and next_stage is not None and self._dap() < next_stage:
            self.obs, _, terminated, truncated, info = self.raw_env.step(self._raw(0.0, 0.0))
            self.state = latest_observation_dict(self.raw_env, self.obs, info)
        if next_stage is not None and not (terminated or truncated) and self._dap() != next_stage:
            raise RuntimeError(f"DSSAT skipped DAP{next_stage}; current DAP{self._dap()}")
        self.stage_index += 1
        done = bool(terminated or truncated)
        if self.stage_index == len(EXECUTABLE_STAGE_DAPS) and not done:
            while not (terminated or truncated):
                self.obs, _, terminated, truncated, info = self.raw_env.step(self._raw(0.0, 0.0))
                self.state = latest_observation_dict(self.raw_env, self.obs, info)
            done = True
        reward = resource_reward
        if done:
            if self.stage_index != len(EXECUTABLE_STAGE_DAPS):
                raise RuntimeError("Season ended before six stage decisions")
            final_yield = float(scalar(self.state.get("grnwt"), 0.0))
            final_biomass = float(scalar(self.state.get("topwt"), 0.0))
            yield_gain = max(0.0, final_yield - self.null_yield)
            gate_bonus = FEASIBILITY_BONUS if final_yield >= self.gate_yield else 0.0
            terminal_reward = (yield_gain + gate_bonus) / 1000.0
            reward += terminal_reward
            row.update({"yield_gain": yield_gain, "gate_bonus": gate_bonus, "terminal_reward": terminal_reward})
            self.last_result = {
                "final_yield": final_yield, "final_biomass": final_biomass,
                "irrigation_total": self.used_i, "nitrogen_total": self.used_n,
            }
            obs_out = np.zeros(OBS_DIM, dtype=np.float32)
            info_out = {**self.last_result, "terminal_observation_available": False}
        else:
            row.update({"yield_gain": 0.0, "gate_bonus": 0.0, "terminal_reward": 0.0})
            obs_out = self._scaled()
            info_out = {"dap": self._dap(), "action_mask": self.action_masks().copy()}
        row["reward"] = reward
        self.stage_rows.append(row)
        return obs_out, float(reward), bool(terminated), bool(truncated), info_out

    def close(self) -> None:
        self.raw_env.close()


def prepare_smoke_run() -> tuple[Path, dict[str, Any]]:
    source = hla.YEAR_INPUTS[2010]
    run_dir = OUT / "smoke" / "hla2010_all_noop"
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)
    for item in sorted(source.iterdir()):
        if item.is_file():
            shutil.copyfile(item, input_dir / item.name)
    filex = hla.validate_input(input_dir, 2010)
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"), "mode": "all", "seed": 0,
        "random_weather": False, "evaluation": True,
        "fileX_template_path": str(filex), "experiment_number": 1,
        "auxiliary_file_paths": [str(path) for path in sorted(input_dir.iterdir()) if path.is_file() and path != filex],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, env_args


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    OUT = args.out if args.out.is_absolute() else ROOT / args.out
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    hashes_before = source_hashes()
    baselines = baseline_rows()
    baselines.to_csv(OUT / "027_01_hla2010_four_baselines.csv", index=False, encoding="utf-8-sig")
    state_rows: list[dict[str, Any]] = []
    rerun_rows: list[dict[str, Any]] = []
    expected_by_scenario = baselines.set_index("scenario").to_dict("index")
    for scenario in SCENARIOS:
        states, rerun = collect_scenario(scenario, expected_by_scenario[scenario])
        state_rows.extend(states)
        rerun_rows.append(rerun)
    states = pd.DataFrame(state_rows)
    reruns = pd.DataFrame(rerun_rows)
    states.to_csv(OUT / "027_01_hla2010_scaler_source_states.csv", index=False, encoding="utf-8-sig")
    reruns.to_csv(OUT / "027_01_hla2010_four_baseline_rerun_audit.csv", index=False, encoding="utf-8-sig")
    local_null = float(reruns.loc[reruns.scenario.eq("null"), "final_gwad"].iloc[0])
    local_gate = float(reruns.loc[reruns.scenario.isin(["dssat_auto", "extension_expert"]), "final_gwad"].max())
    reward_config = {
        "site": "HLA", "year": 2010, "local_null_yield": local_null,
        "local_feasibility_yield": local_gate, "water_cost": 1.0, "nitrogen_cost": 5.0,
        "feasibility_bonus": FEASIBILITY_BONUS,
        "irrigation_budget": IRRIGATION_BUDGET, "nitrogen_budget": NITROGEN_BUDGET,
        "recorded_used_in_reward": False,
        "precision_source": "fresh_deterministic_027_01_rerun_not_integer_020_11_summary",
    }
    (OUT / "027_01_hla2010_reward_config.json").write_text(json.dumps(reward_config, ensure_ascii=False, indent=2), encoding="utf-8")
    scaler, scaler_validation = fit_scaler(states)
    scaler.to_csv(OUT / "027_01_hla2010_observation_scaler.csv", index=False, encoding="utf-8-sig")

    expected = baselines.set_index("scenario")
    actual = reruns.set_index("scenario")
    rerun_max_yield_error = float(max(abs(actual.loc[s, "final_gwad"] - expected.loc[s, "final_gwad"]) for s in SCENARIOS))
    run_dir, env_args = prepare_smoke_run()
    raw_env = extension.make_raw_env(env_args)
    env = HLAStageEnv027(raw_env, scaler, local_null, local_gate)
    smoke_rows = []
    total_reward = 0.0
    snapshot = run_dir / "pdi_tmp_snapshot_eval"
    try:
        obs, info = env.reset()
        reset_finite = bool(obs.shape == (OBS_DIM,) and np.isfinite(obs).all())
        done = False
        while not done:
            mask = env.action_masks()
            obs, reward, terminated, truncated, info = env.step(0)
            total_reward += reward
            smoke_rows.append({**env.stage_rows[-1], "mask_noop_valid": bool(mask[0]), "observation_dimension_after": int(np.asarray(obs).size)})
            done = bool(terminated or truncated)
        if env.last_result is None:
            raise RuntimeError("Smoke ended without result")
    finally:
        tmp = getattr(raw_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
    smoke = pd.DataFrame(smoke_rows)
    smoke.to_csv(OUT / "027_01_hla2010_smoke_stage_actions.csv", index=False, encoding="utf-8-sig")
    smoke_metrics = metric_source.metrics_from_snapshot(
        snapshot, env.last_result["final_yield"], env.last_result["irrigation_total"], env.last_result["nitrogen_total"]
    )
    hashes_after = source_hashes()
    checks = {
        "four_baselines_present": len(baselines) == 4 and set(baselines.scenario) == set(SCENARIOS),
        "baseline_summary_metrics_finite": bool(np.isfinite(baselines[["final_gwad", "WP_ET_kg_m3"]].to_numpy(dtype=float)).all()),
        "rerun_all_four_scenarios": len(reruns) == 4,
        "rerun_yield_max_error_le_2": rerun_max_yield_error <= 2.0,
        "scaler_has_24_states": len(states) == 24,
        "scaler_has_24_dimensions": scaler_validation["dimension"] == 24,
        "scaler_normalized_mean_pass": scaler_validation["max_abs_normalized_mean_nonconstant"] < 1e-4,
        "scaler_normalized_std_pass": scaler_validation["max_abs_normalized_std_minus_one_nonconstant"] < 1e-4,
        "scaler_reconstruction_pass": scaler_validation["reconstruction_max_abs_error"] < 2e-3,
        "smoke_reset_observation_valid": reset_finite,
        "smoke_six_stage_actions": len(smoke) == 6 and smoke["dap"].tolist() == list(EXECUTABLE_STAGE_DAPS),
        "smoke_all_noop": bool(smoke["action_index"].eq(0).all()),
        "smoke_noop_masks_valid": bool(smoke["mask_noop_valid"].all()),
        "smoke_zero_invalid_actions": env.invalid_attempts == 0,
        "smoke_zero_water_nitrogen": env.last_result["irrigation_total"] == 0 and env.last_result["nitrogen_total"] == 0,
        "smoke_null_yield_error_le_2": abs(env.last_result["final_yield"] - local_null) <= 2.0,
        "smoke_summary_match": float(smoke_metrics["summary_match_score"]) <= 4.0,
        "source_hashes_unchanged": hashes_before == hashes_after,
        "training_steps_zero": True,
    }
    branch = "A_ready_for_027_02_seed0" if all(checks.values()) else "B_readiness_or_smoke_failed"
    payload = {
        "status": "completed", "branch": branch, "checks": checks,
        "scaler_validation": scaler_validation, "reward_config": reward_config,
        "rerun_max_yield_error_kg_ha": rerun_max_yield_error,
        "smoke_final_result": env.last_result, "smoke_total_reward": total_reward,
        "smoke_metrics": smoke_metrics, "training_steps": 0, "ppo_models_created": 0,
        "next_step_allowed": branch.startswith("A_"),
        "next_step": "Write 027_02 seed0 240-stage-step preregistration" if branch.startswith("A_") else "Diagnose only failed readiness checks",
    }
    (OUT / "027_01_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
