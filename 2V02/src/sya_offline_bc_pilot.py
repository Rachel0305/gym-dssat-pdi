"""Low-compute SY offline trajectories and behavior cloning pilot.

The module is isolated from PPO training. It reuses the verified 046_10
wrapper for collection/replay and never changes reward, masks, actions, or
existing checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sya_offline_bc_pilot.json"
PPO_CONFIG_PATH = ROOT / "configs" / "046_10_sya_originIC_expanded_action_maskableppo.json"
INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
RECORDED_SOURCE = ROOT / "benchmark_results" / "027_05" / "027_05_daily_values.csv"
EXPERT_SOURCE = ROOT / "benchmark_results" / "046_03_sya_originIC_four_baselines" / "evaluation" / "046_03_baseline_daily.csv"
PPO_CHECKPOINT = ROOT / "benchmark_results" / "046_10_sya_originIC_expanded_action_maskableppo" / "models" / "SYA" / "SYA_half_split_stress_aware_maskableppo_seed0_ckpt100000.zip"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def action_grid(config: dict[str, Any]) -> list[dict[str, float]]:
    actions = config["action_grid"]
    return [{"amir": float(i), "anfer": float(n)} for i in actions["irrigation_levels_mm"] for n in actions["nitrogen_levels_kg_ha"]]


def runtime_action_grid(runtime_cfg: dict[str, Any]) -> list[dict[str, float]]:
    actions = runtime_cfg["discrete_actions"]
    return [{"amir": float(i), "anfer": float(n)} for i in actions["irrigation_levels"] for n in actions["nitrogen_levels"]]


def nearest_legal_action(requested_i: float, requested_n: float, legal_mask: np.ndarray, grid: list[dict[str, float]]) -> int:
    legal = [int(i) for i, ok in enumerate(np.asarray(legal_mask, dtype=bool)) if ok]
    if not legal:
        return 0
    if abs(requested_i) < 1e-9 and abs(requested_n) < 1e-9 and 0 in legal:
        return 0
    scale_i = max(max(x["amir"] for x in grid), 1.0)
    scale_n = max(max(x["anfer"] for x in grid), 1.0)
    scored = []
    for idx in legal:
        candidate = grid[idx]
        distance = ((candidate["amir"] - requested_i) / scale_i) ** 2 + ((candidate["anfer"] - requested_n) / scale_n) ** 2
        scored.append((distance, idx))
    return int(min(scored, key=lambda item: (item[0], item[1]))[1])


def expanded_observation_names(raw_names: list[str], raw_observation: dict[str, Any] | None = None) -> list[str]:
    names: list[str] = []
    for variable in raw_names:
        value = None if raw_observation is None else raw_observation.get(variable)
        size = int(np.asarray(value).size) if value is not None else 1
        if size <= 1:
            names.append(str(variable))
        elif variable == "sw":
            names.extend([f"sw_{index + 1}" for index in range(size)])
        else:
            names.extend([f"{variable}_{index + 1}" for index in range(size)])
    return names


def state_indices(schema: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(schema)}


def state_scalar(state: np.ndarray, indices: dict[str, int], key: str, default: float = 0.0) -> float:
    index = indices.get(key)
    if index is None or index >= len(state):
        return default
    return finite_float(state[index], default)


def state_sw_mean(state: np.ndarray, indices: dict[str, int]) -> float:
    values = [state_scalar(state, indices, f"sw_{index}", math.nan) for index in range(1, 10)]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def read_schedule_source(path: Path, kind: str) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    manifest: dict[str, Any] = {"kind": kind, "path": rel(path), "exists": path.exists()}
    if not path.exists():
        manifest["status"] = "missing_source"
        return {}, manifest
    frame = pd.read_csv(path, keep_default_na=False)
    if kind == "recorded_farmer_template":
        if "scenario" in frame.columns:
            frame = frame[frame["scenario"].astype(str).eq("recorded_farmer")]
        frame = frame[frame.get("site", pd.Series(dtype=str)).astype(str).eq("SY")]
        i_col, n_col = "irrigation_executed_mm", "nitrogen_executed_kg_ha"
        manifest["status"] = "static_template_not_annual_real_farmer_record"
    else:
        frame = frame[frame.get("station_code", pd.Series(dtype=str)).astype(str).eq("SYA") & frame.get("scenario", pd.Series(dtype=str)).astype(str).eq("official_extension_expert")]
        if "year" in frame.columns and frame["year"].astype(str).eq("2014").any():
            frame = frame[frame["year"].astype(str).eq("2014")]
        i_col, n_col = "irrigation_requested_mm", "nitrogen_requested_kg_ha"
        manifest["status"] = "project_extension_expert_schedule_existing_output"
    required = ["dap", i_col, n_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        manifest.update({"status": "missing_schedule_columns", "missing_columns": missing})
        return {}, manifest
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame[(frame[i_col].abs() > 1e-9) | (frame[n_col].abs() > 1e-9)]
    if kind == "recorded_farmer_template":
        # 027_05 repeats the static recorded template in multiple report rows.
        # Match the project's existing deduplication contract before summing.
        frame = frame.assign(dap_key=frame["dap"].round().astype(int).clip(lower=1))
        frame = frame.drop_duplicates(subset=["site", "dap_key", i_col, n_col], keep="first")
    schedule: dict[int, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        dap = max(1, int(round(float(getattr(row, "dap")))))
        schedule.setdefault(dap, {"amir": 0.0, "anfer": 0.0})
        schedule[dap]["amir"] += float(getattr(row, i_col))
        schedule[dap]["anfer"] += float(getattr(row, n_col))
    manifest.update({"event_count": len(schedule), "event_daps": sorted(schedule), "total_irrigation_mm": sum(x["amir"] for x in schedule.values()), "total_nitrogen_kg_ha": sum(x["anfer"] for x in schedule.values())})
    return schedule, manifest


class BehaviorPolicy:
    policy_id = "base"
    policy_type = "unknown"
    include_in_bc = True
    oracle_policy = False

    def reset(self) -> None:
        return None

    def parameters(self) -> dict[str, Any]:
        return {}

    def choose(self, state: np.ndarray, legal_mask: np.ndarray, dap: int, grid: list[dict[str, float]], indices: dict[str, int]) -> dict[str, Any]:
        raise NotImplementedError


class ScheduleProjectionPolicy(BehaviorPolicy):
    policy_type = "existing_schedule_grid_projection"

    def __init__(self, policy_id: str, schedule: dict[int, dict[str, float]]) -> None:
        self.policy_id = policy_id
        self.schedule = schedule

    def parameters(self) -> dict[str, Any]:
        return {"schedule_event_daps": sorted(self.schedule), "projection": "nearest_legal_joint_action"}

    def choose(self, state, legal_mask, dap, grid, indices):
        request = self.schedule.get(dap, {"amir": 0.0, "anfer": 0.0})
        index = nearest_legal_action(request["amir"], request["anfer"], legal_mask, grid)
        return {"action_index": index, "source_requested_amir": request["amir"], "source_requested_anfer": request["anfer"], "action_projection": "nearest_legal_joint_action", "policy_note": "existing schedule projected onto current legal grid; not exact replay", **grid[index]}


class ObservableStateRulePolicy(BehaviorPolicy):
    policy_id = "state_rule_observable"
    policy_type = "deterministic_observable_state_rule"

    def __init__(self, policy_id: str | None = None, water_threshold: float = 0.20, nitrogen_threshold: float = 0.05, early_i: float = 15.0, mid_i: float = 30.0, high_i: float = 45.0, early_n: float = 80.0, regular_n: float = 40.0) -> None:
        self.policy_id = policy_id or "state_rule_observable"
        self.water_threshold = float(water_threshold)
        self.nitrogen_threshold = float(nitrogen_threshold)
        self.early_i, self.mid_i, self.high_i = float(early_i), float(mid_i), float(high_i)
        self.early_n, self.regular_n = float(early_n), float(regular_n)

    def parameters(self) -> dict[str, Any]:
        return {"water_threshold": self.water_threshold, "nitrogen_threshold": self.nitrogen_threshold, "early_i": self.early_i, "mid_i": self.mid_i, "high_i": self.high_i, "early_n": self.early_n, "regular_n": self.regular_n}

    def choose(self, state, legal_mask, dap, grid, indices):
        sw = state_sw_mean(state, indices)
        swfac = state_scalar(state, indices, "swfac", 0.0)
        nstres = state_scalar(state, indices, "nstres", 0.0)
        cumsum_n = state_scalar(state, indices, "cumsumfert", 0.0)
        water_low = (math.isfinite(sw) and sw < self.water_threshold) or swfac > 0.05
        nitrogen_low = nstres > self.nitrogen_threshold or (dap <= 30 and cumsum_n < 40.0)
        if dap <= 30:
            requested_i, requested_n = (self.high_i if water_low else self.early_i), (self.early_n if nitrogen_low else self.regular_n)
        elif dap <= 60:
            requested_i, requested_n = (self.high_i if water_low else self.mid_i), self.regular_n
        elif dap <= 90:
            requested_i, requested_n = (self.high_i if water_low else self.mid_i), self.regular_n
        elif dap <= 120:
            requested_i, requested_n = (self.mid_i if water_low else self.early_i), 0.0
        else:
            requested_i, requested_n = 0.0, 0.0
        index = nearest_legal_action(requested_i, requested_n, legal_mask, grid)
        return {"action_index": index, "source_requested_amir": requested_i, "source_requested_anfer": requested_n, "action_projection": "nearest_legal_joint_action", "policy_note": "uses only dap, sw[1:9], swfac, nstres, cumsumfert from verified PPO observation", "rule_water_low": bool(water_low), "rule_nitrogen_low": bool(nitrogen_low), **grid[index]}


class StructuredGridRoundRobinPolicy(BehaviorPolicy):
    policy_id = "structured_grid_round_robin"
    policy_type = "deterministic_round_robin_over_current_legal_grid"

    def __init__(self) -> None:
        self.opportunity = 0

    def reset(self) -> None:
        self.opportunity = 0

    def choose(self, state, legal_mask, dap, grid, indices):
        legal_positive = [int(index) for index, legal in enumerate(legal_mask) if legal and index > 0]
        index = legal_positive[self.opportunity % len(legal_positive)] if legal_positive else 0
        if legal_positive:
            self.opportunity += 1
        return {"action_index": index, "source_requested_amir": grid[index]["amir"], "source_requested_anfer": grid[index]["anfer"], "action_projection": "none_current_legal_grid", "policy_note": "deterministic legal-action round-robin; no daily management randomness", **grid[index]}


class StructuredTargetGridPolicy(BehaviorPolicy):
    policy_type = "deterministic_targeted_current_legal_grid"

    def __init__(self, target_action_index: int, variant: int) -> None:
        self.target_action_index = int(target_action_index)
        self.variant = int(variant)
        self.policy_id = f"structured_grid_target_a{self.target_action_index:02d}_r{self.variant:02d}"
        self.opportunity = 0

    def reset(self) -> None:
        self.opportunity = 0

    def parameters(self) -> dict[str, Any]:
        return {"target_action_index_at_first_decision": self.target_action_index, "variant": self.variant, "selection": "deterministic_target_then_stride_5_over_legal_grid"}

    def choose(self, state, legal_mask, dap, grid, indices):
        desired = self.target_action_index if self.opportunity == 0 else (self.target_action_index + self.variant + 5 * self.opportunity) % len(grid)
        index = desired if bool(legal_mask[desired]) else nearest_legal_action(grid[desired]["amir"], grid[desired]["anfer"], legal_mask, grid)
        self.opportunity += 1
        return {"action_index": index, "source_requested_amir": grid[desired]["amir"], "source_requested_anfer": grid[desired]["anfer"], "action_projection": "none_or_nearest_legal_target", "policy_note": "structured coverage target on current legal grid; no random daily management", **grid[index]}


class StructuredStageSweepPolicy(BehaviorPolicy):
    policy_type = "deterministic_stage_conditioned_grid_sweep"

    def __init__(self, variant: int, sequence: list[int]) -> None:
        self.variant = int(variant)
        self.sequence = [int(value) for value in sequence]
        self.policy_id = f"structured_stage_sweep_v{self.variant:02d}"

    def parameters(self) -> dict[str, Any]:
        return {"stage_action_sequence": self.sequence, "variant": self.variant, "selection": "dap_stage_then_current_legal_mask"}

    def choose(self, state, legal_mask, dap, grid, indices):
        stage = min(max((int(dap) - 1) // 30, 0), len(self.sequence) - 1)
        desired = self.sequence[stage]
        index = desired if bool(legal_mask[desired]) else nearest_legal_action(grid[desired]["amir"], grid[desired]["anfer"], legal_mask, grid)
        return {"action_index": index, "source_requested_amir": grid[desired]["amir"], "source_requested_anfer": grid[desired]["anfer"], "action_projection": "none_or_nearest_legal_stage_target", "policy_note": "stage-conditioned deterministic grid sweep; no random daily management", **grid[index]}


def build_coverage_plan(years: list[int], recorded: dict[int, dict[str, float]], expert: dict[int, dict[str, float]]) -> list[tuple[BehaviorPolicy, int]]:
    """Create exactly 100 deterministic SY episodes for coverage validation.

    The first decision of 64 target episodes deliberately covers every one of
    the 16 legal joint actions four times. This is structured exploration, not
    random daily management; all later choices still pass through the live
    legal mask.
    """
    plan: list[tuple[BehaviorPolicy, int]] = []
    for action_index in range(16):
        for repeat in range(4):
            plan.append((StructuredTargetGridPolicy(action_index, repeat), years[(action_index + repeat) % len(years)]))
    rule_variants = [
        ("state_rule_conservative", 0.18, 0.10, 15.0, 30.0, 45.0, 40.0, 40.0),
        ("state_rule_water_sensitive", 0.24, 0.05, 15.0, 30.0, 45.0, 80.0, 40.0),
        ("state_rule_n_sensitive", 0.20, 0.01, 15.0, 30.0, 45.0, 120.0, 80.0),
        ("state_rule_balanced", 0.21, 0.03, 30.0, 45.0, 45.0, 80.0, 40.0),
    ]
    for policy_id, water_threshold, n_threshold, early_i, mid_i, high_i, early_n, regular_n in rule_variants:
        for year in years:
            plan.append((ObservableStateRulePolicy(policy_id, water_threshold, n_threshold, early_i, mid_i, high_i, early_n, regular_n), year))
    sequences = [[0, 1, 4, 8, 12], [0, 3, 5, 10, 15], [1, 2, 6, 9, 13], [4, 8, 12, 14, 15], [3, 7, 11, 15, 5], [2, 6, 10, 14, 1]]
    for variant, sequence in enumerate(sequences):
        for year in years:
            plan.append((StructuredStageSweepPolicy(variant, sequence), year))
    for policy_id, schedule in [("recorded_farmer_template_grid_projection", recorded), ("official_extension_expert_grid_projection", expert)]:
        if schedule:
            for year in years:
                plan.append((ScheduleProjectionPolicy(policy_id, schedule), year))
    if len(plan) != 100:
        raise RuntimeError(f"coverage plan expected 100 episodes, got {len(plan)}")
    return plan


class PPOContrastPolicy(BehaviorPolicy):
    policy_id = "ppo_contrast_ckpt100000"
    policy_type = "frozen_existing_maskableppo_contrast"
    include_in_bc = False

    def __init__(self, model: Any) -> None:
        self.model = model

    def parameters(self) -> dict[str, Any]:
        return {"checkpoint": rel(PPO_CHECKPOINT), "deterministic": True, "expert_label": False}

    def choose(self, state, legal_mask, dap, grid, indices):
        action, _ = self.model.predict(state, action_masks=np.asarray(legal_mask, dtype=bool), deterministic=True)
        requested = int(np.asarray(action).reshape(-1)[0])
        if not (0 <= requested < len(grid)):
            requested = 0
        index = requested if bool(legal_mask[requested]) else nearest_legal_action(grid[requested]["amir"], grid[requested]["anfer"], legal_mask, grid)
        return {"action_index": index, "source_requested_amir": grid[requested]["amir"], "source_requested_anfer": grid[requested]["anfer"], "action_projection": "mask_safe_fallback" if index != requested else "none_frozen_ppo", "policy_note": "frozen PPO contrast/source only; not expert", **grid[index]}


class NearestNeighborBC:
    def __init__(self) -> None:
        self.states: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("BC training data is empty")
        self.states = np.asarray([row["state_t"] for row in records], dtype=np.float64)
        self.actions = np.asarray([int(row["action_index"]) for row in records], dtype=np.int64)
        if self.states.ndim != 2 or not np.isfinite(self.states).all():
            raise ValueError("BC training states must be finite 2D")
        self.mean = self.states.mean(axis=0)
        self.scale = self.states.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0

    def predict_raw(self, state: list[float] | np.ndarray) -> int:
        if self.states is None or self.actions is None or self.mean is None or self.scale is None:
            raise RuntimeError("BC model is not fitted")
        vector = (np.asarray(state, dtype=np.float64) - self.mean) / self.scale
        train = (self.states - self.mean) / self.scale
        return int(self.actions[int(np.argmin(np.sum((train - vector) ** 2, axis=1)))])

    def predict_legal(self, state: list[float], legal_mask: list[bool], grid: list[dict[str, float]]) -> int:
        requested = self.predict_raw(state)
        if 0 <= requested < len(legal_mask) and bool(legal_mask[requested]):
            return requested
        candidate = grid[requested] if 0 <= requested < len(grid) else {"amir": 0.0, "anfer": 0.0}
        return nearest_legal_action(candidate["amir"], candidate["anfer"], np.asarray(legal_mask), grid)

    def as_json(self) -> dict[str, Any]:
        if self.states is None or self.actions is None or self.mean is None or self.scale is None:
            raise RuntimeError("BC model is not fitted")
        return {"model": "standardized_1_nearest_neighbor_joint_action_classifier", "state_dimension": int(self.states.shape[1]), "train_rows": int(self.states.shape[0]), "mean": self.mean.tolist(), "scale": self.scale.tolist(), "train_states": self.states.tolist(), "train_action_indices": self.actions.tolist()}


def import_runtime_modules() -> tuple[Any, Any, Any, Any]:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    import ppo_safe_rendering
    import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
    import run_sya_originIC_expanded_action_maskableppo_046_10 as ppo046

    # This is the same wrapper patch used by 046_10 before its own main flow.
    batch.base.StressAwareDiscreteWrapper = ppo046.engine.base04036.LateIrrigationReserveMaskWrapper
    return ppo_safe_rendering, batch, ppo046, batch.direct_ppo


def runtime_config(ppo046: Any) -> dict[str, Any]:
    config = ppo046.engine.load_config()
    config["discrete_actions"]["irrigation_levels"] = [0.0, 15.0, 30.0, 45.0]
    config["discrete_actions"]["nitrogen_levels"] = [0.0, 40.0, 80.0, 120.0]
    return config


def make_env_config(config: dict[str, Any], output_root: Path, direct_ppo: Any) -> dict[str, Any]:
    source = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
    env_config = direct_ppo.load_yaml(source)
    env_config["paths"]["output_root"] = rel(output_root)
    env_config["runtime"]["mode"] = "all"
    env_config["runtime"]["max_steps"] = int(config["compute_guardrails"]["max_steps"])
    env_config["seed"] = 0
    env_config["observed_years"] = {
        "SYA": [
            {
                "year": int(item["year"]),
                "label": f"sya_offline_bc_{item['initial_condition']}",
                "planting_date": item["planting_date"],
                "weather_file": item["weather_file"],
                "source": "sya_offline_bc_pilot_fixed_existing_SY_year_weather",
            }
            for item in config["scope"]["year_inputs"]
        ]
    }
    return env_config


def source_file_inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = [
        ROOT / config["scope"]["source_template"],
        INPUT_ROOT / "SY" / "MZCER048.CUL",
        INPUT_ROOT / "SY" / "SOIL.SOL",
        *[INPUT_ROOT / "SY" / item["weather_file"] for item in config["scope"]["year_inputs"]],
        PPO_CONFIG_PATH,
        PPO_CHECKPOINT,
        RECORDED_SOURCE,
        EXPERT_SOURCE,
    ]
    return [{"path": rel(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256(path) if path.exists() else ""} for path in paths]


def capture_snapshot(env: Any, output_root: Path, episode_id: str) -> dict[str, Any]:
    folder = getattr(env.unwrapped, "_tmp_folder", None)
    if not folder:
        return {"snapshot_status": "missing_tmp_folder", "snapshot_path": "", "snapshot_abs_path": "", "snapshot_files": []}
    source = Path(folder)
    target = output_root / "dssat_snapshots" / episode_id
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ["Summary.OUT", "MgmtEvent.OUT", "PlantGro.OUT", "SoilWat.OUT", "Weather.OUT"]:
        source_file = source / name
        if source_file.exists():
            shutil.copyfile(source_file, target / name)
            copied.append(name)
    return {"snapshot_status": "copied" if copied else "no_required_outputs_found", "snapshot_path": rel(target), "snapshot_abs_path": str(target), "snapshot_files": copied}


def parse_dssat_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith("@")), None)
    if header_index is None:
        return []
    matches = list(re.finditer(r"\S+", lines[header_index]))
    names = [match.group().strip().rstrip(".") for match in matches[1:]]
    ends = [match.end() for match in matches[1:]]
    starts = [0] + ends[:-1]
    rows = []
    for line in lines[header_index + 1 :]:
        if not line.strip() or line.startswith(("@", "!", "*")) or not re.match(r"^\s*\d", line):
            continue
        values = []
        for start, end in zip(starts, ends):
            token = line[start:end].strip()
            if token in {"", "-99"}:
                values.append(None)
            else:
                try:
                    values.append(float(token))
                except ValueError:
                    values.append(token)
        rows.append(dict(zip(names, values)))
    return rows


def summary_metrics(snapshot_path: Path, final_yield: float, online_i: float, online_n: float) -> dict[str, Any]:
    summary_path = snapshot_path / "Summary.OUT"
    rows = parse_dssat_table(summary_path)
    if not rows:
        return {"summary_out_status": "missing_or_unparseable", "summary_out_path": rel(summary_path), "summary_irrigation_total": math.nan, "summary_nitrogen_total": math.nan, "etcp_mm": math.nan, "WP_ET_kg_m3": None, "PFP_N_kg_kg": None, "wp_et_status": "unavailable_without_valid_Summary.OUT_ETCP"}

    def number(row: dict[str, Any], key: str) -> float:
        return finite_float(row.get(key), math.nan)

    def score(row: dict[str, Any]) -> float:
        values = [number(row, "HWAM"), number(row, "IRCM"), number(row, "NICM")]
        targets = [final_yield, online_i, online_n]
        return sum(abs(value - target) for value, target in zip(values, targets) if math.isfinite(value))

    row = min(rows, key=score)
    ircm, nicm, etcp = number(row, "IRCM"), number(row, "NICM"), number(row, "ETCP")
    ypem, ypnam = number(row, "YPEM"), number(row, "YPNAM")
    valid_etcp = math.isfinite(etcp) and etcp > 0
    wp = ypem * 0.1 if valid_etcp and math.isfinite(ypem) and ypem >= 0 else (final_yield / (etcp * 10.0) if valid_etcp else None)
    pfp = ypnam if nicm > 0 and math.isfinite(ypnam) and ypnam >= 0 else (final_yield / nicm if nicm > 0 else None)
    return {"summary_out_status": "parsed", "summary_out_path": rel(summary_path), "summary_yield_hwam": number(row, "HWAM"), "summary_irrigation_total": ircm, "summary_nitrogen_total": nicm, "etcp_mm": etcp, "WP_ET_kg_m3": wp, "PFP_N_kg_kg": pfp, "wp_et_status": "available_from_Summary.OUT_ETCP" if valid_etcp else "unavailable_without_valid_Summary.OUT_ETCP", "summary_accounting_i_difference_vs_online": ircm - online_i if math.isfinite(ircm) else math.nan, "summary_accounting_n_difference_vs_online": nicm - online_n if math.isfinite(nicm) else math.nan}


def choose_policy_action(policy: BehaviorPolicy, state: np.ndarray, mask: np.ndarray, dap: int, grid: list[dict[str, float]], indices: dict[str, int]) -> dict[str, Any]:
    decision = dict(policy.choose(state, mask, dap, grid, indices))
    index = int(decision.get("action_index", 0))
    if index < 0 or index >= len(grid) or not bool(mask[index]):
        index = nearest_legal_action(float(decision.get("amir", 0.0)), float(decision.get("anfer", 0.0)), mask, grid)
    decision.update({"action_index": index, "amir": grid[index]["amir"], "anfer": grid[index]["anfer"], "chosen_action_was_legal": bool(mask[index])})
    return decision


def run_trajectory(policy: BehaviorPolicy, year: int, role: str, runtime_cfg: dict[str, Any], env_config: dict[str, Any], output_root: Path, config: dict[str, Any], schema: list[str] | None, indices: dict[str, int] | None, grid: list[dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], dict[str, int]]:
    _, batch, _, direct_ppo = import_runtime_modules()
    policy.reset()
    policy_id = policy.policy_id
    episode_id = f"{policy_id}_{year}_{role}".replace("-", "_")
    env = None
    planting = pd.Timestamp("1900-01-01")
    transitions: list[dict[str, Any]] = []
    done = False
    error = ""
    snapshot_info: dict[str, Any] = {}
    try:
        env = batch.base.make_env(runtime_cfg, env_config, "SYA", int(year), 0, f"sya_offline_bc_{episode_id}", evaluation=True)
        planting = pd.Timestamp(direct_ppo.find_year(env_config, "SYA", int(year))["planting_date"])
        obs, info = env.reset()
        raw = env.unwrapped
        raw_names = [str(value) for value in raw.observation_variables]
        raw_obs = getattr(raw, "observation", None)
        if not isinstance(raw_obs, dict):
            raw_obs = getattr(env, "last_obs_dict", {})
        live_schema = expanded_observation_names(raw_names, raw_obs if isinstance(raw_obs, dict) else None)
        if schema is None:
            schema = live_schema
        if live_schema != schema:
            raise RuntimeError(f"observation schema drift: expected {schema}, got {live_schema}")
        indices = state_indices(schema)
        for step_index in range(int(config["compute_guardrails"]["max_steps"])):
            state = np.asarray(obs, dtype=np.float64).reshape(-1)
            if len(state) != len(schema) or not np.isfinite(state).all():
                raise RuntimeError(f"invalid state at step {step_index}: {state.shape}")
            mask = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
            if len(mask) != len(grid):
                raise RuntimeError(f"mask dimension {len(mask)} != grid dimension {len(grid)}")
            dap_value = state_scalar(state, indices, "dap", step_index + 1)
            dap = int(round(dap_value)) if dap_value > 0 else step_index + 1
            decision = choose_policy_action(policy, state, mask, dap, grid, indices)
            action_index = int(decision["action_index"])
            next_obs, reward, terminated, truncated, next_info = env.step(action_index)
            done = bool(terminated or truncated)
            next_state = np.asarray(next_obs, dtype=np.float64).reshape(-1)
            action_info = dict(getattr(env, "last_action_info", {}) or {})
            actual_index = int(action_info.get("discrete_action_index", action_index))
            transition = {
                "episode_id": episode_id, "trajectory_role": role, "step_index": int(step_index), "station_code": "SYA", "site": "SY", "year": int(year),
                "weather_file": next(item["weather_file"] for item in config["scope"]["year_inputs"] if int(item["year"]) == int(year)), "cultivar_id": config["scope"]["cultivar_id"], "cultivar_name": config["scope"]["cultivar_name"], "input_profile": config["scope"]["input_profile"],
                "behavior_policy_source": policy_id, "behavior_policy_type": policy.policy_type, "policy_parameters": policy.parameters(), "random_seed": 0, "policy_randomness": "none_deterministic", "oracle_policy": bool(policy.oracle_policy), "state_observation_contract": "046_10_verified_runtime_flat_observation", "state_schema": schema,
                "state_t": state.tolist(), "action_t": {"index": actual_index, "irrigation_mm": grid[actual_index]["amir"], "nitrogen_kg_ha": grid[actual_index]["anfer"]}, "action_index": actual_index,
                "action_amir_mm": grid[actual_index]["amir"], "action_anfer_kg_ha": grid[actual_index]["anfer"], "requested_action_index": action_index,
                "source_requested_amir": finite_float(decision.get("source_requested_amir"), grid[action_index]["amir"]), "source_requested_anfer": finite_float(decision.get("source_requested_anfer"), grid[action_index]["anfer"]), "action_projection": str(decision.get("action_projection", "none")),
                "legal_action_mask_t": mask.tolist(), "legal_action_count_t": int(mask.sum()), "positive_legal_action_count_t": int(mask[1:].sum()), "decision_day": bool(mask[1:].any()), "chosen_action_was_legal": bool(mask[action_index]), "reward_t": float(reward), "next_state_t": next_state.tolist(), "done_t": done,
                "date": (planting + pd.Timedelta(days=max(dap - 1, 0))).strftime("%Y-%m-%d"), "dap": dap, "safe_action_amir": finite_float(action_info.get("safe_action_amir"), grid[actual_index]["amir"]), "safe_action_anfer": finite_float(action_info.get("safe_action_anfer"), grid[actual_index]["anfer"]), "info": next_info if isinstance(next_info, dict) else {}, "policy_note": str(decision.get("policy_note", "")),
            }
            transitions.append(transition)
            obs, info = next_obs, next_info
            if done:
                break
        if transitions:
            snapshot_info = capture_snapshot(env, output_root, episode_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if transitions:
            snapshot_info = capture_snapshot(env, output_root, episode_id)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                if not error:
                    error = f"close_{type(exc).__name__}: {exc}"
    final_state = transitions[-1]["next_state_t"] if transitions else []
    final_yield = state_scalar(np.asarray(final_state, dtype=float), state_indices(schema or []), "grnwt", math.nan) if final_state else math.nan
    online_i = float(sum(row["safe_action_amir"] for row in transitions))
    online_n = float(sum(row["safe_action_anfer"] for row in transitions))
    snapshot_path = Path(snapshot_info["snapshot_abs_path"]) if snapshot_info.get("snapshot_abs_path") else output_root / "missing_snapshot"
    summary = {"episode_id": episode_id, "trajectory_role": role, "behavior_policy_source": policy_id, "station_code": "SYA", "site": "SY", "year": int(year), "weather_file": next(item["weather_file"] for item in config["scope"]["year_inputs"] if int(item["year"]) == int(year)), "cultivar_id": config["scope"]["cultivar_id"], "episode_length": len(transitions), "complete_season": bool(transitions and transitions[-1]["done_t"] and not error), "error": error, "final_grnwt": final_yield, "online_total_irrigation_mm": online_i, "online_total_nitrogen_kg_ha": online_n, **snapshot_info, **summary_metrics(snapshot_path, final_yield, online_i, online_n)}
    return transitions, summary, schema or [], indices or {}


class BCPolicy(BehaviorPolicy):
    policy_id = "bc_combined_nearest_neighbor"
    policy_type = "standardized_1nn_joint_action_classifier"

    def __init__(self, model: NearestNeighborBC) -> None:
        self.model = model

    def parameters(self) -> dict[str, Any]:
        return {"model": "standardized_1nn_joint_action_classifier", "legal_mask_fallback": True, "expert_label": False}

    def choose(self, state, legal_mask, dap, grid, indices):
        raw = self.model.predict_raw(state)
        chosen = self.model.predict_legal(state, legal_mask.tolist(), grid)
        return {"action_index": chosen, "source_requested_amir": grid[raw]["amir"], "source_requested_anfer": grid[raw]["anfer"], "action_projection": "mask_safe_fallback", "policy_note": "BC consumes only state_t and legal_action_mask_t", **grid[chosen]}


def split_for_bc(records: list[dict[str, Any]], config: dict[str, Any], years_run: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    behavior = [row for row in records if row["trajectory_role"] == "behavior_dataset"]
    train_years = set(int(value) for value in config["bc"]["train_years"])
    validation_years = set(int(value) for value in config["bc"]["validation_years"])
    if train_years.intersection(years_run) and validation_years.intersection(years_run):
        train = [row for row in behavior if int(row["year"]) in train_years]
        validation = [row for row in behavior if int(row["year"]) in validation_years]
        if train and validation:
            return train, validation, "cross_year_configured_split"
    train, validation = [], []
    train_fraction = 1.0 - float(config["bc"]["within_episode_holdout_fraction"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior:
        grouped[str(row["episode_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step_index"]))
        if len(rows) <= 1:
            train.extend(rows)
            continue
        cut = max(1, min(len(rows) - 1, int(math.floor(len(rows) * train_fraction))))
        train.extend(rows[:cut])
        validation.extend(rows[cut:])
    return train, validation, "within_episode_temporal_holdout"


def bc_reproduction(model: NearestNeighborBC, records: list[dict[str, Any]], grid: list[dict[str, float]], label: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["trajectory_role"] == "behavior_dataset":
            grouped[str(row["behavior_policy_source"])].append(row)
    output = []
    for source, rows in sorted(grouped.items()):
        raw_match = executed_match = legal_raw = 0
        for row in rows:
            raw = model.predict_raw(row["state_t"])
            executed = model.predict_legal(row["state_t"], row["legal_action_mask_t"], grid)
            raw_match += int(raw == int(row["action_index"]))
            executed_match += int(executed == int(row["action_index"]))
            legal_raw += int(0 <= raw < len(row["legal_action_mask_t"]) and bool(row["legal_action_mask_t"][raw]))
        total = len(rows)
        output.append({"evaluation_set": label, "behavior_policy_source": source, "rows": total, "bc_raw_action_match_rate": raw_match / total if total else math.nan, "bc_executed_action_match_rate": executed_match / total if total else math.nan, "bc_raw_prediction_legal_rate": legal_raw / total if total else math.nan})
    return output


def diversity_row(rows: list[dict[str, Any]], label: str, source: str) -> dict[str, Any]:
    counts = Counter(int(row["action_index"]) for row in rows)
    total = len(rows)
    probabilities = np.asarray(list(counts.values()), dtype=float) / total if total else np.asarray([], dtype=float)
    entropy = float(-np.sum(probabilities * np.log(probabilities))) if len(probabilities) else math.nan
    top_fraction = max(counts.values()) / total if total else math.nan
    positive = [index for index in counts if index > 0]
    return {"dataset": label, "behavior_policy_source": source, "rows": total, "unique_action_count": len(counts), "unique_positive_action_count": len(positive), "noop_fraction": counts.get(0, 0) / total if total else math.nan, "top_action_fraction": top_fraction, "action_entropy_nats": entropy, "action_collapse_flag": bool(len(positive) <= 1 or (math.isfinite(top_fraction) and top_fraction >= 0.90)), "action_counts": json.dumps({str(key): int(value) for key, value in sorted(counts.items())}, ensure_ascii=False)}


def quality_checks(records: list[dict[str, Any]], summaries: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    state_lengths = Counter(len(row.get("state_t", [])) for row in records)
    next_lengths = Counter(len(row.get("next_state_t", [])) for row in records)
    mask_lengths = Counter(len(row.get("legal_action_mask_t", [])) for row in records)
    illegal = [row for row in records if not (0 <= int(row["action_index"]) < len(row["legal_action_mask_t"]) and row["legal_action_mask_t"][int(row["action_index"])])]
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        episodes[str(row["episode_id"])].append(row)
    terminal_counts = {episode: sum(bool(row["done_t"]) for row in rows) for episode, rows in episodes.items()}
    expected_years = {int(item["year"]) for item in config["scope"]["year_inputs"]}
    observed_years = {int(row["year"]) for row in records}
    finite_states = all(np.isfinite(np.asarray(row["state_t"], dtype=float)).all() and np.isfinite(np.asarray(row["next_state_t"], dtype=float)).all() for row in records)
    required = ["state_t", "action_t", "reward_t", "next_state_t", "done_t", "legal_action_mask_t", "policy_parameters", "random_seed", "episode_id", "year", "weather_file", "cultivar_id"]
    missing_fields = sum(sum(field not in row for field in required) for row in records)
    complete_behavior = [row for row in summaries if row["trajectory_role"] == "behavior_dataset"]
    failure_reasons = Counter(str(row.get("error") or "unknown") for row in complete_behavior if not bool(row.get("complete_season")))
    success_rate = sum(bool(row.get("complete_season")) for row in complete_behavior) / len(complete_behavior) if complete_behavior else 0.0
    finite_rewards = all(math.isfinite(finite_float(row.get("reward_t"), math.nan)) for row in records)
    completed_yields = [finite_float(row.get("final_grnwt"), math.nan) for row in summaries if bool(row.get("complete_season"))]
    invalid_yields = [value for value in completed_yields if not math.isfinite(value) or value < 0]
    site_scope_passed = all(row["station_code"] == "SYA" and row["site"] == "SY" for row in records)
    cultivar_scope_passed = all(row["cultivar_id"] == config["scope"]["cultivar_id"] for row in records)
    weather_scope_passed = observed_years.issubset(expected_years)
    yield_quality_passed = not invalid_yields
    return {"transition_count": len(records), "episode_count": len(episodes), "state_length_counts": dict(state_lengths), "next_state_length_counts": dict(next_lengths), "legal_mask_length_counts": dict(mask_lengths), "all_states_finite": finite_states, "all_rewards_finite": finite_rewards, "missing_required_field_count": int(missing_fields), "illegal_action_count": len(illegal), "episodes_with_exactly_one_terminal_transition": sum(value == 1 for value in terminal_counts.values()), "complete_season_count": sum(bool(row.get("complete_season")) for row in summaries), "behavior_episode_count": len(complete_behavior), "behavior_dssat_success_rate": success_rate, "behavior_failure_reasons": dict(failure_reasons), "site_scope_passed": site_scope_passed, "cultivar_scope_passed": cultivar_scope_passed, "weather_scope_passed": weather_scope_passed, "completed_yield_count": len(completed_yields), "invalid_completed_yield_count": len(invalid_yields), "yield_quality_passed": yield_quality_passed, "oracle_policy_count": sum(bool(row.get("oracle_policy")) for row in records), "wp_et_policy": "Summary.OUT ETCP only; unavailable if Summary.OUT/ETCP missing or invalid; never inferred from daily CSV", "pass": bool(records and finite_states and finite_rewards and missing_fields == 0 and not illegal and site_scope_passed and cultivar_scope_passed and weather_scope_passed and yield_quality_passed and success_rate >= float(config["coverage_gate"]["minimum_success_rate"]))}


def balanced_bc_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically cap the no-op majority before fitting BC."""
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[int(row["action_index"])].append(row)
    positive_counts = [len(rows) for action, rows in groups.items() if action > 0]
    cap = max(1, max(positive_counts)) if positive_counts else max(1, len(groups.get(0, [])))
    selected: list[dict[str, Any]] = []
    for action in sorted(groups):
        rows = groups[action]
        if len(rows) <= cap:
            selected.extend(rows)
        else:
            positions = np.linspace(0, len(rows) - 1, cap, dtype=int)
            selected.extend(rows[int(position)] for position in positions)
    return selected


def bc_classification_metrics(model: NearestNeighborBC, records: list[dict[str, Any]], grid: list[dict[str, float]], label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not records:
        return {"evaluation_set": label, "rows": 0, "bc_accuracy": math.nan, "always_noop_accuracy": math.nan, "bc_beats_always_noop": False, "macro_f1": math.nan, "positive_action_recall": math.nan, "legal_rate": math.nan, "unique_predicted_actions": 0, "predicted_action_indices": [], "decision_day_rows": 0}, []
    truth = np.asarray([int(row["action_index"]) for row in records], dtype=int)
    predictions = np.asarray([model.predict_legal(row["state_t"], row["legal_action_mask_t"], grid) for row in records], dtype=int)
    recalls: list[dict[str, Any]] = []
    f1_values = []
    for action in range(len(grid)):
        true_positive = int(((truth == action) & (predictions == action)).sum())
        false_positive = int(((truth != action) & (predictions == action)).sum())
        false_negative = int(((truth == action) & (predictions != action)).sum())
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else math.nan
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else math.nan
        f1 = 2 * precision * recall / (precision + recall) if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0 else 0.0
        if math.isfinite(f1):
            f1_values.append(f1)
        recalls.append({"evaluation_set": label, "action_index": action, "irrigation_mm": grid[action]["amir"], "nitrogen_kg_ha": grid[action]["anfer"], "support": int((truth == action).sum()), "recall": recall, "precision": precision, "f1": f1})
    positive = truth > 0
    positive_recall = float(((positive) & (predictions > 0)).sum() / positive.sum()) if positive.any() else math.nan
    bc_accuracy = float((truth == predictions).mean())
    noop_accuracy = float((truth == 0).mean())
    result = {"evaluation_set": label, "rows": len(records), "bc_accuracy": bc_accuracy, "always_noop_accuracy": noop_accuracy, "bc_beats_always_noop": bool(bc_accuracy > noop_accuracy), "macro_f1": float(np.mean(f1_values)) if f1_values else math.nan, "positive_action_recall": positive_recall, "legal_rate": 1.0, "unique_predicted_actions": int(len(set(predictions.tolist()))), "predicted_action_indices": sorted(set(int(value) for value in predictions.tolist())), "decision_day_rows": int(sum(bool(row.get("decision_day")) for row in records))}
    return result, recalls


def coverage_gate(records: list[dict[str, Any]], summaries: list[dict[str, Any]], config: dict[str, Any], bc_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    behavior = [row for row in records if row["trajectory_role"] == "behavior_dataset"]
    decision = [row for row in behavior if bool(row.get("decision_day"))]
    counts = Counter(int(row["action_index"]) for row in decision)
    families = sorted(set(str(row["behavior_policy_type"]) for row in behavior))
    positive_counts = {str(action): int(counts.get(action, 0)) for action in range(1, 16)}
    holdout = next((row for row in bc_metrics if row.get("evaluation_set") == "bc_holdout_decision_days"), next((row for row in bc_metrics if row.get("evaluation_set") == "bc_holdout"), {}))
    behavior_summaries = [row for row in summaries if row.get("trajectory_role") == "behavior_dataset"]
    positive_rate = float(sum(int(row["action_index"]) > 0 for row in decision) / len(decision)) if decision else math.nan
    minimum_positive_rate = float(config["coverage_gate"].get("minimum_decision_day_positive_action_rate", 0.10))
    coverage = {"behavior_trajectory_count": len(behavior_summaries), "target_trajectory_count": int(config["coverage_gate"]["target_trajectories"]), "all_16_actions_present_on_decision_days": all(counts.get(action, 0) > 0 for action in range(16)), "nonzero_action_counts_on_decision_days": positive_counts, "minimum_nonzero_action_count": min(positive_counts.values()) if positive_counts else 0, "required_minimum_nonzero_action_count": int(config["coverage_gate"]["minimum_each_nonzero_action_count_on_decision_days"]), "decision_day_count": len(decision), "decision_day_positive_action_rate": positive_rate, "minimum_decision_day_positive_action_rate": minimum_positive_rate, "decision_day_positive_action_rate_passed": bool(math.isfinite(positive_rate) and positive_rate >= minimum_positive_rate), "behavior_policy_families": families, "minimum_behavior_policy_families": int(config["coverage_gate"]["minimum_behavior_policy_families"]), "bc_holdout_decision_metrics": holdout}
    coverage["pass"] = bool(coverage["behavior_trajectory_count"] >= coverage["target_trajectory_count"] and coverage["all_16_actions_present_on_decision_days"] and coverage["minimum_nonzero_action_count"] >= coverage["required_minimum_nonzero_action_count"] and coverage["decision_day_positive_action_rate_passed"] and len(families) >= coverage["minimum_behavior_policy_families"])
    return coverage


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def write_audit(output_root: Path, config: dict[str, Any], runtime_cfg: dict[str, Any], schema: list[str], initial_mask: list[bool], model_contract: dict[str, Any], schedule_manifests: list[dict[str, Any]]) -> None:
    ppo_root = ROOT / "benchmark_results" / "046_10_sya_originIC_expanded_action_maskableppo"
    eval_path = ppo_root / "evaluation" / "046_10_checkpoint_validation_summary.csv"
    eval_by_checkpoint: list[dict[str, Any]] = []
    if eval_path.exists():
        eval_frame = pd.read_csv(eval_path, keep_default_na=False)
        for checkpoint, group in eval_frame.groupby("checkpoint_step", sort=True):
            eval_by_checkpoint.append({"checkpoint_step": int(checkpoint), "rows": int(len(group)), "complete_rows": int(group["run_status"].astype(str).str.startswith("ok").sum()), "mean_final_grnwt": float(pd.to_numeric(group["final_grnwt"], errors="coerce").mean()), "mean_total_irrigation": float(pd.to_numeric(group["total_irrigation"], errors="coerce").mean()), "mean_total_n": float(pd.to_numeric(group["total_n"], errors="coerce").mean()), "daily_csv_paths": sorted(set(str(value) for value in group["daily_csv_path"].tolist()))})
    audit = {"task": config["task_id"], "mode": "read_only_contract_audit", "scope": config["scope"], "input_files": source_file_inventory(config), "schedule_sources": schedule_manifests, "runtime_module_contract": {"actual_module": "046_10 originIC + 040_36 wrapper", "discrete_actions": runtime_cfg.get("discrete_actions"), "action_safety": runtime_cfg.get("action_safety"), "reward": runtime_cfg.get("reward"), "paths": runtime_cfg.get("paths")}, "observation_contract": {"expanded_state_schema": schema, "state_dimension": len(schema), "initial_legal_action_mask": initial_mask, "initial_legal_action_count": int(sum(initial_mask))}, "ppo_checkpoint_contract": model_contract, "existing_ppo_evaluation": {"root": rel(ppo_root), "checkpoint_count": len(list((ppo_root / "models" / "SYA").glob("*.zip"))), "checkpoint_files": [rel(path) for path in sorted((ppo_root / "models" / "SYA").glob("*.zip"))], "evaluation_files": [rel(path) for path in sorted((ppo_root / "evaluation").glob("046_10_*.csv"))], "by_checkpoint": eval_by_checkpoint, "note": "frozen PPO is contrast/behavior source only, not expert; config label and actual inherited runtime contract are both retained"}, "changes_to_reward_action_space_ppo_main": False, "offline_rl_implemented": False}
    (output_root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_root / "state_schema.json").write_text(json.dumps({"schema": schema, "dimension": len(schema), "initial_legal_action_mask": initial_mask}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# SY offline BC pilot: read-only contract audit", "", "## 结论", "", f"- 状态维度：`{len(schema)}`；动作数：`{len(runtime_action_grid(runtime_cfg))}`；初始合法动作数：`{sum(initial_mask)}`。", "- 品种固定 FY0985；年份/天气只限 SY 2012、2014、2015。", "- 现有 PPO 只作为 frozen contrast/source，不作为 expert。", "- 实际 safety/reward 按 040_36 wrapper 记录，未修改原 PPO 流程。", "", "## 展平状态变量", "", "`" + ", ".join(schema) + "`", "", "## 约束与来源", "", "```json", json.dumps({"input_files": audit["input_files"], "schedule_sources": schedule_manifests}, ensure_ascii=False, indent=2), "```", "", "- 真实年度农户记录不存在；recorded farmer 仅为 static template。"]
    (output_root / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(config: dict[str, Any], output_root: Path) -> None:
    ppo_safe_rendering, batch, ppo046, direct_ppo = import_runtime_modules()
    runtime_cfg = runtime_config(ppo046)
    env_config = make_env_config(config, output_root, direct_ppo)
    schedule_manifests = [read_schedule_source(RECORDED_SOURCE, "recorded_farmer_template")[1], read_schedule_source(EXPERT_SOURCE, "official_extension_expert")[1]]
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
    env = batch.base.make_env(runtime_cfg, env_config, "SYA", 2014, 0, "sya_offline_bc_contract_audit", evaluation=True)
    try:
        observation, _ = env.reset()
        raw = env.unwrapped
        raw_names = [str(value) for value in raw.observation_variables]
        raw_observation = getattr(raw, "observation", {})
        schema = expanded_observation_names(raw_names, raw_observation if isinstance(raw_observation, dict) else None)
        model_contract: dict[str, Any] = {"checkpoint": rel(PPO_CHECKPOINT), "exists": PPO_CHECKPOINT.exists(), "runtime_state_shape": list(np.asarray(observation).shape)}
        if PPO_CHECKPOINT.exists():
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(str(PPO_CHECKPOINT), device="cpu")
            model_contract.update({"observation_space": str(model.observation_space), "action_space": str(model.action_space), "observation_shape": list(model.observation_space.shape), "action_n": int(model.action_space.n)})
        write_audit(output_root, config, runtime_cfg, schema, np.asarray(env.action_masks(), dtype=bool).tolist(), model_contract, schedule_manifests)
    finally:
        env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def write_pilot_record(output_root: Path, config: dict[str, Any], summaries: list[dict[str, Any]], diversity: list[dict[str, Any]], reproduction: list[dict[str, Any]], checks: dict[str, Any], split_label: str, bc_summary: dict[str, Any]) -> None:
    behavior_summaries = [row for row in summaries if row["trajectory_role"] == "behavior_dataset"]
    years = sorted({int(row["year"]) for row in summaries})
    def text_frame(rows: list[dict[str, Any]]) -> str:
        return pd.DataFrame(rows).to_string(index=False) if rows else "无记录。"
    lines = [
        "# SY 单站点低算力离线轨迹与 BC pilot 记录", "", "## 结论先说", "",
        f"- 本轮只运行 SYA/SY，品种 `{config['scope']['cultivar_id']}`，年份：`{','.join(map(str, years))}`。",
        f"- 行为轨迹：`{len(behavior_summaries)}`；完整生育期：`{sum(bool(x.get('complete_season')) for x in behavior_summaries)}/{len(behavior_summaries)}`；离线 RL/在线 PPO 训练：`未执行`。",
        f"- 数据质量门：`{'PASS' if checks.get('pass') else 'FAIL'}`；非法动作数：`{checks.get('illegal_action_count')}`；oracle 主数据条数：`{checks.get('oracle_policy_count')}`。",
        f"- BC 切分：`{split_label}`；BC DSSAT replay 完整：`{bc_summary.get('complete_season')}`。", "",
        "## 策略来源边界", "",
        "- recorded farmer 是项目已有静态 template，不是年度真实农户记录；本轮只投影到当前合法联合动作网格。",
        "- official extension expert 来自项目已有 SYA 2014 baseline daily 输出；投影后不称 exact expert replay。",
        "- state rule 只使用 dap、9 层 sw、swfac、nstres、cumsumfert 这些已验证 PPO 可观测状态变量。",
        "- structured exploration 是 legal-mask 约束下的确定性 round-robin、target-grid 和 stage-sweep，不是每日随机灌溉/施氮。",
        "- 现有 PPO 仅允许作为单独 contrast/source，默认不进入 BC 训练集，不视为专家。", "",
        "## DSSAT 指标边界", "",
        "- WP_ET 只接受本次保存的 Summary.OUT 中有效 ETCP；缺失/无效即 unavailable，不从 daily CSV 反推。",
        "- online action totals 与 Summary.OUT 的 IRCM/NICM 分开保存，检查动作通道和 DSSAT 实际管理是否闭合。", "",
        "## 扩展通过门槛", "", json.dumps({"data_quality_gate": checks.get("pass"), "coverage_gate": checks.get("coverage_gate", {}), "bc_gate": checks.get("bc_gate", {})}, ensure_ascii=False, indent=2, default=str), "",
        "## episode summary", "", text_frame(summaries), "", "## action diversity", "", text_frame(diversity), "", "## BC reproduction", "", text_frame(reproduction), "", "## quality checks", "", json.dumps(checks, ensure_ascii=False, indent=2, default=str), "", "## 解释边界", "", "这是 SY 单站点、低算力、少量轨迹的行为数据/BC 可行性 pilot，不支持跨站点、跨品种或普遍优越性结论。",
    ]
    (output_root / "pilot_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(config: dict[str, Any], output_root: Path, smoke: bool, include_ppo: bool, coverage100: bool = False) -> None:
    ppo_safe_rendering, _, ppo046, direct_ppo = import_runtime_modules()
    runtime_cfg = runtime_config(ppo046)
    env_config = make_env_config(config, output_root, direct_ppo)
    grid = runtime_action_grid(runtime_cfg)
    if len(grid) != int(config["action_grid"]["joint_action_count"]):
        raise RuntimeError("runtime action grid does not match pilot contract")
    years = [int(value) for value in (config["bc"]["smoke_years"] if smoke else config["scope"]["years"])]
    if smoke:
        years = years[: int(config["compute_guardrails"]["smoke_max_years"])]
    recorded, recorded_manifest = read_schedule_source(RECORDED_SOURCE, "recorded_farmer_template")
    expert, expert_manifest = read_schedule_source(EXPERT_SOURCE, "official_extension_expert")
    for subdir in ["configs", "evaluation", "models", "trajectories"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)
    (output_root / "configs" / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "configs" / "runtime_contract.json").write_text(json.dumps(runtime_cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_root / "configs" / "source_file_inventory.json").write_text(json.dumps(source_file_inventory(config), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([recorded_manifest, expert_manifest]).to_csv(output_root / "configs" / "source_schedule_inventory.csv", index=False, encoding="utf-8-sig")

    policies: list[BehaviorPolicy] = []
    if recorded:
        policies.append(ScheduleProjectionPolicy("recorded_farmer_template_grid_projection", recorded))
    if expert:
        policies.append(ScheduleProjectionPolicy("official_extension_expert_grid_projection", expert))
    policies.extend([ObservableStateRulePolicy(), StructuredGridRoundRobinPolicy()])
    plan = build_coverage_plan(years, recorded, expert) if coverage100 else [(policy, year) for policy in policies for year in years]
    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    schema: list[str] | None = None
    indices: dict[str, int] | None = None
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
    try:
        for policy, year in plan:
            rows, summary, schema, indices = run_trajectory(policy, year, "behavior_dataset", runtime_cfg, env_config, output_root, config, schema, indices, grid)
            all_records.extend(rows)
            summaries.append(summary)
        if include_ppo:
            if not PPO_CHECKPOINT.exists():
                raise FileNotFoundError(PPO_CHECKPOINT)
            from sb3_contrib import MaskablePPO

            ppo_model = MaskablePPO.load(str(PPO_CHECKPOINT), device="cpu")
            for year in years:
                ppo_policy = PPOContrastPolicy(ppo_model)
                rows, summary, schema, indices = run_trajectory(ppo_policy, year, "ppo_contrast", runtime_cfg, env_config, output_root, config, schema, indices, grid)
                all_records.extend(rows)
                summaries.append(summary)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
    if not all_records or schema is None:
        # Preserve failure evidence even when the first/all environment builds fail;
        # a failed task must never disappear silently just because BC cannot run.
        pd.DataFrame(summaries).to_csv(output_root / "evaluation" / "episode_summary_failures.csv", index=False, encoding="utf-8-sig")
        (output_root / "evaluation" / "failure_reasons.json").write_text(json.dumps(dict(Counter(str(row.get("error") or "unknown") for row in summaries)), ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("no offline transitions were generated; failure evidence was written")

    behavior_records = [row for row in all_records if row["trajectory_role"] == "behavior_dataset"]
    train_rows, validation_rows, split_label = split_for_bc(behavior_records, config, years)
    balanced_train_rows = balanced_bc_rows(train_rows)
    model = NearestNeighborBC()
    model.fit(balanced_train_rows)
    model_payload = model.as_json()
    model_payload.update({"unbalanced_train_rows": len(train_rows), "balanced_train_rows": len(balanced_train_rows), "balanced_action_counts": dict(Counter(int(row["action_index"]) for row in balanced_train_rows))})
    (output_root / "models" / "bc_combined_nearest_neighbor.json").write_text(json.dumps(model_payload, ensure_ascii=False), encoding="utf-8")
    reproduction = bc_reproduction(model, train_rows, grid, "bc_train") + bc_reproduction(model, validation_rows, grid, "bc_holdout")
    pd.DataFrame(reproduction).to_csv(output_root / "evaluation" / "bc_reproduction.csv", index=False, encoding="utf-8-sig")
    diversity = [diversity_row([row for row in behavior_records if row["behavior_policy_source"] == source], "behavior_dataset", source) for source in sorted(set(str(row["behavior_policy_source"]) for row in behavior_records))]
    diversity.extend([diversity_row([row for row in behavior_records if row.get("decision_day")], "behavior_decision_days", "combined"), diversity_row(train_rows, "bc_train_labels", "combined"), diversity_row(validation_rows, "bc_holdout_labels", "combined")])
    pd.DataFrame(diversity).to_csv(output_root / "evaluation" / "action_diversity.csv", index=False, encoding="utf-8-sig")
    bc_results, recalls = [], []
    for label, rows in [("bc_train", train_rows), ("bc_holdout", validation_rows), ("bc_train_decision_days", [row for row in train_rows if row.get("decision_day")]), ("bc_holdout_decision_days", [row for row in validation_rows if row.get("decision_day")])]:
        result_row, recall_rows = bc_classification_metrics(model, rows, grid, label)
        bc_results.append(result_row)
        recalls.extend(recall_rows)
    pd.DataFrame(bc_results).to_csv(output_root / "evaluation" / "bc_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(recalls).to_csv(output_root / "evaluation" / "bc_action_recall.csv", index=False, encoding="utf-8-sig")
    validation_year = next((int(value) for value in config["bc"]["validation_years"] if int(value) in years), years[0])
    bc_policy = BCPolicy(model)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
    try:
        bc_records, bc_summary, schema, indices = run_trajectory(bc_policy, validation_year, "bc_validation", runtime_cfg, env_config, output_root, config, schema, indices, grid)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
    all_records.extend(bc_records)
    summaries.append(bc_summary)
    write_jsonl(output_root / "trajectories" / "offline_transitions_with_bc_validation.jsonl", all_records)
    # The compact behavior-only file is useful for BC readers and excludes the replay trajectory.
    write_jsonl(output_root / "trajectories" / "offline_behavior_transitions.jsonl", behavior_records)
    pd.DataFrame(summaries).to_csv(output_root / "evaluation" / "episode_summary.csv", index=False, encoding="utf-8-sig")
    checks = quality_checks(all_records, summaries, config)
    coverage = coverage_gate(all_records, summaries, config, bc_results)
    checks["coverage_gate"] = coverage
    holdout_metrics = next((row for row in bc_results if row["evaluation_set"] == "bc_holdout"), {})
    holdout_decision_metrics = next((row for row in bc_results if row["evaluation_set"] == "bc_holdout_decision_days"), {})
    recorded_action_indices = sorted({int(row["action_index"]) for row in behavior_records if str(row["behavior_policy_source"]) == "recorded_farmer_template_grid_projection"})
    holdout_predictions = set(int(value) for value in holdout_metrics.get("predicted_action_indices", []))
    if recorded_action_indices:
        bc_not_template_only = bool(holdout_predictions) and not holdout_predictions.issubset(set(recorded_action_indices))
    else:
        bc_not_template_only = len(holdout_predictions) > 1
    checks["bc_gate"] = {"holdout_bc_beats_always_noop": bool(holdout_metrics.get("bc_beats_always_noop", False)), "holdout_decision_bc_beats_always_noop": bool(holdout_decision_metrics.get("bc_beats_always_noop", False)), "holdout_positive_action_recall": holdout_metrics.get("positive_action_recall", math.nan), "holdout_decision_positive_action_recall": holdout_decision_metrics.get("positive_action_recall", math.nan), "holdout_macro_f1": holdout_metrics.get("macro_f1", math.nan), "holdout_decision_macro_f1": holdout_decision_metrics.get("macro_f1", math.nan), "holdout_legal_rate": holdout_metrics.get("legal_rate", math.nan), "holdout_decision_unique_predicted_actions": holdout_decision_metrics.get("unique_predicted_actions", 0), "bc_not_single_action_on_decision_days": bool(holdout_decision_metrics.get("unique_predicted_actions", 0) > 1), "recorded_template_action_indices": recorded_action_indices, "holdout_predicted_action_indices": sorted(holdout_predictions), "bc_not_template_only": bc_not_template_only}
    checks["bc_gate"]["pass"] = bool(checks["bc_gate"]["holdout_bc_beats_always_noop"] and checks["bc_gate"]["holdout_decision_bc_beats_always_noop"] and float(checks["bc_gate"]["holdout_legal_rate"]) == 1.0 and checks["bc_gate"]["bc_not_single_action_on_decision_days"] and checks["bc_gate"]["bc_not_template_only"])
    (output_root / "evaluation" / "data_quality_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_root / "evaluation" / "coverage_gate.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_root / "evaluation" / "bc_gate.json").write_text(json.dumps(checks["bc_gate"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([bc_summary]).to_csv(output_root / "evaluation" / "bc_validation_summary.csv", index=False, encoding="utf-8-sig")
    write_pilot_record(output_root, config, summaries, diversity, reproduction, checks, split_label, bc_summary)
    result = {"task": config["task_id"], "mode": "coverage100" if coverage100 else ("smoke" if smoke else "pilot"), "years_run": years, "behavior_episode_count": len([row for row in summaries if row["trajectory_role"] == "behavior_dataset"]), "bc_validation_episode_count": 1, "complete_season_count": int(sum(bool(row.get("complete_season")) for row in summaries)), "data_quality_pass": bool(checks.get("pass")), "coverage_gate_pass": bool(coverage.get("pass")), "bc_gate_pass": bool(checks["bc_gate"].get("pass")), "expansion_gate_pass": bool(checks.get("pass") and coverage.get("pass") and checks["bc_gate"].get("pass")), "bc_gate_holdout_beats_always_noop": bool(checks["bc_gate"]["holdout_bc_beats_always_noop"]), "offline_rl": False, "online_ppo_training": False, "output_root": rel(output_root), "pilot_record": rel(output_root / "pilot_record.md")}
    (output_root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refuse to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SY low-compute offline trajectory and BC pilot")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--coverage100", action="store_true", help="run exactly 100 deterministic SY behavior trajectories for the expansion gate")
    parser.add_argument("--include-ppo", action="store_true", help="optional frozen PPO contrast trajectory")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    selected_modes = [args.audit_only, args.smoke, args.pilot, args.coverage100]
    if sum(bool(mode) for mode in selected_modes) != 1:
        parser.error("choose exactly one of --audit-only, --smoke, --pilot, or --coverage100")
    config = load_json(CONFIG_PATH)
    if args.output_root is not None:
        output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    elif args.audit_only:
        output_root = ROOT / "benchmark_results" / "sya_offline_bc_pilot_audit"
    elif args.smoke:
        output_root = ROOT / "benchmark_results" / "sya_offline_bc_pilot_smoke"
    elif args.coverage100:
        output_root = ROOT / "benchmark_results" / "sya_offline_bc_pilot_coverage100_v1"
    else:
        output_root = ROOT / "benchmark_results" / "sya_offline_bc_pilot"
    prepare_output(output_root)
    if args.audit_only:
        run_audit(config, output_root)
        print(json.dumps({"mode": "audit", "output_root": rel(output_root), "audit": rel(output_root / "audit.json")}, ensure_ascii=False, indent=2))
    else:
        if args.include_ppo and args.coverage100:
            parser.error("--include-ppo is intentionally disabled for --coverage100")
        run_pilot(config, output_root, smoke=args.smoke, include_ppo=args.include_ppo, coverage100=args.coverage100)


if __name__ == "__main__":
    main()
