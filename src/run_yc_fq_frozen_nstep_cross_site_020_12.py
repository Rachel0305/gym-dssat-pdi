from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gymnasium_base
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frozen_nstep_dqn_config_020_11 import (
    ACTION_TABLE_9,
    CHECKPOINT_SELECTION,
    CONFIG_ID,
    DQN_FIXED_KWARGS,
    IRRIGATION_BUDGET,
    IRRIGATION_WINDOWS,
    MIN_INTERVAL_DAYS,
    NITROGEN_BUDGET,
    NITROGEN_WINDOWS,
    apply_environment_constants,
    dqn_kwargs,
    manifest,
)
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_all_year_screen_and_dqn_transfer_014_01 import prepare_text_for_shifted_scenario
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    parse_dssat_table,
    prepare_text_for_scenario,
    set_management_for_treatment,
)
import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared


OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "frozen_nstep_cross_site_020_12"
DOC_ROOT = PROJECT_ROOT / "docs"
ENVIRONMENT_SEED = 0


@dataclass(frozen=True)
class SiteSpec:
    code: str
    station: str
    year: int
    treatment: int
    input_root: Path
    mzx_name: str
    weather_name: str
    soil_id: str

    @property
    def case_id(self) -> str:
        return f"{self.code}{self.year}"


SITE_SPECS = {
    "YC": SiteSpec(
        code="YC",
        station="Yucheng",
        year=2014,
        treatment=2,
        input_root=PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "YC",
        mzx_name="CNYC0801.MZX",
        weather_name="CNYC1401.WTH",
        soil_id="YC99001200",
    ),
    "FQ": SiteSpec(
        code="FQ",
        station="Fengqiu",
        year=2016,
        treatment=2,
        input_root=PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "FQ",
        mzx_name="CNFQ0801.MZX",
        weather_name="CNFQ1601.WTH",
        soil_id="FQ99001200",
    ),
}


class BaselineRelativeRewardWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(self, env: gymnasium_base.Env, null_baseline_yield: float):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.null_baseline_yield = float(null_baseline_yield)
        self.last_reward_components: dict[str, float] = {}
        self._reset_components()

    def _reset_components(self) -> None:
        self.last_reward_components = {
            "yield_gain": 0.0,
            "water_cost_term": 0.0,
            "nitrogen_cost_term": 0.0,
            "baseline_relative_reward": 0.0,
        }

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self._reset_components()
        return obs, info

    def step(self, action):
        obs, _old_reward, terminated, truncated, info = self.env.step(action)
        latest = latest_observation_dict(self.env, obs, info)
        grnwt = float(scalar(latest.get("grnwt", 0.0)) or 0.0)
        safe = dict(getattr(self.env, "last_safe_real_action", {}) or {})
        irrigation = float(safe.get("amir", 0.0))
        nitrogen = float(safe.get("anfer", 0.0))
        water_cost = float(shared.WATER_COST * irrigation)
        nitrogen_cost = float(shared.NITROGEN_COST * nitrogen)
        yield_gain = (
            max(0.0, grnwt - self.null_baseline_yield)
            if bool(terminated or truncated)
            else 0.0
        )
        reward = float(yield_gain - water_cost - nitrogen_cost)
        self.last_reward_components = {
            "yield_gain": yield_gain,
            "water_cost_term": water_cost,
            "nitrogen_cost_term": nitrogen_cost,
            "baseline_relative_reward": reward,
        }
        info = dict(info) if isinstance(info, dict) else {}
        info.update(self.last_reward_components)
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self):
        return None

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header_tokens(line: str) -> list[str]:
    tokens = line.split()
    if tokens and tokens[0] == "@N":
        tokens[0] = "N"
    return [re.sub(r"\.+$", "", token).upper() for token in tokens]


def treatment_row(text: str, treatment: int) -> dict[str, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("@N R O C TNAME"):
            continue
        header = _header_tokens(line)
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if stripped.startswith("*"):
                break
            parts = candidate.split()
            if parts and parts[0] == str(treatment):
                if len(parts) < len(header):
                    raise RuntimeError(f"Treatment {treatment} row is shorter than its header")
                return dict(zip(header, parts))
    raise RuntimeError(f"Treatment {treatment} was not found")


def control_row(text: str, treatment: int, section: str) -> dict[str, str]:
    lines = text.splitlines()
    marker = f"@N {section.upper()}"
    for index, line in enumerate(lines):
        if not line.strip().startswith(marker):
            continue
        header = _header_tokens(line)
        expected_code = section[:2].upper()
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if stripped.startswith("@") or stripped.startswith("*"):
                break
            parts = candidate.split()
            if len(parts) >= 2 and parts[0] == str(treatment) and parts[1].upper() == expected_code:
                if len(parts) < len(header):
                    raise RuntimeError(f"{section} row for treatment {treatment} is shorter than header")
                return dict(zip(header, parts))
    raise RuntimeError(f"{section} row for treatment {treatment} was not found")


def prepare_site_text(spec: SiteSpec, scenario: str) -> str:
    source = (spec.input_root / spec.mzx_name).read_text(encoding="latin-1", errors="ignore")
    if spec.code == "YC":
        if scenario == "dqn":
            return set_management_for_treatment(source, spec.treatment, "L", "L")
        if scenario == "null":
            return prepare_text_for_scenario(source, spec.treatment, "null")
    if spec.code == "FQ":
        source_scenario = "dqn_linked_free_daily" if scenario == "dqn" else "null"
        return prepare_text_for_shifted_scenario(spec.year, source_scenario)
    raise ValueError(f"Unsupported site/scenario: {spec.code}/{scenario}")


def inspect_text(spec: SiteSpec, text: str, expected_irrig: str, expected_ferti: str) -> dict[str, Any]:
    treatment = treatment_row(text, spec.treatment)
    options = control_row(text, spec.treatment, "OPTIONS")
    management = control_row(text, spec.treatment, "MANAGEMENT")
    checks = {
        "treatment_ic_is_1": treatment.get("IC") == "1",
        "water_enabled": options.get("WATER") == "Y",
        "nitrogen_enabled": options.get("NITRO") == "Y",
        "irrigation_mode_expected": management.get("IRRIG") == expected_irrig,
        "fertilizer_mode_expected": management.get("FERTI") == expected_ferti,
        "weather_reference_present": spec.weather_name.removesuffix(".WTH") in text,
        "soil_reference_present": spec.soil_id in text,
    }
    return {
        "treatment_row": treatment,
        "options_row": options,
        "management_row": management,
        "checks": checks,
        "passed": all(checks.values()),
    }


def static_site_audit(spec: SiteSpec) -> dict[str, Any]:
    required = {
        "source_mzx": spec.input_root / spec.mzx_name,
        "target_weather": spec.input_root / spec.weather_name,
        "cultivar": spec.input_root / "MZCER048.CUL",
        "soil": spec.input_root / "SOIL.SOL",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required site input(s): " + ", ".join(missing))
    dqn_text = prepare_site_text(spec, "dqn")
    null_text = prepare_site_text(spec, "null")
    dqn = inspect_text(spec, dqn_text, "L", "L")
    null = inspect_text(spec, null_text, "N", "N")
    checks = {
        "dqn_input_passed": bool(dqn["passed"]),
        "null_input_passed": bool(null["passed"]),
        "action_count_is_9": len(ACTION_TABLE_9) == 9,
        "n_steps_is_5": int(DQN_FIXED_KWARGS.get("n_steps", -1)) == 5,
        "environment_seed_is_0": ENVIRONMENT_SEED == 0,
        "irrigation_budget_is_120": float(IRRIGATION_BUDGET) == 120.0,
        "nitrogen_budget_is_300": float(NITROGEN_BUDGET) == 300.0,
        "minimum_interval_is_7": int(MIN_INTERVAL_DAYS) == 7,
        "irrigation_window_is_dap1_120": IRRIGATION_WINDOWS == [(1, 120)],
        "nitrogen_window_is_dap1_120": NITROGEN_WINDOWS == [(1, 120)],
    }
    return {
        "site": spec.code,
        "station": spec.station,
        "year": spec.year,
        "treatment": spec.treatment,
        "config_id": CONFIG_ID,
        "required_files": {key: str(path) for key, path in required.items()},
        "required_file_hashes": {key: sha256(path) for key, path in required.items()},
        "dqn_input": dqn,
        "null_input": null,
        "checks": checks,
        "passed": all(checks.values()),
        "notes": [
            (
                "YC2014 keeps the source ICDAT=08153 with SDATE=14152 to isolate the framework change."
                if spec.code == "YC"
                else "FQ2016 keeps the shifted source ICDAT=07152 with SDATE=16153 to isolate the framework change."
            ),
            "The fixed-profile date choice is a recorded caveat, not changed in experiment 020_12.",
        ],
    }


def copy_site_inputs(spec: SiteSpec, destination: Path, text: str, file_name: str) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    filex = destination / file_name
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for source in spec.input_root.iterdir():
        if source.is_file() and source.name != spec.mzx_name:
            shutil.copyfile(source, destination / source.name)
    return filex


def build_env_args(spec: SiteSpec, run_dir: Path, filex: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    auxiliary = [
        str(path)
        for path in filex.parent.iterdir()
        if path.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}
    ]
    return {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": ENVIRONMENT_SEED,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": spec.treatment,
        "auxiliary_file_paths": auxiliary,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }


def make_raw_env(env_args: dict[str, Any]):
    return shared.make_raw_env(env_args)


def make_train_env(env_args: dict[str, Any], null_baseline_yield: float):
    linked = shared.YCDiscreteBudgetedWrapper(
        make_raw_env(env_args),
        list(IRRIGATION_WINDOWS),
        list(NITROGEN_WINDOWS),
    )
    return BaselineRelativeRewardWrapper(linked, null_baseline_yield)


def parse_events(snapshot: Path) -> pd.DataFrame:
    path = snapshot / "MgmtEvent.OUT"
    columns = ["dap", "amount", "unit", "operation"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        parts = raw.split()
        dap: float = np.nan
        if len(parts) >= 7:
            try:
                dap = float(parts[6])
            except ValueError:
                pass
        match = re.search(r"([-+]?\d+(?:\.\d*)?)\s*(mm|kg(?:\[[A-Za-z]+\])?/ha|kg)", raw)
        rows.append(
            {
                "dap": dap,
                "amount": float(match.group(1)) if match else 0.0,
                "unit": match.group(2) if match else "",
                "operation": raw.strip(),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.drop_duplicates() if not frame.empty else frame


def final_outputs(snapshot: Path) -> tuple[float, float]:
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    grain = pd.to_numeric(plantgro.get("GWAD", pd.Series(dtype=float)), errors="coerce").dropna()
    biomass = pd.to_numeric(plantgro.get("CWAD", pd.Series(dtype=float)), errors="coerce").dropna()
    return (
        float(grain.iloc[-1]) if not grain.empty else np.nan,
        float(biomass.iloc[-1]) if not biomass.empty else np.nan,
    )


def copy_snapshot(env: gymnasium_base.Env, destination: Path) -> None:
    temporary = getattr(env.unwrapped, "_tmp_folder", None)
    if not temporary or not Path(temporary).exists():
        raise RuntimeError("PDI temporary output directory was not available")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(temporary), destination, dirs_exist_ok=False)


def run_local_null(spec: SiteSpec, env_args: dict[str, Any], run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = make_raw_env(env_args)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(380):
            normalized = normalize_action(
                env.formator.action_names,
                env.formator.action_space_dict,
                {"amir": 0.0, "anfer": 0.0},
            )
            obs, _reward, terminated, truncated, info = env.step(normalized)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": spec.code,
                    "year": spec.year,
                    "scenario": "local_same_input_null",
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": 0.0,
                    "fertilizer_kg_ha": 0.0,
                    "reward": 0.0,
                }
            )
            if terminated or truncated:
                break
    finally:
        snapshot = run_dir / "pdi_tmp_snapshot"
        copy_snapshot(env, snapshot)
        env.close()
    daily = pd.DataFrame(rows)
    grain, biomass = final_outputs(snapshot)
    summary = {
        "site": spec.code,
        "year": spec.year,
        "scenario": "local_same_input_null",
        "final_grain_kg_ha": grain,
        "final_biomass_kg_ha": biomass,
        "irrigation_total_mm": 0.0,
        "nitrogen_total_kg_ha": 0.0,
        "total_reward": 0.0,
        "steps": len(daily),
    }
    daily.to_csv(run_dir / "null_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(run_dir / "null_summary.csv", index=False, encoding="utf-8-sig")
    return daily, summary


def evaluate_checkpoint(
    model: Any,
    spec: SiteSpec,
    env_args: dict[str, Any],
    null_yield: float,
    checkpoint: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    env = make_train_env(env_args, null_yield)
    if int(env.action_space.n) != 9:
        env.close()
        raise RuntimeError(f"Action space is {env.action_space.n}, expected 9")
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(380):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            components = dict(getattr(env, "last_reward_components", {}) or {})
            budget_dap = scalar(info.get("budget_dap")) if isinstance(info, dict) else np.nan
            rows.append(
                {
                    "site": spec.code,
                    "station": spec.station,
                    "year": spec.year,
                    "scenario": "frozen_nstep5_dqn",
                    "checkpoint_step": checkpoint,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "operation_dap": budget_dap,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "action_index": int(np.asarray(action).item()),
                    "irrigation_mm": float(safe.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(safe.get("anfer", 0.0)),
                    "reward": float(reward),
                    "yield_gain": float(components.get("yield_gain", 0.0)),
                    "water_cost_term": float(components.get("water_cost_term", 0.0)),
                    "nitrogen_cost_term": float(components.get("nitrogen_cost_term", 0.0)),
                }
            )
            if terminated or truncated:
                break
    finally:
        snapshot = checkpoint_dir / "pdi_tmp_snapshot_eval"
        copy_snapshot(env, snapshot)
        env.close()
    daily = pd.DataFrame(rows)
    daily["cumulative_reward"] = daily["reward"].cumsum()
    events = parse_events(snapshot)
    events.to_csv(checkpoint_dir / "management_events.csv", index=False, encoding="utf-8-sig")
    grain, biomass = final_outputs(snapshot)
    positive = daily.loc[(daily["irrigation_mm"] > 0) | (daily["fertilizer_kg_ha"] > 0)].copy()
    operation_daps = [
        int(value)
        for value in sorted(
            pd.to_numeric(positive["operation_dap"], errors="coerce").dropna().astype(int).unique()
        )
    ]
    intervals = np.diff(operation_daps) if len(operation_daps) > 1 else np.asarray([], dtype=float)
    action_i = float(daily["irrigation_mm"].sum())
    action_n = float(daily["fertilizer_kg_ha"].sum())
    event_i = float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0
    event_n = float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0
    summary = {
        "site": spec.code,
        "station": spec.station,
        "year": spec.year,
        "checkpoint_step": checkpoint,
        "null_baseline_yield": null_yield,
        "final_grain_kg_ha": grain,
        "final_biomass_kg_ha": biomass,
        "action_irrigation_total_mm": action_i,
        "action_nitrogen_total_kg_ha": action_n,
        "mgmt_event_irrigation_total_mm": event_i,
        "mgmt_event_nitrogen_total_kg_ha": event_n,
        "max_water_stress": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
        "max_nitrogen_stress": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
        "total_reward": float(daily["reward"].sum()),
        "steps": len(daily),
    }
    transmission_n_comparable = action_n == 0.0 or event_n > 0.0
    runtime_checks = {
        "action_space_is_9": True,
        "irrigation_within_budget": action_i <= IRRIGATION_BUDGET + 1e-9,
        "nitrogen_within_budget": action_n <= NITROGEN_BUDGET + 1e-9,
        "single_irrigation_within_cap": float(daily["irrigation_mm"].max()) <= 30.0 + 1e-9,
        "single_nitrogen_within_cap": float(daily["fertilizer_kg_ha"].max()) <= 100.0 + 1e-9,
        "operation_interval_at_least_7": bool(intervals.size == 0 or intervals.min() >= MIN_INTERVAL_DAYS),
        "irrigation_action_reaches_mgmtevent": bool(action_i == 0.0 or event_i > 0.0),
        "nitrogen_action_reaches_mgmtevent": bool(transmission_n_comparable),
        "grain_output_is_finite": bool(np.isfinite(grain)),
        "biomass_output_is_finite": bool(np.isfinite(biomass)),
    }
    runtime_audit = {
        "checkpoint_step": checkpoint,
        "operation_daps": operation_daps,
        "intervals": intervals.tolist(),
        "checks": runtime_checks,
        "passed": all(runtime_checks.values()),
    }
    daily.to_csv(checkpoint_dir / "eval_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(checkpoint_dir / "eval_summary.csv", index=False, encoding="utf-8-sig")
    (checkpoint_dir / "runtime_audit.json").write_text(
        json.dumps(runtime_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return daily, summary, runtime_audit


def write_input_hash_audit(run_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    for input_group in ("dqn_input", "null_input"):
        for path in sorted((run_dir / input_group).iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "input_group": input_group,
                        "file": path.name,
                        "suffix": path.suffix.upper(),
                        "sha256": sha256(path),
                    }
                )
    pd.DataFrame(rows).to_csv(run_dir / "input_hash_audit.csv", index=False, encoding="utf-8-sig")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                rendered = "NA"
            elif isinstance(value, (float, np.floating)):
                rendered = f"{float(value):.3f}"
            elif isinstance(value, (int, np.integer)):
                rendered = str(int(value))
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_record(
    spec: SiteSpec,
    seed: int,
    timesteps: int,
    checkpoint_interval: int,
    run_dir: Path,
    null_summary: dict[str, Any],
    summaries: pd.DataFrame,
    selected: dict[str, Any],
) -> None:
    lines = [
        f"# 020_12 {spec.case_id} 冻结 n-step DQN 跨站点验证记录",
        "",
        "## 设计",
        "",
        f"- 冻结配置：`{CONFIG_ID}`。",
        "- 与 HLA 保持同一奖励、9 动作、预算、操作间隔、n-step=5 和 DQN 超参数。",
        "- 仅替换目标站点输入，并在目标站点同一输入上重新计算 null baseline 后独立训练。",
        "- 环境 seed 固定为 0；模型 seed 单独记录。",
        "",
        "## 运行参数",
        "",
        f"- site/year: {spec.code}/{spec.year}",
        f"- model seed: {seed}",
        f"- timesteps: {timesteps}",
        f"- checkpoint interval: {checkpoint_interval}",
        f"- local null GWAD: {float(null_summary['final_grain_kg_ha']):.3f} kg/ha",
        f"- output: `{run_dir.relative_to(PROJECT_ROOT)}`",
        "",
        "## 结果",
        "",
        dataframe_to_markdown(summaries),
        "",
        "## 最佳 checkpoint",
        "",
        f"- checkpoint: {int(selected['checkpoint_step'])}",
        f"- total reward: {float(selected['total_reward']):.3f}",
        f"- GWAD: {float(selected['final_grain_kg_ha']):.3f} kg/ha",
        f"- irrigation: {float(selected['action_irrigation_total_mm']):.3f} mm",
        f"- nitrogen: {float(selected['action_nitrogen_total_kg_ha']):.3f} kg/ha",
        "",
        "## 解释边界",
        "",
        "该结果只能说明冻结训练框架在该目标站点重新训练后的表现；不是 HLA 模型权重的直接跨站点迁移。",
    ]
    record = DOC_ROOT / (
        f"2026-07-10_020_12_{spec.case_id.lower()}_seed{seed}_frozen_nstep_cross_site_record.md"
    )
    record.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    spec: SiteSpec,
    seed: int,
    timesteps: int,
    checkpoint_interval: int,
    run_tag: str = "",
) -> Path:
    if timesteps <= 0 or checkpoint_interval <= 0:
        raise ValueError("timesteps and checkpoint interval must be positive")
    if timesteps % checkpoint_interval != 0:
        raise ValueError("timesteps must be divisible by checkpoint interval")
    audit = static_site_audit(spec)
    if not audit["passed"]:
        raise RuntimeError("Static input/config audit failed: " + json.dumps(audit, ensure_ascii=False))

    apply_environment_constants(shared)
    safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", run_tag.strip())
    suffix = f"_{safe_tag}" if safe_tag else ""
    run_dir = OUT_ROOT / spec.case_id / f"seed{seed}_{timesteps}steps{suffix}"
    if run_dir.exists():
        raise RuntimeError(f"Output exists; refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    dqn_filex = copy_site_inputs(
        spec,
        run_dir / "dqn_input",
        prepare_site_text(spec, "dqn"),
        f"{spec.case_id}_frozen_nstep5_dqn.MZX",
    )
    null_filex = copy_site_inputs(
        spec,
        run_dir / "null_input",
        prepare_site_text(spec, "null"),
        f"{spec.case_id}_same_input_null.MZX",
    )
    dqn_args = build_env_args(spec, run_dir / "dqn", dqn_filex)
    null_args = build_env_args(spec, run_dir / "null", null_filex)
    (run_dir / "dqn_env_args.json").write_text(json.dumps(dqn_args, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "null_env_args.json").write_text(json.dumps(null_args, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "static_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "frozen_config.json").write_text(json.dumps(manifest(), indent=2, ensure_ascii=False), encoding="utf-8")
    write_input_hash_audit(run_dir)

    _null_daily, null_summary = run_local_null(spec, null_args, run_dir / "null")
    null_yield = float(null_summary["final_grain_kg_ha"])
    if not np.isfinite(null_yield):
        raise RuntimeError("Local same-input null yield is not finite")

    from stable_baselines3 import DQN

    train_env = make_train_env(dqn_args, null_yield)
    if int(train_env.action_space.n) != 9:
        train_env.close()
        raise RuntimeError(f"Action space is {train_env.action_space.n}, expected 9")
    model = DQN("MlpPolicy", train_env, verbose=0, **dqn_kwargs(seed))
    all_daily: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []
    all_audits: list[dict[str, Any]] = []
    try:
        previous = 0
        for checkpoint in range(checkpoint_interval, timesteps + 1, checkpoint_interval):
            model.learn(
                total_timesteps=checkpoint - previous,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            previous = checkpoint
            checkpoint_dir = run_dir / "checkpoints" / f"checkpoint_{checkpoint}"
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            model.save(str(checkpoint_dir / "model"))
            daily, summary, runtime_audit = evaluate_checkpoint(
                model,
                spec,
                dqn_args,
                null_yield,
                checkpoint,
                checkpoint_dir,
            )
            all_daily.append(daily)
            all_summaries.append(summary)
            all_audits.append(runtime_audit)
    finally:
        train_env.close()

    daily_frame = pd.concat(all_daily, ignore_index=True)
    summary_frame = pd.DataFrame(all_summaries)
    audit_frame = pd.DataFrame(
        [
            {
                "checkpoint_step": item["checkpoint_step"],
                "passed": item["passed"],
                **item["checks"],
            }
            for item in all_audits
        ]
    )
    daily_frame.to_csv(run_dir / "all_checkpoint_daily.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(run_dir / "checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    audit_frame.to_csv(run_dir / "runtime_audit_summary.csv", index=False, encoding="utf-8-sig")

    ranked = summary_frame.sort_values(
        ["total_reward", "checkpoint_step"], ascending=[False, True], kind="stable"
    )
    selected = ranked.iloc[0].to_dict()
    selected_checkpoint = int(selected["checkpoint_step"])
    selected_daily = daily_frame.loc[daily_frame["checkpoint_step"].eq(selected_checkpoint)].copy()
    selected_daily.to_csv(run_dir / "selected_checkpoint_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([selected]).to_csv(run_dir / "selected_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    selection = {
        "selection_rule": CHECKPOINT_SELECTION,
        "checkpoint_step": selected_checkpoint,
        "model_path": str(run_dir / "checkpoints" / f"checkpoint_{selected_checkpoint}" / "model.zip"),
        "summary": selected,
    }
    (run_dir / "selected_checkpoint.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_record(spec, seed, timesteps, checkpoint_interval, run_dir, null_summary, summary_frame, selected)
    return run_dir


def print_static_audit(site: str) -> None:
    sites = SITE_SPECS.values() if site == "ALL" else [SITE_SPECS[site]]
    audits = [static_site_audit(spec) for spec in sites]
    print(json.dumps(audits, indent=2, ensure_ascii=False))
    if not all(item["passed"] for item in audits):
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="020_12 frozen n-step DQN cross-site validation")
    parser.add_argument("--site", choices=["YC", "FQ", "ALL"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--run-tag", default="", help="Optional suffix for a non-overwriting retry")
    parser.add_argument("--static-check", action="store_true")
    args = parser.parse_args()
    if args.static_check:
        print_static_audit(args.site)
        return
    if args.site == "ALL":
        raise ValueError("Training must be launched one site at a time; --site ALL is static-check only")
    output = run_experiment(
        SITE_SPECS[args.site],
        args.seed,
        args.timesteps,
        args.checkpoint_interval,
        args.run_tag,
    )
    print(f"completed: {output}")


if __name__ == "__main__":
    main()
