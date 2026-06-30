from __future__ import annotations

import json
import re
import shutil
import sys
import argparse
from pathlib import Path
from typing import Any

import gymnasium as gymnasium_base
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT,
    SITE_CONFIG,
    parse_dssat_table,
    parse_weather,
    prepare_text_for_scenario,
    set_management_for_treatment,
)


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2008_linked_dqn_5k_multiseed_013_08"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_013_08_yc2008_linked_dqn_5k_multiseed_record.md"

SITE = "YC"
STATION = "Yucheng"
YEAR = 2008
TRNO = SITE_CONFIG[SITE]["treatments"][YEAR]
MZX_NAME = SITE_CONFIG[SITE]["mzx"]

TIMESTEPS = 5000
SEED = 0
WATER_COST = 1.0
NITROGEN_COST = 5.0

IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7
FREE_DAILY_WINDOWS = {
    "irrigation": [(1, 120)],
    "nitrogen": [(1, 120)],
}
AGRONOMIC_WINDOWS = {
    "irrigation": [(35, 65)],
    "nitrogen": [(1, 10), (35, 55)],
}

SCENARIO_ORDER = ["dqn_linked_free_daily", "dqn_linked_agronomic_window"]
SCENARIO_LABELS = {
    "null": "Null",
    "recorded": "Recorded",
    "dssat_auto": "DSSAT auto",
    "dqn_linked_free_daily": "DQN linked free daily",
    "dqn_linked_agronomic_window": "DQN linked agronomic window",
}
SCENARIO_COLORS = {
    "null": "#464C55",
    "recorded": "#CC6F47",
    "dssat_auto": "#5477C4",
    "dqn_linked_free_daily": "#386411",
    "dqn_linked_agronomic_window": "#7A2E8E",
}

ACTION_TABLE: dict[int, dict[str, float]] = {
    0: {"amir": 0.0, "anfer": 0.0},
    1: {"amir": 30.0, "anfer": 0.0},
    2: {"amir": 0.0, "anfer": 100.0},
    3: {"amir": 30.0, "anfer": 100.0},
}


class LazyScalarGymDssatWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.formator = env.formator
        self.last_observation_dict = {}

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.last_observation_dict = dict(getattr(self.env, "observation", {}) or {})
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation_dict = dict(getattr(self.env, "observation", {}) or {})
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


class YCDiscreteBudgetedWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(self, env, irrigation_windows: list[tuple[int, int]], nitrogen_windows: list[tuple[int, int]]):
        super().__init__()
        self.env = env
        self.irrigation_windows = irrigation_windows
        self.nitrogen_windows = nitrogen_windows
        self.action_space = gymnasium_base.spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = env.observation_space
        self.formator = env.formator
        self.reset_budget_state()
        self.last_action_index = 0
        self.last_raw_real_action = {"amir": 0.0, "anfer": 0.0}
        self.last_safe_real_action = {"amir": 0.0, "anfer": 0.0}

    def reset_budget_state(self):
        self.used_irrigation = 0.0
        self.used_nitrogen = 0.0
        self.last_operation_dap: int | None = None

    def reset(self, *args, **kwargs):
        self.reset_budget_state()
        self.last_action_index = 0
        self.last_raw_real_action = {"amir": 0.0, "anfer": 0.0}
        self.last_safe_real_action = {"amir": 0.0, "anfer": 0.0}
        return self.env.reset(*args, **kwargs)

    def _current_dap(self) -> int:
        raw = getattr(self.env.unwrapped, "observation", None)
        if isinstance(raw, dict) and "dap" in raw:
            return int(round(scalar(raw.get("dap", 0)) or 0))
        return 0

    def _normalize_real_action(self, real_action: dict[str, float]) -> np.ndarray:
        values = []
        spaces = getattr(self.formator.action_space_dict, "spaces", self.formator.action_space_dict)
        for name in self.formator.action_names:
            space = spaces[name]
            low = float(np.asarray(space.low).flatten()[0])
            high = float(np.asarray(space.high).flatten()[0])
            real = float(real_action.get(name, 0.0))
            values.append(2.0 * ((real - low) / (high - low)) - 1.0)
        return np.asarray(values, dtype=np.float32)

    def _safe_real_action(self, raw_real: dict[str, float], dap: int) -> dict[str, float]:
        if dap < 1:
            return {"amir": 0.0, "anfer": 0.0}
        can_operate = self.last_operation_dap is None or (dap - self.last_operation_dap) >= MIN_INTERVAL_DAYS
        if not can_operate:
            return {"amir": 0.0, "anfer": 0.0}
        in_irrigation_window = any(left <= dap <= right for left, right in self.irrigation_windows)
        in_nitrogen_window = any(left <= dap <= right for left, right in self.nitrogen_windows)
        remaining_i = max(0.0, IRRIGATION_BUDGET - self.used_irrigation)
        remaining_n = max(0.0, NITROGEN_BUDGET - self.used_nitrogen)
        safe_i = min(max(0.0, raw_real.get("amir", 0.0)), DAILY_IRRIGATION_CAP, remaining_i) if in_irrigation_window else 0.0
        safe_n = min(max(0.0, raw_real.get("anfer", 0.0)), DAILY_NITROGEN_CAP, remaining_n) if in_nitrogen_window else 0.0
        return {"amir": safe_i, "anfer": safe_n}

    def step(self, action):
        action_index = int(np.asarray(action).item())
        raw_real = dict(ACTION_TABLE[action_index])
        dap = self._current_dap()
        safe_real = self._safe_real_action(raw_real, dap)
        safe_norm = self._normalize_real_action(safe_real)
        obs, reward, terminated, truncated, info = self.env.step(safe_norm)
        if safe_real["amir"] > 0 or safe_real["anfer"] > 0:
            self.last_operation_dap = dap
        self.used_irrigation += safe_real["amir"]
        self.used_nitrogen += safe_real["anfer"]
        self.last_action_index = action_index
        self.last_raw_real_action = raw_real
        self.last_safe_real_action = safe_real
        info = info if isinstance(info, dict) else {}
        info.update(
            {
                "used_irrigation": self.used_irrigation,
                "used_nitrogen": self.used_nitrogen,
                "budget_dap": dap,
            }
        )
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


class EconomicRewardWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.last_grnwt = 0.0
        self.last_reward_components = {
            "delta_grnwt": 0.0,
            "water_cost_term": 0.0,
            "nitrogen_cost_term": 0.0,
            "economic_reward": 0.0,
        }

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        latest = latest_observation_dict(self.env, obs, info)
        self.last_grnwt = float(scalar(latest.get("grnwt", 0.0)) or 0.0)
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        latest = latest_observation_dict(self.env, obs, info)
        grnwt = float(scalar(latest.get("grnwt", self.last_grnwt)) or self.last_grnwt)
        delta_grnwt = max(0.0, grnwt - self.last_grnwt)
        irrigation = float(getattr(self.env, "last_safe_real_action", {}).get("amir", 0.0))
        nitrogen = float(getattr(self.env, "last_safe_real_action", {}).get("anfer", 0.0))
        water_cost_term = WATER_COST * irrigation
        nitrogen_cost_term = NITROGEN_COST * nitrogen
        reward = float(delta_grnwt - water_cost_term - nitrogen_cost_term)
        self.last_grnwt = grnwt
        self.last_reward_components = {
            "delta_grnwt": delta_grnwt,
            "water_cost_term": water_cost_term,
            "nitrogen_cost_term": nitrogen_cost_term,
            "economic_reward": reward,
        }
        info = info if isinstance(info, dict) else {}
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


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            m = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if m:
                values.append(float(m.group(1)))
    return values


def parse_events_eval(run_dir: Path, scenario: str) -> pd.DataFrame:
    path = run_dir / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["scenario", "dap", "amount", "unit", "operation"])
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        parts = raw.split()
        dap = np.nan
        if len(parts) >= 7:
            try:
                dap = int(parts[6])
            except ValueError:
                pass
        amount = 0.0
        unit = ""
        m = re.search(r"([-+]?\d+(?:\.\d*)?)\s*(mm|kg(?:\[[A-Za-z]+\])?/ha|kg)", raw)
        if m:
            amount = float(m.group(1))
            unit = m.group(2)
        rows.append({"scenario": scenario, "dap": dap, "amount": amount, "unit": unit, "operation": raw.strip()})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["scenario", "dap", "amount", "unit", "operation"])
    return out


def prepare_case_for_scenario(scenario: str) -> Path:
    input_src = INPUT_ROOT / SITE
    run_dir = OUT_DIR / f"seed{SEED}" / scenario
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    source = (input_src / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    if scenario.startswith("dqn_"):
        # Dynamic RL actions require linked management.  Keep the original
        # management tables/pointers available, and only switch the treatment
        # management mode to IRRIG=L and FERTI=L.  Do not reuse the null
        # template here: null sets MI/MF to 0, which leaves linked fertilizer
        # without a valid fertilizer table and triggers DSSAT FertType_mod
        # index-0 errors.
        text = set_management_for_treatment(source, TRNO, "L", "L")
    else:
        text = prepare_text_for_scenario(source, TRNO, scenario)
    filex = input_dir / f"{scenario}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in input_src.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)
    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": SEED,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": TRNO,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def make_raw_env(env_args: dict):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def run_zero_action_scenario(scenario: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"[baseline] prepare {scenario}", flush=True)
    run_dir = prepare_case_for_scenario(scenario)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_raw_env(env_args)
    rows = []
    try:
        print(f"[baseline] reset {scenario}", flush=True)
        obs, info = env.reset()
        print(f"[baseline] stepping {scenario}", flush=True)
        for step in range(380):
            action = {"amir": 0.0, "anfer": 0.0}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "scenario": scenario,
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
                    "action_index": np.nan,
                    "reward": reward,
                }
            )
            if step % 50 == 0:
                print(f"[baseline] {scenario} step={step} dap={rows[-1]['dap']}", flush=True)
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    print(f"[baseline] postprocess {scenario}", flush=True)
    events = parse_events_eval(run_dir, scenario)
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    if not events.empty:
        event_map = events.copy()
        event_map["irrigation_mm"] = np.where(event_map["unit"].eq("mm"), event_map["amount"], 0.0)
        event_map["fertilizer_kg_ha"] = np.where(event_map["unit"].str.contains("kg", na=False), event_map["amount"], 0.0)
        event_map = event_map.groupby("dap", as_index=False)[["irrigation_mm", "fertilizer_kg_ha"]].sum()
        daily = daily.merge(event_map, on="dap", how="left", suffixes=("", "_event"))
        daily["irrigation_mm"] = daily["irrigation_mm_event"].fillna(daily["irrigation_mm"])
        daily["fertilizer_kg_ha"] = daily["fertilizer_kg_ha_event"].fillna(daily["fertilizer_kg_ha"])
        daily = daily.drop(columns=["irrigation_mm_event", "fertilizer_kg_ha_event"])
    summary = {
        "scenario": scenario,
        "action_irrigation_total": 0.0,
        "action_fertilizer_total": 0.0,
        "mgmt_event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmt_event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
    }
    return daily, summary


def run_dqn_smoke(scenario: str, windows: dict[str, list[tuple[int, int]]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"[dqn] prepare {scenario}", flush=True)
    run_dir = prepare_case_for_scenario(scenario)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    from stable_baselines3 import DQN

    print("[dqn] train env init", flush=True)
    train_env = EconomicRewardWrapper(
        YCDiscreteBudgetedWrapper(make_raw_env(env_args), windows["irrigation"], windows["nitrogen"])
    )
    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=SEED,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=50,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )
    try:
        print("[dqn] learn start", flush=True)
        model.learn(total_timesteps=int(TIMESTEPS), progress_bar=False)
        print("[dqn] learn done", flush=True)
        model.save(str(run_dir / "dqn_model"))
    finally:
        train_env.close()

    print("[dqn] eval init", flush=True)
    eval_env = EconomicRewardWrapper(
        YCDiscreteBudgetedWrapper(make_raw_env(env_args), windows["irrigation"], windows["nitrogen"])
    )
    rows = []
    try:
        print("[dqn] eval reset", flush=True)
        obs, info = eval_env.reset()
        print("[dqn] eval stepping", flush=True)
        for step in range(240):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "scenario": scenario,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": eval_env.last_safe_real_action.get("amir", 0.0),
                    "fertilizer_kg_ha": eval_env.last_safe_real_action.get("anfer", 0.0),
                    "action_index": eval_env.last_action_index,
                    "reward": reward,
                }
            )
            if step % 50 == 0:
                print(f"[dqn] step={step} dap={rows[-1]['dap']} action={rows[-1]['action_index']}", flush=True)
            if terminated or truncated:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        eval_env.close()

    daily = pd.DataFrame(rows)
    events = parse_events_eval(run_dir, scenario)
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = {
        "scenario": scenario,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "mgmt_event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmt_event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
    }
    daily.to_csv(run_dir / f"{scenario}_eval_daily.csv", index=False, encoding="utf-8-sig")
    return daily, summary


def build_plot_df(all_daily: pd.DataFrame) -> pd.DataFrame:
    rain = parse_weather(SITE, YEAR)
    merged = all_daily.merge(rain, on="doy", how="left")
    merged["rain"] = merged["rain"].fillna(0.0)
    return merged


def process_plot(data: pd.DataFrame, out_path: Path) -> None:
    max_dap = int(np.nanmax(data["dap"])) if not data.empty else 130
    x_max = max(10, max_dap + 5)
    x_ticks = np.arange(0, x_max + 1, 25)
    fig, axes = plt.subplots(
        5, 1, figsize=(15.8, 13.4), sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0, 1.0, 1.1], "hspace": 0.25},
    )

    rain = data[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")

    for scenario in SCENARIO_ORDER:
        sub = data[data["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        label = SCENARIO_LABELS[scenario]
        axes[1].plot(sub["dap"], sub["swfac"], color=color, linewidth=2.0, label=label)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, linewidth=2.0, label=label)
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, linewidth=2.0)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, linewidth=1.6, linestyle="--", alpha=0.65)
        mg_i = sub[sub["irrigation_mm"].fillna(0) > 1e-6]
        mg_n = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-6]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, linewidth=2.4, alpha=0.95)
        if not mg_n.empty:
            axes[3].scatter(mg_n["dap"], mg_n["fertilizer_kg_ha"], marker="^", s=48, color=color, edgecolor="#FFFFFF", linewidth=0.6, zorder=4)

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[1].set_title("YC 2014 linked DQN smoke process plot", loc="left", fontsize=14)
    axes[2].set_title("Water and nitrogen stress", loc="left", fontsize=10)
    axes[3].set_title("Management events", loc="left", fontsize=10)
    axes[4].set_title("Crop outcome: solid=grain, dashed=biomass", loc="left", fontsize=10)

    for ax in axes:
        ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x_ticks)

    legend_lines = [Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2, label=SCENARIO_LABELS[s]) for s in SCENARIO_ORDER]
    axes[1].legend(handles=legend_lines, loc="upper left", ncol=2, frameon=False, fontsize=10)
    mgmt_legend = [
        Line2D([0], [0], color="#333333", lw=2, label="Irrigation"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#333333", markeredgecolor="#FFFFFF", markersize=8, label="Fertilization"),
    ]
    axes[3].legend(handles=mgmt_legend, loc="upper left", frameon=False, fontsize=9)
    outcome_legend = [
        Line2D([0], [0], color="#333333", lw=2, label="Grain"),
        Line2D([0], [0], color="#333333", lw=1.5, linestyle="--", label="Biomass"),
    ]
    axes[4].legend(handles=outcome_legend, loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_doc(summary_df: pd.DataFrame) -> None:
    headers = list(summary_df.columns)
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in summary_df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}" if not float(val).is_integer() else f"{int(val)}")
            else:
                vals.append(str(val))
        md_lines.append("| " + " | ".join(vals) + " |")
    lines = [
        "# 013_08 YC2008 linked DQN 5K 多 seed 记录",
        "",
        "## 设置",
        "",
        "- 情景：dqn_linked_free_daily / dqn_linked_agronomic_window",
        f"- RL：DQN, timesteps={TIMESTEPS}, seed={SEED}",
        f"- 奖励：delta_grnwt - {WATER_COST}*I - {NITROGEN_COST}*N",
        f"- 预算：I<={IRRIGATION_BUDGET}, N<={NITROGEN_BUDGET}",
        "- 关键修复：DQN 输入保留原始管理表/指针，只设置 IRRIG=L, FERTI=L，确保动作进入 MgmtEvent.OUT。",
        "",
        "## 汇总",
        "",
        "\n".join(md_lines),
        "",
    ]
    seed_doc = OUT_DIR / f"seed{SEED}" / f"013_08_yc2008_linked_dqn_5k_seed{SEED}_record.md"
    seed_doc.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", choices=["free", "window", "both"], default="both")
    args = parser.parse_args()
    SEED = int(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_out_dir = OUT_DIR / f"seed{SEED}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_daily = []
    summaries = []
    if args.only in ("both", "free"):
        dqn_daily, dqn_summary = run_dqn_smoke("dqn_linked_free_daily", FREE_DAILY_WINDOWS)
        all_daily.append(dqn_daily)
        summaries.append(dqn_summary)
    if args.only in ("both", "window"):
        dqn_daily, dqn_summary = run_dqn_smoke("dqn_linked_agronomic_window", AGRONOMIC_WINDOWS)
        all_daily.append(dqn_daily)
        summaries.append(dqn_summary)

    all_daily_df = pd.concat(all_daily, ignore_index=True, sort=False)
    all_daily_df = build_plot_df(all_daily_df)
    summary_df = pd.DataFrame(summaries)
    summary_df["scenario"] = pd.Categorical(summary_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    summary_df = summary_df.sort_values("scenario").reset_index(drop=True)

    all_daily_df.to_csv(run_out_dir / f"013_08_yc2008_linked_dqn_5k_seed{SEED}_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_out_dir / f"013_08_yc2008_linked_dqn_5k_seed{SEED}_summary.csv", index=False, encoding="utf-8-sig")
    process_plot(all_daily_df, figures_dir / f"yc2008_linked_dqn_5k_seed{SEED}_process.png")
    write_doc(summary_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
