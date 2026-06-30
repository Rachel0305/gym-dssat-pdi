from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gymnasium_base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


SOURCE_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_new_cultivar_candidate_year_screening"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_2015_official_reward_restart"
OFFICIAL_REWARD = PROJECT_ROOT / "references" / "rewards.py"
NEW_CUL = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "cultivar_calibration_HLA2004_480" / "input_corrected_package" / "MZCER048.CUL"


def install_official_reward_module() -> None:
    """Load references/rewards.py as the package reward module for this process."""
    if not OFFICIAL_REWARD.exists():
        raise FileNotFoundError(str(OFFICIAL_REWARD))
    module_name = "gym_dssat_pdi.envs.configs.rewards"
    spec = importlib.util.spec_from_file_location(module_name, OFFICIAL_REWARD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import reward file: {OFFICIAL_REWARD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    original_all_reward = module.all_reward
    original_get_reward_function = module.get_reward_function

    def scalar_all_reward(*args, **kwargs):
        value = original_all_reward(*args, **kwargs)
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(np.nansum(np.asarray(value, dtype=float)))
        return value

    def scalar_get_reward_function(mode):
        if mode == "all":
            return scalar_all_reward
        return original_get_reward_function(mode)

    module.all_reward = scalar_all_reward
    module.get_reward_function = scalar_get_reward_function


def set_treatment_mi_mf(text: str, mi: str = "1", mf: str = "1") -> str:
    out = []
    changed = False
    for line in text.splitlines():
        if re.match(r"^\s*1\s+1\s+1\s+0\s+\S+", line):
            parts = line.split()
            while len(parts) < 18:
                parts.append("0")
            parts[10] = mi
            parts[11] = mf
            out.append(
                f" {parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]:<25} "
                f"{parts[5]:>2} {parts[6]:>2} {parts[7]:>2} {parts[8]:>2} {parts[9]:>2} {parts[10]:>2} {parts[11]:>2} "
                f"{parts[12]:>2} {parts[13]:>2} {parts[14]:>2} {parts[15]:>2} {parts[16]:>2} {parts[17]:>2}"
            )
            changed = True
        else:
            out.append(line)
    if not changed:
        raise RuntimeError("Treatment line not found")
    return "\n".join(out) + "\n"


def set_management_line(text: str, irrig: str = "R", ferti: str = "R") -> str:
    new_text, n = re.subn(
        r"(?m)^(\s*1\s+MA\s+R\s+)\S+(\s+)\S+(\s+R\s+M)\s*$",
        rf"\1{irrig}\2{ferti}\3",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Management line not found")
    return new_text


def set_pdi_jinja_placeholders(text: str) -> str:
    """Insert the official gym-DSSAT Jinja placeholders into a static MZX copy.

    DssatPdi renders only `{{ wther }}`, `{{ plant }}`, `{{ irrig }}`, and
    `{{ ferti }}`. Static DSSAT MZX files do not contain these placeholders,
    so mode='all' cannot switch MANAGEMENT to linked mode unless we insert
    them before passing the file to gym-DSSAT.
    """
    text, n_methods = re.subn(
        r"(?m)^(\s*1\s+ME\s+)\S+(\s+M\s+E\s+R\s+S\s+\S\s+R\s+1\s+G\s+\S\s+2\s*)$",
        r"\1{{ wther }}\2",
        text,
        count=1,
    )
    if n_methods != 1:
        raise RuntimeError("METHODS/WTHER line not found for Jinja placeholder insertion")

    text, n_management = re.subn(
        r"(?m)^(\s*1\s+MA\s+)\S+(\s+)\S+(\s+)\S+(\s+\S\s+\S\s*)$",
        r"\1{{ plant }}\2{{ irrig }}\3{{ ferti }}\4",
        text,
        count=1,
    )
    if n_management != 1:
        raise RuntimeError("MANAGEMENT line not found for Jinja placeholder insertion")
    return text


def read_pdate(text: str) -> str:
    in_plant = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@P PDATE"):
            in_plant = True
            continue
        if in_plant and re.match(r"^\s*1\s+\d{5}\b", line):
            return line.split()[1]
        if in_plant and (stripped.startswith("@") or stripped.startswith("*")):
            in_plant = False
    raise RuntimeError("PDATE not found")


def replace_zero_management_rows(text: str) -> str:
    pdate = read_pdate(text)
    out = []
    in_ir = False
    in_fe = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@I IDATE"):
            in_ir = True
            in_fe = False
            out.append(line)
            out.append(f" 1 {pdate} IR001     0")
            continue
        if stripped.startswith("@F FDATE"):
            in_fe = True
            in_ir = False
            out.append(line)
            out.append(f" 1 {pdate} FE005 AP002     0     0   -99   -99   -99   -99   -99 null")
            continue
        if in_ir:
            if stripped.startswith("@") or stripped.startswith("*"):
                in_ir = False
                out.append(line)
            elif re.match(r"^\s*1\s+\d{5}\b", line):
                continue
            else:
                out.append(line)
            continue
        if in_fe:
            if stripped.startswith("@") or stripped.startswith("*"):
                in_fe = False
                out.append(line)
            elif re.match(r"^\s*1\s+\d{5}\b", line):
                continue
            else:
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_case(year: int, tag: str) -> Path:
    src_dir = SOURCE_ROOT / "null" / str(year) / "input"
    if not src_dir.exists():
        raise FileNotFoundError(str(src_dir))
    case_dir = OUT_DIR / tag / str(year)
    input_dir = case_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_file():
            shutil.copyfile(src, input_dir / src.name)
    shutil.copyfile(NEW_CUL, input_dir / "MZCER048.CUL")
    filex = next(input_dir.glob("*.MZX"))
    text = filex.read_text(encoding="latin1", errors="ignore")
    text = set_treatment_mi_mf(text, "1", "1")
    text = set_pdi_jinja_placeholders(text)
    text = replace_zero_management_rows(text)
    filex.write_text(text, encoding="latin1")
    aux = [str(p) for p in sorted(input_dir.iterdir()) if p.is_file() and p.name != filex.name]
    env_args = {
        "log_saving_path": str(case_dir / f"{tag}_{year}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_dir


def prepare_case_at(year: int, case_dir: Path) -> Path:
    prepared = prepare_case(year, "_tmp_prepare_case")
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(prepared, case_dir)
    shutil.rmtree(prepared)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    input_dir = case_dir / "input"
    filex = next(input_dir.glob("*.MZX"))
    env_args["log_saving_path"] = str(case_dir / f"{case_dir.name}.log")
    env_args["fileX_template_path"] = str(filex)
    env_args["auxiliary_file_paths"] = [str(p) for p in sorted(input_dir.iterdir()) if p.is_file() and p.name != filex.name]
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_dir


def parse_events(path: Path) -> dict[str, Any]:
    irrigation = 0.0
    fertilizer = 0.0
    irrigation_events = 0
    fertilizer_events = 0
    if path.exists():
        for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
            if "Irrigation" in line:
                m = re.search(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*mm", line)
                if m:
                    irrigation += float(m.group(1))
                    irrigation_events += 1
            if "Fertilizer" in line:
                m = re.search(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*kg", line)
                if m:
                    fertilizer += float(m.group(1))
                    fertilizer_events += 1
    return {
        "irrigation_total_mgmtevent": irrigation,
        "fertilizer_total_mgmtevent": fertilizer,
        "irrigation_events_mgmtevent": irrigation_events,
        "fertilizer_events_mgmtevent": fertilizer_events,
    }


def child_action_smoke(year: int, max_steps: int = 150) -> None:
    install_official_reward_module()
    import gym
    from sb3_wrapper import GymDssatWrapper

    case_dir = OUT_DIR / "action_channel_smoke" / str(year)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows = []
    try:
        obs, info = env.reset()
        done = False
        for step in range(max_steps):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step))))
            # Expert-like manual action schedule, delivered through gym actions.
            # 2007 shifted reference: N at planting, irrigation near DAP 49/70/95.
            real = {
                "amir": 10.0 if dap in [49, 70, 95] else 0.0,
                "anfer": 165.0 if dap in [1] else 0.0,
            }
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "step": step,
                    "dap_before": dap,
                    "amir": real["amir"],
                    "anfer": real["anfer"],
                    "reward_raw": repr(reward),
                    "dap_after": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
    pd.DataFrame(rows).to_csv(case_dir / "action_channel_smoke_daily.csv", index=False, encoding="utf-8-sig")
    event_summary = parse_events(snapshot / "MgmtEvent.OUT")
    # Add final yield if PlantGro was produced.
    plant = snapshot / "PlantGro.OUT"
    final_gwad = np.nan
    final_topwt = np.nan
    if plant.exists():
        header = None
        for line in plant.read_text(encoding="latin1", errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("@"):
                header = s.replace("@", "", 1).split()
            elif header and re.match(r"^\d{4}\s+\d+", s):
                parts = s.split()
                if "GWAD" in header:
                    final_gwad = float(parts[header.index("GWAD")])
                if "CWAD" in header:
                    final_topwt = float(parts[header.index("CWAD")])
    event_summary["final_gwad"] = final_gwad
    event_summary["final_cwad"] = final_topwt
    (case_dir / "event_summary.json").write_text(json.dumps(event_summary, indent=2), encoding="utf-8")


class ScalarRewardWrapper:
    """Convert official multi-objective/list reward to scalar for SB3."""

    def __init__(self, env):
        self.env = env

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_reward = reward
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = float(np.nansum(np.asarray(reward, dtype=float)))
        else:
            reward = float(reward) if reward is not None else 0.0
        if isinstance(info, dict):
            info = {**info, "raw_official_reward": raw_reward}
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class LazyScalarGymDssatWrapper(gymnasium_base.Env):
    """Gymnasium-compatible wrapper for SB3 without an extra reset in __init__.

    The project-level sb3_wrapper.GymDssatWrapper calls raw env.reset() inside
    __init__ to infer observation shape. That is risky for the socket-based
    DSSAT-PDI environment because SB3/Monitor will reset again before rollout.
    This wrapper uses the observation already produced by DssatPdi.__init__.
    """

    metadata = {"render_modes": []}

    def __init__(self, env):
        from sb3_wrapper import Formator

        super().__init__()
        self.env = env
        self.formator = Formator(env)
        self.action_space = gymnasium_base.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.formator.action_names),),
            dtype=np.float32,
        )
        raw_obs = getattr(env, "observation", None)
        if raw_obs is None:
            raw_obs = env.get_state()
        obs_array = np.asarray(self.formator.format_observation(raw_obs), dtype=np.float32)
        self.observation_space = gymnasium_base.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=obs_array.shape,
            dtype=np.float32,
        )
        self._last_obs = obs_array
        self._last_info = {}

    def reset(self, *, seed=None, options=None):
        raw_obs = self.env.reset(seed=seed)
        obs = np.asarray(self.formator.format_observation(raw_obs), dtype=np.float32)
        self._last_obs = obs
        self._last_info = {}
        return obs, self._last_info

    def step(self, action):
        denormalized = self.formator.denormalize_actions(action)
        formatted_action = self.formator.format_actions(denormalized)
        result = self.env.step(formatted_action)
        if result is None or (isinstance(result, tuple) and result[0] is None):
            return self._last_obs, 0.0, True, False, self._last_info
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result
        obs = np.asarray(self.formator.format_observation(obs), dtype=np.float32)
        raw_reward = reward
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = float(np.nansum(np.asarray(reward, dtype=float)))
        else:
            reward = float(reward) if reward is not None else 0.0
        info = info if isinstance(info, dict) else {}
        info = {**info, "raw_official_reward": raw_reward}
        self._last_obs = obs
        self._last_info = info
        return obs, reward, bool(done), bool(truncated), info

    def close(self):
        return self.env.close()

    def render(self):
        return None

    @property
    def unwrapped(self):
        inner = self.env
        while hasattr(inner, "env"):
            inner = inner.env
        return inner

    def __getattr__(self, name):
        return getattr(self.env, name)


class BudgetedDailyActionWrapper(gymnasium_base.Env):
    """Keep daily interaction but enforce simple agronomic safety limits."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        irrigation_budget: float = 120.0,
        nitrogen_budget: float = 150.0,
        daily_irrigation_cap: float = 30.0,
        daily_nitrogen_cap: float = 50.0,
        min_interval_days: int = 7,
        allow_irrigation: bool = True,
        allow_nitrogen: bool = True,
        irrigation_windows: list[tuple[int, int]] | None = None,
        nitrogen_windows: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.env = env
        self.irrigation_budget = float(irrigation_budget)
        self.nitrogen_budget = float(nitrogen_budget)
        self.daily_irrigation_cap = float(daily_irrigation_cap)
        self.daily_nitrogen_cap = float(daily_nitrogen_cap)
        self.min_interval_days = int(min_interval_days)
        self.allow_irrigation = bool(allow_irrigation)
        self.allow_nitrogen = bool(allow_nitrogen)
        self.irrigation_windows = irrigation_windows
        self.nitrogen_windows = nitrogen_windows
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.formator = env.formator
        self.reset_budget_state()
        self.last_raw_real_action = {"amir": 0.0, "anfer": 0.0}
        self.last_safe_real_action = {"amir": 0.0, "anfer": 0.0}

    def reset_budget_state(self):
        self.used_irrigation = 0.0
        self.used_nitrogen = 0.0
        self.last_operation_dap: int | None = None

    def reset(self, *args, **kwargs):
        self.reset_budget_state()
        self.last_raw_real_action = {"amir": 0.0, "anfer": 0.0}
        self.last_safe_real_action = {"amir": 0.0, "anfer": 0.0}
        return self.env.reset(*args, **kwargs)

    def _current_dap(self) -> int:
        raw = getattr(self.env.unwrapped, "observation", None)
        if isinstance(raw, dict) and "dap" in raw:
            return int(round(scalar(raw.get("dap", 0))))
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
        can_operate = self.last_operation_dap is None or (dap - self.last_operation_dap) >= self.min_interval_days
        if not can_operate:
            return {"amir": 0.0, "anfer": 0.0}
        in_irrigation_window = (
            True
            if self.irrigation_windows is None
            else any(left <= dap <= right for left, right in self.irrigation_windows)
        )
        in_nitrogen_window = (
            True
            if self.nitrogen_windows is None
            else any(left <= dap <= right for left, right in self.nitrogen_windows)
        )
        remaining_i = max(0.0, self.irrigation_budget - self.used_irrigation)
        remaining_n = max(0.0, self.nitrogen_budget - self.used_nitrogen)
        safe_i = (
            min(max(0.0, float(raw_real.get("amir", 0.0))), self.daily_irrigation_cap, remaining_i)
            if self.allow_irrigation and in_irrigation_window
            else 0.0
        )
        safe_n = (
            min(max(0.0, float(raw_real.get("anfer", 0.0))), self.daily_nitrogen_cap, remaining_n)
            if self.allow_nitrogen and in_nitrogen_window
            else 0.0
        )
        return {"amir": safe_i, "anfer": safe_n}

    def step(self, action):
        raw_real_vals = self.formator.denormalize_actions(action)
        raw_real = {name: float(value) for name, value in zip(self.formator.action_names, raw_real_vals)}
        dap = self._current_dap()
        safe_real = self._safe_real_action(raw_real, dap)
        safe_norm = self._normalize_real_action(safe_real)
        obs, reward, terminated, truncated, info = self.env.step(safe_norm)
        if safe_real["amir"] > 0 or safe_real["anfer"] > 0:
            self.last_operation_dap = dap
        self.used_irrigation += safe_real["amir"]
        self.used_nitrogen += safe_real["anfer"]
        self.last_raw_real_action = raw_real
        self.last_safe_real_action = safe_real
        info = info if isinstance(info, dict) else {}
        info.update(
            {
                "raw_real_action_amir": raw_real.get("amir", 0.0),
                "raw_real_action_anfer": raw_real.get("anfer", 0.0),
                "safe_real_action_amir": safe_real.get("amir", 0.0),
                "safe_real_action_anfer": safe_real.get("anfer", 0.0),
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


def child_train_smoke(year: int, timesteps: int, seed: int, variant: str = "joint") -> None:
    install_official_reward_module()
    import gym
    from stable_baselines3 import PPO

    if variant == "joint":
        run_name = f"seed{seed}_{timesteps}steps"
        allow_irrigation, allow_nitrogen = True, True
        irrigation_windows, nitrogen_windows = None, None
    elif variant == "joint_windowed":
        run_name = f"windowed_seed{seed}_{timesteps}steps"
        allow_irrigation, allow_nitrogen = True, True
        irrigation_windows = [(20, 35), (45, 65), (70, 95)]
        nitrogen_windows = [(25, 40), (55, 70)]
    elif variant == "irrigation_only":
        run_name = f"irrigation_seed{seed}_{timesteps}steps"
        allow_irrigation, allow_nitrogen = True, False
        irrigation_windows, nitrogen_windows = None, None
    elif variant == "fertilization_only":
        run_name = f"fertilization_seed{seed}_{timesteps}steps"
        allow_irrigation, allow_nitrogen = False, True
        irrigation_windows, nitrogen_windows = None, None
    else:
        raise ValueError(f"Unknown variant: {variant}")

    case_dir = OUT_DIR / "ppo_smoke" / str(year) / run_name
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    debug_log = case_dir / "ppo_smoke_debug.log"
    debug_log.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{pd.Timestamp.now().isoformat()} {msg}"
        print(line, flush=True)
        with debug_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("child_train_smoke:start")
    log(f"python={sys.executable}")
    log("gym.make:start")
    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    log("gym.make:done")
    log("wrapper:start")
    env = BudgetedDailyActionWrapper(
        LazyScalarGymDssatWrapper(raw),
        allow_irrigation=allow_irrigation,
        allow_nitrogen=allow_nitrogen,
        irrigation_windows=irrigation_windows,
        nitrogen_windows=nitrogen_windows,
    )
    log("wrapper:done")
    model_dir = case_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Smoke-only settings: keep rollout/update tiny to avoid wasting DSSAT/PDI
        # compute. This is not a training-quality configuration.
        log("ppo_init:start")
        model = PPO("MlpPolicy", env, verbose=0, seed=seed, n_steps=5, batch_size=5, n_epochs=1, gamma=0.99)
        log("ppo_init:done")
        log("learn:start")
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        log("learn:done")
        model.save(str(model_dir / "ppo_official_reward_smoke"))
        log("model_save:done")
    finally:
        log("train_env_close:start")
        env.close()
        log("train_env_close:done")

    log("eval_env_make:start")
    eval_env = BudgetedDailyActionWrapper(
        LazyScalarGymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped),
        allow_irrigation=allow_irrigation,
        allow_nitrogen=allow_nitrogen,
        irrigation_windows=irrigation_windows,
        nitrogen_windows=nitrogen_windows,
    )
    log("eval_env_make:done")
    rows = []
    try:
        log("eval_reset:start")
        obs, info = eval_env.reset()
        log("eval_reset:done")
        done = False
        for step in range(220):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(eval_env, obs, info)
            real = eval_env.formator.denormalize_actions(action)
            names = eval_env.formator.action_names
            act = {name: float(value) for name, value in zip(names, real)}
            rows.append(
                {
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "raw_amir": act.get("amir", np.nan),
                    "raw_anfer": act.get("anfer", np.nan),
                    "safe_amir": eval_env.last_safe_real_action.get("amir", np.nan),
                    "safe_anfer": eval_env.last_safe_real_action.get("anfer", np.nan),
                    "used_irrigation": getattr(eval_env, "used_irrigation", np.nan),
                    "used_nitrogen": getattr(eval_env, "used_nitrogen", np.nan),
                    "reward": reward,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        log("eval_env_close:start")
        eval_env.close()
        log("eval_env_close:done")
    pd.DataFrame(rows).to_csv(case_dir / "ppo_smoke_eval_daily.csv", index=False, encoding="utf-8-sig")
    event_summary = parse_events(snapshot / "MgmtEvent.OUT")
    event_summary["variant"] = variant
    event_summary["allow_irrigation"] = allow_irrigation
    event_summary["allow_nitrogen"] = allow_nitrogen
    event_summary["irrigation_windows"] = irrigation_windows
    event_summary["nitrogen_windows"] = nitrogen_windows
    (case_dir / "event_summary.json").write_text(json.dumps(event_summary, indent=2), encoding="utf-8")
    log("child_train_smoke:done")


def run_subprocess(args: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], cwd=str(PROJECT_ROOT), timeout=timeout, capture_output=True, text=True)
        return {"args": " ".join(args), "returncode": proc.returncode, "timed_out": False, "stdout_tail": proc.stdout[-1200:], "stderr_tail": proc.stderr[-1200:]}
    except subprocess.TimeoutExpired as exc:
        return {"args": " ".join(args), "returncode": None, "timed_out": True, "stdout_tail": "", "stderr_tail": str(exc)[-1200:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-action-smoke", type=int)
    parser.add_argument("--child-train-smoke", nargs="+", metavar="CHILD_ARG")
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--variant", choices=["joint", "joint_windowed", "irrigation_only", "fertilization_only"], default="joint")
    args = parser.parse_args()
    if args.child_action_smoke:
        child_action_smoke(int(args.child_action_smoke))
        return
    if args.child_train_smoke:
        y, t, s, *rest = args.child_train_smoke
        variant = rest[0] if rest else "joint"
        child_train_smoke(int(y), int(t), int(s), variant)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    prepare_case(args.year, "action_channel_smoke")
    statuses.append(run_subprocess(["--child-action-smoke", str(args.year)], timeout=args.timeout))
    action_event = OUT_DIR / "action_channel_smoke" / str(args.year) / "event_summary.json"
    action_ok = False
    if action_event.exists():
        event_data = json.loads(action_event.read_text(encoding="utf-8"))
        # MgmtEvent may not record PDI step actions. Treat the channel as usable
        # only if either events are recorded or final yield moves clearly above
        # the 2010 null reference (~6956 kg/ha).
        action_ok = (
            (event_data.get("irrigation_events_mgmtevent", 0) > 0)
            or (event_data.get("fertilizer_events_mgmtevent", 0) > 0)
            or (float(event_data.get("final_gwad", 0) or 0) > 7200)
        )
    if action_ok:
        if args.variant == "joint":
            run_name = f"seed{args.seed}_{args.timesteps}steps"
        elif args.variant == "joint_windowed":
            run_name = f"windowed_seed{args.seed}_{args.timesteps}steps"
        elif args.variant == "irrigation_only":
            run_name = f"irrigation_seed{args.seed}_{args.timesteps}steps"
        else:
            run_name = f"fertilization_seed{args.seed}_{args.timesteps}steps"
        dst_case = OUT_DIR / "ppo_smoke" / str(args.year) / run_name
        prepare_case_at(args.year, dst_case)
        statuses.append(
            run_subprocess(
                ["--child-train-smoke", str(args.year), str(args.timesteps), str(args.seed), args.variant],
                timeout=args.timeout,
            )
        )
    status = pd.DataFrame(statuses)
    status.to_csv(OUT_DIR / "official_reward_restart_smoke_status.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# HLA official reward PPO restart smoke",
        "",
        f"- year: {args.year}",
        f"- timesteps: {args.timesteps}",
        f"- official reward: `{OFFICIAL_REWARD.relative_to(PROJECT_ROOT)}`",
        "- scalarization: `sum([fertilization_reward, irrigation_reward])` for SB3 compatibility",
        f"- new cultivar: `{NEW_CUL.relative_to(PROJECT_ROOT)}`",
        "",
        "## Status",
        "",
        "| args | returncode | timed_out | stderr_tail |",
        "| --- | --- | --- | --- |",
    ]
    for row in status.itertuples(index=False):
        lines.append(f"| {row.args} | {row.returncode} | {row.timed_out} | {str(row.stderr_tail).replace('|','/')[:500]} |")
    if action_event.exists():
        lines += ["", "## Action channel event summary", "", "```json", action_event.read_text(encoding="utf-8"), "```"]
    train_event = OUT_DIR / "ppo_smoke" / str(args.year) / f"seed{args.seed}_{args.timesteps}steps" / "event_summary.json"
    if train_event.exists():
        lines += ["", "## PPO smoke eval event summary", "", "```json", train_event.read_text(encoding="utf-8"), "```"]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(json.dumps({"out_dir": str(OUT_DIR), "status_csv": str(OUT_DIR / "official_reward_restart_smoke_status.csv"), "readme": str(OUT_DIR / "README.md")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
