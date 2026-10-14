"""Lightweight YC/YCA weather pipeline diagnostic.

This script does not train PPO.  It audits the 055_00 YCA/YC pipeline,
renders one working env_args payload, and checks whether episode-level
weather selection changes under the current RandomYearEnv year sampler.
Use --gym-reset only when a real DSSAT/gym reset probe is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222


def load_yca055_module():
    path = ROOT / "src" / "055_yca_lowIC_site_transfer" / "run_055_00_yca_lowIC_expanded_action_maskableppo.py"
    spec = importlib.util.spec_from_file_location("run_055_00_yca_lowIC_expanded_action_maskableppo", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


yca055 = load_yca055_module()


OUT_DIR = ROOT / "results" / "yc_weather_audit"
SNAPSHOT = OUT_DIR / "yc_weather_config_snapshot.json"
REPORT_JSON = OUT_DIR / "yc_weather_reset_diagnostic.json"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def weather_summary(path: Path) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    in_data = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@"):
            in_data = True
            continue
        if not in_data or not line.strip() or line.startswith("*") or line.startswith("!"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "date_code": float(parts[0]),
                    "srad": float(parts[1]),
                    "tmax": float(parts[2]),
                    "tmin": float(parts[3]),
                    "rain": float(parts[4]),
                }
            )
        except ValueError:
            continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rows": 0, "rain_sum": None, "tmax_mean": None, "tmin_mean": None, "sha256": sha256(path)}
    return {
        "rows": int(len(frame)),
        "first_date_code": int(frame["date_code"].iloc[0]),
        "last_date_code": int(frame["date_code"].iloc[-1]),
        "rain_sum": float(frame["rain"].sum()),
        "tmax_mean": float(frame["tmax"].mean()),
        "tmin_mean": float(frame["tmin"].mean()),
        "sha256": sha256(path),
    }


def package_info() -> dict[str, Any]:
    try:
        pkg = importlib.import_module("gym_dssat_pdi")
        mod = importlib.import_module("gym_dssat_pdi.envs.dssat_pdi")
        cls = getattr(mod, "DssatPdi")
        return {
            "gym_dssat_version": getattr(pkg, "__version__", ""),
            "gym_dssat_package_file": getattr(pkg, "__file__", ""),
            "gym_dssat_env_file": inspect.getfile(cls),
            "gym_dssat_init_signature": str(inspect.signature(cls.__init__)),
            "supports_random_weather_arg": "random_weather" in str(inspect.signature(cls.__init__)),
            "supports_auxiliary_file_paths_arg": "auxiliary_file_paths" in str(inspect.signature(cls.__init__)),
        }
    except Exception as exc:
        return {"gym_dssat_version": "", "gym_dssat_probe_error": repr(exc)}


def rng_sequence(years: list[int], seed: int, episodes: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    return [int(rng.choice(years)) for _ in range(int(episodes))]


def build_current_env_args(cfg: dict[str, Any], train_years: list[int]) -> dict[str, Any]:
    config = base03222.load_config()
    config["paths"]["output_root"] = "results/yc_weather_audit/rendered_probe"
    config["seed"] = int(cfg.get("seed", 0))
    split = base03222.load_split()
    selection = base03222.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = yca055.INPUT_PROFILES[str(cfg["input_profile"])]
    year = int(train_years[0])
    info = direct_ppo.find_year(env_config, "YCA", year)
    return ppo_safe_rendering.build_env_args(
        station="YCA",
        year=year,
        planting_date=info["planting_date"],
        seed=int(cfg.get("seed", 0)),
        config=env_config,
        run_tag="yc_weather_audit_probe",
        evaluation=False,
        mode=env_config.get("runtime", {}).get("mode", "all"),
        linked_management=True,
    )


def optional_gym_reset_probe(cfg: dict[str, Any], train_years: list[int], resets: int) -> dict[str, Any]:
    try:
        config = base03222.load_config()
        config["paths"]["output_root"] = "results/yc_weather_audit/gym_reset_probe"
        config["seed"] = int(cfg.get("seed", 0))
        split = base03222.load_split()
        selection = base03222.build_selection(split)
        env_config = direct_ppo.build_env_config(config, selection)
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = yca055.INPUT_PROFILES[str(cfg["input_profile"])]
        env = base03222.RandomYearEnv(config, env_config, "YCA", train_years, int(cfg.get("seed", 0)))
        sequence: list[int] = []
        try:
            for _ in range(int(resets)):
                _obs, info = env.reset()
                sequence.append(int(info.get("active_year")))
        finally:
            env.close()
        return {"gym_reset_attempted": True, "gym_reset_sequence": sequence, "gym_reset_error": ""}
    except Exception as exc:
        return {"gym_reset_attempted": True, "gym_reset_sequence": [], "gym_reset_error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--weather-seed", type=int, default=None)
    parser.add_argument("--gym-reset", action="store_true")
    parser.add_argument("--gym-resets", type=int, default=3)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yca055.read_config(ROOT / "configs" / "055_00_yca_lowIC_expanded_action_maskableppo.json")
    preflight = yca055.preflight(cfg)
    train_years = list(map(int, cfg["scope"]["train_years"]))
    ppo_seed = int(cfg.get("seed", 0))
    weather_seed = ppo_seed if args.weather_seed is None else int(args.weather_seed)
    seq_a = rng_sequence(train_years, weather_seed, args.episodes)
    seq_b = rng_sequence(train_years, weather_seed, args.episodes)
    seq_c = rng_sequence(train_years, weather_seed + 1, args.episodes)
    env_args = build_current_env_args(cfg, train_years)
    weather_path = Path(env_args["auxiliary_file_paths"][1])
    template_path = Path(env_args["fileX_template_path"])
    soil_path = Path(env_args["auxiliary_file_paths"][2])
    cultivar_path = Path(env_args["auxiliary_file_paths"][0])
    climate_files = sorted((ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual" / "YC").glob("*.CLI"))

    diagnostic = {
        "diagnostic_mode": "rng_year_selection_and_env_arg_render_no_ppo",
        "site": "YC",
        "station_code": "YCA",
        "training_entry": "src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py",
        "config_path": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json",
        "preflight": preflight,
        "package_info": package_info(),
        "ppo_seed": ppo_seed,
        "weather_seed": weather_seed,
        "weather_seed_currently_separate_from_ppo_seed": args.weather_seed is not None,
        "train_years": train_years,
        "rng_sequence_seed_a": seq_a,
        "rng_sequence_seed_a_repeat": seq_b,
        "rng_sequence_seed_a_plus_1": seq_c,
        "episode_weather_changes": len(set(seq_a)) > 1,
        "weather_sequence_reproducible": seq_a == seq_b,
        "different_weather_seed_changes_sequence": seq_a != seq_c,
        "env_args": env_args,
        "rendered_template": rel(template_path),
        "rendered_weather_file": rel(weather_path),
        "rendered_weather_summary": weather_summary(weather_path),
        "cultivar_file": rel(cultivar_path),
        "soil_file": rel(soil_path),
        "climate_files_in_current_yc_lowic_input_dir": [rel(path) for path in climate_files],
    }
    if args.gym_reset:
        diagnostic["gym_reset_probe"] = optional_gym_reset_probe(cfg, train_years, args.gym_resets)

    snapshot = {
        "site": "YC",
        "gym_dssat_version": diagnostic["package_info"].get("gym_dssat_version", ""),
        "training_entry": diagnostic["training_entry"],
        "random_weather": bool(env_args.get("random_weather")),
        "ppo_seed": ppo_seed,
        "weather_seed": weather_seed,
        "experiment_file": rel(template_path),
        "weather_file": rel(weather_path),
        "climate_file": climate_files[0].as_posix() if climate_files else "",
        "soil_file": rel(soil_path),
        "wsta": ppo_safe_rendering.target_weather_stem("YCA", int(train_years[0])),
        "episode_weather_changes": bool(diagnostic["episode_weather_changes"]),
        "weather_reproducible": bool(diagnostic["weather_sequence_reproducible"]),
    }
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"snapshot": rel(SNAPSHOT), "diagnostic": rel(REPORT_JSON), **snapshot}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
