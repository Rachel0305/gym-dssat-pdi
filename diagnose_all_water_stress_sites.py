from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from sb3_safe_action_wrapper import SafeActionCaps, SafeActionGymDssatWrapper
from diagnose_water_stress_sites import (
    ETCP_FROM_DSSAT_485,
    SITE_NAMES,
    crop_season_prcp,
    parse_pdate,
    parse_template_irrigation,
    safe_float,
)
from dssat_site_config import SITE_CONFIGS, build_env_args
from sb3_wrapper import GymDssatWrapper, Formator


class NullAllAgent:
    def __init__(self, env):
        self.action_formator = Formator(env.unwrapped)

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        real_actions = [0.0 for _ in self.action_formator.action_names]
        return np.asarray(self.action_formator.normalize_actions(real_actions), dtype=np.float32), obs


class ExpertAllAgent:
    def __init__(self, env):
        self.action_formator = Formator(env.unwrapped)
        obs_vars = env.unwrapped.observation_variables
        if "dap" not in obs_vars:
            raise ValueError(f"'dap' not found in observation variables: {obs_vars}")
        self.dap_index = obs_vars.index("dap")
        self.fertilization_policy = {1: 165.0}
        self.irrigation_policy = {49: 10.0, 70: 10.0, 95: 10.0}

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        flat_obs = np.asarray(obs, dtype=float).flatten()
        dap = int(flat_obs[self.dap_index]) if flat_obs.size > self.dap_index else 0
        action_by_name = {name: 0.0 for name in self.action_formator.action_names}
        if "anfer" in action_by_name:
            action_by_name["anfer"] = self.fertilization_policy.get(dap, 0.0)
        if "amir" in action_by_name:
            action_by_name["amir"] = self.irrigation_policy.get(dap, 0.0)
        real_actions = [action_by_name[name] for name in self.action_formator.action_names]
        return np.asarray(self.action_formator.normalize_actions(real_actions), dtype=np.float32), obs


def latest_observation(env: GymDssatWrapper) -> dict:
    history = getattr(env.unwrapped, "history", {})
    if isinstance(history, dict):
        observations = history.get("observation", [])
        if observations and isinstance(observations[-1], dict):
            return observations[-1]
    return {}


def latest_action(env: GymDssatWrapper) -> dict:
    history = getattr(env.unwrapped, "history", {})
    if isinstance(history, dict):
        actions = history.get("action", [])
        if actions and isinstance(actions[-1], dict):
            return actions[-1]
    return {}


def make_agent(agent_name: str, env: GymDssatWrapper):
    if agent_name == "null":
        return NullAllAgent(env)
    if agent_name == "expert":
        return ExpertAllAgent(env)
    raise ValueError(f"Unsupported agent for this diagnostic: {agent_name}")


def make_agent_with_model(agent_name: str, env: GymDssatWrapper, model_path: str | None):
    if agent_name == "ppo":
        if not model_path:
            raise ValueError("--model-path is required when agents include ppo")
        return PPO.load(model_path)
    return make_agent(agent_name, env)


def evaluate_site_agent(
    site: str,
    agent_name: str,
    env_args: dict,
    output_dir: Path,
    max_steps: int,
    model_path: str | None = None,
    safe_anfer_cap: float | None = None,
    safe_amir_cap: float | None = None,
) -> dict:
    source_env = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args)
    if agent_name == "ppo" and (safe_anfer_cap is not None or safe_amir_cap is not None):
        env = SafeActionGymDssatWrapper(
            source_env.unwrapped,
            SafeActionCaps(anfer=safe_anfer_cap, amir=safe_amir_cap),
        )
    else:
        env = GymDssatWrapper(source_env.unwrapped)
    try:
        observation, _ = env.reset()
        print(f"    obs_vars={env.unwrapped.observation_variables}", flush=True)
        agent = make_agent_with_model(agent_name, env, model_path)
        rows = []
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            step_count += 1
            normalized_action = agent.predict(observation)[0]
            real_action_values = env.formator.denormalize_actions(normalized_action)
            real_action_dict = env.formator.format_actions(real_action_values)
            observation, reward, terminated, truncated, _info = env.step(normalized_action)
            done = terminated or truncated

            obs = latest_observation(env)
            if not obs:
                obs_vars = getattr(env.unwrapped, "observation_variables", [])
                obs = dict(zip(obs_vars, np.asarray(observation).flatten()))
            action = latest_action(env)
            dap = safe_float(obs.get("dap"))
            if np.isnan(dap) or dap == 0:
                continue
            rows.append(
                {
                    "site": site,
                    "agent": agent_name,
                    "dap": dap,
                    "swfac": safe_float(obs.get("swfac")),
                    "turfac": safe_float(obs.get("turfac")),
                    "nstres": safe_float(obs.get("nstres")),
                    "trnu": safe_float(obs.get("trnu")),
                    "topwt": safe_float(obs.get("topwt")),
                    "grnwt": safe_float(obs.get("grnwt")),
                    "xlai": safe_float(obs.get("xlai")),
                    "ep": safe_float(obs.get("ep")),
                    "totir": safe_float(obs.get("totir")),
                    "reward": safe_float(reward),
                    "raw_action_anfer": safe_float(np.asarray(normalized_action).flatten()[0]),
                    "raw_action_amir": safe_float(np.asarray(normalized_action).flatten()[1])
                    if len(np.asarray(normalized_action).flatten()) > 1
                    else np.nan,
                    "real_action_anfer": safe_float(action.get("anfer", real_action_dict.get("anfer"))),
                    "real_action_amir": safe_float(action.get("amir", real_action_dict.get("amir"))),
                }
            )

        df = pd.DataFrame(rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{site}_{agent_name}_all_water_stress_trace.csv", index=False)
        summary = summarize_trace(site, agent_name, df, env_args)
        summary["terminated_normally"] = bool(done)
        summary["max_steps"] = max_steps
        return summary
    finally:
        env.close()
        gc.collect()


def count_gt(series: pd.Series, threshold: float) -> int:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return int((values > threshold).sum()) if not values.empty else 0


def summarize_trace(site: str, agent_name: str, df: pd.DataFrame, env_args: dict) -> dict:
    pdate = parse_pdate(env_args["fileX_template_path"])
    prcp = crop_season_prcp(env_args["auxiliary_file_paths"][1], pdate, len(df) or 1)
    etcp = ETCP_FROM_DSSAT_485.get(site, np.nan)
    swfac = pd.to_numeric(df.get("swfac"), errors="coerce")
    turfac = pd.to_numeric(df.get("turfac"), errors="coerce")
    nstres = pd.to_numeric(df.get("nstres"), errors="coerce")
    trnu = pd.to_numeric(df.get("trnu"), errors="coerce")
    grnwt = pd.to_numeric(df.get("grnwt"), errors="coerce")
    anfer = pd.to_numeric(df.get("real_action_anfer"), errors="coerce")
    amir = pd.to_numeric(df.get("real_action_amir"), errors="coerce")
    final_totir = pd.to_numeric(df.get("totir"), errors="coerce").dropna()
    return {
        "site": site,
        "site_name": SITE_NAMES.get(site, site),
        "agent": agent_name,
        "mode": "all",
        "template": env_args["fileX_template_path"],
        "weather": env_args["auxiliary_file_paths"][1],
        "soil": env_args["auxiliary_file_paths"][2],
        "PDATE": pdate,
        "n_days": len(df),
        "PRCP": prcp,
        "ETCP": etcp,
        "PRCP_minus_ETCP": prcp - etcp if not np.isnan(etcp) else np.nan,
        "swfac_stress_days_gt_0.05": count_gt(swfac, 0.05),
        "swfac_stress_days_gt_0.10": count_gt(swfac, 0.10),
        "max_swfac": swfac.max(skipna=True),
        "mean_swfac": swfac.mean(skipna=True),
        "turfac_stress_days_gt_0.05": count_gt(turfac, 0.05),
        "turfac_stress_days_gt_0.10": count_gt(turfac, 0.10),
        "max_turfac": turfac.max(skipna=True),
        "mean_turfac": turfac.mean(skipna=True),
        "nstres_days_gt_0.05": count_gt(nstres, 0.05),
        "max_nstres": nstres.max(skipna=True),
        "mean_nstres": nstres.mean(skipna=True),
        "mean_trnu": trnu.mean(skipna=True),
        "final_trnu": trnu.dropna().iloc[-1] if not trnu.dropna().empty else np.nan,
        "max_grnwt": grnwt.max(skipna=True),
        "total_anfer": anfer.sum(skipna=True),
        "total_amir": amir.sum(skipna=True),
        "final_totir": final_totir.iloc[-1] if not final_totir.empty else np.nan,
        "template_irrigation_mm": parse_template_irrigation(env_args["fileX_template_path"]),
        "total_reward": pd.to_numeric(df.get("reward"), errors="coerce").sum(skipna=True),
        "note": "maize post-processing uses 1-original_value, so larger swfac/nstres means stronger stress; turfac is saved only when safely exposed by the environment",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default="HL,SY,LC,FQ,YC")
    parser.add_argument("--agents", default="null,expert")
    parser.add_argument("--output-dir", default="output_hl/all_water_stress_diagnostics")
    parser.add_argument("--data-dir", default="./my_data")
    parser.add_argument("--run-dssat-location", default="/opt/dssat_pdi/run_dssat")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--safe-anfer-cap", type=float, default=None)
    parser.add_argument("--safe-amir-cap", type=float, default=None)
    args = parser.parse_args()

    sites = [value.strip().upper() for value in args.sites.split(",") if value.strip()]
    agents = [value.strip().lower() for value in args.agents.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    summaries = []
    metadata = {}

    for site in sites:
        if site not in SITE_CONFIGS:
            raise ValueError(f"Unknown site: {site}")
        env_args = build_env_args(
            site=site,
            mode="all",
            seed=123,
            data_dir=args.data_dir,
            prefer_suffix=None,
            run_dssat_location=args.run_dssat_location,
        )
        metadata[site] = env_args
        print(f"Running {site}: {env_args['fileX_template_path']}", flush=True)
        for agent_name in agents:
            print(f"  agent={agent_name}", flush=True)
            summaries.append(
                evaluate_site_agent(
                    site,
                    agent_name,
                    env_args,
                    output_dir,
                    args.max_steps,
                    args.model_path,
                    args.safe_anfer_cap,
                    args.safe_amir_cap,
                )
            )

    summary_df = pd.DataFrame(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "all_water_stress_summary.csv", index=False)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    md = [
        "# All-mode Water/Nitrogen Stress Diagnostic Summary",
        "",
        "`swfac` and `nstres` are available in maize `all` mode. `turfac` is kept as a column and will be saved if a safe environment exposes it. Because the installed maize post-processing applies `1 - value` to maize stress variables, larger values mean stronger stress in this table.",
        "",
        "```text",
        summary_df.round(3).to_string(index=False),
        "```",
    ]
    (output_dir / "all_water_stress_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
