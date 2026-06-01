from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import gym
import sys

from gym_dssat_pdi.envs.utils import utils as dssat_utils

from dssat_site_config import build_env_args, describe_env_args
from sb3_safe_action_wrapper import SafeActionCaps, SafeActionGymDssatWrapper
from train_hl_all_multisite import assert_all_reward_is_scalar_ready, parse_args, set_reward_env


def make_safe_env(env_args: dict, anfer_cap: float, amir_cap: float):
    source_env = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return Monitor(SafeActionGymDssatWrapper(source_env, SafeActionCaps(anfer=anfer_cap, amir=amir_cap)))


if __name__ == "__main__":
    import argparse

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--safe-anfer-cap", type=float, default=60.0)
    extra_parser.add_argument("--safe-amir-cap", type=float, default=20.0)
    safe_args, _ = extra_parser.parse_known_args()
    filtered_argv = [sys.argv[0]]
    skip_next = False
    for index, value in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if value in {"--safe-anfer-cap", "--safe-amir-cap"}:
            skip_next = True
            continue
        if value.startswith("--safe-anfer-cap=") or value.startswith("--safe-amir-cap="):
            continue
        filtered_argv.append(value)
    original_argv = sys.argv
    sys.argv = filtered_argv
    args = parse_args()
    sys.argv = original_argv

    set_reward_env(args)
    assert_all_reward_is_scalar_ready()
    env = None
    eval_env = None
    try:
        for directory in ["./output_hl", args.output_dir, args.log_dir]:
            dssat_utils.make_folder(directory)

        env_args = build_env_args(
            site=args.site,
            mode="all",
            seed=args.seed,
            data_dir=args.data_dir,
            prefer_suffix=args.prefer_suffix,
            log_saving_path=f"{args.log_dir}/dssat_pdi_{args.site}_all_safe_coef{args.coef}_pen{args.penality}.log",
            run_dssat_location=args.run_dssat_location,
        )

        print(f'###########################\n## MODE: {env_args["mode"]} SAFE ##\n###########################')
        print(f"## site={args.site}, prefer_suffix={args.prefer_suffix}")
        print("## DSSAT input files:\n" + describe_env_args(env_args))
        print(f"## reward coef={args.coef}, penality={args.penality}")
        print(f"## all reward weights: fertilization={args.all_fert_weight}, irrigation={args.all_irrig_weight}")
        print(f"## all costs: anfer={args.all_anfer_cost}, amir={args.all_amir_cost}")
        print(
            f"## all stress cost: amir_no_stress={args.all_amir_no_stress_cost}, "
            f"water_stress_threshold={args.all_water_stress_threshold}"
        )
        print(
            f"## all excess: anfer_limit={args.all_anfer_excess_limit}, anfer_cost={args.all_anfer_excess_cost}, "
            f"amir_limit={args.all_amir_excess_limit}, amir_cost={args.all_amir_excess_cost}"
        )
        print(f"## safe action caps: anfer<={safe_args.safe_anfer_cap}, amir<={safe_args.safe_amir_cap}")
        print(f"## output_dir={args.output_dir}")

        env = make_safe_env(env_args, safe_args.safe_anfer_cap, safe_args.safe_amir_cap)
        raw_obs, _ = env.unwrapped.reset(), {}
        print("底层原始 observation 是 dict，keys：", sorted(raw_obs.keys()))

        ppo_args = {
            "seed": args.train_seed,
            "gamma": 0.99,
            "n_steps": args.ppo_n_steps,
            "batch_size": args.ppo_batch_size,
            "n_epochs": args.ppo_n_epochs,
        }

        if args.resume_model:
            print(f"Resuming PPO agent from: {args.resume_model}")
            ppo_agent = PPO.load(
                args.resume_model,
                env=env,
                verbose=args.sb3_verbose,
                tensorboard_log=args.tensorboard_log,
            )
        else:
            ppo_agent = PPO("MlpPolicy", env, verbose=args.sb3_verbose, tensorboard_log=args.tensorboard_log, **ppo_args)

        eval_env_args = {**env_args, "seed": args.eval_seed}
        eval_env = make_safe_env(eval_env_args, safe_args.safe_anfer_cap, safe_args.safe_amir_cap)
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=args.eval_freq,
            best_model_save_path=args.output_dir,
            deterministic=True,
            n_eval_episodes=args.n_eval_episodes,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=f"{args.output_dir}/checkpoints",
            name_prefix="ppo_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        print("Training PPO agent with safe action caps...")
        ppo_agent.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name=f"PPO_dssat_all_safe_coef{args.coef}_pen{args.penality}",
            reset_num_timesteps=not bool(args.resume_model),
        )
        ppo_agent.save(f"{args.output_dir}/final_model")
        print("Training done")
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()
