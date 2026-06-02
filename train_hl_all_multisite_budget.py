from __future__ import annotations

import argparse
import sys

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from gym_dssat_pdi.envs.utils import utils as dssat_utils

from dssat_site_config import build_env_args, describe_env_args
from sb3_safe_action_wrapper import (
    BudgetedSafeActionGymDssatWrapper,
    SafeActionCaps,
    SeasonalBudgetCaps,
)
from train_hl_all_multisite import assert_all_reward_is_scalar_ready, parse_args, set_reward_env


EXTRA_ARGS = {
    "--safe-anfer-cap",
    "--safe-amir-cap",
    "--season-anfer-budget",
    "--season-amir-budget",
}


def parse_budget_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--safe-anfer-cap", type=float, default=20.0)
    parser.add_argument("--safe-amir-cap", type=float, default=5.0)
    parser.add_argument("--season-anfer-budget", type=float, default=220.0)
    parser.add_argument("--season-amir-budget", type=float, default=100.0)
    budget_args, _ = parser.parse_known_args()

    filtered_argv = [sys.argv[0]]
    skip_next = False
    for value in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if value in EXTRA_ARGS:
            skip_next = True
            continue
        if any(value.startswith(f"{name}=") for name in EXTRA_ARGS):
            continue
        filtered_argv.append(value)
    original_argv = sys.argv
    sys.argv = filtered_argv
    base_args = parse_args()
    sys.argv = original_argv
    return base_args, budget_args


def make_budget_env(env_args: dict, budget_args):
    source_env = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return Monitor(
        BudgetedSafeActionGymDssatWrapper(
            source_env,
            SafeActionCaps(anfer=budget_args.safe_anfer_cap, amir=budget_args.safe_amir_cap),
            SeasonalBudgetCaps(anfer=budget_args.season_anfer_budget, amir=budget_args.season_amir_budget),
        )
    )


if __name__ == "__main__":
    args, budget_args = parse_budget_args()
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
            log_saving_path=f"{args.log_dir}/dssat_pdi_{args.site}_all_budget_coef{args.coef}_pen{args.penality}.log",
            run_dssat_location=args.run_dssat_location,
        )

        print(f'#############################\n## MODE: {env_args["mode"]} BUDGET ##\n#############################')
        print(f"## site={args.site}, prefer_suffix={args.prefer_suffix}")
        print("## DSSAT input files:\n" + describe_env_args(env_args))
        print(f"## reward coef={args.coef}, penality={args.penality}")
        print(f"## all costs: amir={args.all_amir_cost}, amir_no_stress={args.all_amir_no_stress_cost}")
        print(
            f"## all excess: anfer_limit={args.all_anfer_excess_limit}, anfer_cost={args.all_anfer_excess_cost}, "
            f"amir_limit={args.all_amir_excess_limit}, amir_cost={args.all_amir_excess_cost}"
        )
        print(f"## daily caps: anfer<={budget_args.safe_anfer_cap}, amir<={budget_args.safe_amir_cap}")
        print(
            f"## seasonal budgets: anfer<={budget_args.season_anfer_budget}, "
            f"amir<={budget_args.season_amir_budget}"
        )
        print(f"## output_dir={args.output_dir}")

        env = make_budget_env(env_args, budget_args)
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
        eval_env = make_budget_env(eval_env_args, budget_args)
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

        print("Training PPO agent with daily caps and seasonal budgets...")
        ppo_agent.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name=f"PPO_dssat_all_budget_coef{args.coef}_pen{args.penality}",
            reset_num_timesteps=not bool(args.resume_model),
        )
        ppo_agent.save(f"{args.output_dir}/final_model")
        print("Training done")
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()
