from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_wrapper import GymDssatWrapper
from gym_dssat_pdi.envs.utils import utils as dssat_utils
import argparse
import gym
import os
import inspect


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coef', type=float, default=1.0)
    parser.add_argument('--penality', type=float, default=0.5)
    parser.add_argument('--all-fert-weight', type=float, default=1.0)
    parser.add_argument('--all-irrig-weight', type=float, default=1.0)
    parser.add_argument('--all-anfer-cost', type=float, default=0.0)
    parser.add_argument('--all-amir-cost', type=float, default=0.0)
    parser.add_argument('--all-anfer-excess-limit', type=float, default=1e12)
    parser.add_argument('--all-amir-excess-limit', type=float, default=1e12)
    parser.add_argument('--all-anfer-excess-cost', type=float, default=0.0)
    parser.add_argument('--all-amir-excess-cost', type=float, default=0.0)
    parser.add_argument('--total-timesteps', type=int, default=10_000)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--output-dir', default='./output_hl/all')
    parser.add_argument('--log-dir', default='./logs_hl')
    parser.add_argument('--tensorboard-log', default='./tensorboard/')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--train-seed', type=int, default=123)
    parser.add_argument('--eval-seed', type=int, default=345)
    parser.add_argument('--sb3-verbose', type=int, default=0)
    parser.add_argument('--resume-model', default=None)
    parser.add_argument('--checkpoint-freq', type=int, default=50000)
    parser.add_argument('--ppo-n-steps', type=int, default=2048)
    parser.add_argument('--ppo-batch-size', type=int, default=64)
    parser.add_argument('--ppo-n-epochs', type=int, default=10)
    return parser.parse_args()


def set_reward_env(args):
    coef = args.coef
    penality = args.penality
    os.environ['GYM_DSSAT_REWARD_COEF'] = str(coef)
    os.environ['GYM_DSSAT_REWARD_PENALITY'] = str(penality)
    os.environ['GYM_DSSAT_ALL_FERT_WEIGHT'] = str(args.all_fert_weight)
    os.environ['GYM_DSSAT_ALL_IRRIG_WEIGHT'] = str(args.all_irrig_weight)
    os.environ['GYM_DSSAT_ALL_ANFER_COST'] = str(args.all_anfer_cost)
    os.environ['GYM_DSSAT_ALL_AMIR_COST'] = str(args.all_amir_cost)
    os.environ['GYM_DSSAT_ALL_ANFER_EXCESS_LIMIT'] = str(args.all_anfer_excess_limit)
    os.environ['GYM_DSSAT_ALL_AMIR_EXCESS_LIMIT'] = str(args.all_amir_excess_limit)
    os.environ['GYM_DSSAT_ALL_ANFER_EXCESS_COST'] = str(args.all_anfer_excess_cost)
    os.environ['GYM_DSSAT_ALL_AMIR_EXCESS_COST'] = str(args.all_amir_excess_cost)


def assert_all_reward_is_scalar_ready():
    from gym_dssat_pdi.envs.configs import rewards

    source = inspect.getsource(rewards.all_reward)
    if 'return [ferti_reward_value, irrig_reward_value]' in source:
        raise RuntimeError(
            "Installed all_reward still returns a list. Run this inside the DSSAT "
            "environment first: python tools/patch_dssat_rewards.py --patch-all-reward"
        )


if __name__ == '__main__':
    args = parse_args()
    set_reward_env(args)
    assert_all_reward_is_scalar_ready()
    env = None
    eval_env = None
    try:
        for dir in ['./output_hl', args.output_dir, args.log_dir]:
            dssat_utils.make_folder(dir)

        # Create environment
        env_args = {
            'log_saving_path': f'{args.log_dir}/dssat_pdi_HL_all_coef{args.coef}_pen{args.penality}.log',
            'mode': 'all',
            'seed': args.seed,
            'random_weather': False,           # 用固定天气，保持 False
            # 'evaluation': True,
            'fileX_template_path': './my_data/UFGA8201-HL.jinja2',  # 海伦专用模板
            # 'fileX_template_path': None,

            'experiment_number': 1,

            'auxiliary_file_paths': [
                './my_data/MZCER048.CUL',         # 海伦站的玉米品种参数（必须）
                './my_data/CNHL0701.WTH',      # 海伦站2008年的天气文件
                './my_data/HL.SOL',       # 海伦站的土壤文件（必须）
            ],
            'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        }


        print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')
        print(f'## reward coef={args.coef}, penality={args.penality}')
        print(f'## all reward weights: fertilization={args.all_fert_weight}, irrigation={args.all_irrig_weight}')
        print(f'## all costs: anfer={args.all_anfer_cost}, amir={args.all_amir_cost}')
        print(f'## all excess: anfer_limit={args.all_anfer_excess_limit}, anfer_cost={args.all_anfer_excess_cost}, '
              f'amir_limit={args.all_amir_excess_limit}, amir_cost={args.all_amir_excess_cost}')
        print(f'## output_dir={args.output_dir}')

        env = Monitor(GymDssatWrapper(gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args).unwrapped))

        unwrapped_env = env.unwrapped  # 剥掉 Monitor 和 GymDssatWrapper
        reset_result = unwrapped_env.reset()
        if isinstance(reset_result, tuple):
            raw_obs, info = reset_result
        else:
            raw_obs = reset_result
        print("底层原始 observation 是 dict，keys：", sorted(raw_obs.keys()))

        # Training arguments for PPO agent
        ppo_args = {
            'seed': args.train_seed,  # seed training for reproducibility
            'gamma': 0.99,
            'n_steps': args.ppo_n_steps,
            'batch_size': args.ppo_batch_size,
            'n_epochs': args.ppo_n_epochs,
        }

        # Create or resume the agent
        if args.resume_model:
            print(f'Resuming PPO agent from: {args.resume_model}')
            ppo_agent = PPO.load(
                args.resume_model,
                env=env,
                verbose=args.sb3_verbose,
                tensorboard_log=args.tensorboard_log,
            )
        else:
            ppo_agent = PPO('MlpPolicy', env, verbose=args.sb3_verbose, tensorboard_log=args.tensorboard_log, **ppo_args)

        # path to save best model found
        path = args.output_dir

        # eval callback
        eval_env_args = {**env_args, 'seed': args.eval_seed}
        eval_env = Monitor(GymDssatWrapper(gym.make('gym_dssat_pdi:GymDssatPdi-v0', **eval_env_args).unwrapped))
        eval_callback = EvalCallback(eval_env,
                                     eval_freq=args.eval_freq,
                                     best_model_save_path=f'{path}',
                                     deterministic=True,
                                     n_eval_episodes=args.n_eval_episodes)
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=f'{path}/checkpoints',
            name_prefix='ppo_checkpoint',
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # Train
        print('Training PPO agent...')
        ppo_agent.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name=f"PPO_dssat_all_coef{args.coef}_pen{args.penality}",
            reset_num_timesteps=not bool(args.resume_model),
        )
        ppo_agent.save(f'{path}/final_model')
        print('Training done')
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()
