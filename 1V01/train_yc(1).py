from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sb3_wrapper import GymDssatWrapper
from gym_dssat_pdi.envs.utils import utils as dssat_utils
import gymnasium as gym
import gym as old_gym      # 旧版
import gymnasium as gym    # 新版
# from gymnasium.utils.compatibility import EnvCompatibilityV0
import gym
import gym_dssat_pdi
from gym_dssat_pdi.envs.configs import rewards
import inspect



if __name__ == '__main__':
    try:
        for dir in ['./output_yc', './logs_yc']:
            dssat_utils.make_folder(dir)

        # Create environment
        env_args = {
            'log_saving_path': './logs_yc/dssat_pdi_YC_fer.log',
            'mode': 'irrigation',
            'seed': 1234,
            'random_weather': False,           # 用固定天气，保持 False
            # 'evaluation': True,
            'fileX_template_path': './my_data/UFGA8201-YC.jinja2',  # sy专用模板
            # 'fileX_template_path': None,

            'experiment_number': 1,

            'auxiliary_file_paths': [
                './my_data/MZCER048.CUL',         # 禹城站的玉米品种参数（必须）
                './my_data/CNYC0801.WTH',      # 禹城站2008年的天气文件
                './my_data/YC.SOL',       # 禹城站的土壤文件（必须）
            ],
        }


        print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')

        raw_env = old_gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        env = Monitor(GymDssatWrapper(raw_env))
        # env = Monitor(GymDssatWrapper(gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)))

        unwrapped_env = env.unwrapped  # 剥掉 Monitor 和 GymDssatWrapper
        reset_result = unwrapped_env.reset()
        if isinstance(reset_result, tuple):
            raw_obs, info = reset_result
        else:
            raw_obs = reset_result
        print("底层原始 observation 是 dict，keys：", sorted(raw_obs.keys()))

        # Training arguments for PPO agent
        ppo_args = {
            'seed': 123,  # seed training for reproducibility
            'gamma': 0.99,
        }

        # Create the agent
        ppo_agent = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/", **ppo_args)

        # path to save best model found
        path = f'./output_yc/{env_args["mode"]}'

        # eval callback
        eval_freq = 1000
        eval_env_args = {**env_args, 'seed': 345}
        # eval_env = Monitor(GymDssatWrapper(gym.make('GymDssatPdi-v0', **eval_env_args)))
        # 正确：和训练环境一样，用旧版 gym.make()
        eval_env = Monitor(GymDssatWrapper(old_gym.make('gym_dssat_pdi:GymDssatPdi-v0', **eval_env_args)))
        eval_callback = EvalCallback(eval_env,
                                     eval_freq=eval_freq,
                                     best_model_save_path=f'{path}',
                                     deterministic=True,
                                     n_eval_episodes=10)

        # Train
        # total_timesteps = 5_000
        total_timesteps = 1000_000
        print('Training PPO agent...')
        ppo_agent.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name="PPO_dssat" )
        # ppo_agent.save(f'{path}/final_model')
        print('Training done')
    finally:
        env.close()