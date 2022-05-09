from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sb3_wrapper import GymDssatWrapper
import gym

if __name__ == '__main__':
    # Create environment
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'mode': 'fertilization',
        'seed': 123,
        'random_weather': True,
    }

    env = Monitor(GymDssatWrapper(gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)))

    # Training arguments for PPO agent
    ppo_args = {
        'seed': 123,  # seed training for reproducibility
    }

    # Create the agent
    ppo_agent = PPO('MlpPolicy', env, **ppo_args)

    # path to save best model found
    path = 'ppo'

    # eval callback
    eval_freq = 1000
    eval_env_args = {**env_args, 'seed': 345}
    eval_env = Monitor(GymDssatWrapper(gym.make('GymDssatPdi-v0', **eval_env_args)))
    eval_callback = EvalCallback(eval_env,
                                 eval_freq=eval_freq,
                                 best_model_save_path=f'./{path}',
                                 deterministic=True)

    # Train
    print('Training PPO agent...')
    ppo_agent.learn(total_timesteps=1_000_000, callback=eval_callback)
    ppo_agent.save(f'./{path}/final_model')
    print('Training done')