from stable_baselines3 import PPO
from baseline_policies import NullAgent, ExpertAgent
from sb3_wrapper import GymDssatWrapper
from stable_baselines3.common.monitor import Monitor
from gym_dssat_pdi.envs.utils import utils
from copy import deepcopy
import gym
import pickle
import os
import numpy as np
import pdb


def evaluate(agent, eval_args, n_episodes=100):
    # Create eval env
    source_env = gym.make('GymDssatPdi-v0', **eval_args)
    env = GymDssatWrapper(source_env)
    all_histories = []
    try:
        for ep in range(n_episodes):
            done = False
            observation, _ = env.reset()  # Gymnasium 格式
            while not done:
                action = agent.predict(observation)[0]
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            # Episode 結束後，從原始 env 拿 history
            episode_history = env.env.history   # ← 這裡修正！
            
            # 如果需要轉置成方便分析的格式（官方推薦）
            # transposed = utils.transpose_dicts([episode_history])  # 如果是 list of dicts
            # 但大多數情況下 history 已經是 dict of lists 或類似
            
            all_histories.append(episode_history)
            
            print(f"Episode {ep+1} finished, history length: {len(episode_history)}")  # debug 用
            
    finally:
        env.close()
    return all_histories


if __name__ == '__main__':

    env_args = {
        # 'mode': 'fertilization',
        'mode': 'irrigation',
        'seed': 123,
        'random_weather': True,
        'evaluation': True,  # isolated seeds for weather generation
    }

    print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')

    assert os.path.exists(f'./output/{env_args["mode"]}/best_model.zip')


    source_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    env = Monitor(GymDssatWrapper(source_env))
    n_episodes = 1000
    try:
        ppo_best = PPO.load(f'./output/{env_args["mode"]}/best_model')
        agents = {
            'null': NullAgent(env),
            'ppo': ppo_best,
            'expert': ExpertAgent(env)
        }

        all_histories = {}
        for agent_name in [*agents]:
            agent = agents[agent_name]
            print(f'Evaluating {agent_name} agent...')
            histories = evaluate(agent=agent, eval_args=env_args, n_episodes=n_episodes)
            histories = utils.transpose_dicts(histories)
            all_histories[agent_name] = histories
            print('Done')

        saving_path = f'./output/{env_args["mode"]}/evaluation_histories.pkl'
        with open(saving_path, 'wb') as handle:
            pickle.dump(all_histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        print("env type:", type(env))
        print("env.env type:", type(env.env))
        print("dir(env.env):", [attr for attr in dir(env.env) if not attr.startswith('__')])
        # 如果你懷疑 history 在 wrapper 層級
        print("dir(env):", [attr for attr in dir(env) if 'hist' in attr.lower() or 'record' in attr.lower()])
        env.close()
