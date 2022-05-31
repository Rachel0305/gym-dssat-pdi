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
        for _ in range(n_episodes):
            done = False
            observation = env.reset()
            while not done:
                action = agent.predict(observation)[0]
                observation, reward, done, _ = env.step(action=action)
            all_histories.append(env.env._history)
    finally:
        env.close()
    return all_histories


if __name__ == '__main__':

    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        # 'mode': 'fertilization',
        'mode': 'irrigation',
        'seed': 123,
        'random_weather': True,
        'evaluation': True,
    }

    print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')

    assert os.path.exists(f'./output/{env_args["mode"]}/best_model.zip')


    eval_args = deepcopy(env_args)
    eval_args['seed'] = 456

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
            histories = evaluate(agent=agent, eval_args=eval_args, n_episodes=n_episodes)
            histories = utils.transpose_dicts(histories)
            all_histories[agent_name] = histories
            print('Done')

        saving_path = f'./output/{env_args["mode"]}/evaluation_histories.pkl'
        with open(saving_path, 'wb') as handle:
            pickle.dump(all_histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        env.close()
