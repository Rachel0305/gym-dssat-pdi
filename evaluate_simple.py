from stable_baselines3 import PPO
from baseline_policies import NullAgent, ExpertAgent
from sb3_wrapper import GymDssatWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import pickle
import os
import pdb

def evaluate(agent, n_episodes=100):
    # Create eval env
    eval_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'mode': 'fertilization',
        'seed': 456,
        'random_weather': True,
    }
    env = Monitor(GymDssatWrapper(gym.make('GymDssatPdi-v0', **eval_args)))

    env.eval()
    returns, _ = evaluate_policy(
        agent, env, n_eval_episodes=n_episodes, return_episode_rewards=True)

    env.close()

    return returns


if __name__ == '__main__':

    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        # 'mode': 'fertilization',
        'mode': 'irrigation',
        'seed': 123,
        'random_weather': True,
    }

    assert os.path.exists(f'./output/{env_args["mode"]}/best_model.zip')

    source_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    env = Monitor(GymDssatWrapper(source_env))
    n_episodes = 100
    try:
        ppo_best = PPO.load(f'./output/{env_args["mode"]}/best_model')
        agents = {
            'null': NullAgent(env),
            'ppo': ppo_best,
            'expert': ExpertAgent(env)}

        results = {}
        for agent_name in [*agents]:
            agent = agents[agent_name]
            print(f'Evaluating {agent_name} agent...')
            histories = evaluate(agent, n_episodes=n_episodes)
            results[agent_name] = histories
            print('Done')

        saving_path = f'./output/{env_args["mode"]}/evaluation_rewards.pkl'
        with open(saving_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        env.close()