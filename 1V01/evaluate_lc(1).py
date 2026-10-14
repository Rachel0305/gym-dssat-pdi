from stable_baselines3 import PPO
from baseline_policies import NullAgent, ExpertAgent
from sb3_wrapper import GymDssatWrapper
from stable_baselines3.common.monitor import Monitor
from gym_dssat_pdi.envs.utils import utils
import gym
import pickle
import os
import numpy as np
import pandas as pd


def get_base_env(env):
    """可靠地获取最底层的 GymDssatPdi 环境"""
    current = env
    visited = set()
    
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        
        if hasattr(current, 'history') and hasattr(current, 'observation_variables'):
            return current
            
        if hasattr(current, 'unwrapped') and current.unwrapped is not current:
            current = current.unwrapped
            continue
            
        if hasattr(current, 'env') and current.env is not current:
            current = current.env
            continue
            
        break
    
    # 兜底策略
    for candidate in [current, getattr(env, 'env', None), getattr(env, 'unwrapped', None)]:
        if candidate and hasattr(candidate, 'history'):
            return candidate
    
    raise AttributeError("无法找到包含 'history' 的底层环境")


def make_env(env_args):
    """统一的环境创建函数"""
    source_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    wrapped = GymDssatWrapper(source_env)
    return Monitor(wrapped)


def evaluate(agent, eval_args, agent_name, n_episodes=1):
    source_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **eval_args)
    env = GymDssatWrapper(source_env)          # 有 .history / .observation_variables
    print(env.observation_variables)

    test_obs, _ = env.reset()
    obs_keys = env.observation_variables       # ← 直接用，不再需要 get_base_env

    output_dir = f'./output_lc/{eval_args["mode"]}'
    os.makedirs(output_dir, exist_ok=True)

    all_histories = []
    try:
        for ep in range(n_episodes):
            done = False
            observation, _ = env.reset()
            daily_records = []

        while not done:
            normalized_action = agent.predict(observation)[0]
            observation, reward, terminated, truncated, info = env.step(normalized_action)
            done = terminated or truncated

            history = env.history
            raw_obs = history['observation'][-1] if history.get('observation') else {}

            # 跳过播前阶段
            if raw_obs.get('dap', 0) == 0:
                continue

            record = {
                'dap':    raw_obs.get('dap',    np.nan),
                'topwt':  raw_obs.get('topwt',  np.nan),
                'grnwt':  raw_obs.get('grnwt',  np.nan),
                'xlai':   raw_obs.get('xlai',   np.nan),
                'totir':  raw_obs.get('totir',  np.nan),
                'swfac':  raw_obs.get('swfac',  np.nan),
                'nstres': raw_obs.get('nstres', np.nan),
                'reward': reward,
            }

            if eval_args['mode'] == 'irrigation':
                real_amir = np.nan
                if history.get('action'):
                    last_action = history['action'][-1]
                    if isinstance(last_action, dict):
                        val = last_action.get('amir')
                        if val is None:
                            val = last_action.get('AMIR')
                        real_amir = float(val) if val is not None else np.nan
                    elif isinstance(last_action, (int, float, np.number)):
                        real_amir = float(last_action)

                record['real_action_amir'] = real_amir
                record['normalized_action_amir'] = (
                    float(normalized_action[0])
                    if isinstance(normalized_action, np.ndarray)
                    else float(normalized_action)
                )

            daily_records.append(record)
            # episode 结束
            all_histories.append(env.history)
            df_daily = pd.DataFrame(daily_records)
            save_path = f'{output_dir}/{agent_name}_decision_trace_ep{ep+1}.csv'
            df_daily.to_csv(save_path, index=False)
            print(f"Saved: {save_path}  (days={len(daily_records)})")

    finally:
        env.close()
    return all_histories

if __name__ == '__main__':
    env_args = {
        'mode': 'irrigation',
        'seed': 123,
        'random_weather': False,
        'evaluation': False,
        'fileX_template_path': './my_data/UFGA8201-LC.jinja2',
        'experiment_number': 1,
        'auxiliary_file_paths': [
            './my_data/MZCER048.CUL',
            './my_data/CNLC0801.WTH',
            './my_data/LC.SOL',
        ],
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
    }

    print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')

    assert os.path.exists(f'./output_lc/{env_args["mode"]}/best_model.zip')

    # ==================== 主环境测试 ====================
    print("before reset")
    env = make_env(env_args)          # 使用统一函数
    test_obs, _ = env.reset()

    base_env = get_base_env(env)
    obs_keys = base_env.observation_variables
    obs_dict = dict(zip(obs_keys, test_obs))

    print("=== DAY 0 / RESET STATE ===")
    print("DAP:", obs_dict.get('dap'))
    print("SNO3:", obs_dict.get('sno3'))
    print("SNH4:", obs_dict.get('snh4'))
    print("XLAT:", obs_dict.get('xlai'))
    print("TOPWT:", obs_dict.get('topwt'))
    print("NSTRES:", obs_dict.get('nstres'))

    n_episodes = 1   # 先用小数字测试，确认没问题再改大

    try:
        ppo_best = PPO.load(f'./output_lc/{env_args["mode"]}/best_model.zip')
        
        agents = {
            'null': NullAgent(env),
            # 'ppo': ppo_best,
            'expert': ExpertAgent(env)
        }

        all_histories = {}
        for agent_name in list(agents.keys()):
            agent = agents[agent_name]
            print(f'\nEvaluating {agent_name} agent...')
            histories = evaluate(agent=agent, eval_args=env_args, agent_name=agent_name, n_episodes=n_episodes)
            histories = utils.transpose_dicts(histories)
            all_histories[agent_name] = histories
            print(f'{agent_name} evaluation Done')

        saving_path = f'./output_lc/{env_args["mode"]}/evaluation_histories.pkl'
        with open(saving_path, 'wb') as handle:
            pickle.dump(all_histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Evaluation completed and saved to {saving_path}")

    finally:
        try:
            env.close()
        except:
            pass