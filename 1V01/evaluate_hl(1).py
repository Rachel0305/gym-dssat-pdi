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


def evaluate(agent, eval_args, n_episodes=1):
    print("STEP 1: make env")
    source_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **eval_args)
    print("STEP 2: wrap env")
    env = GymDssatWrapper(source_env)

    # 检查 observation
    test_obs, _ = env.reset()
    base_env = get_base_env(env)
    obs_keys = base_env.observation_variables
    
    print("obs_keys:", obs_keys)
    print(f"Actual observation shape: {test_obs.shape}")

    all_histories = []
    try:
        for ep in range(n_episodes):
            done = False
            observation, _ = env.reset()
            daily_records = []
            
            while not done:
                action = agent.predict(observation)[0]
                # print("RAW ACTION FROM AGENT:", action)
                
                observation, reward, terminated, truncated, info = env.step(action)
                
                base_env = get_base_env(env)
                # print("LAST DSSAT ACTION:", base_env.history['action'][-1])
                # print("info keys:", list(info.keys()))
                
                done = terminated or truncated
                
                # 记录数据
                obs_dict = dict(zip(obs_keys, observation))
                
                record = {
                    'dap': obs_dict.get('dap', np.nan),
                    'swfac': obs_dict.get('swfac', np.nan),
                    'nstres': obs_dict.get('nstres', np.nan),
                    'topwt': obs_dict.get('topwt', np.nan),
                    'grnwt': obs_dict.get('grnwt', np.nan),
                    'xlai': obs_dict.get('xlai', np.nan),
                    'reward': reward,
                }

                if eval_args['mode'] == 'irrigation':
                    record['action_amir'] = float(action[0])

                daily_records.append(record)
            
            # Episode 结束
            base_env = get_base_env(env)
            episode_history = base_env.history
            all_histories.append(episode_history)
            
            # 保存每日记录
            df_daily = pd.DataFrame(daily_records)
            save_path = f'./output_hl/{eval_args["mode"]}/{agent_name}_decision_trace_ep{ep+1}.csv'
            df_daily.to_csv(save_path, index=False)
            print(f"Saved daily trace to: {save_path}")
            print(f"Episode {ep+1} finished, history length: {len(episode_history)}\n")
            
    finally:
        env.close()
    return all_histories


if __name__ == '__main__':
    env_args = {
        'mode': 'irrigation',
        'seed': 123,
        'random_weather': False,
        'evaluation': False,
        'fileX_template_path': './my_data/UFGA8201-HL.jinja2',
        'experiment_number': 1,
        'auxiliary_file_paths': [
            './my_data/MZCER048.CUL',
            './my_data/CNHL0701.WTH',
            './my_data/HL.SOL',
        ],
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
    }

    print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')

    assert os.path.exists(f'./output_hl/{env_args["mode"]}/best_model.zip')

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

    n_episodes = 1000   # 先用小数字测试，确认没问题再改大

    try:
        ppo_best = PPO.load(f'./output_hl/{env_args["mode"]}/best_model.zip')
        
        agents = {
            'null': NullAgent(env),
            'ppo': ppo_best,
            'expert': ExpertAgent(env)
        }

        all_histories = {}
        for agent_name in list(agents.keys()):
            agent = agents[agent_name]
            print(f'\nEvaluating {agent_name} agent...')
            histories = evaluate(agent=agent, eval_args=env_args, n_episodes=n_episodes)
            histories = utils.transpose_dicts(histories)
            all_histories[agent_name] = histories
            print(f'{agent_name} evaluation Done')

        saving_path = f'./output_hl/{env_args["mode"]}/evaluation_histories.pkl'
        with open(saving_path, 'wb') as handle:
            pickle.dump(all_histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Evaluation completed and saved to {saving_path}")

    finally:
        try:
            env.close()
        except:
            pass