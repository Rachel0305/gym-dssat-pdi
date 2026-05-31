from stable_baselines3 import PPO
from sb3_wrapper import GymDssatWrapper, Formator
from stable_baselines3.common.monitor import Monitor
from gym_dssat_pdi.envs.utils import utils
from copy import deepcopy
import argparse
import gc
import gym
import pickle
import os
import numpy as np
import pdb
import inspect


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def get_latest_observation_dict(env, info=None, obs_dict=None):
    sources = []
    unwrapped_env = env.unwrapped
    history = getattr(unwrapped_env, 'history', {})

    if isinstance(history, dict):
        observations = history.get('observation', [])
        if observations and isinstance(observations[-1], dict):
            sources.append(observations[-1])

    for attr in ['observation', '_observation', 'state', '_state', 'next_state', '_next_state']:
        value = getattr(unwrapped_env, attr, None)
        if isinstance(value, dict):
            sources.append(value)

    if isinstance(info, dict):
        sources.append(info)
    if isinstance(obs_dict, dict):
        sources.append(obs_dict)

    merged = {}
    for source in sources:
        merged.update(source)
    return merged


def sync_trnu_to_history(env, trnu):
    if np.isnan(trnu):
        return
    history = getattr(env.unwrapped, 'history', {})
    if not isinstance(history, dict):
        return
    observations = history.get('observation', [])
    if observations and isinstance(observations[-1], dict):
        observations[-1]['trnu'] = trnu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coef', type=float, default=1.0)
    parser.add_argument('--penality', type=float, default=0.5)
    parser.add_argument('--all-fert-weight', type=float, default=1.0)
    parser.add_argument('--all-irrig-weight', type=float, default=1.0)
    parser.add_argument('--n-episodes', type=int, default=1)
    parser.add_argument('--output-dir', default='./output_hl/all')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--agents', default='null,ppo,expert')
    parser.add_argument('--verbose-step', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no-history-pickle', action='store_true')
    return parser.parse_args()


def set_reward_env(coef, penality, all_fert_weight, all_irrig_weight):
    os.environ['GYM_DSSAT_REWARD_COEF'] = str(coef)
    os.environ['GYM_DSSAT_REWARD_PENALITY'] = str(penality)
    os.environ['GYM_DSSAT_ALL_FERT_WEIGHT'] = str(all_fert_weight)
    os.environ['GYM_DSSAT_ALL_IRRIG_WEIGHT'] = str(all_irrig_weight)


def assert_all_reward_is_scalar_ready():
    from gym_dssat_pdi.envs.configs import rewards

    source = inspect.getsource(rewards.all_reward)
    if 'return [ferti_reward_value, irrig_reward_value]' in source:
        raise RuntimeError(
            "Installed all_reward still returns a list. Run this inside the DSSAT "
            "environment first: python tools/patch_dssat_rewards.py --patch-all-reward"
        )


class NullAllAgent:
    def __init__(self, env):
        self.action_formator = Formator(self._get_dssat_env(env))

    def _get_dssat_env(self, env):
        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env
        return inner

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        real_actions = [0.0 for _ in self.action_formator.action_names]
        normalized_action = self.action_formator.normalize_actions(real_actions)
        return np.array(normalized_action, dtype=np.float32), obs


class ExpertAllAgent:
    def __init__(self, env):
        dssat_env = self._get_dssat_env(env)
        self.action_formator = Formator(dssat_env)
        obs_vars = self._get_obs_vars(env)
        assert 'dap' in obs_vars, f"'dap' not found in observation_variables: {obs_vars}"
        self.dap_index = obs_vars.index('dap')
        self.fertilization_policy = {1: 165}
        self.irrigation_policy = {49: 10, 70: 10, 95: 10}

    def _get_dssat_env(self, env):
        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env
        return inner

    def _get_obs_vars(self, env):
        inner = env
        while inner is not None:
            if hasattr(inner, 'observation_variables'):
                return inner.observation_variables
            inner = getattr(inner, 'env', None)
        raise AttributeError("Cannot find observation_variables in env stack")

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        obs = np.concatenate(obs, axis=None)
        dap = int(obs[self.dap_index])
        action_by_name = {name: 0.0 for name in self.action_formator.action_names}
        if 'anfer' in action_by_name:
            action_by_name['anfer'] = self.fertilization_policy.get(dap, 0.0)
        if 'amir' in action_by_name:
            action_by_name['amir'] = self.irrigation_policy.get(dap, 0.0)
        real_actions = [action_by_name[name] for name in self.action_formator.action_names]
        normalized_action = self.action_formator.normalize_actions(real_actions)
        return np.array(normalized_action, dtype=np.float32), obs


def evaluate(
    agent,
    eval_args,
    n_episodes=100,
    agent_name='agent',
    output_dir=None,
    verbose_step=False,
    quiet=False,
    collect_history=True,
):
    # Create eval env
    print("STEP 1: make env")
    source_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0',**eval_args)
    print("STEP 2: wrap env")
    env = GymDssatWrapper(source_env.unwrapped)

# 检查维度是否匹配
    test_obs, _ = env.reset()
# 替换那一行，改成：
    obs_keys = env.unwrapped.observation_variables
    print("obs_keys:", obs_keys)
    print(f"Actual observation shape: {test_obs.shape}")
    
    if hasattr(agent, 'observation_space'):
        print(f"Agent expected shape: {agent.observation_space.shape}")
        if agent.observation_space.shape != test_obs.shape:
            print(f"Dimension mismatch! Updating agent's observation space...")
            agent.observation_space = gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=test_obs.shape,
                dtype="float32"
            )
            # 对于 PPO，还需要更新 policy
            if hasattr(agent, 'policy'):
                agent.policy.observation_space = agent.observation_space

    print("STEP 3: before unwrapped")
    unwrapped_env = env.unwrapped  # 剥掉 Monitor 和 GymDssatWrapper
    print("STEP 4: before reset")
    reset_result = unwrapped_env.reset()
    print("STEP 5: after reset")
    if isinstance(reset_result, tuple):
        raw_obs, info = reset_result
    else:
        raw_obs = reset_result
    print("底层原始 observation 是 dict，keys：", sorted(raw_obs.keys()))

    all_histories = []
    try:
        for ep in range(n_episodes):
            done = False
            observation, _ = env.reset()
            import pandas as pd
            # 新增：记录每日数据
            daily_records = []
            
            while not done:
                action = agent.predict(observation)[0]
                raw_action = np.asarray(action, dtype=float).flatten()
                real_action_dict = {}
                try:
                    real_action_values = env.formator.denormalize_actions(raw_action)
                    real_action_dict = env.formator.format_actions(real_action_values)
                except Exception:
                    real_action_dict = {}
                if verbose_step:
                    print("RAW ACTION FROM AGENT:", action)
                observation, reward, terminated, truncated, info = env.step(action)
                if verbose_step:
                    print("LAST DSSAT ACTION:", env.unwrapped.history['action'][-1])
                    print(info.keys())
                # print(f"step: terminated={terminated}, truncated={truncated}, done={done}")
                done = terminated or truncated
                
                # 新增：每天记录关键变量
                # observation是numpy数组，需要对应obs_keys
                # obs_keys = ['cleach','cumsumfert','dap','dtt','ep',
                #             'grnwt','istage','nstres','pcngrn',
                #             'swfac','topwt','vstage','xlai']
                obs_keys = env.unwrapped.observation_variables
                obs_dict = dict(zip(obs_keys, observation))
                latest_obs = get_latest_observation_dict(env, info=info, obs_dict=obs_dict)
                trnu = safe_float(latest_obs.get('trnu', np.nan))
                sync_trnu_to_history(env, trnu)
                
                record = {
                    'dap': obs_dict.get('dap', np.nan),
                    'swfac': obs_dict.get('swfac', np.nan),
                    'nstres': obs_dict.get('nstres', np.nan),
                    'topwt': obs_dict.get('topwt', np.nan),
                    'grnwt': obs_dict.get('grnwt', np.nan),
                    'xlai': obs_dict.get('xlai', np.nan),
                    'trnu': trnu,
                    'reward': reward,
                }

                # =========================
                # 根据 mode 解析 action
                # =========================

                if eval_args['mode'] == 'fertilization':

                    # record['action_anfer'] = float(action[0])
                    raw_action = float(action[0])

                    real_action = ((raw_action + 1) / 2) * 200

                    record['raw_action_anfer'] = raw_action
                    record['real_action_anfer'] = real_action

                elif eval_args['mode'] == 'irrigation':

                    record['action_amir'] = float(action[0])

                elif eval_args['mode'] == 'all':

                    record['raw_action_anfer'] = float(raw_action[0]) if len(raw_action) > 0 else np.nan
                    record['raw_action_amir'] = float(raw_action[1]) if len(raw_action) > 1 else np.nan
                    record['real_action_anfer'] = safe_float(real_action_dict.get('anfer', np.nan))
                    record['real_action_amir'] = safe_float(real_action_dict.get('amir', np.nan))

                daily_records.append(record)
            
            # episode结束后打印摘要
            if not quiet:
                print(f"\nEpisode {ep+1} 每日记录（前10天）:")

            for d in ([] if quiet else daily_records[:10]):

                base_info = (
                    f"DAP={float(d['dap']):.0f} | "
                    f"swfac={float(d['swfac']):.3f} | "
                    f"nstres={float(d['nstres']):.3f} | "
                    f"topwt={float(d['topwt']):.1f} | "
                    f"grnwt={float(d['grnwt']):.1f} | "
                    f"trnu={float(d['trnu']):.3f} | "
                )

                # =========================
                # fertilization
                # =========================

                if eval_args['mode'] == 'fertilization':

                    action_info = (
                        # f"anfer={float(d.get('action_anfer', 0)):.1f} | "
                        f"raw={float(d.get('raw_action_anfer', 0)):.2f} |"
                        f"realN={float(d.get('real_action_anfer', 0)):.1f}|"
                    )

                # =========================
                # irrigation
                # =========================

                elif eval_args['mode'] == 'irrigation':

                    action_info = (
                        f"amir={float(d.get('action_amir', 0)):.1f} | "
                    )

                # =========================
                # all
                # =========================

                elif eval_args['mode'] == 'all':

                    action_info = (
                        f"anfer={float(d.get('real_action_anfer', 0)):.1f} | "
                        f"amir={float(d.get('real_action_amir', 0)):.1f} | "
                    )

                else:

                    action_info = ""

                reward_info = (
                    f"reward={float(d['reward']):.3f}"
                )

                print(base_info + action_info + reward_info)

            df_daily = pd.DataFrame(daily_records)
            trnu_series = df_daily['trnu'].dropna()
            total_fert = np.nan
            if eval_args['mode'] == 'fertilization' and 'real_action_anfer' in df_daily:
                total_fert = df_daily['real_action_anfer'].sum()
            elif eval_args['mode'] == 'all' and 'real_action_anfer' in df_daily:
                total_fert = df_daily['real_action_anfer'].sum()
            total_irrig = np.nan
            if eval_args['mode'] == 'irrigation' and 'action_amir' in df_daily:
                total_irrig = df_daily['action_amir'].sum()
            elif eval_args['mode'] == 'all' and 'real_action_amir' in df_daily:
                total_irrig = df_daily['real_action_amir'].sum()

            if not quiet:
                if trnu_series.empty:
                    print(f"Episode {ep+1} TRNU summary: trnu not found in environment outputs.")
                else:
                    print(
                        f"Episode {ep+1} TRNU summary: "
                        f"final_trnu={trnu_series.iloc[-1]:.3f} | "
                        f"mean_trnu={trnu_series.mean():.3f} | "
                        f"max_trnu={trnu_series.max():.3f} | "
                        f"total_anfer={total_fert:.1f} | "
                        f"total_amir={total_irrig:.1f}"
                    )
            
            episode_history = env.env.history
            if collect_history:
                all_histories.append(deepcopy(episode_history))
            # 保存 episode 每日记录

            if output_dir is None:
                output_dir = f'./output_hl/{eval_args["mode"]}'
            os.makedirs(output_dir, exist_ok=True)
            save_path = f'{output_dir}/{agent_name}_decision_trace_ep{ep+1}.csv'

            df_daily.to_csv(save_path, index=False)

            if quiet:
                if (ep + 1) % 100 == 0 or ep == 0:
                    print(f"{agent_name}: episode {ep+1}/{n_episodes} saved")
            else:
                print(f"Saved daily trace to: {save_path}")
                print(f"Episode {ep+1} finished, history length: {len(episode_history)}")  # debug 用
            del df_daily, daily_records
            if not collect_history:
                del episode_history
            gc.collect()
            
    finally:
        env.close()
    return all_histories


if __name__ == '__main__':
    args = parse_args()
    set_reward_env(args.coef, args.penality, args.all_fert_weight, args.all_irrig_weight)
    assert_all_reward_is_scalar_ready()
    os.makedirs(args.output_dir, exist_ok=True)

    env_args = {
        'mode': 'all',
        'seed': args.seed,
        'random_weather': False,
        'evaluation': False,  # isolated seeds for weather generation
        'fileX_template_path': './my_data/UFGA8201-HL.jinja2',
        'experiment_number': 1,
        'auxiliary_file_paths': [
            './my_data/MZCER048.CUL',
            './my_data/CNHL0701.WTH',
            './my_data/HL.SOL',
        ],
        
    }

    print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')
    print(f'## reward coef={args.coef}, penality={args.penality}')
    print(f'## all reward weights: fertilization={args.all_fert_weight}, irrigation={args.all_irrig_weight}')
    print(f'## output_dir={args.output_dir}')

    agent_names = [name.strip() for name in args.agents.split(',') if name.strip()]
    model_path = args.model_path or f'{args.output_dir}/best_model.zip'
    if 'ppo' in agent_names:
        assert os.path.exists(model_path), f'Model not found: {model_path}'

    print("before reset")
    # 第210行附近，改成：
    source_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    print("after reset")
    env = Monitor(GymDssatWrapper(source_env.unwrapped))
    test_obs, _ = env.reset()

    obs_keys = env.unwrapped.observation_variables
    obs_dict = dict(zip(obs_keys, test_obs))

    print("=== DAY 0 / RESET STATE ===")
    print("DAP:", obs_dict.get('dap'))
    print("SNO3:", obs_dict.get('sno3'))
    print("SNH4:", obs_dict.get('snh4'))
    print("XLAT:", obs_dict.get('xlai'))
    print("TOPWT:", obs_dict.get('topwt'))
    print("NSTRES:", obs_dict.get('nstres'))
    n_episodes = args.n_episodes
    try:
        ppo_best = PPO.load(model_path) if 'ppo' in agent_names else None
        all_agents = {
            'null': NullAllAgent(env),
            'expert': ExpertAllAgent(env)
        }
        if ppo_best is not None:
            all_agents['ppo'] = ppo_best
        unknown_agents = sorted(set(agent_names) - set(all_agents))
        if unknown_agents:
            raise ValueError(f'Unknown agents: {unknown_agents}. Valid agents: {sorted(all_agents)}')
        agents = {name: all_agents[name] for name in agent_names}
        env.close()

        all_histories = {}
        for agent_name in [*agents]:
            agent = agents[agent_name]
            print(f'Evaluating {agent_name} agent...')
            histories = evaluate(
                agent=agent,
                eval_args=env_args,
                n_episodes=n_episodes,
                agent_name=agent_name,
                output_dir=args.output_dir,
                verbose_step=args.verbose_step,
                quiet=args.quiet,
                collect_history=not args.no_history_pickle,
            )
            if not args.no_history_pickle:
                histories = utils.transpose_dicts(histories)
                all_histories[agent_name] = histories
            print('Done')

        if not args.no_history_pickle:
            saving_path = f'{args.output_dir}/evaluation_histories.pkl'
            with open(saving_path, 'wb') as handle:
                pickle.dump(all_histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Skip evaluation_histories.pkl because --no-history-pickle is enabled.')
    finally:
        print("env type:", type(env))
        print("env.env type:", type(env.env))
        print("dir(env.env):", [attr for attr in dir(env.env) if not attr.startswith('__')])
        # 如果你懷疑 history 在 wrapper 層級
        print("dir(env):", [attr for attr in dir(env) if 'hist' in attr.lower() or 'record' in attr.lower()])
        env.close()
