import argparse
import gc
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from plot_hl import plot_actions, plot_rewards


AGENTS = ['null', 'ppo', 'expert']
OBS_COLUMNS = ['dap', 'swfac', 'nstres', 'topwt', 'grnwt', 'xlai', 'trnu', 'cumsumfert']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-output-dir', default='./output_hl/reward_sweep')
    parser.add_argument('--base-figure-dir', default='./figures_hl/reward_sweep')
    parser.add_argument('--source-dir', default=None)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def parse_run_name(name):
    match = re.fullmatch(r'coef([0-9.]+)_pen([0-9.]+)', name)
    if not match:
        raise ValueError(f'Cannot parse reward setting from directory name: {name}')
    return float(match.group(1)), float(match.group(2))


def count_traces(run_dir, agent):
    return len(list(run_dir.glob(f'{agent}_decision_trace_ep*.csv')))


def trace_path(run_dir, agent, episode):
    return run_dir / f'{agent}_decision_trace_ep{episode}.csv'


def find_source_dir(base_dir, episodes):
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if count_traces(run_dir, 'null') >= episodes and count_traces(run_dir, 'expert') >= episodes:
            return run_dir
    raise FileNotFoundError('No complete null/expert trace source was found.')


def read_trace(run_dir, source_dir, agent, episode):
    path = trace_path(run_dir, agent, episode)
    if not path.exists() and agent in {'null', 'expert'}:
        path = trace_path(source_dir, agent, episode)
    if not path.exists():
        raise FileNotFoundError(f'Missing trace for {agent} episode {episode}: {path}')
    return pd.read_csv(path)


def recompute_reward(df, coef, penality):
    df = df.copy()
    trnu = pd.to_numeric(df.get('trnu'), errors='coerce').fillna(0.0)
    action = pd.to_numeric(df.get('real_action_anfer'), errors='coerce').fillna(0.0)
    df['reward'] = trnu * coef - penality * action
    return df


def episode_to_history(df):
    action_values = pd.to_numeric(df.get('real_action_anfer'), errors='coerce').fillna(0.0)
    cumsum = action_values.cumsum()
    observations = []
    actions = []
    rewards = []
    for idx, row in df.iterrows():
        obs = {}
        for col in OBS_COLUMNS:
            value = cumsum.iloc[idx] if col == 'cumsumfert' else row.get(col, np.nan)
            try:
                obs[col] = float(value)
            except (TypeError, ValueError):
                obs[col] = np.nan
        observations.append(obs)
        actions.append({'anfer': float(action_values.iloc[idx])})
        rewards.append(float(row.get('reward', 0.0)))
    return observations, actions, rewards


def build_histories(run_dir, source_dir, coef, penality, episodes):
    histories = {agent: {'observation': [], 'action': [], 'reward': []} for agent in AGENTS}
    for agent in AGENTS:
        for episode in range(1, episodes + 1):
            df = read_trace(run_dir, source_dir, agent, episode)
            df = recompute_reward(df, coef, penality)
            obs, actions, rewards = episode_to_history(df)
            histories[agent]['observation'].append(obs)
            histories[agent]['action'].append(actions)
            histories[agent]['reward'].append(rewards)
    return histories


def save_histories(run_dir, histories):
    with (run_dir / 'evaluation_histories.pkl').open('wb') as handle:
        pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    base_dir = Path(args.base_output_dir)
    figure_base = Path(args.base_figure_dir)
    source_dir = Path(args.source_dir) if args.source_dir else find_source_dir(base_dir, args.episodes)
    print(f'Using baseline source traces from: {source_dir}', flush=True)

    for run_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        coef, penality = parse_run_name(run_dir.name)
        plots_marker = run_dir / 'PLOTS_COMPLETE'
        pkl_path = run_dir / 'evaluation_histories.pkl'
        figure_dir = figure_base / run_dir.name
        apps_path = figure_dir / 'fertilizationApplications.pdf'
        rewards_path = figure_dir / 'fertilizationRewards.pdf'

        if (
            not args.force
            and plots_marker.exists()
            and pkl_path.exists()
            and apps_path.exists()
            and rewards_path.exists()
        ):
            print(f'Skip complete: {run_dir.name}', flush=True)
            continue

        print(f'Building {run_dir.name}: coef={coef}, penality={penality}', flush=True)
        histories = build_histories(run_dir, source_dir, coef, penality, args.episodes)
        save_histories(run_dir, histories)
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_actions(
            history_dict=histories,
            mode='fertilization',
            saving_path=str(figure_dir / 'fertilizationApplications.pdf'),
            keys=['ppo', 'expert'],
        )
        plot_rewards(
            history_dict=histories,
            mode='fertilization',
            quantile_range_legend=False,
            saving_path=str(figure_dir / 'fertilizationRewards.pdf'),
        )
        plots_marker.write_text('complete\n', encoding='utf-8')
        print(f'Done {run_dir.name}', flush=True)
        del histories
        gc.collect()


if __name__ == '__main__':
    main()
