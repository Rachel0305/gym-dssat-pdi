import argparse
import gc
import os
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def find_source_dir(base_dir, episodes):
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if count_traces(run_dir, 'null') >= episodes and count_traces(run_dir, 'expert') >= episodes:
            return run_dir
    raise FileNotFoundError('No source directory with complete null and expert traces was found.')


def trace_path(run_dir, agent, episode):
    return run_dir / f'{agent}_decision_trace_ep{episode}.csv'


def recompute_reward(df, coef, penality):
    trnu = pd.to_numeric(df.get('trnu'), errors='coerce').fillna(0.0)
    action = pd.to_numeric(df.get('real_action_anfer'), errors='coerce').fillna(0.0)
    df['reward'] = trnu * coef - penality * action
    return df


def ensure_agent_traces(run_dir, source_dir, agent, coef, penality, episodes, force=False):
    existing = count_traces(run_dir, agent)
    if existing >= episodes and not force:
        return

    print(f'  preparing {agent} traces: {existing}/{episodes}')
    source_agent_dir = run_dir if existing >= episodes else source_dir
    for episode in range(1, episodes + 1):
        source_path = trace_path(source_agent_dir, agent, episode)
        if not source_path.exists():
            source_path = trace_path(source_dir, agent, episode)
        if not source_path.exists():
            raise FileNotFoundError(f'Missing source trace: {source_path}')

        df = pd.read_csv(source_path)
        df = recompute_reward(df, coef, penality)
        df.to_csv(trace_path(run_dir, agent, episode), index=False)


def normalize_existing_rewards(run_dir, coef, penality, episodes, force=False):
    for agent in AGENTS:
        if count_traces(run_dir, agent) < episodes:
            continue
        marker = run_dir / f'{agent}_REWARD_NORMALIZED'
        if marker.exists() and not force:
            continue
        print(f'  normalizing {agent} rewards')
        for episode in range(1, episodes + 1):
            path = trace_path(run_dir, agent, episode)
            df = pd.read_csv(path)
            df = recompute_reward(df, coef, penality)
            df.to_csv(path, index=False)
        marker.write_text('done\n', encoding='utf-8')


def episode_to_history(df):
    cumsum = pd.to_numeric(df.get('real_action_anfer'), errors='coerce').fillna(0.0).cumsum()
    observations = []
    actions = []
    rewards = []

    for idx, row in df.iterrows():
        obs = {}
        for col in OBS_COLUMNS:
            if col == 'cumsumfert':
                value = cumsum.iloc[idx]
            else:
                value = row[col] if col in df.columns else np.nan
            try:
                obs[col] = float(value)
            except (TypeError, ValueError):
                obs[col] = np.nan
        observations.append(obs)

        action_value = row['real_action_anfer'] if 'real_action_anfer' in df.columns else 0.0
        actions.append({'anfer': float(action_value)})
        rewards.append(float(row['reward'] if 'reward' in df.columns else 0.0))

    return observations, actions, rewards


def build_evaluation_histories(run_dir, episodes):
    histories = {
        agent: {'observation': [], 'action': [], 'reward': []}
        for agent in AGENTS
    }
    for agent in AGENTS:
        for episode in range(1, episodes + 1):
            df = pd.read_csv(trace_path(run_dir, agent, episode))
            observations, actions, rewards = episode_to_history(df)
            histories[agent]['observation'].append(observations)
            histories[agent]['action'].append(actions)
            histories[agent]['reward'].append(rewards)
    with (run_dir / 'evaluation_histories.pkl').open('wb') as handle:
        pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return histories


def plot_applications(histories, figure_dir):
    rows = []
    for agent in ['ppo', 'expert']:
        for episode_actions in histories[agent]['action']:
            for step, action in enumerate(episode_actions, start=1):
                value = float(action.get('anfer', 0.0))
                if value > 0:
                    rows.append({'step': step, 'action': value, 'policy': agent})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    if df.empty:
        ax.text(0.5, 0.5, 'No fertilizer applications', ha='center', va='center')
    else:
        for policy, group in df.groupby('policy'):
            ax.scatter(group['step'], group['action'], s=4, alpha=0.15, label=policy)
        ax.legend(loc='upper left')
    ax.set_xlabel('day of simulation')
    ax.set_ylabel('fertilizer quantity (kg/ha)')
    ax.set_title('Nitrogen fertilizer applications (1000 episodes)')
    ax.set_xlim(left=0)
    ax.grid(True, linestyle='--', alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / 'fertilizationApplications.pdf')
    plt.close(fig)


def plot_rewards(histories, figure_dir):
    colors = {'null': '#7f7f7f', 'ppo': '#1f77b4', 'expert': '#ff7f0e'}
    fig, ax = plt.subplots(figsize=(8, 6))
    for agent in AGENTS:
        rewards = []
        for episode_rewards in histories[agent]['reward']:
            rewards.append(np.asarray(episode_rewards, dtype=float).cumsum())
        min_len = min(len(values) for values in rewards)
        rewards = np.asarray([values[:min_len] for values in rewards])
        x = np.arange(1, min_len + 1)
        mean = np.nanmean(rewards, axis=0)
        low = np.nanquantile(rewards, 0.05, axis=0)
        high = np.nanquantile(rewards, 0.95, axis=0)
        ax.plot(x, mean, label=agent, color=colors.get(agent))
        ax.fill_between(x, low, high, color=colors.get(agent), alpha=0.12)
    ax.set_xlabel('day of simulation')
    ax.set_ylabel('cumulated return')
    ax.set_title('Policy returns (1000 episodes)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / 'fertilizationRewards.pdf')
    plt.close(fig)


def main():
    args = parse_args()
    base_dir = Path(args.base_output_dir)
    figure_base = Path(args.base_figure_dir)
    source_dir = Path(args.source_dir) if args.source_dir else find_source_dir(base_dir, args.episodes)
    print(f'Using baseline source traces from: {source_dir}')

    for run_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        coef, penality = parse_run_name(run_dir.name)
        print(f'\n=== {run_dir.name}: coef={coef}, penality={penality} ===')

        for agent in ['null', 'expert']:
            ensure_agent_traces(run_dir, source_dir, agent, coef, penality, args.episodes, force=args.force)
        normalize_existing_rewards(run_dir, coef, penality, args.episodes, force=args.force)

        histories = build_evaluation_histories(run_dir, args.episodes)
        figure_dir = figure_base / run_dir.name
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_applications(histories, figure_dir)
        plot_rewards(histories, figure_dir)
        (run_dir / 'PLOTS_COMPLETE').write_text('complete\n', encoding='utf-8')
        print(f'  saved {run_dir / "evaluation_histories.pkl"}')
        print(f'  saved {figure_dir / "fertilizationApplications.pdf"}')
        print(f'  saved {figure_dir / "fertilizationRewards.pdf"}')
        del histories
        gc.collect()


if __name__ == '__main__':
    main()
