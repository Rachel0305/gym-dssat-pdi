import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(',') if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='output_hl/all_reward_sweep')
    parser.add_argument('--base-model', default='output_hl/all_200k/coef1_pen0.5/final_model.zip')
    parser.add_argument('--coef', type=float, default=1.0)
    parser.add_argument('--penalities', default='1.0,2.0,5.0,10.0')
    parser.add_argument('--all-fert-weight', type=float, default=1.0)
    parser.add_argument('--all-irrig-weight', type=float, default=1.0)
    parser.add_argument('--anfer-costs', default='0.0,0.5,1.0')
    parser.add_argument('--amir-costs', default='0.0')
    parser.add_argument('--anfer-excess-limits', default='300')
    parser.add_argument('--anfer-excess-costs', default='0.0,1.0,5.0')
    parser.add_argument('--amir-excess-limits', default='80')
    parser.add_argument('--amir-excess-costs', default='0.0,1.0')
    parser.add_argument('--total-timesteps', type=int, default=30000)
    parser.add_argument('--eval-episodes', type=int, default=30)
    parser.add_argument('--eval-agents', default='null,ppo,expert')
    parser.add_argument('--eval-freq', type=int, default=10000)
    parser.add_argument('--n-eval-episodes', type=int, default=2)
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--ppo-n-steps', type=int, default=512)
    parser.add_argument('--ppo-batch-size', type=int, default=128)
    parser.add_argument('--ppo-n-epochs', type=int, default=5)
    parser.add_argument('--max-combos', type=int, default=0)
    parser.add_argument('--skip-existing', action='store_true')
    return parser.parse_args()


def run_name(setting):
    return (
        f"pen{setting['penality']:g}_"
        f"ncost{setting['anfer_cost']:g}_"
        f"icost{setting['amir_cost']:g}_"
        f"nlim{setting['anfer_excess_limit']:g}_"
        f"nex{setting['anfer_excess_cost']:g}_"
        f"ilim{setting['amir_excess_limit']:g}_"
        f"iex{setting['amir_excess_cost']:g}"
    )


def run_command(command, log_path, env):
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write('\n$ ' + ' '.join(command) + '\n')
        log.flush()
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if process.returncode != 0:
        raise RuntimeError(f'Command failed with exit code {process.returncode}: {command}')


def trace_metrics(output_dir, agents):
    rows = []
    for agent in agents:
        traces = sorted(
            output_dir.glob(f'{agent}_decision_trace_ep*.csv'),
            key=lambda path: int(path.stem.split('ep')[-1]),
        )
        for trace in traces:
            df = pd.read_csv(trace)
            trnu = df['trnu'].dropna() if 'trnu' in df else pd.Series(dtype=float)
            real_anfer = df.get('real_action_anfer', pd.Series(dtype=float))
            real_amir = df.get('real_action_amir', pd.Series(dtype=float))
            rows.append({
                'agent': agent,
                'episode': int(trace.stem.split('ep')[-1]),
                'final_trnu': trnu.iloc[-1] if len(trnu) else np.nan,
                'mean_trnu': trnu.mean() if len(trnu) else np.nan,
                'max_trnu': trnu.max() if len(trnu) else np.nan,
                'total_anfer': real_anfer.sum(),
                'total_amir': real_amir.sum(),
                'napp_anfer': (real_anfer > 0).sum(),
                'napp_amir': (real_amir > 0).sum(),
                'max_grnwt': df['grnwt'].max() if 'grnwt' in df else np.nan,
                'final_grnwt': df['grnwt'].iloc[-1] if 'grnwt' in df else np.nan,
                'max_topwt': df['topwt'].max() if 'topwt' in df else np.nan,
                'total_reward': df['reward'].sum() if 'reward' in df else np.nan,
            })
    return pd.DataFrame(rows)


def summarize_episode_metrics(df):
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby('agent', as_index=False).agg({
        'final_trnu': 'mean',
        'mean_trnu': 'mean',
        'total_anfer': 'mean',
        'total_amir': 'mean',
        'napp_anfer': 'mean',
        'napp_amir': 'mean',
        'max_grnwt': 'mean',
        'final_grnwt': 'mean',
        'max_topwt': 'mean',
        'total_reward': 'mean',
        'episode': 'count',
    })
    return grouped.rename(columns={'episode': 'episodes'})


def score_ppo(summary):
    ppo = summary[summary['agent'] == 'ppo']
    if ppo.empty:
        return np.nan
    row = ppo.iloc[0]
    return (
        0.35 * row['mean_trnu']
        + 0.35 * (row['max_grnwt'] / 7000.0)
        - 0.20 * (row['total_anfer'] / 300.0)
        - 0.10 * (row['total_amir'] / 80.0)
    )


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    agents = [agent.strip() for agent in args.eval_agents.split(',') if agent.strip()]

    settings = []
    for penality, anfer_cost, amir_cost, anfer_limit, anfer_excess_cost, amir_limit, amir_excess_cost in itertools.product(
        parse_float_list(args.penalities),
        parse_float_list(args.anfer_costs),
        parse_float_list(args.amir_costs),
        parse_float_list(args.anfer_excess_limits),
        parse_float_list(args.anfer_excess_costs),
        parse_float_list(args.amir_excess_limits),
        parse_float_list(args.amir_excess_costs),
    ):
        settings.append({
            'penality': penality,
            'anfer_cost': anfer_cost,
            'amir_cost': amir_cost,
            'anfer_excess_limit': anfer_limit,
            'anfer_excess_cost': anfer_excess_cost,
            'amir_excess_limit': amir_limit,
            'amir_excess_cost': amir_excess_cost,
        })
    if args.max_combos:
        settings = settings[:args.max_combos]

    all_rows = []
    failures = []
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    for index, setting in enumerate(settings, start=1):
        name = run_name(setting)
        combo_dir = output_root / name
        combo_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== [{index}/{len(settings)}] {name} ===', flush=True)

        if args.skip_existing and (combo_dir / 'SWEEP_COMPLETE').exists():
            print('Skip existing complete setting.')
        else:
            train_cmd = [
                sys.executable, 'train_hl_all.py',
                '--coef', str(args.coef),
                '--penality', str(setting['penality']),
                '--all-fert-weight', str(args.all_fert_weight),
                '--all-irrig-weight', str(args.all_irrig_weight),
                '--all-anfer-cost', str(setting['anfer_cost']),
                '--all-amir-cost', str(setting['amir_cost']),
                '--all-anfer-excess-limit', str(setting['anfer_excess_limit']),
                '--all-amir-excess-limit', str(setting['amir_excess_limit']),
                '--all-anfer-excess-cost', str(setting['anfer_excess_cost']),
                '--all-amir-excess-cost', str(setting['amir_excess_cost']),
                '--total-timesteps', str(args.total_timesteps),
                '--eval-freq', str(args.eval_freq),
                '--n-eval-episodes', str(args.n_eval_episodes),
                '--checkpoint-freq', str(args.checkpoint_freq),
                '--ppo-n-steps', str(args.ppo_n_steps),
                '--ppo-batch-size', str(args.ppo_batch_size),
                '--ppo-n-epochs', str(args.ppo_n_epochs),
                '--output-dir', str(combo_dir),
                '--log-dir', 'logs_hl',
                '--sb3-verbose', '1',
            ]
            if args.base_model:
                train_cmd.extend(['--resume-model', args.base_model])

            eval_cmd = [
                sys.executable, 'evaluate_hl_all.py',
                '--coef', str(args.coef),
                '--penality', str(setting['penality']),
                '--all-fert-weight', str(args.all_fert_weight),
                '--all-irrig-weight', str(args.all_irrig_weight),
                '--all-anfer-cost', str(setting['anfer_cost']),
                '--all-amir-cost', str(setting['amir_cost']),
                '--all-anfer-excess-limit', str(setting['anfer_excess_limit']),
                '--all-amir-excess-limit', str(setting['amir_excess_limit']),
                '--all-anfer-excess-cost', str(setting['anfer_excess_cost']),
                '--all-amir-excess-cost', str(setting['amir_excess_cost']),
                '--n-episodes', str(args.eval_episodes),
                '--output-dir', str(combo_dir),
                '--model-path', str(combo_dir / 'best_model.zip'),
                '--agents', args.eval_agents,
                '--quiet',
                '--no-history-pickle',
            ]

            try:
                run_command(train_cmd, combo_dir / 'train.log', env)
                run_command(eval_cmd, combo_dir / 'evaluate.log', env)
                (combo_dir / 'SWEEP_COMPLETE').write_text('ok\n', encoding='utf-8')
            except Exception as exc:
                failures.append({'run': name, 'error': repr(exc), **setting})
                pd.DataFrame(failures).to_csv(output_root / 'failures.csv', index=False)
                print(f'FAILED {name}: {exc}', flush=True)
                continue

        episode_df = trace_metrics(combo_dir, agents)
        episode_df.to_csv(combo_dir / 'episode_metrics.csv', index=False)
        summary = summarize_episode_metrics(episode_df)
        if not summary.empty:
            summary['run'] = name
            for key, value in setting.items():
                summary[key] = value
            summary['score'] = score_ppo(summary)
            summary.to_csv(combo_dir / 'summary_metrics.csv', index=False)
            all_rows.extend(summary.to_dict(orient='records'))
            pd.DataFrame(all_rows).to_csv(output_root / 'all_summary_metrics.csv', index=False)
            ppo = summary[summary['agent'] == 'ppo']
            if not ppo.empty:
                cols = ['run', 'mean_trnu', 'total_anfer', 'total_amir', 'max_grnwt', 'total_reward', 'score']
                print(ppo[cols].round(3).to_string(index=False), flush=True)

    if all_rows:
        all_summary = pd.DataFrame(all_rows)
        ppo_summary = all_summary[all_summary['agent'] == 'ppo'].sort_values('score', ascending=False)
        ppo_summary.to_csv(output_root / 'ppo_ranking.csv', index=False)
        print('\nTop PPO settings:')
        print(ppo_summary.head(10)[[
            'run', 'penality', 'anfer_cost', 'anfer_excess_limit', 'anfer_excess_cost',
            'amir_excess_limit', 'amir_excess_cost', 'mean_trnu', 'total_anfer',
            'total_amir', 'max_grnwt', 'total_reward', 'score',
        ]].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
