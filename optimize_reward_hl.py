import argparse
import gc
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefs', default='0.5,1.0,1.5,2.0')
    parser.add_argument('--penalities', default='0.1,0.25,0.5,0.75,1.0')
    parser.add_argument('--total-timesteps', type=int, default=5000)
    parser.add_argument('--train-chunk-timesteps', type=int, default=50000)
    parser.add_argument('--eval-episodes', type=int, default=1)
    parser.add_argument('--base-output-dir', default='./output_hl/reward_sweep')
    parser.add_argument('--base-figure-dir', default='./figures_hl/reward_sweep')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--eval-agents', default='ppo')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--stop-on-error', action='store_true')
    parser.add_argument('--cleanup-processes', action='store_true')
    return parser.parse_args()


def run_command(command, env, log_path):
    print(f'\n$ {" ".join(command)}')
    print(f'  log: {log_path}', flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open('a', encoding='utf-8', errors='replace') as log_file:
        log_file.write('\n\n' + '=' * 80 + '\n')
        log_file.write('$ ' + ' '.join(command) + '\n')
        log_file.flush()
        subprocess.run(
            command,
            check=True,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    print(f'  finished in {(time.time() - start) / 60:.1f} min', flush=True)


def cleanup_processes(enabled):
    if not enabled:
        return
    if os.name == 'nt':
        commands = [
            ['taskkill', '/F', '/IM', 'dscsm048.exe'],
            ['taskkill', '/F', '/IM', 'run_dssat.exe'],
        ]
    else:
        commands = [
            ['pkill', '-f', 'dscsm048'],
            ['pkill', '-f', 'run_dssat'],
        ]
    for command in commands:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cleanup_tmp(enabled=True):
    if not enabled:
        return
    command = (
        "find /tmp -mindepth 1 -maxdepth 1 "
        "\\( -name 'pip-*' -o -name 'tmp*' -o -name 'train.log' \\) "
        "-print0 | xargs -0 -r rm -rf"
    )
    subprocess.run(
        ['sh', '-lc', command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def latest_checkpoint(output_dir):
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoints = sorted(
        checkpoint_dir.glob('ppo_checkpoint_*_steps.zip'),
        key=lambda path: path.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None


def latest_resume_model(output_dir):
    candidates = [
        output_dir / 'final_model.zip',
        latest_checkpoint(output_dir),
        output_dir / 'best_model.zip',
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def read_completed_timesteps(progress_path):
    try:
        return int(progress_path.read_text(encoding='utf-8').strip())
    except (FileNotFoundError, ValueError):
        return 0


def write_completed_timesteps(progress_path, value):
    progress_path.write_text(f'{value}\n', encoding='utf-8')


def run_one_setting(coef, penality, args):
    run_name = f'coef{coef:g}_pen{penality:g}'
    output_dir = Path(args.base_output_dir) / run_name
    figure_dir = Path(args.base_figure_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = output_dir / 'SWEEP_COMPLETE'
    if complete_marker.exists():
        print(f'Skip completed setting: {run_name}')
        return summarize_setting(coef, penality, output_dir)
    existing_ppo_traces = sorted(output_dir.glob('ppo_decision_trace_ep*.csv'))
    if args.skip_existing and len(existing_ppo_traces) >= args.eval_episodes and latest_resume_model(output_dir) is not None:
        print(f'Mark existing completed setting: {run_name}')
        rows = summarize_setting(coef, penality, output_dir)
        complete_marker.write_text('complete\n', encoding='utf-8')
        return rows

    env = os.environ.copy()
    env['GYM_DSSAT_REWARD_COEF'] = str(coef)
    env['GYM_DSSAT_REWARD_PENALITY'] = str(penality)

    best_model_path = output_dir / 'best_model.zip'
    final_model_path = output_dir / 'final_model.zip'
    progress_path = output_dir / 'TRAIN_PROGRESS.txt'
    completed_timesteps = read_completed_timesteps(progress_path)
    if args.skip_existing and final_model_path.exists() and completed_timesteps >= args.total_timesteps:
        print(f'Skip existing training result: {output_dir}')
    else:
        while completed_timesteps < args.total_timesteps:
            chunk = min(args.train_chunk_timesteps, args.total_timesteps - completed_timesteps)
            resume_model = latest_resume_model(output_dir)
            command = [
                sys.executable,
                'train_hl.py',
                '--coef', str(coef),
                '--penality', str(penality),
                '--total-timesteps', str(chunk),
                '--output-dir', str(output_dir),
                '--sb3-verbose', '0',
                '--checkpoint-freq', str(max(1000, min(chunk, args.train_chunk_timesteps))),
            ]
            if resume_model is not None:
                command.extend(['--resume-model', str(resume_model)])
            run_command(command, env, output_dir / 'train.log')
            completed_timesteps += chunk
            write_completed_timesteps(progress_path, completed_timesteps)
            cleanup_processes(args.cleanup_processes)
            cleanup_tmp()
            gc.collect()

    model_path = best_model_path if best_model_path.exists() else final_model_path
    if not model_path.exists():
        model_path = latest_resume_model(output_dir)
    if not model_path.exists():
        raise FileNotFoundError(f'No trained model found in {output_dir}')

    ppo_traces = sorted(output_dir.glob('ppo_decision_trace_ep*.csv'))
    if args.skip_existing and len(ppo_traces) >= args.eval_episodes:
        print(f'Skip existing evaluation traces: {output_dir}')
    else:
        run_command([
            sys.executable,
            'evaluate_hl.py',
            '--coef', str(coef),
            '--penality', str(penality),
            '--n-episodes', str(args.eval_episodes),
            '--output-dir', str(output_dir),
            '--model-path', str(model_path),
            '--agents', args.eval_agents,
            '--no-history-pickle',
        ], env, output_dir / 'evaluate.log')
        cleanup_processes(args.cleanup_processes)
        cleanup_tmp()
        gc.collect()

    plot_agents = {name.strip() for name in args.eval_agents.split(',') if name.strip()}
    can_plot = {'null', 'ppo', 'expert'}.issubset(plot_agents)
    if not args.no_plot and can_plot:
        run_command([
            sys.executable,
            'plot_hl.py',
            '--mode', 'fertilization',
            '--input-dir', str(output_dir),
            '--figure-dir', str(figure_dir),
        ], env, output_dir / 'plot.log')
        gc.collect()
    elif not args.no_plot:
        print('Skip plot: plot_hl.py needs null, ppo, expert histories. Use --eval-agents null,ppo,expert to plot.')

    rows = summarize_setting(coef, penality, output_dir)
    complete_marker.write_text('complete\n', encoding='utf-8')
    return rows


def summarize_setting(coef, penality, output_dir):
    trace_paths = sorted(Path(output_dir).glob('ppo_decision_trace_ep*.csv'))
    if not trace_paths:
        raise FileNotFoundError(f'No PPO decision trace found in {output_dir}')

    rows = []
    for ep, path in enumerate(trace_paths, start=1):
        df = pd.read_csv(path)
        trnu = pd.to_numeric(df.get('trnu'), errors='coerce')
        rewards = pd.to_numeric(df.get('reward'), errors='coerce')
        anfer = pd.to_numeric(df.get('real_action_anfer'), errors='coerce')
        rows.append({
            'coef': coef,
            'penality': penality,
            'episode': ep,
            'final_trnu': trnu.dropna().iloc[-1] if trnu.notna().any() else np.nan,
            'mean_trnu': trnu.mean(),
            'max_trnu': trnu.max(),
            'total_anfer': anfer.sum() if anfer.notna().any() else np.nan,
            'n_applications': int((anfer.fillna(0) > 0).sum()) if anfer.notna().any() else 0,
            'total_reward': rewards.sum(),
            'output_dir': str(output_dir),
        })
    return rows


def add_scores(df):
    df = df.copy()
    grouped = df.groupby(['coef', 'penality'], as_index=False).agg({
        'final_trnu': 'mean',
        'mean_trnu': 'mean',
        'max_trnu': 'mean',
        'total_anfer': 'mean',
        'n_applications': 'mean',
        'total_reward': 'mean',
        'output_dir': 'first',
    })

    trnu_min = grouped['final_trnu'].min()
    trnu_span = grouped['final_trnu'].max() - trnu_min
    fert_min = grouped['total_anfer'].min()
    fert_span = grouped['total_anfer'].max() - fert_min

    grouped['trnu_norm'] = (
        (grouped['final_trnu'] - trnu_min) / trnu_span
        if trnu_span > 0 else 1.0
    )
    grouped['fert_saving_norm'] = (
        (grouped['total_anfer'].max() - grouped['total_anfer']) / fert_span
        if fert_span > 0 else 1.0
    )
    grouped['score'] = 0.5 * grouped['trnu_norm'] + 0.5 * grouped['fert_saving_norm']

    grouped['pareto_best'] = False
    for idx, row in grouped.iterrows():
        dominated = (
            (grouped['final_trnu'] >= row['final_trnu'])
            & (grouped['total_anfer'] <= row['total_anfer'])
            & (
                (grouped['final_trnu'] > row['final_trnu'])
                | (grouped['total_anfer'] < row['total_anfer'])
            )
        ).any()
        grouped.loc[idx, 'pareto_best'] = not dominated

    return grouped.sort_values(['pareto_best', 'score'], ascending=[False, False])


def main():
    args = parse_args()
    coefs = parse_float_list(args.coefs)
    penalities = parse_float_list(args.penalities)
    all_rows = []
    failures = []

    for coef, penality in itertools.product(coefs, penalities):
        print(f'\n=== Reward setting: coef={coef}, penality={penality} ===')
        try:
            rows = run_one_setting(coef, penality, args)
            all_rows.extend(rows)
        except Exception as exc:
            print(f'FAILED coef={coef}, penality={penality}: {type(exc).__name__}: {exc}')
            failures.append({'coef': coef, 'penality': penality, 'error': repr(exc)})
            Path(args.base_output_dir).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(failures).to_csv(Path(args.base_output_dir) / 'reward_sweep_failures.csv', index=False)
            cleanup_processes(args.cleanup_processes)
            gc.collect()
            if args.stop_on_error:
                raise
            continue

        Path(args.base_output_dir).mkdir(parents=True, exist_ok=True)
        if all_rows:
            raw_df = pd.DataFrame(all_rows)
            raw_df.to_csv(Path(args.base_output_dir) / 'reward_sweep_episode_results.csv', index=False)
            add_scores(raw_df).to_csv(Path(args.base_output_dir) / 'reward_sweep_summary.csv', index=False)

    if not all_rows:
        raise RuntimeError('No successful reward settings. Check per-setting train/evaluate logs.')
    summary = add_scores(pd.DataFrame(all_rows))
    print('\n=== Reward sweep summary ===')
    print(summary[['coef', 'penality', 'final_trnu', 'total_anfer', 'score', 'pareto_best']])
    print(f'\nSaved summary to: {Path(args.base_output_dir) / "reward_sweep_summary.csv"}')


if __name__ == '__main__':
    main()
