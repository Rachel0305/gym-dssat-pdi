import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_hl import (
    check_dap,
    dssat_utils,
    get_growing_stage_occurences,
    load_data,
    make_df_from_dict,
    plot_actions,
    plot_rewards,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='./output_hl/all')
    parser.add_argument('--figure-dir', default='./figures_hl/all')
    return parser.parse_args()


def get_all_statistics(history_dict):
    features_subset = ['topwt', 'grnwt', 'cumsumfert', 'totir', 'runoff', 'trnu']
    features = [
        *features_subset,
        'trnu_mean',
        'trnu_max',
        'fertilizer_efficiency',
        'irrigation_efficiency',
        'duration',
        'napp_anfer',
        'napp_amir',
        'total_anfer',
        'total_amir',
    ]
    state_features_dic = {key: {feature: [] for feature in features} for key in [*history_dict]}

    for key in [*history_dict]:
        for repetition in history_dict[key]['observation']:
            last_state = repetition[-1]
            for feature in features_subset:
                state_features_dic[key][feature].append(last_state.get(feature, np.nan))

            trnu_values = [
                state.get('trnu', np.nan)
                for state in repetition
                if isinstance(state, dict)
            ]
            trnu_values = np.asarray(trnu_values, dtype=float)
            if trnu_values.size == 0 or np.isnan(trnu_values).all():
                state_features_dic[key]['trnu_mean'].append(np.nan)
                state_features_dic[key]['trnu_max'].append(np.nan)
            else:
                state_features_dic[key]['trnu_mean'].append(np.nanmean(trnu_values))
                state_features_dic[key]['trnu_max'].append(np.nanmax(trnu_values))

    null_grnwt = state_features_dic.get('null', {}).get('grnwt', [])
    for key in [*history_dict]:
        actions_history = history_dict[key]['action']
        grnwt_rep = state_features_dic[key]['grnwt']
        cumsumfert_rep = state_features_dic[key]['cumsumfert']
        totir_rep = state_features_dic[key]['totir']

        for idx, applications in enumerate(actions_history):
            anfers = np.asarray([action.get('anfer', 0.0) for action in applications], dtype=float)
            amirs = np.asarray([action.get('amir', 0.0) for action in applications], dtype=float)
            total_anfer = np.nansum(anfers)
            total_amir = np.nansum(amirs)
            state_features_dic[key]['napp_anfer'].append((anfers > 0).sum())
            state_features_dic[key]['napp_amir'].append((amirs > 0).sum())
            state_features_dic[key]['total_anfer'].append(total_anfer)
            state_features_dic[key]['total_amir'].append(total_amir)
            state_features_dic[key]['duration'].append(len(applications))

            grnwt_0 = null_grnwt[idx] if idx < len(null_grnwt) else np.nan
            yield_gain = grnwt_rep[idx] - grnwt_0
            state_features_dic[key]['fertilizer_efficiency'].append(
                np.nan if total_anfer == 0 else yield_gain / total_anfer
            )
            state_features_dic[key]['irrigation_efficiency'].append(
                np.nan if total_amir == 0 else yield_gain / total_amir
            )

    return make_df_from_dict(state_features_dic)


if __name__ == '__main__':
    args = parse_args()
    history_dict = load_data(path=f'{args.input_dir}/evaluation_histories.pkl')

    print('###########################')
    print('## MODE: all')
    print('###########################')
    print('Top-level keys:', history_dict.keys())

    os.makedirs(args.figure_dir, exist_ok=True)
    check_dap(history_dict)

    df_stages = get_growing_stage_occurences(history_dict)
    df_stages.transpose().round(0).to_csv(f'{args.input_dir}/growing_stages.csv', index_label='istage')

    action_keys = [key for key in ['ppo', 'expert'] if key in history_dict]
    if not action_keys:
        action_keys = [key for key in history_dict if key != 'null']

    plot_actions(
        history_dict=history_dict,
        mode='fertilization',
        saving_path=f'{args.figure_dir}/allFertilizationApplications.pdf',
        keys=action_keys,
    )
    plt.close('all')

    plot_actions(
        history_dict=history_dict,
        mode='irrigation',
        saving_path=f'{args.figure_dir}/allIrrigationApplications.pdf',
        keys=action_keys,
    )
    plt.close('all')

    plot_rewards(
        history_dict=history_dict,
        mode='all',
        quantile_range_legend=False,
        saving_path=f'{args.figure_dir}/allRewards.pdf',
    )
    plt.close('all')

    df_stats = get_all_statistics(history_dict=history_dict)
    df_stats.describe().round(1).to_csv(f'{args.input_dir}/advanced_evaluation_all.csv')
