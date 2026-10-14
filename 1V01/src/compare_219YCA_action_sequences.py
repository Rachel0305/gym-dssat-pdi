from pathlib import Path
import re
import pandas as pd

ROOT = Path.cwd()
OLD = ROOT / 'benchmark_results/218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k_five_scenario_figures_ckpt5000_attempt2/tables/218YCA_five_scenario_management_events.csv'
NEW = ROOT / 'benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/tables/219YCA_development_validation_episode_metrics.csv'
OUT = NEW.parent

def fmt(x):
    return f'{float(x):g}'

def signature(g):
    rows = g.groupby('dap', as_index=False)[['I', 'N']].sum().sort_values('dap')
    rows = rows[(rows.I > 0) | (rows.N > 0)]
    return ';'.join(f"DAP{int(r.dap)}:I{fmt(r.I)}/N{fmt(r.N)}" for _, r in rows.iterrows())

old = pd.read_csv(OLD, keep_default_na=False)
old = old[old.scenario.eq('rl_candidate')].copy()
old['I'] = pd.to_numeric(old.irrigation_executed_mm, errors='coerce').fillna(0)
old['N'] = pd.to_numeric(old.nitrogen_executed_kg_ha, errors='coerce').fillna(0)
old = old[(old.I > 0) | (old.N > 0)]
old_year = old.groupby('year').apply(lambda g: pd.Series({
    'old_signature': signature(g),
    'old_irrigation_events': int((g.groupby('dap').I.sum() > 0).sum()),
    'old_n_events': int((g.groupby('dap').N.sum() > 0).sum()),
    'old_positive_action_rows': int(len(g.groupby('dap'))),
    'old_unique_pairs': int(len(set(zip(g.I[g.I > 0], g.N[g.N > 0])))),
}), include_groups=False).reset_index()

new = pd.read_csv(NEW, keep_default_na=False)
new = new[(new.seed == 1) & (new.checkpoint_step == 20160)].copy()
new = new.rename(columns={'year': 'year', 'action_signature': 'new_signature'})
new['new_irrigation_events'] = pd.to_numeric(new.irrigation_event_count)
new['new_n_events'] = pd.to_numeric(new.n_event_count)
new['new_positive_action_rows'] = pd.to_numeric(new.positive_action_rows)
new['new_unique_pairs'] = pd.to_numeric(new.unique_nonzero_action_pairs)
cols = ['year', 'new_signature', 'new_irrigation_events', 'new_n_events', 'new_positive_action_rows', 'new_unique_pairs', 'total_irrigation', 'total_n']
new = new[cols]
out = old_year.merge(new, on='year', how='inner')
out['signature_same'] = out.old_signature == out.new_signature

def parse(s):
    ans = {}
    for item in str(s).split(';'):
        m = re.match(r'DAP(\d+):I([^/]+)/N(.+)', item)
        if m:
            ans[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return ans

def overlap(row):
    a, b = parse(row.old_signature), parse(row.new_signature)
    keys = set(a) | set(b)
    exact = sum(k in a and k in b and a[k] == b[k] for k in keys)
    return pd.Series({'same_dap_count': sum(k in a and k in b for k in keys), 'exact_dap_action_count': exact})
out = pd.concat([out, out.apply(overlap, axis=1)], axis=1)
out.to_csv(OUT / '219YCA_action_sequence_pre_vs_normobs_seed1_ck20160.csv', index=False, encoding='utf-8-sig')

summary = {
    'years': int(len(out)),
    'old_unique_year_signatures': int(out.old_signature.nunique()),
    'new_unique_year_signatures': int(out.new_signature.nunique()),
    'same_signature_years': int(out.signature_same.sum()),
    'changed_signature_years': int((~out.signature_same).sum()),
    'mean_same_dap_count': float(out.same_dap_count.mean()),
    'mean_exact_dap_action_count': float(out.exact_dap_action_count.mean()),
    'mean_old_irrigation_events': float(out.old_irrigation_events.mean()),
    'mean_new_irrigation_events': float(out.new_irrigation_events.mean()),
    'mean_old_n_events': float(out.old_n_events.mean()),
    'mean_new_n_events': float(out.new_n_events.mean()),
}
(OUT / '219YCA_action_sequence_pre_vs_normobs_seed1_ck20160_summary.txt').write_text('\n'.join(f'{k}: {v}' for k, v in summary.items()), encoding='utf-8')
print(pd.Series(summary).to_string())
print(out[['year', 'old_irrigation_events', 'new_irrigation_events', 'old_n_events', 'new_n_events', 'same_dap_count', 'exact_dap_action_count', 'signature_same']].to_string(index=False))
