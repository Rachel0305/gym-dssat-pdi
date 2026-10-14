from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'benchmark_results/215HLA_hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo_five_scenario_figures_ckpt2000'
target = out / 'tables/215HLA_five_scenario_season_summary.csv'
native = ROOT / 'benchmark_results/080_00_hla_lowIC_nepochs3_10k_native_auto_replay/tables/080_00_native_auto_summary.csv'

df = pd.read_csv(target)
nat = pd.read_csv(native)
nat = nat[nat.scenario.eq('dssat_auto')].copy()
nat['scenario'] = 'dssat_auto'
nat = nat.rename(columns={'unified_reward': 'unified_cumulative_reward'})
for c in ['biomass_kg_ha', 'simple_profit']:
    if c not in nat.columns:
        nat[c] = float('nan')
nat = nat[df.columns]
df = df[~df.scenario.eq('dssat_auto_external_n')]
df = pd.concat([df, nat], ignore_index=True).sort_values(['year', 'scenario'])
df.to_csv(target, index=False, encoding='utf-8-sig')
print(df.groupby('scenario')[['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg']].mean(numeric_only=True).to_string())
