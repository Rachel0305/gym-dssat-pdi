"""Generate the two missing 215HLA five-scenario summary bar figures."""
from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
run_dir = ROOT / 'benchmark_results/215HLA_hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo_five_scenario_figures_ckpt2000'
table = run_dir / 'tables/215HLA_five_scenario_season_summary_native_auto.csv'
fig_dir = run_dir / 'figures'

spec = importlib.util.spec_from_file_location('hla054', ROOT / 'src/054_hla_lowIC_site_transfer/run_054_03_hla_lowIC_five_scenario_figures.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 215HL uses native DSSAT automatic irrigation; no external-N scenario.
mod.SCENARIOS = ['null', 'recorded_farmer_template', 'dssat_auto', 'official_extension_expert', 'rl_candidate']
mod.LABELS['dssat_auto'] = 'DSSAT auto irrigation'
mod.COLORS['dssat_auto'] = mod.COLORS.get('dssat_auto_external_n', '#D8A305')

summary = pd.read_csv(table)
years = sorted(pd.to_numeric(summary['year']).astype(int).unique().tolist())
mod.draw_grouped(summary, years, mod.RESOURCE_COLUMNS, fig_dir / '215HLA_five_scenario_management.png')
mod.draw_grouped(summary, years, mod.METRIC_COLUMNS, fig_dir / '215HLA_five_scenario_metrics.png')
print('created:', fig_dir / '215HLA_five_scenario_management.png')
print('created:', fig_dir / '215HLA_five_scenario_metrics.png')
