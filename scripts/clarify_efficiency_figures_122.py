"""Clarify undefined N productivity; preserve all source values and backup figures."""
from pathlib import Path
import shutil,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'results/forecast_five_scenario_120_corrected_HL_20261027'
AUD=ROOT/'results/forecast_publication_audit_122_20261027'
BACK=ROOT/'backups/figure122'
d=pd.read_csv(PKG/'all_five_sites_seed_year_scenario.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
metrics=['grain_yield_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg']
labels=['Grain yield (kg/ha)','Water productivity\nyield / ET (kg/m3)','N partial factor productivity\nyield / N (kg/kg)']
colors=['#555555','#C44E52','#D8A305','#7E63B6','#2A9D55']
count=0
for (site,seed),q in d.groupby(['site','seed']):
    scenarios=['null','recorded_farmer_template','dssat_auto' if site=='HL' else 'dssat_auto_external_n','official_extension_expert','rl_candidate']
    names=['Null','Recorded template','DSSAT auto (N=0)' if site=='HL' else 'DSSAT auto + external N','Official expert','PPO + perfect 7-day summary']
    fig,axes=plt.subplots(3,1,figsize=(15,11),sharex=True)
    x=np.arange(10); years=list(range(2014,2024))
    for ax,metric,label in zip(axes,metrics,labels):
        for j,(scenario,name,color) in enumerate(zip(scenarios,names,colors)):
            z=q[q.scenario.eq(scenario)].set_index('year').reindex(years)
            pos=x+(j-2)*.16
            ax.bar(pos,z[metric],width=.15,color=color,label=name)
            if metric=='PFP_N_kg_kg':
                for xx in pos[z[metric].isna().to_numpy()]:
                    ax.text(xx,.025,'N=0',rotation=90,ha='center',va='bottom',fontsize=6,color=color,transform=ax.get_xaxis_transform())
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[-1].set_xticks(x,years)
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3,frameon=False)
    fig.suptitle(f'{site} | seed {int(seed):02d} | five scenarios',y=.985)
    fig.text(.06,.018,'N=0: PFP_N is undefined (missing bar, not zero). WP_ET uses seasonal ET; it is not irrigation-only productivity.\nAll zero-N scenarios are marked. No undefined value is imputed.',fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.09,left=.09,right=.99,hspace=.23)
    stem=PKG/site/f'seed_{int(seed):02d}'/'summary/five_scenario_metrics'
    for ext in ['png','pdf','svg']:
        target=stem.with_suffix('.'+ext)
        backup=BACK/target.relative_to(PKG);backup.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and not backup.exists():shutil.copy2(target,backup)
        fig.savefig(target,dpi=150)
    plt.close(fig);count+=1
    print(f'{site} seed {int(seed):02d}',flush=True)
(AUD/'figure_repair.json').write_text(json.dumps({'seed_figures_updated':count,'formats':['png','pdf','svg'],'source_values_changed':False,'undefined_values_imputed':False,'backup':str(BACK)},indent=2),encoding='utf-8')
