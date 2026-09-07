from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "benchmark_results/218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k_five_scenario_figures_ckpt5000_attempt2"
OUT = ROOT / "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot/figures"
OUT.mkdir(parents=True, exist_ok=True)
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert", "rl_candidate"]
LABELS = {"null": "Null", "recorded_farmer_template": "Recorded template", "dssat_auto_external_n": "DSSAT auto + external N", "official_extension_expert": "Official expert", "rl_candidate": "PPO"}
COLORS = {"null": "#555555", "recorded_farmer_template": "#C44E52", "dssat_auto_external_n": "#D8A305", "official_extension_expert": "#7E63B6", "rl_candidate": "#2A9D55"}
STYLES = {"null": "-", "recorded_farmer_template": "--", "dssat_auto_external_n": "-.", "official_extension_expert": ":", "rl_candidate": "-"}


def daily_plot(d, year):
    fig, ax = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    w = d[d.scenario.eq("null")].sort_values("dap")
    ax[0, 0].bar(w.dap, w.rainfall_mm, color="#80A9C7", alpha=.85, label="Rain")
    t = ax[0, 0].twinx(); t.plot(w.dap, w.tmax_c, color="#C23B32", lw=1.2, label="Tmax"); t.plot(w.dap, w.tmin_c, color="#666", ls="--", lw=1.0, label="Tmin")
    ax[0, 0].set_title("Weather"); ax[0, 0].set_ylabel("Rain (mm)"); t.set_ylabel("Temperature (C)")
    for s in SCENARIOS:
        q = d[d.scenario.eq(s)].sort_values("dap"); c, ls, lab = COLORS[s], STYLES[s], LABELS[s]
        ax[0, 1].plot(q.dap, q.soil_water_mm, c=c, ls=ls, lw=1.3, label=lab)
        ax[1, 0].plot(q.dap, q.water_stress_index_wspd, c=c, ls=ls, lw=1.3, label=lab)
        ax[1, 1].plot(q.dap, q.nitrogen_stress_index_nstd, c=c, ls=ls, lw=1.3, label=lab)
        qi = q[q.irrigation_executed_mm.gt(0)]; qn = q[q.nitrogen_executed_kg_ha.gt(0)]
        if len(qi): ax[2, 0].stem(qi.dap, qi.irrigation_executed_mm, linefmt=c, markerfmt="o", basefmt=" ", label=lab)
        if len(qn): ax[2, 1].stem(qn.dap, qn.nitrogen_executed_kg_ha, linefmt=c, markerfmt="o", basefmt=" ", label=lab)
        ax[3, 0].plot(q.dap, q.grain_yield_kg_ha, c=c, ls=ls, lw=1.4, label=f"{lab} grain")
        ax[3, 0].plot(q.dap, q.biomass_kg_ha, c=c, ls=ls, lw=.8, alpha=.4)
        ax[3, 1].plot(q.dap, q.unified_cumulative_reward, c=c, ls=ls, lw=1.3, label=lab)
    for a, title, ylabel in [(ax[0,1],"Soil water","SWTD (mm)"),(ax[1,0],"Water stress index","WSPD"),(ax[1,1],"Nitrogen stress index","NSTD"),(ax[2,0],"Irrigation events","mm/event"),(ax[2,1],"Nitrogen application events","kg/ha/event"),(ax[3,0],"Grain and biomass trajectories","kg/ha"),(ax[3,1],"Unified cumulative reward","common units")]:
        a.set_title(title); a.set_ylabel(ylabel); a.grid(alpha=.2)
    for a in [ax[0,1], ax[2,0], ax[2,1], ax[3,1]]: a.legend(fontsize=7, ncol=2)
    ax[3,0].set_xlabel("DAP"); ax[3,1].set_xlabel("DAP")
    fig.suptitle(f"YCA{year} five-scenario daily process (218YCA-5K replay)", x=.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(OUT / f"219YCA_yca{year}_five_scenario_daily.png", dpi=220); plt.close(fig)


def bars(s):
    years = sorted(s.year.unique()); x = np.arange(len(years)); width = .16
    def draw(cols, title, name):
        fig, axes = plt.subplots(len(cols), 1, figsize=(16, 4*len(cols)), sharex=True); axes = np.atleast_1d(axes)
        for a, col in zip(axes, cols):
            for i, sc in enumerate(SCENARIOS):
                z = s[s.scenario.eq(sc)].set_index("year").reindex(years)[col]
                a.bar(x+(i-2)*width, z, width, label=LABELS[sc], color=COLORS[sc])
            a.set_ylabel(col); a.grid(axis="y", alpha=.25); a.legend(ncol=3, fontsize=8)
        axes[-1].set_xticks(x); axes[-1].set_xticklabels(years); axes[-1].set_xlabel("Validation year")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(OUT/name, dpi=220); plt.close(fig)
    draw(["actual_irrigation_mm", "actual_nitrogen_kg_ha"], "YCA yearly water and nitrogen inputs by scenario", "219YCA_five_scenario_management_bars.png")
    draw(["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"], "YCA yearly yield and resource-use efficiency by scenario", "219YCA_five_scenario_metrics_bars.png")


def main():
    d = pd.read_csv(SRC / "tables/218YCA_five_scenario_daily.csv", keep_default_na=False)
    s = pd.read_csv(SRC / "tables/218YCA_five_scenario_season_summary.csv", keep_default_na=False)
    for year in sorted(d.year.unique()): daily_plot(d[d.year.eq(year)].copy(), int(year))
    bars(s)
    print(f"wrote {d.year.nunique()} yearly daily figures and 2 grouped bar figures")


if __name__ == "__main__": main()
