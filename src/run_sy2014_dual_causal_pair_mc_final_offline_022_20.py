from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_sy2014_pairwise_mc_gradient_conflict_022_12 as a12
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import QNetwork


OUT = ROOT / "benchmark_results" / "022_20"
L65_WEIGHT = 0.003002436945892297
L110_WEIGHT = 0.06764715488665898
UPDATES = 3000
CHECKPOINTS = (0, 500, 1500, 3000)
SEEDS = (0, 1, 2)


def compute_losses(model, train_obs, train_actions, train_targets, obs65, target65, obs110, target110):
    q = model(train_obs).gather(1, train_actions[:, None]).squeeze(1)
    lmc = torch.nn.functional.smooth_l1_loss(q, train_targets)
    q65 = model(obs65)
    l65 = torch.nn.functional.smooth_l1_loss(q65[:, 1] - q65[:, 7], target65)
    q110 = model(obs110)
    l110 = torch.nn.functional.smooth_l1_loss(q110[:, 0] - q110[:, 1], target110)
    return lmc, l65, l110


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = np.load(a12.NPZ_PATH)
    obs, actions, targets = data["observations"], data["actions"], data["targets"]
    scenarios = data["scenarios"].astype(str)
    stage_indices = data["stage_indices"]
    daps = np.asarray([(1, 30, 50, 65, 85, 110)[int(i)] for i in stage_indices])
    manifest = pd.read_csv(a12.MANIFEST_PATH)
    split = pd.read_csv(a12.SPLIT_PATH)
    train_set = set(split.loc[split["split"] == "train", "scenario"].astype(str))
    clean_set = set(split.loc[split["split"] == "test", "scenario"].astype(str)) - {"W120_critical__N200_early"}
    train_mask = np.asarray([s in train_set for s in scenarios])
    clean_mask = np.asarray([s in clean_set for s in scenarios])
    train_obs = torch.tensor(obs[train_mask], dtype=torch.float32)
    train_actions = torch.tensor(actions[train_mask], dtype=torch.long)
    train_targets = torch.tensor(targets[train_mask], dtype=torch.float32)
    clean_obs = torch.tensor(obs[clean_mask], dtype=torch.float32)
    clean_daps = daps[clean_mask]
    support = a12.support_map()

    _, obs65, target65 = a12.build_pair_targets(obs, manifest)
    t110 = pd.read_csv(ROOT / "benchmark_results" / "022_19" / "022_19_dap110_causal_targets.csv")
    idx110 = []
    for scenario in t110["scenario"].astype(str):
        idx = np.where((scenarios == scenario) & (daps == 110))[0]
        if len(idx) != 1:
            raise ValueError(f"{scenario}: invalid DAP110 state count")
        idx110.append(int(idx[0]))
    obs110 = torch.tensor(obs[idx110], dtype=torch.float32)
    target110 = torch.tensor(t110["target_q0_minus_q1_scaled"].to_numpy(np.float32), dtype=torch.float32)

    loss_rows, metric_rows, margin_rows, stage_rows, final_rows = [], [], [], [], []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        payload = torch.load(a12.CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = QNetwork(); model.load_state_dict(payload["model_state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        baseline_choices, _ = a12.support_argmax(model, clean_obs, clean_daps, support)
        initial_mc = float(compute_losses(model, train_obs, train_actions, train_targets, obs65, target65, obs110, target110)[0].detach())
        intermediate_pass = False
        for update in range(UPDATES + 1):
            if update in CHECKPOINTS:
                model.eval()
                lmc, l65, l110 = compute_losses(model, train_obs, train_actions, train_targets, obs65, target65, obs110, target110)
                with torch.no_grad():
                    q65 = model(obs65); m65 = (q65[:, 1] - q65[:, 7]).cpu().numpy()
                    q110 = model(obs110); m110 = (q110[:, 0] - q110[:, 1]).cpu().numpy()
                choices, _ = a12.support_argmax(model, clean_obs, clean_daps, support)
                mc_rel = (float(lmc.detach()) - initial_mc) / max(abs(initial_mc), 1e-12)
                finite = all(
                    math.isfinite(v)
                    for v in [float(lmc.detach()), float(l65.detach()), float(l110.detach()), mc_rel]
                ) and np.isfinite(m65).all() and np.isfinite(m110).all()
                passed = bool((m65 > 0).all() and (m110 > 0).all() and mc_rel <= 0.01 and finite)
                if update not in (0, UPDATES) and passed: intermediate_pass = True
                metric_rows.append({"seed":seed,"update":update,"mc_loss":float(lmc.detach()),"l65":float(l65.detach()),"l110":float(l110.detach()),"mc_relative_change":mc_rel,"dap65_all_correct":bool((m65>0).all()),"dap110_all_correct":bool((m110>0).all()),"finite":finite,"checkpoint_pass":passed})
                for i,v in enumerate(m65): margin_rows.append({"seed":seed,"update":update,"group":"DAP65","case":i,"margin":float(v),"correct":bool(v>0)})
                for i,v in enumerate(m110): margin_rows.append({"seed":seed,"update":update,"group":"DAP110","case":i,"margin":float(v),"correct":bool(v>0)})
                for dap in (1,85):
                    mask=clean_daps==dap
                    stage_rows.append({"seed":seed,"update":update,"dap":dap,"argmax_changes_from_update0":int(np.sum(choices[mask]!=baseline_choices[mask]))})
                torch.save({"model_state_dict":model.state_dict(),"seed":seed,"additional_updates":update,"lambda65":L65_WEIGHT,"lambda110":L110_WEIGHT,"optimizer_reinitialized":True},OUT/"checkpoints"/f"dual_pair_mc_seed{seed}_update{update}.pt")
                model.train()
            if update == UPDATES: break
            lmc,l65,l110=compute_losses(model,train_obs,train_actions,train_targets,obs65,target65,obs110,target110)
            total=lmc+L65_WEIGHT*l65+L110_WEIGHT*l110
            optimizer.zero_grad(set_to_none=True); total.backward(); optimizer.step()
            loss_rows.append({"seed":seed,"update":update+1,"mc_loss":float(lmc.detach()),"l65":float(l65.detach()),"l110":float(l110.detach()),"total_loss":float(total.detach())})
        final=metric_rows[-1]
        final_rows.append({"seed":seed,"final_pass":bool(final["checkpoint_pass"]),"intermediate_pass":intermediate_pass,"intermediate_pass_then_regressed":bool(intermediate_pass and not final["checkpoint_pass"]),"final_dap65_all_correct":bool(final["dap65_all_correct"]),"final_dap110_all_correct":bool(final["dap110_all_correct"]),"final_mc_relative_change":float(final["mc_relative_change"])})

    losses=pd.DataFrame(loss_rows); metrics=pd.DataFrame(metric_rows); margins=pd.DataFrame(margin_rows); stages=pd.DataFrame(stage_rows); finals=pd.DataFrame(final_rows)
    losses.to_csv(OUT/"022_20_training_loss.csv",index=False); metrics.to_csv(OUT/"022_20_checkpoint_metrics.csv",index=False); margins.to_csv(OUT/"022_20_causal_margin_trajectory.csv",index=False); stages.to_csv(OUT/"022_20_descriptive_stage_changes.csv",index=False); finals.to_csv(OUT/"022_20_seed_final_status.csv",index=False)
    count=int(finals["final_pass"].sum()); branch="A_offline_causal_learning_success" if count>=2 else ("B_single_seed" if count==1 else "C_failed")
    result={"status":"completed","branch":branch,"final_seed_pass_count":count,"total_seeds":3,"lambda65":L65_WEIGHT,"lambda110":L110_WEIGHT,"updates_per_seed":UPDATES,"total_offline_updates":9000,"dssat_calls":0,"online_interactions":0,"next_step_allowed":branch=="A_offline_causal_learning_success"}
    (OUT/"022_20_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for (seed,group),g in margins.groupby(["seed","group"]): axes[0].plot(g.groupby("update")["margin"].min(),marker="o",label=f"s{seed}-{group}")
    axes[0].axhline(0,color="black",ls="--"); axes[0].set(title="Worst causal margin",xlabel="Update",ylabel="Minimum margin"); axes[0].legend(frameon=False,fontsize=7,ncol=2)
    for seed,g in metrics.groupby("seed"): axes[1].plot(g["update"],g["mc_relative_change"]*100,marker="o",label=f"seed{seed}")
    axes[1].axhline(1,color="red",ls="--"); axes[1].set(title="MC loss guardrail",xlabel="Update",ylabel="Change (%)"); axes[1].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/"022_20_final_offline_training.png",dpi=220,bbox_inches="tight"); fig.savefig(OUT/"022_20_final_offline_training.svg",bbox_inches="tight"); plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
