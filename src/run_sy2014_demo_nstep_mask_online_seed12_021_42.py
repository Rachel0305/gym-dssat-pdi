from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_event_balanced_online_retention_1k_021_35 as ab
import run_sy2014_online_demo_nstep_mask_ab_021_41 as treatment


OUT = ROOT / "benchmark_results" / "021_42"
DOC = ROOT / "docs" / "2026-07-15_021_42_sy2014_demo_nstep_mask_online_seed12_1k.md"
SEEDS = [(1, 21036), (2, 21037)]


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    if DOC.exists(): raise FileExistsError(f"Refusing to overwrite {DOC}")
    OUT.mkdir(parents=True)
    all_rows = []; seed_summaries = []; validations = []
    for seed, sample_seed in SEEDS:
        seed_out = OUT / f"online_seed{seed}"
        ab.OUT = seed_out; ab.online_update = treatment.treatment_update
        interactions, updates, audit, scaler = ab.train_online(action_seed=seed, sample_seed=sample_seed)
        interactions.to_csv(seed_out / "training_interactions.csv", index=False, encoding="utf-8-sig")
        updates.to_csv(seed_out / "update_log.csv", index=False, encoding="utf-8-sig")
        prior = json.loads(ab.PRIOR_SUMMARY.read_text(encoding="utf-8"))
        rows = [{
            "online_seed": seed, "checkpoint": 0,
            "yield_kg_ha": prior["yield_kg_ha"], "biomass_kg_ha": prior["biomass_kg_ha"],
            "irrigation_mm": prior["irrigation_mm"], "nitrogen_kg_ha": prior["nitrogen_kg_ha"],
            "late_n_after_dap90_kg_ha": prior["late_n_after_dap90_kg_ha"],
            "expert_efficiency_gate": True, "q_values_finite": True, "terminated_or_truncated": True,
        }]
        for checkpoint in ab.CHECKPOINTS:
            _daily, result = ab.evaluate_checkpoint(checkpoint, scaler["mean"], scaler["scale"])
            result["online_seed"] = seed; rows.append(result)
        frame = pd.DataFrame(rows).sort_values("checkpoint")
        frame.to_csv(seed_out / "checkpoint_trajectory.csv", index=False, encoding="utf-8-sig")
        online = frame[frame.checkpoint > 0]
        passes = int(online.expert_efficiency_gate.astype(bool).sum())
        final_pass = bool(online.loc[online.checkpoint == 1000, "expert_efficiency_gate"].iloc[0])
        success = bool(passes >= 3 and final_pass and float(online.yield_kg_ha.min()) >= 9613)
        seed_summaries.append({
            "online_seed": seed, "sample_seed": sample_seed, "pass_count": passes,
            "final_pass": final_pass, "minimum_yield": float(online.yield_kg_ha.min()),
            "success": success,
        })
        validations.append({
            "online_seed": seed, "initial_hash_matches": audit["initial_hash_matches"],
            "demonstrations_unchanged": audit["demonstrations_unchanged"],
            "all_batches_16_16": audit["all_batches_16_demo_16_agent"],
            "all_updates_finite": audit["all_updates_finite"],
            "replay_agent_count_final": audit["replay_agent_count_final"],
            "action_seed_recorded": audit["online_action_seed"] == seed,
            "sample_seed_recorded": audit["online_sample_seed"] == sample_seed,
        })
        all_rows.extend(frame.to_dict("records"))
    trajectory = pd.DataFrame(all_rows)
    summary_frame = pd.DataFrame(seed_summaries)
    validation_frame = pd.DataFrame(validations)
    trajectory.to_csv(OUT / "021_42_checkpoint_trajectories.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(OUT / "021_42_seed_summary.csv", index=False, encoding="utf-8-sig")
    validation_frame.to_csv(OUT / "021_42_seed_validation.csv", index=False, encoding="utf-8-sig")
    successes = int(summary_frame.success.sum())
    branch = "A" if successes == 2 else ("B" if successes == 1 else "C")
    validation = {
        "seed_count": 2,
        "all_start_hashes_match": bool(validation_frame.initial_hash_matches.all()),
        "all_demonstrations_unchanged": bool(validation_frame.demonstrations_unchanged.all()),
        "all_batches_valid": bool(validation_frame.all_batches_16_16.all()),
        "all_updates_finite": bool(validation_frame.all_updates_finite.all()),
        "seeds_recorded_correctly": bool(validation_frame.action_seed_recorded.all() and validation_frame.sample_seed_recorded.all()),
        "all_required_checks_pass": bool(
            validation_frame.initial_hash_matches.all() and validation_frame.demonstrations_unchanged.all()
            and validation_frame.all_batches_16_16.all() and validation_frame.all_updates_finite.all()
            and validation_frame.action_seed_recorded.all() and validation_frame.sample_seed_recorded.all()
        ),
    }
    (OUT / "021_42_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch, "successful_online_seeds": successes,
        "scope": "same offline seed0 start; online stochasticity only",
        "online_5k_started": False,
        "interpretation": {
            "A": "demo n-step屏蔽在两个额外在线随机轨迹上均保持成功，可进入独立5K seed0验证。",
            "B": "仅一个额外在线seed成功，保持性仍有随机敏感性。",
            "C": "两个额外在线seed均失败，seed0正结果未复现。",
        }[branch],
    }
    (OUT / "021_42_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13,4))
    for seed, frame in trajectory.groupby("online_seed"):
        frame = frame.sort_values("checkpoint")
        axes[0].plot(frame.checkpoint, frame.yield_kg_ha, marker="o", label=f"seed{seed}")
        axes[1].plot(frame.checkpoint, frame.irrigation_mm, marker="o", label=f"seed{seed}")
        axes[2].plot(frame.checkpoint, frame.nitrogen_kg_ha, marker="o", label=f"seed{seed}")
    axes[0].axhline(11077, color="#333333", linestyle="--"); axes[0].set_ylabel("Yield (kg/ha)")
    axes[1].set_ylabel("Irrigation (mm)"); axes[2].set_ylabel("Nitrogen (kg/ha)")
    for ax in axes: ax.grid(alpha=0.2); ax.set_xlabel("Online steps"); ax.legend(frameon=False)
    fig.suptitle("SY2014 demo n-step mask: online seed1/2 retention")
    fig.tight_layout(rect=(0,0,1,0.92))
    fig.savefig(OUT / "021_42_online_seed12_retention.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_42_online_seed12_retention.svg", bbox_inches="tight")
    plt.close(fig)

    display = trajectory[["online_seed","checkpoint","yield_kg_ha","irrigation_mm","nitrogen_kg_ha","late_n_after_dap90_kg_ha","expert_efficiency_gate"]].round(3)
    doc = f"""# 021_42 SY2014 demo n-step屏蔽在线seed1/2复核记录

## 范围

固定同一个021_34 seed0离线起点，只改变在线action RNG和sampling RNG；这不是不同网络初始化复现。

## 结果

{markdown_table(display)}

{markdown_table(summary_frame)}

预注册分支：**{branch}**。{summary['interpretation']}

未启动5K。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation, "seeds": seed_summaries}, indent=2))


if __name__ == "__main__": main()
