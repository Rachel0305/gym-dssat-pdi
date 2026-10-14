from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "018_07_sy2014_seed_stability_audit"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_07_sy2014_dqn_seed_stability_audit_record.md"


def path_exists(path: Path) -> bool:
    return path.exists()


def build_inventory() -> pd.DataFrame:
    checks = [
        {
            "artifact": "seed0_training_run",
            "path": PROJECT_ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08" / "train_runs" / "2014" / "seed0" / "dqn_train",
        },
        {
            "artifact": "seed0_checkpoint_summary",
            "path": PROJECT_ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08" / "017_08_sy_dqn_checkpoint_summary.csv",
        },
        {
            "artifact": "seed0_four_scenario_summary",
            "path": PROJECT_ROOT / "DSSAT_auto_validation" / "sy2014_dqn_resource_space_017_09" / "017_09_sy2014_four_scenario_summary.csv",
        },
        {
            "artifact": "seed1_training_run",
            "path": PROJECT_ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08" / "train_runs" / "2014" / "seed1" / "dqn_train",
        },
        {
            "artifact": "seed1_checkpoint_summary",
            "path": PROJECT_ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08" / "017_08_sy_dqn_checkpoint_summary_seed1.csv",
        },
        {
            "artifact": "seed1_eval_run",
            "path": PROJECT_ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08" / "eval_runs" / "2014" / "seed1",
        },
    ]
    rows = []
    for item in checks:
        rows.append(
            {
                "artifact": item["artifact"],
                "path": str(item["path"].relative_to(PROJECT_ROOT)),
                "exists": path_exists(item["path"]),
            }
        )
    return pd.DataFrame(rows)


def build_summary() -> pd.DataFrame:
    summary_rows = []

    four_path = PROJECT_ROOT / "DSSAT_auto_validation" / "sy2014_dqn_resource_space_017_09" / "017_09_sy2014_four_scenario_summary.csv"
    if four_path.exists():
        df = pd.read_csv(four_path)
        dqn = df[df["scenario"].eq("dqn")].iloc[0]
        ext_yield = 11077.0
        ext_irrigation = 266.1
        ext_n = 300.0
        summary_rows.append(
            {
                "seed": 0,
                "source": str(four_path.relative_to(PROJECT_ROOT)),
                "final_gwad": float(dqn["final_gwad"]),
                "final_cwad": float(dqn["final_cwad"]),
                "irrigation_mm": float(dqn["irrigation_total_mm"]),
                "nitrogen_kg_ha": float(dqn["fertilizer_total_kg_ha"]),
                "max_water_stress": float(dqn["max_water_stress"]),
                "max_nitrogen_stress": float(dqn["max_nitrogen_stress"]),
                "yield_diff_vs_extension": float(dqn["final_gwad"] - ext_yield),
                "irrigation_diff_vs_extension": float(dqn["irrigation_total_mm"] - ext_irrigation),
                "nitrogen_diff_vs_extension": float(dqn["fertilizer_total_kg_ha"] - ext_n),
                "status": "available",
            }
        )

    if not summary_rows:
        summary_rows.append(
            {
                "seed": 0,
                "source": "",
                "final_gwad": None,
                "final_cwad": None,
                "irrigation_mm": None,
                "nitrogen_kg_ha": None,
                "max_water_stress": None,
                "max_nitrogen_stress": None,
                "yield_diff_vs_extension": None,
                "irrigation_diff_vs_extension": None,
                "nitrogen_diff_vs_extension": None,
                "status": "missing",
            }
        )

    summary_rows.append(
        {
            "seed": 1,
            "source": "",
            "final_gwad": None,
            "final_cwad": None,
            "irrigation_mm": None,
            "nitrogen_kg_ha": None,
            "max_water_stress": None,
            "max_nitrogen_stress": None,
            "yield_diff_vs_extension": None,
            "irrigation_diff_vs_extension": None,
            "nitrogen_diff_vs_extension": None,
            "status": "missing_seed_result",
        }
    )
    return pd.DataFrame(summary_rows)


def md_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for c in view.columns:
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if isinstance(x, float) else str(x)))
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, r in view.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory()
    summary = build_summary()
    inventory.to_csv(OUT_DIR / "018_07_sy2014_seed_inventory.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "018_07_sy2014_seed_stability_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 018_07 SY2014 DQN 跨 seed 稳定性复核记录",
        "",
        "## 结论",
        "",
        "SY2014 目前只有 seed0 的明确 DQN 结果证据，没有发现 seed1 的训练输出或 checkpoint 评估结果。因此当前不能判断 SY2014 是否跨 seed 稳定成功。",
        "",
        "## 已有证据盘点",
        "",
        md_table(inventory),
        "",
        "## 当前可用 seed 结果",
        "",
        md_table(summary),
        "",
        "## 当前判断",
        "",
        "1. `seed0` 证据很强：DQN 产量约 11216 kg/ha，高于官方推广 expert 11077 kg/ha；灌溉约 120 mm，低于官方推广 expert 266.1 mm；施氮 300 kg/ha，与官方推广 expert 基本相同。",
        "2. `seed1` 证据缺失：现在不是 seed1 表现不好，而是我们根本还没有对应结果。",
        "3. 因此，SY2014 当前应标记为 `promising but missing seed1 evidence`，不能直接升级为稳定成功案例。",
        "",
        "## 如果要最小补充实验",
        "",
        "最小补充应是：在 SY2014 现有框架下补跑一个 `seed1`，不必先加长训练，也不必先改奖励函数。先做与 seed0 同长度、同设置的最小复现实验即可。",
        "",
        "## 输出文件",
        "",
        f"- inventory: `{(OUT_DIR / '018_07_sy2014_seed_inventory.csv').relative_to(PROJECT_ROOT)}`",
        f"- summary: `{(OUT_DIR / '018_07_sy2014_seed_stability_summary.csv').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(inventory.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

