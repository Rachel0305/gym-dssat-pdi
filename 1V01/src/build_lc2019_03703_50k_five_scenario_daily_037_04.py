from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_lca_early_starter_cap_maskableppo_smoke_037_03 as cap037
from build_relaxed_success_five_scenario_daily_evidence_027_05 import Case, build_case, plot_daily
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
TASK = "037_04_lc2019_early_starter_cap_50k_five_scenario_daily"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
OUT = ROOT / "benchmark_results" / TASK
FIG = OUT / "figures"
TAB = OUT / "tables"
SNAP = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"

STATION = "LCA"
SITE = "LC"
YEAR = 2019
SEED = 0
CHECKPOINT = 50_000
MODEL = (
    ROOT
    / "benchmark_results"
    / "037_03_lca_early_starter_cap_maskableppo_smoke_clean_retrain"
    / "models"
    / "LCA"
    / "LCA_half_split_stress_aware_maskableppo_seed0_ckpt50000.zip"
)

BASE34 = ROOT / "benchmark_results" / "034_00_multisite_input_ic1_four_baseline_rebuild" / "snapshots" / "LCA" / str(YEAR)
PPO_SNAPSHOT = SNAP / "LCA" / str(YEAR) / "rl_candidate_seed0_ckpt50000_early_starter_cap"

REQUIRED = ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]


def ensure_dirs() -> None:
    for path in [FIG, TAB, SNAP, OUT / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    shutil.copy2(batch.CONFIG, OUT / "configs" / batch.CONFIG.name)


def find_attr(obj: Any, attr: str) -> Any | None:
    seen: set[int] = set()
    cur = obj
    for _ in range(12):
        if id(cur) in seen:
            return None
        seen.add(id(cur))
        if hasattr(cur, attr):
            try:
                return getattr(cur, attr)
            except Exception:
                return None
        nxt = None
        for name in ["env", "unwrapped", "current_env"]:
            try:
                cand = getattr(cur, name)
                if cand is not cur:
                    nxt = cand
                    break
            except Exception:
                pass
        if nxt is None:
            return None
        cur = nxt
    return None


def copy_required_snapshot(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        source = src / name
        if not source.exists():
            raise FileNotFoundError(f"missing {source}")
        shutil.copy2(source, dst / name)


def run_ppo_and_save_snapshot() -> pd.DataFrame:
    cap037.patch_base_wrapper()
    config = batch.load_config()
    config = json.loads(json.dumps(config))
    config["seed"] = SEED
    config["action_safety"]["early_starter_cap_enabled"] = True
    config["action_safety"]["early_starter_cap_dap_max"] = cap037.EARLY_DAP_MAX
    config["action_safety"]["early_starter_irrigation_cap_mm"] = cap037.EARLY_IRRIGATION_CAP_MM
    config["action_safety"]["early_starter_n_cap_kg_ha"] = cap037.EARLY_N_CAP_KG_HA
    split = batch.load_split()
    selection = batch.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(config, OUT / "configs" / "037_04_config_snapshot.yaml")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "037_04_resolved_env_config.yaml")

    model = MaskablePPO.load(str(MODEL), device="cpu")
    env = base032.make_env(
        config,
        env_config,
        STATION,
        YEAR,
        SEED,
        "LCA_2019_037_04_seed0_ckpt50000_eval",
        evaluation=True,
    )
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(config["runtime"]["max_steps"]):
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base032.latest_observation_dict(env, obs, info)
            last_action = dict(getattr(env, "last_action_info", {}))
            records.append(
                {
                    "step": step_count + 1,
                    "reward": float(reward),
                    "done": done,
                    "grnwt": float(latest.get("grnwt", np.nan)),
                    "topwt": float(latest.get("topwt", np.nan)),
                    "swfac": float(latest.get("swfac", np.nan)),
                    "nstres": float(latest.get("nstres", np.nan)),
                    **last_action,
                }
            )
            step_count += 1
        tmp = find_attr(env, "_tmp_folder")
        if tmp is None:
            raise RuntimeError("could not locate DSSAT _tmp_folder for PPO evaluation")
        copy_required_snapshot(Path(tmp), PPO_SNAPSHOT)
    finally:
        env.close()
    return pd.DataFrame(records)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "无记录"
    show = df.head(max_rows).copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = pd.to_numeric(show[col], errors="coerce").round(4)
    show = show.astype(object).where(pd.notna(show), "")
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    eval_trace = run_ppo_and_save_snapshot()
    eval_trace.to_csv(TAB / "037_04_lc2019_ppo_eval_trace.csv", index=False, encoding="utf-8-sig")

    snapshots = {
        "null": BASE34 / "null",
        "recorded_farmer": BASE34 / "recorded_farmer_template",
        "dssat_auto": BASE34 / "dssat_auto",
        "official_extension_expert": BASE34 / "official_extension_expert",
        "rl_candidate": PPO_SNAPSHOT,
    }
    checks = []
    for scenario, path in snapshots.items():
        checks.append(
            {
                "scenario": scenario,
                "snapshot_path": path.relative_to(ROOT).as_posix() if path.exists() else str(path),
                "exists": path.exists(),
                "required_complete": all((path / name).exists() for name in REQUIRED),
            }
        )
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(TAB / "037_04_snapshot_checks.csv", index=False, encoding="utf-8-sig")
    if not bool(checks_df["required_complete"].all()):
        raise RuntimeError("snapshot checks failed; see 037_04_snapshot_checks.csv")

    case = Case(
        site=SITE,
        station=STATION,
        year=YEAR,
        seed=SEED,
        checkpoint=CHECKPOINT,
        model_path=MODEL,
        snapshots=snapshots,
        selection_source=PROMPT,
        note="037_03 LCA early starter cap clean retrain seed0 checkpoint 50k; frozen deterministic LC2019 evaluation.",
    )

    import build_relaxed_success_five_scenario_daily_evidence_027_05 as old

    old.FIG = FIG
    daily, summary, evidence_checks = build_case(case, algorithm="MaskablePPO")
    figures = plot_daily(daily, case, algorithm="MaskablePPO")

    daily_path = TAB / "037_04_lc2019_50k_early_cap_five_scenario_daily.csv"
    summary_path = TAB / "037_04_lc2019_50k_early_cap_five_scenario_summary.csv"
    checks_path = TAB / "037_04_lc2019_50k_early_cap_evidence_checks.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(evidence_checks).to_csv(checks_path, index=False, encoding="utf-8-sig")

    rl = summary[summary["scenario"].eq("rl_candidate")]
    lines = [
        "# 037_04：LC2019 early starter cap 50K 五情景日过程图记录",
        "",
        "## 状态",
        "",
        "- 完成。",
        "- 本任务不训练，只对 037_03 的 LCA seed0 50K checkpoint 做冻结确定性评估并绘图。",
        "- 四基线 snapshot 复用已有 LC2019 基线；RL candidate snapshot 本轮重新生成。",
        "",
        "## RL candidate 摘要",
        "",
        markdown_table(rl),
        "",
        "## Snapshot 检查",
        "",
        markdown_table(checks_df),
        "",
        "## 输出文件",
        "",
        f"- daily CSV：`{daily_path.relative_to(ROOT).as_posix()}`",
        f"- summary CSV：`{summary_path.relative_to(ROOT).as_posix()}`",
        f"- evidence checks：`{checks_path.relative_to(ROOT).as_posix()}`",
        *[f"- figure：`{path.relative_to(ROOT).as_posix()}`" for path in figures],
        "",
        "## 初步说明",
        "",
        "- 037_03 50K 的 LC2019 策略为小剂量 starter：DAP1 灌溉 30mm、DAP2 施氮 40kg/ha。",
        "- 本图用于检查该小剂量 starter 是否导致明显水氮胁迫、产量损失或过程异常。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    result = {
        "task": TASK,
        "status": "ok",
        "daily_csv": daily_path.relative_to(ROOT).as_posix(),
        "summary_csv": summary_path.relative_to(ROOT).as_posix(),
        "figures": [path.relative_to(ROOT).as_posix() for path in figures],
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "037_04_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
