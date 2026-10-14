from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
from build_relaxed_success_five_scenario_daily_evidence_027_05 import Case, build_case, plot_daily
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "032_24_lc2019_75k_ppo_five_scenario_daily_figure.md"
OUT = ROOT / "benchmark_results" / "032_24_lc2019_75k_ppo_five_scenario_daily_figure"
FIG = OUT / "figures"
TAB = OUT / "tables"
SNAP = OUT / "snapshots"
DOC = ROOT / "docs" / "032_24_lc2019_75k_ppo_five_scenario_daily_figure_record.md"

STATION = "LCA"
SITE = "LC"
YEAR = 2019
SEED = 0
CHECKPOINT = 75_000
MODEL = (
    ROOT
    / "benchmark_results"
    / "032_22_five_site_half_split_stress_aware_maskableppo_batch"
    / "models"
    / "LCA"
    / "LCA_half_split_stress_aware_maskableppo_seed0_ckpt75000.zip"
)

BASE35 = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134" / "snapshots" / "LCA" / "2019"
AUTO36 = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "snapshots" / "LCA" / "2019" / "dssat_auto"
PPO_SNAPSHOT = SNAP / "LCA" / "2019" / "rl_candidate_seed0_ckpt75000"

REQUIRED = ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]


def ensure_dirs() -> None:
    for path in [FIG, TAB, SNAP, OUT / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    shutil.copy2(batch.CONFIG, OUT / "configs" / batch.CONFIG.name)


def find_attr(obj: Any, attr: str) -> Any | None:
    seen: set[int] = set()
    cur = obj
    for _ in range(10):
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
    config = batch.load_config()
    split = batch.load_split()
    selection = batch.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    model = MaskablePPO.load(str(MODEL), device="cpu")
    env = base.make_env(
        config,
        env_config,
        STATION,
        YEAR,
        SEED,
        "LCA_2019_032_24_seed0_ckpt75000_eval",
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
            latest = base.latest_observation_dict(env, obs, info)
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


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_空表_"
    show = df.head(max_rows).copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = pd.to_numeric(show[col], errors="coerce").round(4)
    show = show.astype(object).where(pd.notna(show), "")
    header = "| " + " | ".join(map(str, show.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in show.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    ensure_dirs()
    eval_trace = run_ppo_and_save_snapshot()
    eval_trace.to_csv(TAB / "032_24_lc2019_ppo_eval_trace.csv", index=False, encoding="utf-8-sig")

    snapshots = {
        "null": BASE35 / "null",
        "recorded_farmer": BASE35 / "recorded_farmer_template_02705",
        "dssat_auto": AUTO36,
        "official_extension_expert": BASE35 / "official_extension_expert",
        "rl_candidate": PPO_SNAPSHOT,
    }
    checks: list[dict[str, Any]] = []
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
    snapshot_checks_path = TAB / "032_24_snapshot_checks.csv"
    checks_df.to_csv(snapshot_checks_path, index=False, encoding="utf-8-sig")
    if not bool(checks_df["required_complete"].all()):
        raise RuntimeError("snapshot checks failed; see 032_24_snapshot_checks.csv")

    case = Case(
        site=SITE,
        station=STATION,
        year=YEAR,
        seed=SEED,
        checkpoint=CHECKPOINT,
        model_path=MODEL,
        snapshots=snapshots,
        selection_source=PROMPT,
        note="032_22 LC half-split stress-aware free-timing MaskablePPO seed0 checkpoint 75k; snapshot regenerated by frozen deterministic evaluation.",
    )

    import build_relaxed_success_five_scenario_daily_evidence_027_05 as old

    old.FIG = FIG
    daily, summary, evidence_checks = build_case(case, algorithm="MaskablePPO")
    figures = plot_daily(daily, case, algorithm="MaskablePPO")

    daily_path = TAB / "032_24_lc2019_75k_ppo_five_scenario_daily.csv"
    summary_path = TAB / "032_24_lc2019_75k_ppo_five_scenario_summary.csv"
    checks_path = TAB / "032_24_lc2019_75k_ppo_daily_evidence_checks.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(evidence_checks).to_csv(checks_path, index=False, encoding="utf-8-sig")

    rl = summary[summary["scenario"].eq("rl_candidate")].copy()
    lines = [
        "# 032_24 LC2019 75K PPO 五情景日过程图记录",
        "",
        "## 结论先说",
        "",
        "- 状态：完成。",
        "- 本任务没有训练，只对 032_22 的 LC seed0 75K checkpoint 做冻结确定性回放，并保存 DSSAT snapshot。",
        "- 代表年份：LC2019。选择理由：032_22 中 LC 75K/100K 在验证年均为 10/10 any-metric win；LC2019 同时具有产量和 PFP_N 相对四情景最高值的小幅优势，适合作为 LC 决策过程展示样本。",
        "",
        "## PPO 候选摘要",
        "",
        md_table(rl),
        "",
        "## Snapshot 完整性检查",
        "",
        md_table(checks_df),
        "",
        "## 输出文件",
        "",
        f"- daily CSV：`{daily_path.relative_to(ROOT).as_posix()}`",
        f"- summary CSV：`{summary_path.relative_to(ROOT).as_posix()}`",
        f"- evidence checks：`{checks_path.relative_to(ROOT).as_posix()}`",
        f"- snapshot checks：`{snapshot_checks_path.relative_to(ROOT).as_posix()}`",
        f"- PPO snapshot：`{PPO_SNAPSHOT.relative_to(ROOT).as_posix()}`",
        "",
        "## 图件",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{fig.relative_to(ROOT).as_posix()}`")
    lines += [
        "",
        "## 备注",
        "",
        "- recorded_farmer 使用 031_35 中的 `recorded_farmer_template_02705` snapshot。",
        "- dssat_auto 使用 031_36 补齐后的 snapshot。",
        "- 所有五情景均从 DSSAT snapshot 解析日过程，避免混用不完整 daily 表。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")

    result = {
        "task": "032_24_lc2019_75k_ppo_five_scenario_daily_figure",
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "daily_csv": daily_path.relative_to(ROOT).as_posix(),
        "summary_csv": summary_path.relative_to(ROOT).as_posix(),
        "figures": [fig.relative_to(ROOT).as_posix() for fig in figures],
    }
    (OUT / "032_24_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
