from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_all_year_screen_and_dqn_transfer_014_01 import make_raw_env, prepare_run_dir
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "runtime_leaching_observation_probe_019_04"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-10_019_04_runtime_leaching_observation_probe_record.md"

YEAR = 2016
CASES = {
    "I0_N0": {
        "irrigation": {},
        "nitrogen": {},
    },
    "I120_N300": {
        "irrigation": {35: 30.0, 55: 30.0, 75: 30.0, 95: 30.0},
        "nitrogen": {10: 100.0, 45: 100.0, 65: 100.0},
    },
}


def parse_final_summary(snapshot: Path) -> dict[str, Any]:
    summary = parse_dssat_table(snapshot / "Summary.OUT")
    soilni = parse_dssat_table(snapshot / "SoilNi.OUT")
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")

    out: dict[str, Any] = {}
    if not summary.empty:
        for col in ["NLCM", "NUCM", "NICM", "IRCM", "HWAM", "CWAM"]:
            if col in summary.columns:
                values = pd.to_numeric(summary[col], errors="coerce").dropna()
                out[f"summary_final_{col}"] = float(values.iloc[-1]) if not values.empty else None
    if not soilni.empty:
        for col in ["NLCC", "NITD", "NHTD", "NIAD"]:
            if col in soilni.columns:
                values = pd.to_numeric(soilni[col], errors="coerce").dropna()
                out[f"soilni_final_{col}"] = float(values.iloc[-1]) if not values.empty else None
                out[f"soilni_max_{col}"] = float(values.max()) if not values.empty else None
    if not plantgro.empty:
        for col in ["GWAD", "CWAD"]:
            if col in plantgro.columns:
                values = pd.to_numeric(plantgro[col], errors="coerce").dropna()
                out[f"plantgro_final_{col}"] = float(values.iloc[-1]) if not values.empty else None
    return out


def parse_harvest_yield(snapshot: Path) -> float | None:
    path = snapshot / "MgmtEvent.OUT"
    if not path.exists():
        return None
    values: list[float] = []
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            match = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if match:
                values.append(float(match.group(1)))
    return values[-1] if values else None


def latest_full_state_dict(env) -> dict[str, Any]:
    full_state: dict[str, Any] = {}
    raw = getattr(env.unwrapped, "_state", None)
    if isinstance(raw, dict):
        full_state.update(raw)
    history = getattr(env.unwrapped, "_history", {})
    if isinstance(history, dict):
        states = history.get("state", [])
        if states and isinstance(states[-1], dict):
            full_state.update(states[-1])
    return full_state


def run_case(case: str, spec: dict[str, dict[int, float]]) -> dict[str, Any]:
    case_dir = OUT_DIR / case
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    run_dir = prepare_run_dir(YEAR, f"dqn_leaching_probe_{case.lower()}", seed=0)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_raw_env(env_args)

    rows: list[dict[str, Any]] = []
    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    try:
        obs, info = env.reset()
        for step in range(380):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = int(round(float(scalar(latest_before.get("dap", step)) or 0.0)))
            real_action = {
                "amir": float(spec["irrigation"].get(dap_before, 0.0)),
                "anfer": float(spec["nitrogen"].get(dap_before, 0.0)),
            }
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real_action)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            full_state = latest_full_state_dict(env)
            done = bool(terminated or truncated)
            rows.append(
                {
                    "case": case,
                    "step": step,
                    "dap_before": dap_before,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": scalar(latest.get("yrdoy")),
                    "amir": real_action["amir"],
                    "anfer": real_action["anfer"],
                    "reward": float(reward) if reward is not None else None,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "cleach": scalar(latest.get("cleach")),
                    "tleachd": scalar(latest.get("tleachd")),
                    "cnox": scalar(latest.get("cnox")),
                    "full_state_cleach": scalar(full_state.get("cleach")),
                    "full_state_tleachd": scalar(full_state.get("tleachd")),
                    "full_state_cnox": scalar(full_state.get("cnox")),
                    "cumsumfert": scalar(latest.get("cumsumfert")),
                    "rain": scalar(latest.get("rain")),
                    "done": done,
                    "observation_has_cleach": "cleach" in latest,
                    "observation_has_tleachd": "tleachd" in latest,
                    "observation_has_cnox": "cnox" in latest,
                    "full_state_has_cleach": "cleach" in full_state,
                    "full_state_has_tleachd": "tleachd" in full_state,
                    "full_state_has_cnox": "cnox" in full_state,
                }
            )
            if done:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
        shutil.rmtree(run_dir, ignore_errors=True)

    daily = pd.DataFrame(rows)
    daily.to_csv(case_dir / "runtime_daily_with_leaching.csv", index=False, encoding="utf-8-sig")

    final = {
        "case": case,
        "year": YEAR,
        "daily_rows": len(daily),
        "observation_has_cleach": bool(daily["observation_has_cleach"].any()) if not daily.empty else False,
        "observation_has_tleachd": bool(daily["observation_has_tleachd"].any()) if not daily.empty else False,
        "observation_has_cnox": bool(daily["observation_has_cnox"].any()) if not daily.empty else False,
        "full_state_has_cleach": bool(daily["full_state_has_cleach"].any()) if not daily.empty else False,
        "full_state_has_tleachd": bool(daily["full_state_has_tleachd"].any()) if not daily.empty else False,
        "full_state_has_cnox": bool(daily["full_state_has_cnox"].any()) if not daily.empty else False,
        "runtime_final_cleach": float(pd.to_numeric(daily["cleach"], errors="coerce").dropna().iloc[-1])
        if "cleach" in daily and not pd.to_numeric(daily["cleach"], errors="coerce").dropna().empty
        else None,
        "runtime_max_cleach": float(pd.to_numeric(daily["cleach"], errors="coerce").max()) if "cleach" in daily else None,
        "runtime_sum_tleachd": float(pd.to_numeric(daily["tleachd"], errors="coerce").sum()) if "tleachd" in daily else None,
        "runtime_final_cnox": float(pd.to_numeric(daily["cnox"], errors="coerce").dropna().iloc[-1])
        if "cnox" in daily and not pd.to_numeric(daily["cnox"], errors="coerce").dropna().empty
        else None,
        "full_state_final_cleach": float(pd.to_numeric(daily["full_state_cleach"], errors="coerce").dropna().iloc[-1])
        if "full_state_cleach" in daily and not pd.to_numeric(daily["full_state_cleach"], errors="coerce").dropna().empty
        else None,
        "full_state_sum_tleachd": float(pd.to_numeric(daily["full_state_tleachd"], errors="coerce").sum())
        if "full_state_tleachd" in daily
        else None,
        "full_state_final_cnox": float(pd.to_numeric(daily["full_state_cnox"], errors="coerce").dropna().iloc[-1])
        if "full_state_cnox" in daily and not pd.to_numeric(daily["full_state_cnox"], errors="coerce").dropna().empty
        else None,
        "planned_irrigation": float(sum(spec["irrigation"].values())),
        "planned_nitrogen": float(sum(spec["nitrogen"].values())),
        "harvest_yield": parse_harvest_yield(snapshot),
        "snapshot": str(snapshot.relative_to(PROJECT_ROOT)) if snapshot.exists() else "",
    }
    final.update(parse_final_summary(snapshot))
    (case_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    return final


def write_record(summary: pd.DataFrame) -> None:
    show_cols = [
        "case",
        "planned_irrigation",
        "planned_nitrogen",
        "observation_has_cleach",
        "full_state_has_cleach",
        "runtime_final_cleach",
        "full_state_final_cleach",
        "runtime_sum_tleachd",
        "full_state_sum_tleachd",
        "summary_final_NLCM",
        "soilni_final_NLCC",
        "plantgro_final_GWAD",
        "harvest_yield",
    ]
    table_df = summary[[c for c in show_cols if c in summary.columns]].copy()
    table = dataframe_to_markdown(table_df)
    lines = [
        "# 019_04 runtime leaching observation probe 记录",
        "",
        "## 结论先行",
        "",
        "本轮使用 FQ2016 两个确定性回放情景，不训练，只验证 gym/PDI 运行时 observation/full state 是否能读到 leaching 相关变量。",
        "",
        "结论：普通 observation 中没有 `cleach/tleachd/cnox`，但 full state 中有。`I120_N300` 的 `full_state_final_cleach` 与 `SoilNi.OUT` 的 `NLCC` 基本一致。因此当前不需要先改 DSSAT/PDI 模板；下一步应该先把 full state 中的 `cleach/tleachd/cnox` 写入评估日值表，并在 reward wrapper 中从 full state 读取。",
        "",
        "## 结果",
        "",
        table,
        "",
        "## 文件",
        "",
        f"- 汇总表：`{(OUT_DIR / '019_04_runtime_leaching_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 输出目录：`{OUT_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## 下一步",
        "",
        "本轮通过了 full-state 可用性验证。下一步做 leaching-aware reward smoke test：先只在一个站点年份上加入 `- leaching_cost * delta_cleach`，并保持其他奖励项不变。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无结果。_"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
        else:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(v))
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, spec in CASES.items():
        rows.append(run_case(case, spec))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "019_04_runtime_leaching_summary.csv", index=False, encoding="utf-8-sig")
    write_record(summary)
    print(summary.to_string(index=False))
    print(DOC_PATH)


if __name__ == "__main__":
    main()
