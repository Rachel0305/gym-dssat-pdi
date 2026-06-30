from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_evaluate import latest_observation_dict, scalar
from run_hla2010_dqn_unified_recheck_014_03 import WINDOWS, make_env


SOURCE_ROOT = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla2010_dqn_unified_recheck_014_03"
    / "2010"
)
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_noop_q_diagnosis_014_04"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_014_04_hla2010_dqn_noop_q_diagnosis_record.md"

CASES = {
    "free_daily_200steps": SOURCE_ROOT / "free_daily_seed0_200steps",
    "free_daily_5000steps": SOURCE_ROOT / "free_daily_seed0_5000steps",
}


def q_values_for_obs(model: Any, obs: Any) -> np.ndarray:
    import torch

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
    return q_values.detach().cpu().numpy().reshape(-1)


def diagnose_case(label: str, run_dir: Path, max_steps: int = 150) -> pd.DataFrame:
    from stable_baselines3 import DQN

    env_args_path = run_dir / "env_args.json"
    model_path = run_dir / "models" / "dqn_hla2010_unified_recheck.zip"
    if not env_args_path.exists():
        raise FileNotFoundError(env_args_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    env_args = json.loads(env_args_path.read_text(encoding="utf-8"))
    env = make_env(env_args, WINDOWS["free_daily"])
    model = DQN.load(str(model_path), env=env, print_system_info=False)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(max_steps):
            q = q_values_for_obs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            sorted_q = np.sort(q)
            rows.append(
                {
                    "case": label,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "chosen_action": action_int,
                    "q_action0_noop": float(q[0]) if len(q) > 0 else np.nan,
                    "q_action1_irrig": float(q[1]) if len(q) > 1 else np.nan,
                    "q_action2_fert": float(q[2]) if len(q) > 2 else np.nan,
                    "q_action3_both": float(q[3]) if len(q) > 3 else np.nan,
                    "q_best": float(np.max(q)) if len(q) else np.nan,
                    "q_second": float(sorted_q[-2]) if len(q) >= 2 else np.nan,
                    "q_gap_best_second": float(sorted_q[-1] - sorted_q[-2]) if len(q) >= 2 else np.nan,
                    "safe_amir": float(env.last_safe_real_action.get("amir", 0.0)),
                    "safe_anfer": float(env.last_safe_real_action.get("anfer", 0.0)),
                    "used_irrigation": float(env.used_irrigation),
                    "used_nitrogen": float(env.used_nitrogen),
                    "reward": float(reward),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": bool(terminated or truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, sub in daily.groupby("case", sort=False):
        q_cols = ["q_action0_noop", "q_action1_irrig", "q_action2_fert", "q_action3_both"]
        action_counts = sub["chosen_action"].value_counts().to_dict()
        first_ops = sub[(sub["safe_amir"] > 0) | (sub["safe_anfer"] > 0)].head(8)
        rows.append(
            {
                "case": case,
                "steps": len(sub),
                "final_grnwt": float(sub["grnwt"].dropna().iloc[-1]) if not sub.empty else np.nan,
                "final_topwt": float(sub["topwt"].dropna().iloc[-1]) if not sub.empty else np.nan,
                "irrigation_total": float(sub["safe_amir"].sum()),
                "nitrogen_total": float(sub["safe_anfer"].sum()),
                "max_swfac": float(sub["swfac"].max()),
                "max_nstres": float(sub["nstres"].max()),
                "action_counts": json.dumps({int(k): int(v) for k, v in action_counts.items()}, ensure_ascii=False),
                "mean_q_noop": float(sub["q_action0_noop"].mean()),
                "mean_q_irrig": float(sub["q_action1_irrig"].mean()),
                "mean_q_fert": float(sub["q_action2_fert"].mean()),
                "mean_q_both": float(sub["q_action3_both"].mean()),
                "mean_q_gap_best_second": float(sub["q_gap_best_second"].mean()),
                "first_operations": first_ops[["step", "dap", "chosen_action", "safe_amir", "safe_anfer"]].to_dict("records"),
            }
        )
    return pd.DataFrame(rows)


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无数据。_"
    keep = [
        "case",
        "steps",
        "final_grnwt",
        "irrigation_total",
        "nitrogen_total",
        "max_swfac",
        "max_nstres",
        "action_counts",
        "mean_q_noop",
        "mean_q_irrig",
        "mean_q_fert",
        "mean_q_both",
        "mean_q_gap_best_second",
    ]
    show = df[[c for c in keep if c in df.columns]].copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            show[col] = show[col].astype(str)
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_record(summary: pd.DataFrame) -> None:
    lines = [
        "# 014_04 HLA2010 DQN no-op 退化 Q 值诊断记录",
        "",
        "## 目的",
        "",
        "014_03 中 200 step smoke 有操作，但 5K 确定性评估退化为 no-op。本轮不训练，只读取两个模型在同一环境轨迹上的 Q 值。",
        "",
        "## 结果摘要",
        "",
        df_to_md(summary),
        "",
        "## 初步判读",
        "",
    ]
    if not summary.empty:
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['case']}: 动作计数 {row['action_counts']}，"
                f"I={row['irrigation_total']:.1f}, N={row['nitrogen_total']:.1f}, "
                f"final GRNWT={row['final_grnwt']:.1f}。"
            )
    lines += [
        "",
        "如果 5K 模型 action_counts 几乎全是 0，且 mean_q_noop 高于其他动作，说明不是动作链路问题，而是 DQN 学到 no-op 价值最高。",
        "如果 Q 值差距很小，则说明模型没有学清楚，可能需要改奖励尺度、探索策略或动作/时间窗口，而不是继续直接加 seed。",
        "",
        "## 输出文件",
        "",
        f"- 日值 Q 表：`{(OUT_DIR / '014_04_hla2010_dqn_q_values_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- 汇总表：`{(OUT_DIR / '014_04_hla2010_dqn_q_summary.csv').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for label, run_dir in CASES.items():
        frames.append(diagnose_case(label, run_dir))
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(OUT_DIR / "014_04_hla2010_dqn_q_values_daily.csv", index=False, encoding="utf-8-sig")
    summary = summarize(daily)
    summary.to_csv(OUT_DIR / "014_04_hla2010_dqn_q_summary.csv", index=False, encoding="utf-8-sig")
    write_record(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
