from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import (
    OUT_DIR,
    install_official_reward_module,
    parse_events,
    prepare_case_at,
)


DIAG_DIR = OUT_DIR / "action_channel_diagnosis"
RAW_FILES = ["PlantGro.OUT", "PlantN.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_final_plantgro(path: Path) -> dict[str, float]:
    out = {"final_gwad": np.nan, "final_cwad": np.nan, "final_dap": np.nan}
    if not path.exists():
        return out
    header = None
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("@"):
            header = s.replace("@", "", 1).split()
            continue
        if header and len(s) and s[0].isdigit():
            parts = s.split()
            if len(parts) < len(header):
                continue
            if "DAP" in header:
                out["final_dap"] = float(parts[header.index("DAP")])
            if "GWAD" in header:
                out["final_gwad"] = float(parts[header.index("GWAD")])
            if "CWAD" in header:
                out["final_cwad"] = float(parts[header.index("CWAD")])
    return out


def parse_summary_yield(path: Path) -> dict[str, float]:
    out = {"summary_hwams": np.nan, "summary_cwams": np.nan}
    if not path.exists():
        return out
    header = None
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("@"):
            header = s.replace("@", "", 1).split()
            continue
        if header and s and s[0].isdigit():
            parts = s.split()
            if len(parts) < len(header):
                continue
            if "HWAM" in header:
                out["summary_hwams"] = float(parts[header.index("HWAM")])
            if "CWAM" in header:
                out["summary_cwams"] = float(parts[header.index("CWAM")])
    return out


def simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.astype(str).values.tolist()]
    return "\n".join([header, sep, *rows])


def freeze_jinja_controls_for_reported_management(text: str) -> str:
    """Turn the copied Jinja-capable MZX into explicit reported management.

    This preserves the original R/R diagnostic after prepare_case began adding
    official Jinja placeholders for normal PPO runs.
    """
    return (
        text.replace("{{ wther }}", "M")
        .replace("{{ plant }}", "R")
        .replace("{{ irrig }}", "R")
        .replace("{{ ferti }}", "R")
    )


def run_episode(year: int, scenario: str, forced: bool, linked_management: bool = False, max_steps: int = 180) -> dict[str, Any]:
    install_official_reward_module()
    import gym
    from sb3_wrapper import GymDssatWrapper

    case_dir = DIAG_DIR / str(year) / scenario
    prepare_case_at(year, case_dir)
    if not linked_management:
        env_args0 = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
        filex0 = Path(env_args0["fileX_template_path"])
        text0 = filex0.read_text(encoding="latin1", errors="ignore")
        text0 = freeze_jinja_controls_for_reported_management(text0)
        filex0.write_text(text0, encoding="latin1")
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        for step in range(max_steps):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step))))
            if forced:
                real = {
                    "amir": 10.0 if dap in [49, 70, 95] else 0.0,
                    "anfer": 165.0 if dap in [1] else 0.0,
                }
            else:
                real = {"amir": 0.0, "anfer": 0.0}
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, real)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "step": step,
                    "dap_before": dap,
                    "amir": real["amir"],
                    "anfer": real["anfer"],
                    "reward": reward,
                    "dap_after": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    daily.to_csv(case_dir / "daily_action_trace.csv", index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "year": year,
        "scenario": scenario,
        "forced": forced,
        "linked_management": linked_management,
        "steps": len(rows),
    }
    summary.update(parse_events(snapshot / "MgmtEvent.OUT"))
    summary.update(parse_final_plantgro(snapshot / "PlantGro.OUT"))
    summary.update(parse_summary_yield(snapshot / "Summary.OUT"))
    summary["nonzero_action_rows"] = int(((daily["amir"] > 0) | (daily["anfer"] > 0)).sum()) if len(daily) else 0
    summary["total_action_irrigation"] = float(daily["amir"].sum()) if len(daily) else 0.0
    summary["total_action_n"] = float(daily["anfer"].sum()) if len(daily) else 0.0
    summary["raw_hashes"] = {
        name: sha256_file(snapshot / name) if (snapshot / name).exists() else None
        for name in RAW_FILES
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    year = 2010
    null_summary = run_episode(year, "null_action", forced=False)
    forced_summary = run_episode(year, "forced_action", forced=True)
    linked_null_summary = run_episode(year, "linked_null_action", forced=False, linked_management=True)
    linked_forced_summary = run_episode(year, "linked_forced_action", forced=True, linked_management=True)

    rows = [null_summary, forced_summary, linked_null_summary, linked_forced_summary]
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "raw_hashes"}
        for name, digest in row["raw_hashes"].items():
            flat[f"sha256_{name}"] = digest
        flat_rows.append(flat)
    summary_df = pd.DataFrame(flat_rows)
    summary_df.to_csv(DIAG_DIR / "action_channel_diagnosis_summary.csv", index=False, encoding="utf-8-sig")

    compare_rows = []
    for name in RAW_FILES:
        compare_rows.append(
            {
                "file": name,
                "rr_null_sha256": null_summary["raw_hashes"].get(name),
                "rr_forced_sha256": forced_summary["raw_hashes"].get(name),
                "rr_identical": null_summary["raw_hashes"].get(name) == forced_summary["raw_hashes"].get(name),
                "ll_null_sha256": linked_null_summary["raw_hashes"].get(name),
                "ll_forced_sha256": linked_forced_summary["raw_hashes"].get(name),
                "ll_identical": linked_null_summary["raw_hashes"].get(name)
                == linked_forced_summary["raw_hashes"].get(name),
            }
        )
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(DIAG_DIR / "raw_file_hash_comparison.csv", index=False, encoding="utf-8-sig")

    readme = [
        "# 011_02 action-channel diagnosis",
        "",
        "Same 2010 input was run twice through gym/PDI:",
        "",
        "- `null_action`: all actions zero",
        "- `forced_action`: DAP 1 N=165, DAP 49/70/95 irrigation=10 mm",
        "- `linked_null_action`: same as null, but `IRRIG=L, FERTI=L`",
        "- `linked_forced_action`: same as forced, but `IRRIG=L, FERTI=L`",
        "",
        "## Summary",
        "",
        simple_markdown_table(summary_df.drop(columns=[c for c in summary_df.columns if c.startswith("sha256_")])),
        "",
        "## Raw file hash comparison",
        "",
        simple_markdown_table(compare_df),
        "",
        "If raw files are identical or final yield is unchanged, the gym action channel is not validated for PPO training.",
    ]
    (DIAG_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
