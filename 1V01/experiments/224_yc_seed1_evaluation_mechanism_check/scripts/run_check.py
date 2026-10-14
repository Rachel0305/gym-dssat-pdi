"""Read-only YC seed1 evaluation-mechanism diagnostic.

The script replays the frozen Original seed1 @100K action traces to construct
one shared batch of YC validation observations.  It then evaluates the frozen
Original/Augmented seed1 @100K policies on exactly those observations and
records raw and masked action distributions.  No training or formal-pipeline
file is modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
YCA_SRC = SRC / "055_yca_lowIC_site_transfer"
for import_root in (SRC, YCA_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

OUT = ROOT / "experiments" / "224_yc_seed1_evaluation_mechanism_check"
RESULTS = ROOT / "results"
DOC = ROOT / "docs" / "yc_seed1_evaluation_mechanism_check.md"
PROBABILITY_CSV = RESULTS / "yc_seed1_policy_probability_comparison.csv"
STATE_SUMMARY_CSV = RESULTS / "yc_seed1_policy_state_comparison_summary.csv"
MASK_AUDIT_CSV = RESULTS / "yc_seed1_mask_audit.csv"
GRID_CSV = RESULTS / "yc_seed1_action_grid_mapping.csv"
INTEGRITY_CSV = RESULTS / "yc_seed1_evaluation_mechanism_integrity_checks.csv"
INTERNAL_CSV = RESULTS / "yc_seed1_augmented_internal_phase_summary.csv"
MANIFEST = OUT / "yc_seed1_evaluation_mechanism_check_manifest.json"
RUN_LOG = OUT / "run_log.md"

MANUAL_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
SOURCE_DAILY_ROOT = ROOT / "benchmark_results" / "222YCA_O1_yc_v1_original_seed1" / "daily_outputs" / "YCA"
ORIGINAL_MODEL = ROOT / "benchmark_results" / "222YCA_O1_yc_v1_original_seed1" / "models" / "YCA" / "YCA_half_split_stress_aware_maskableppo_seed1_ckpt100000.zip"
AUGMENTED_MODEL = ROOT / "benchmark_results" / "222YCA_A1_yc_v1_augmented_seed1" / "models" / "YCA" / "YCA_half_split_stress_aware_maskableppo_seed1_ckpt100000.zip"
AUGMENTED_UPDATE_LOG = ROOT / "benchmark_results" / "222YCA_A1_yc_v1_augmented_seed1" / "logs" / "222YCA_A1_training_update_metrics.csv"

YEARS = list(range(2014, 2024))
CHECKPOINT = 100_000
I_LEVELS = [0.0, 15.0, 30.0, 45.0]
N_LEVELS = [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def action_grid() -> list[dict[str, float]]:
    return [
        {"action_index": index, "irrigation": irrigation, "nitrogen": nitrogen}
        for index, (irrigation, nitrogen) in enumerate((i, n) for i in I_LEVELS for n in N_LEVELS)
    ]


def action_label(index: int) -> str:
    item = action_grid()[int(index)]
    return f"I{item['irrigation']:g}/N{item['nitrogen']:g}"


def read_daily(year: int) -> pd.DataFrame:
    path = SOURCE_DAILY_ROOT / f"YCA_{year}_seed1_ckpt{CHECKPOINT}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, keep_default_na=False)
    required = {"dap", "discrete_action_index", "safe_action_amir", "safe_action_anfer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    return df


def setup_shared_env_contract() -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Load the frozen 055/040/042/032 environment contract without training."""
    import ppo_safe_rendering
    import run_055_00_yca_lowIC_expanded_action_maskableppo as original_runner
    import run_all_year_direct_action_safe_ppo as direct_ppo

    engine = original_runner.engine
    engine.TASK_ID = "224"
    engine.TASK_NAME = "yc_seed1_evaluation_mechanism_check"
    engine.BASE_OUT = OUT
    engine.BASE_DOC = OUT / "diagnostic_record.md"
    engine.LOWIC_INPUT_ROOT = MANUAL_INPUT_ROOT
    engine.STATION = "YCA"
    engine.SITES = ["YCA"]
    engine.BINARY_IRRIGATION_LEVELS = list(I_LEVELS)
    engine.BINARY_NITROGEN_LEVELS = list(N_LEVELS)
    engine.base03222.SEED = 1
    engine.base03222.SITE_NAMES["YCA"] = "YC"
    engine.patch_base_module(OUT, OUT / "diagnostic_record.md", 100_000, [25_000, 50_000, 75_000, 100_000])
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = MANUAL_INPUT_ROOT

    config = engine.load_config()
    split = engine.base03222.load_split()
    selection = engine.base03222.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    return config, env_config, engine


def latest(env: Any, obs: Any, info: Any) -> dict[str, Any]:
    return env.last_obs_dict if isinstance(getattr(env, "last_obs_dict", None), dict) else {}


def collect_shared_states(config: dict[str, Any], env_config: dict[str, Any], engine: Any) -> list[dict[str, Any]]:
    """Replay source actions and retain pre-action observations and masks."""
    states: list[dict[str, Any]] = []
    for year in YEARS:
        source = read_daily(year)
        env = engine.base03222.base.make_env(
            config,
            env_config,
            "YCA",
            int(year),
            1,
            f"224_shared_state_{year}",
            evaluation=True,
        )
        try:
            obs, info = env.reset()
            for source_row, row in source.iterrows():
                if bool(getattr(env, "last_obs_dict", {})):
                    obs_dict = dict(env.last_obs_dict)
                else:
                    obs_dict = latest(env, obs, info)
                reported_dap = float(pd.to_numeric(pd.Series([obs_dict.get("dap", np.nan)]), errors="coerce").iloc[0])
                # The production wrapper uses _dap(), which maps non-positive
                # raw DAP values to effective DAP1.  Reuse that exact rule.
                dap = int(env._dap())
                mask = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
                action_index = int(float(row["discrete_action_index"]))
                obs_array = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
                stress_w = float(pd.to_numeric(pd.Series([obs_dict.get("swfac", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                stress_n = float(pd.to_numeric(pd.Series([obs_dict.get("nstres", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                role = "dap1" if dap == 1 else ("dap2_10" if 2 <= dap <= 10 else "later")
                if stress_w > 0.05 or stress_n > 0.05:
                    role += "+stress"
                states.append(
                    {
                        "state_id": f"YCA_{year}_step{source_row:03d}",
                        "year": int(year),
                        "source_row": int(source_row),
                        "source_dap": int(float(row["dap"])),
                        "dap": dap,
                        "reported_dap": reported_dap,
                        "state_role": role,
                        "state_swfac": stress_w,
                        "state_nstres": stress_n,
                        "source_action_index": action_index,
                        "source_selected_action": action_label(action_index),
                        "valid_action_count": int(mask.sum()),
                        "valid_action_indices": ",".join(map(str, np.flatnonzero(mask).tolist())),
                        "mask_hash": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()[:16],
                        "state_hash": hashlib.sha256(obs_array.tobytes()).hexdigest()[:16],
                        "observation": obs_array,
                        "mask": mask,
                    }
                )
                obs, _reward, terminated, truncated, info = env.step(action_index)
                if bool(terminated or truncated):
                    if source_row != len(source) - 1:
                        raise RuntimeError(f"YCA{year}: episode ended at source row {source_row}, expected {len(source) - 1}")
                    break
        finally:
            env.close()
    if not states:
        raise RuntimeError("No shared states were collected")
    return states


def policy_distribution(model: Any, observation: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Return action-net logits, raw probabilities, masked probabilities and argmax."""
    import torch

    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        features = model.policy.extract_features(obs_tensor, model.policy.pi_features_extractor)
        latent_pi = model.policy.mlp_extractor.forward_actor(features)
        raw_logits_t = model.policy.action_net(latent_pi)
        raw_logits = raw_logits_t.detach().cpu().numpy().reshape(-1)
        raw_prob = torch.softmax(raw_logits_t, dim=-1).detach().cpu().numpy().reshape(-1)
        masked_logits = np.where(mask, raw_logits, -1e8)
        masked_prob = torch.softmax(torch.as_tensor(masked_logits, dtype=torch.float32), dim=-1).numpy()
    selected = int(np.argmax(masked_logits))
    return {
        "raw_logits": raw_logits,
        "raw_probability": raw_prob,
        "masked_logits": masked_logits,
        "masked_probability": masked_prob,
        "raw_argmax": int(np.argmax(raw_logits)),
        "masked_argmax": selected,
    }


def extract_policy_rows(states: list[dict[str, Any]]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    models = [
        ("Original", ORIGINAL_MODEL),
        ("Augmented", AUGMENTED_MODEL),
    ]
    rows: list[dict[str, Any]] = []
    for method, model_path in models:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = MaskablePPO.load(str(model_path), device="cpu")
        try:
            for state in states:
                dist = policy_distribution(model, state["observation"], state["mask"])
                predicted, _ = model.predict(state["observation"], action_masks=state["mask"], deterministic=True)
                predicted_index = int(np.asarray(predicted).reshape(-1)[0])
                if predicted_index != dist["masked_argmax"]:
                    raise AssertionError(f"{method} {state['state_id']}: predict != masked argmax")
                for item in action_grid():
                    index = int(item["action_index"])
                    rows.append(
                        {
                            "method": method,
                            "seed": 1,
                            "checkpoint": CHECKPOINT,
                            "year": state["year"],
                            "source_row": state["source_row"],
                            "state_id": state["state_id"],
                            "source_dap": state["source_dap"],
                            "dap": state["dap"],
                            "state_role": state["state_role"],
                            "state_swfac": state["state_swfac"],
                            "state_nstres": state["state_nstres"],
                            "action_index": index,
                            "irrigation": item["irrigation"],
                            "nitrogen": item["nitrogen"],
                            "valid_by_mask": bool(state["mask"][index]),
                            "logit": float(dist["raw_logits"][index]),
                            "probability": float(dist["raw_probability"][index]),
                            "masked_logit": float(dist["masked_logits"][index]),
                            "masked_probability": float(dist["masked_probability"][index]),
                            "raw_argmax_action": int(dist["raw_argmax"]),
                            "masked_argmax_action": int(dist["masked_argmax"]),
                            "is_argmax": bool(index == predicted_index),
                            "selected_action_index": predicted_index,
                            "selected_action": action_label(predicted_index),
                            "source_selected_action": state["source_selected_action"],
                            "valid_action_count": state["valid_action_count"],
                            "valid_action_indices": state["valid_action_indices"],
                            "mask_hash": state["mask_hash"],
                            "state_hash": state["state_hash"],
                            "source_daily_file": rel(SOURCE_DAILY_ROOT / f"YCA_{state['year']}_seed1_ckpt{CHECKPOINT}_daily.csv"),
                            "model_file": rel(model_path),
                        }
                    )
        finally:
            del model
    return pd.DataFrame(rows)


def make_state_summary(prob: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (state_id, year, source_row, dap, role), group in prob.groupby(
        ["state_id", "year", "source_row", "dap", "state_role"], sort=True
    ):
        by_method = {m: g.sort_values("action_index") for m, g in group.groupby("method")}
        original = by_method["Original"]
        augmented = by_method["Augmented"]
        raw_p_o = original["probability"].to_numpy(dtype=float)
        raw_p_a = augmented["probability"].to_numpy(dtype=float)
        masked_p_o = original["masked_probability"].to_numpy(dtype=float)
        masked_p_a = augmented["masked_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "state_id": state_id,
                "year": int(year),
                "source_row": int(source_row),
                "dap": int(dap),
                "state_role": role,
                "state_swfac": float(original["state_swfac"].iloc[0]),
                "state_nstres": float(original["state_nstres"].iloc[0]),
                "valid_action_count": int(original["valid_action_count"].iloc[0]),
                "valid_action_indices": original["valid_action_indices"].iloc[0],
                "original_selected_action": int(original["selected_action_index"].iloc[0]),
                "augmented_selected_action": int(augmented["selected_action_index"].iloc[0]),
                "same_selected_action": bool(original["selected_action_index"].iloc[0] == augmented["selected_action_index"].iloc[0]),
                "original_raw_argmax": int(original["raw_argmax_action"].iloc[0]),
                "augmented_raw_argmax": int(augmented["raw_argmax_action"].iloc[0]),
                "same_raw_argmax": bool(original["raw_argmax_action"].iloc[0] == augmented["raw_argmax_action"].iloc[0]),
                "raw_total_variation": float(0.5 * np.abs(raw_p_o - raw_p_a).sum()),
                "masked_total_variation": float(0.5 * np.abs(masked_p_o - masked_p_a).sum()),
                "raw_max_abs_probability_diff": float(np.abs(raw_p_o - raw_p_a).max()),
                "masked_max_abs_probability_diff": float(np.abs(masked_p_o - masked_p_a).max()),
                "original_selected_probability": float(original.loc[original["is_argmax"], "masked_probability"].iloc[0]),
                "augmented_selected_probability": float(augmented.loc[augmented["is_argmax"], "masked_probability"].iloc[0]),
                "original_top_raw_probability": float(original["probability"].max()),
                "augmented_top_raw_probability": float(augmented["probability"].max()),
                "state_hash_original": original["state_hash"].iloc[0],
                "state_hash_augmented": augmented["state_hash"].iloc[0],
                "mask_hash_original": original["mask_hash"].iloc[0],
                "mask_hash_augmented": augmented["mask_hash"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def make_mask_audit(prob: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "state_id",
        "year",
        "source_row",
        "source_dap",
        "dap",
        "state_role",
        "valid_action_count",
        "valid_action_indices",
        "mask_hash",
        "method",
    ]
    return prob[cols].drop_duplicates().sort_values(["year", "source_row", "method"]).reset_index(drop=True)


def internal_phase_summary() -> pd.DataFrame:
    expected = ["num_timesteps", "approx_kl", "clip_fraction", "entropy_loss", "explained_variance", "value_loss"]
    if not AUGMENTED_UPDATE_LOG.exists():
        return pd.DataFrame([{"availability": "missing", "source_file": rel(AUGMENTED_UPDATE_LOG)}])
    df = pd.read_csv(AUGMENTED_UPDATE_LOG, keep_default_na=False)
    missing = [col for col in expected if col not in df.columns]
    if missing:
        return pd.DataFrame([{"availability": "schema_missing", "source_file": rel(AUGMENTED_UPDATE_LOG), "missing": ",".join(missing)}])
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bins = [0, 25_000, 50_000, 75_000, 100_000, np.inf]
    labels = ["0-25K", "25-50K", "50-75K", "75-100K", "post-100K"]
    df["phase"] = pd.cut(df["num_timesteps"], bins=bins, labels=labels, include_lowest=True).astype(str)
    rows = []
    for phase, group in df.groupby("phase", sort=False, observed=False):
        rows.append(
            {
                "availability": "available",
                "source_file": rel(AUGMENTED_UPDATE_LOG),
                "phase": phase,
                "updates": int(len(group)),
                "timesteps_min": float(group["num_timesteps"].min()),
                "timesteps_max": float(group["num_timesteps"].max()),
                "approx_kl_max": float(group["approx_kl"].max()),
                "clip_fraction_max": float(group["clip_fraction"].max()),
                "entropy_loss_first": float(group["entropy_loss"].iloc[0]),
                "entropy_loss_last": float(group["entropy_loss"].iloc[-1]),
                "entropy_loss_min": float(group["entropy_loss"].min()),
                "explained_variance_min": float(group["explained_variance"].min()),
                "value_loss_max": float(group["value_loss"].max()),
            }
        )
    return pd.DataFrame(rows)


def write_grid() -> None:
    grid = pd.DataFrame(action_grid())
    grid["action_label"] = grid.apply(lambda r: f"I{r.irrigation:g}/N{r.nitrogen:g}", axis=1)
    grid.to_csv(GRID_CSV, index=False, encoding="utf-8-sig")


def write_integrity(prob: pd.DataFrame, state_summary: pd.DataFrame, mask_audit: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("Original model exists", ORIGINAL_MODEL.exists(), rel(ORIGINAL_MODEL)),
        ("Augmented model exists", AUGMENTED_MODEL.exists(), rel(AUGMENTED_MODEL)),
        ("Source daily files cover 2014-2023", all((SOURCE_DAILY_ROOT / f"YCA_{year}_seed1_ckpt{CHECKPOINT}_daily.csv").exists() for year in YEARS), str(len(YEARS))),
        ("Probability rows are 2 methods x 16 actions per shared state", len(prob) == len(state_summary) * 2 * 16, f"rows={len(prob)}, states={len(state_summary)}"),
        ("Every shared state has identical observation hash across methods", bool((state_summary["state_hash_original"] == state_summary["state_hash_augmented"]).all()), str(state_summary["state_hash_original"].nunique())),
        ("Every shared state has identical mask hash across methods", bool((state_summary["mask_hash_original"] == state_summary["mask_hash_augmented"]).all()), str(state_summary["mask_hash_original"].nunique())),
        ("Raw probabilities sum to one", bool(np.allclose(prob.groupby(["method", "state_id"])["probability"].sum().to_numpy(), 1.0, atol=1e-6)), "group sums"),
        ("Masked probabilities sum to one", bool(np.allclose(prob.groupby(["method", "state_id"])["masked_probability"].sum().to_numpy(), 1.0, atol=1e-6)), "group sums"),
        ("Masked probabilities are zero for invalid actions", bool((prob.loc[~prob["valid_by_mask"], "masked_probability"].abs() < 1e-7).all()), f"invalid_rows={int((~prob['valid_by_mask']).sum())}"),
        ("Initial DAP1 state per year allows all 16 actions", bool(len(state_summary[state_summary["source_row"].eq(0)]) == len(YEARS) and (state_summary.loc[state_summary["source_row"].eq(0), "valid_action_count"] == 16).all()), f"initial_dap1_states={int(state_summary['source_row'].eq(0).sum())}"),
        ("Deterministic model.predict agrees with masked argmax", bool((prob["selected_action_index"] == prob["masked_argmax_action"]).all()), "all policy rows"),
        ("No recorded source action was forced to fallback noop", bool(all((read_daily(year)["mask_forced_noop"].astype(str).str.lower() != "true").all() for year in YEARS)), "source daily files"),
    ]
    return pd.DataFrame([{"check": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in checks])


def fmt(x: Any, digits: int = 3) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def make_report(prob: pd.DataFrame, states: list[dict[str, Any]], state_summary: pd.DataFrame, mask_audit: pd.DataFrame, checks: pd.DataFrame, internal: pd.DataFrame) -> str:
    code_refs = {
        "evaluation": "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py:217-226",
        "mask_wrapper": "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:67-145",
        "action_grid": "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:61-64",
    }
    dap1 = state_summary[state_summary["dap"].eq(1)]
    initial_dap1 = state_summary[state_summary["source_row"].eq(0)]
    repeated_dap1 = dap1[~dap1["source_row"].eq(0)]
    dap2_10 = state_summary[state_summary["dap"].between(2, 10)]
    stress = state_summary[state_summary["state_role"].str.contains("stress", na=False)]
    same_selected_rate = float(state_summary["same_selected_action"].mean())
    mean_tv = float(state_summary["raw_total_variation"].mean())
    median_tv = float(state_summary["raw_total_variation"].median())
    max_tv = float(state_summary["raw_total_variation"].max())
    selected_disagreements = state_summary[~state_summary["same_selected_action"]]
    raw_argmax_disagreements = state_summary[~state_summary["same_raw_argmax"]]
    method_selected = (
        prob[prob["is_argmax"]]
        .groupby(["method", "selected_action"], as_index=False)
        .size()
        .sort_values(["method", "size"], ascending=[True, False])
    )
    selected_lines = "\n".join(f"| {r.method} | {r.selected_action} | {int(r.size)} |" for r in method_selected.itertuples(index=False))
    checks_lines = "\n".join(f"| {'PASS' if r.passed else 'FAIL'} | {r.check} | {r.evidence} |" for r in checks.itertuples(index=False))
    internal_lines = "\n".join(
        f"| {r.phase} | {int(r.updates)} | {fmt(r.entropy_loss_first, 4)} -> {fmt(r.entropy_loss_last, 4)} | {fmt(r.approx_kl_max, 6)} | {fmt(r.clip_fraction_max, 4)} | {fmt(r.explained_variance_min, 3)} | {fmt(r.value_loss_max, 3)} |"
        for r in internal[internal.get("availability", pd.Series(dtype=str)).eq("available")].itertuples(index=False)
    ) or "| NA | NA | NA | NA | NA | NA | NA |"
    stress_text = f"stress states={len(stress)}, selected-action agreement={float(stress['same_selected_action'].mean()):.3f}" if len(stress) else "No stress-labelled shared states were collected."
    if same_selected_rate >= 0.95 and mean_tv < 0.10:
        case = "Case A — truly similar policy"
        case_sentence = "The two seed1 policies genuinely converged to very similar low-action policies on the shared state batch."
    elif same_selected_rate < 0.80 and mean_tv >= 0.10:
        case = "Case B — evaluation artifact / argmax compression"
        case_sentence = "Observed action similarity is partly or mainly an evaluation artifact."
    else:
        case = "Case D — mixed mechanism"
        case_sentence = "The result is mixed: deterministic argmax exposes a strong common low-action preference, while the full distributions retain measurable policy differences; masking is an important state-dependent constraint but not the sole explanation."
    return f"""# YC seed1 evaluation mechanism check

日期：2026-09-11  
状态：`completed; read-only checkpoint and evaluation audit`  
本轮没有训练、没有修改 PPO/reward/action space/wrapper/mask/data，也没有重定义正式结果。

## 结论先行

判定：**{case}**。

{case_sentence}

在同一批 `{len(states)}` 个 shared observations 上，Original seed1 与 Augmented seed1 的 deterministic selected action 一致率为 `{same_selected_rate:.3f}`；原始 16-action probability 的平均 total variation 为 `{mean_tv:.3f}`、中位数 `{median_tv:.3f}`、最大值 `{max_tv:.3f}`。因此，不能把“validation 看起来同样退化”简单归结为 fallback；也不能只凭同一 argmax 宣称两个 policy 的完整分布相同。

关键分层：每年首次 DAP1 状态 `{len(initial_dap1)}` 个，均允许 16 个动作；重放中另有 `{len(repeated_dap1)}` 个重复的有效 DAP1 早期状态，这些状态在首次施加资源后因最小间隔只允许 `{int(repeated_dap1['valid_action_count'].min()) if len(repeated_dap1) else 'NA'}–{int(repeated_dap1['valid_action_count'].max()) if len(repeated_dap1) else 'NA'}` 个动作；DAP2–10 状态 `{len(dap2_10)}` 个；{stress_text}。两个模型在 shared batch 上的 raw argmax 不同状态数为 `{len(raw_argmax_disagreements)}`，masked selected action 不同状态数为 `{len(selected_disagreements)}`。

## 1. validation action selection 机制

正式 YC five-scenario evaluation 在 `{code_refs['evaluation']}` 中加载 `MaskablePPO`，调用：

```python
mask = get_action_masks(env)
action, _ = model.predict(obs, action_masks=mask, deterministic=True)
```

所以 validation 是 deterministic=True 的 masked argmax，而不是 stochastic sampling。训练阶段 `MaskablePPO.learn()` 使用 rollout distribution 进行采样；当前评估阶段将采样噪声关闭，train-time 与 eval-time 的 action selection 不同。因而 validation 行为是策略分布经 mask 后 argmax 的结果，不能代表完整 stochastic policy 的平均行为。

## 2. action grid / index 映射

grid 由 `{code_refs['action_grid']}` 的 `for i in I_LEVELS for n in N_LEVELS` 生成，index 为：

| index | irrigation (mm) | nitrogen (kg/ha) | label |
|---:|---:|---:|---|
{chr(10).join(f"| {int(r.action_index)} | {r.irrigation:g} | {r.nitrogen:g} | I{r.irrigation:g}/N{r.nitrogen:g} |" for r in pd.DataFrame(action_grid()).itertuples(index=False))}

因此 `(45,40)` = index 13，`(45,80)` = index 14，zero action `(0,0)` = index 0。45 mm 是最大 irrigation 档，不是第一个非零档。因为 irrigation 是外层、N 是内层，索引相邻性主要沿 N 变化；index 顺序本身没有把 45 mm 放到“第一个非零 irrigation”位置，也没有单独构造一个 45-mm fallback。

## 3. action mask / safety wrapper

wrapper 的 legal rule 位于 `{code_refs['mask_wrapper']}`：zero action 总是合法；正灌溉受 DAP 范围、季节剩余量、最小间隔和 040_26 staged reserve 约束；正施氮受 DAP 范围、季节剩余量和最小间隔约束。mask 最后强制 `mask[0] = True`，如果没有合法动作也保留 index 0。

当收到非法 action 时，wrapper 执行 `action_index = 0` 并记录 `mask_forced_noop=True`；这就是唯一确认的 fallback/default action，映射为 `(0,0)`。正式 validation 本身也传入 action mask，因此正常情况下 policy 不应选择 invalid index；已有 Original/Augmented seed1 100K daily logs 中 `mask_forced_noop` 均未出现 true。

本次同一 shared replay 的“每年首次 DAP1”mask 审计结果为：`{int(initial_dap1['valid_action_count'].min()) if len(initial_dap1) else 'NA'}–{int(initial_dap1['valid_action_count'].max()) if len(initial_dap1) else 'NA'}` 个合法动作，且每年首次 DAP1 都是全 16-action mask。故 DAP1 的 `(45,40)` 与 `(45,80)` 并非由初始 DAP1 mask 只剩下 45 mm，也不是由 fallback 生成。重放中出现的重复早期记录由 wrapper 的非正 DAP→DAP1 规则和最小间隔共同造成，应与真正的初始 DAP1 决策分开解释。后续状态的 mask 会随累计资源和最小间隔变化；其 Original/Augmented mask hash 在 shared states 上全部一致。

## 4. 两个 policy 的 raw/masked probabilities

比较使用 Original seed1 @100K 的既有 2014–2023 validation action trace 重放生成 shared observations；两个 checkpoint 读取同一 observation array、同一 mask，不比较两条独立 rollout 的状态。完整逐 action 结果见 `{rel(PROBABILITY_CSV)}`；state-level 汇总见 `{rel(STATE_SUMMARY_CSV)}`。

| method | selected action | shared-state count |
|---|---|---:|
{selected_lines}

解释边界：`logit/probability` 是 action network 的 raw output/softmax；`masked_logit/masked_probability` 将 invalid action 的 logit 设为 `-1e8` 后重新归一化。`selected_action` 是实际 deterministic `model.predict(..., action_masks=mask)`，并逐行核对等于 masked argmax。

## 5. 45 mm 是否结构性偏好

没有证据表明 45 mm 是 index ordering 或 fallback 的结构性产物：index 13/14 只是普通联合动作，DAP1 全 16 动作可选，且 fallback 是 index 0 而不是 13/14。若某一 policy 在 DAP1 选择 index 13 或 14，那是该 policy 在全合法集合上的概率/argmax 结果。

后续 DAP 的固定动作可能受到共同 mask 约束，尤其是 DAP1 施肥/灌溉后最小间隔、staged irrigation reserve 和季节预算会压缩合法集合。这可以造成 argmax 的“动作压平”，但本次 shared-state probability 仍保留了 mask 前分布，因此其贡献可与 policy 本身区分。

## 6. 100K Augmented seed1 的渐进性辅助检查

已有 Augmented seed1 update log 可用；Original seed1 没有 update-level log，本报告不补造。下表按 update timesteps 分段，仅作辅助，不把 entropy/KL 单独当成根因：

| phase | updates | entropy first -> last | approx_kl max | clip_fraction max | explained_variance min | value_loss max |
|---|---:|---:|---:|---:|---:|---:|
{internal_lines}

请结合 `{rel(INTERNAL_CSV)}` 检查 75K→100K 的变化。此结果用于判断是否存在渐进确定性化或一次性跳变，不能替代直接的 policy-distribution 证据。

## 7. 对上一轮 seed-dependent local optimum / policy basin 判断的影响

本轮不推翻上一轮判断，但把机制表述收紧：Original seed1 与 Augmented seed1 的单次 DAP1/45-mm 现象不是一个已证实的 evaluation bug；shared DAP1 mask 为全 16，且两个 checkpoint 的完整概率分布可直接比较。与此同时，deterministic masked argmax 确实会把分布差异压缩为一个离散动作，后期共同 safety mask 也会进一步压缩选择空间。因此更准确的表述是：**seed-dependent policy basin/local optimum 是主要解释，deterministic evaluation 与 shared mask 是把该差异/相似性投影到少数执行动作上的次级机制；目前没有证据支持 fallback 是主因。**

## 8. 下一步建议

仍建议做最小化的 PPO exploration/optimization-stability 对照，但不应直接进入 G×E V2，也不应先改正式 evaluation 定义。下一步应保持 reward、action grid、wrapper、mask 和 validation contract 不变，预先固定 checkpoint selection 与 distribution diagnostics，再做小规模、低内存的稳定性对照；是否扩展训练必须等待该对照的响应证据。

## 9. 数据质量与复现检查

| result | check | evidence |
|---|---|---|
{checks_lines}

本轮主要产物：

- `{rel(DOC)}`
- `{rel(PROBABILITY_CSV)}`
- `{rel(STATE_SUMMARY_CSV)}`
- `{rel(MASK_AUDIT_CSV)}`
- `{rel(GRID_CSV)}`
- `{rel(INTERNAL_CSV)}`
- `{rel(INTEGRITY_CSV)}`
- `{rel(Path(__file__))}`
- `{rel(RUN_LOG)}`
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    config, env_config, engine = setup_shared_env_contract()
    states = collect_shared_states(config, env_config, engine)
    probability = extract_policy_rows(states)
    state_summary = make_state_summary(probability)
    mask_audit = make_mask_audit(probability)
    internal = internal_phase_summary()
    write_grid()
    checks = write_integrity(probability, state_summary, mask_audit)

    probability.to_csv(PROBABILITY_CSV, index=False, encoding="utf-8-sig")
    state_summary.to_csv(STATE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    mask_audit.to_csv(MASK_AUDIT_CSV, index=False, encoding="utf-8-sig")
    internal.to_csv(INTERNAL_CSV, index=False, encoding="utf-8-sig")
    checks.to_csv(INTEGRITY_CSV, index=False, encoding="utf-8-sig")
    DOC.write_text(make_report(probability, states, state_summary, mask_audit, checks, internal), encoding="utf-8")
    manifest = {
        "status": "completed_read_only",
        "task": "yc_seed1_evaluation_mechanism_check",
        "shared_states": len(states),
        "probability_rows": len(probability),
        "methods": ["Original seed1 @100K", "Augmented seed1 @100K"],
        "source_daily_root": rel(SOURCE_DAILY_ROOT),
        "source_daily_sha256": {str(year): sha256(SOURCE_DAILY_ROOT / f"YCA_{year}_seed1_ckpt{CHECKPOINT}_daily.csv") for year in YEARS},
        "model_sha256": {"Original": sha256(ORIGINAL_MODEL), "Augmented": sha256(AUGMENTED_MODEL)},
        "output_files": [rel(DOC), rel(PROBABILITY_CSV), rel(STATE_SUMMARY_CSV), rel(MASK_AUDIT_CSV), rel(GRID_CSV), rel(INTERNAL_CSV), rel(INTEGRITY_CSV), rel(Path(__file__)), rel(RUN_LOG)],
        "all_integrity_checks_passed": bool(checks["passed"].all()),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "shared_states": len(states), "probability_rows": len(probability), "all_integrity_checks_passed": manifest["all_integrity_checks_passed"], "report": rel(DOC)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
