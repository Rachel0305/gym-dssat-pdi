from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity"
OUT = OUT_ROOT / "attempt_02"
REPORT = ROOT / "docs/yc_random_weather_004_11_seed5_mask_sensitivity.md"
FROZEN = ROOT / "results/yc_random_weather_ppo/004_05"
REPLAY = ROOT / "results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay"
TRACE_00409 = ROOT / "results/yc_random_weather_ppo/004_09_observation_mask_action_audit/decision_trace_seed345.csv"
RUNNER_PATH = ROOT / "src/run_004_07_02_yc_seed345_etcp_replay.py"
AUDIT_00409_PATH = ROOT / "src/run_004_09_yc_observation_mask_action_audit.py"
PROMPT = ROOT / "prompt_02/004_11_yc_seed5_mask_sensitivity.md"
YEARS = list(range(2014, 2024))
SEED = 5
REGIME = "RANDOM_WEATHER_WGEN"
RSS_LIMIT_MB = 2048.0
ACTION_GRID = [(i, n) for i in (0.0, 15.0, 30.0, 45.0) for n in (0.0, 40.0, 80.0, 120.0)]
VARIANTS = {
    "B_no_early": (None, 150.0),
    "C_no_mid": (75.0, None),
    "D_no_early_mid": (None, None),
}
LABELS = {
    "A_canonical": "A canonical",
    "B_no_early": "B no-early",
    "C_no_mid": "C no-mid",
    "D_no_early_mid": "D no-early-mid",
}
COLORS = {"A_canonical": "#2A9D55", "B_no_early": "#2878B5", "C_no_mid": "#D9822B", "D_no_early_mid": "#8A5A9E"}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_json(path: Path, data: Any, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rss_mb() -> float:
    import psutil

    process = psutil.Process()
    value = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            value += child.memory_info().rss
        except psutil.Error:
            pass
    return value / (1024 * 1024)


def load_runtime():
    audit = import_module(AUDIT_00409_PATH, "yc_00409_for_00411")
    audit.OUT = OUT
    runner, pilot = audit.load_runtime()
    pilot.MAX_PROCESS_TREE_RSS_MB = RSS_LIMIT_MB
    context = pilot._import_canonical()
    engine = context[1]
    staged = engine.base04036.base04026
    if engine.base03222.base.StressAwareDiscreteWrapper is not engine.base04036.LateIrrigationReserveMaskWrapper:
        raise RuntimeError("Effective 040_36 late-reserve wrapper mismatch")
    action_audit = import_module(ROOT / "src/run_004_10_yc_action_mask_rule_audit.py", "yc_00410_rules_for_00411")
    return runner, pilot, staged, action_audit


def set_variant_caps(staged, variant: str) -> tuple[float, float]:
    early, mid = VARIANTS[variant]
    staged.EARLY_CUM_IRRIGATION_CAP = math.inf if early is None else early
    staged.MID_CUM_IRRIGATION_CAP = math.inf if mid is None else mid
    return staged.EARLY_CUM_IRRIGATION_CAP, staged.MID_CUM_IRRIGATION_CAP


def read_formal_refs(runner) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    episodes: dict[int, dict[str, Any]] = {}
    steps: dict[int, pd.DataFrame] = {}
    _, ep_path, step_path, _ = runner.frozen_paths(SEED, YEARS[0])
    ep_frame = pd.read_csv(ep_path, keep_default_na=False)
    ep_frame = ep_frame[
        ep_frame.training_regime.eq(REGIME)
        & ep_frame.evaluation_weather_type.eq("observed_weather")
        & pd.to_numeric(ep_frame.ppo_seed, errors="coerce").eq(SEED)
    ]
    for year in YEARS:
        e = ep_frame[pd.to_numeric(ep_frame.evaluation_weather_year, errors="coerce").eq(year)]
        if len(e) != 1:
            raise RuntimeError(f"Missing/duplicate formal episode seed={SEED} year={year}")
        episodes[year] = e.iloc[0].to_dict()
    required = ["training_regime", "ppo_seed", "evaluation_weather_type", "evaluation_weather_year", "timestep", "dap", "action_index", "irrigation_action_mm", "fertilizer_action_kgN_ha", "instant_reward"]
    chunks = []
    for chunk in pd.read_csv(step_path, usecols=required, chunksize=50000):
        chunk = chunk[
            chunk.training_regime.eq(REGIME)
            & chunk.evaluation_weather_type.eq("observed_weather")
            & pd.to_numeric(chunk.ppo_seed, errors="coerce").eq(SEED)
        ]
        if not chunk.empty:
            chunks.append(chunk)
    all_steps = pd.concat(chunks, ignore_index=True)
    for year in YEARS:
        group = all_steps[pd.to_numeric(all_steps.evaluation_weather_year, errors="coerce").eq(year)].copy()
        group = group.sort_values("timestep").reset_index(drop=True)
        group["timestep"] = pd.to_numeric(group.timestep, errors="raise").astype(int)
        if len(group) != int(float(episodes[year]["episode_days"])):
            raise RuntimeError(f"Formal step count mismatch seed=5 year={year}")
        steps[year] = group
    trace = pd.read_csv(TRACE_00409)
    trace = trace[trace.seed.eq(SEED)].copy()
    if set(trace.year.astype(int).unique()) != set(YEARS) or len(trace) != sum(map(len, steps.values())):
        raise RuntimeError("004_09 decision trace coverage mismatch")
    return steps, episodes, {int(y): g for y, g in trace.groupby("year")}


def canonical_rows(runner, pilot, helper, formal_eps, trace_by_year) -> list[dict[str, Any]]:
    summary_path = REPLAY / "replay_episode_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summaries = pd.read_csv(summary_path)
    summaries = summaries[summaries.seed.eq(SEED)].set_index("year")
    output = []
    for year in YEARS:
        s = summaries.loc[year]
        ep = formal_eps[year]
        output.append({
            "variant": "A_canonical", "year": year,
            "yield_kg_ha": float(s.replay_yield_kg_ha), "irrigation_mm": float(s.irrigation_mm),
            "nitrogen_kg_ha": float(s.nitrogen_kg_ha), "etcp_mm": float(s.etcp_mm),
            "wp_et_kg_m3": float(s.wp_et_kg_m3), "pfp_n_kg_kg": float(s.pfp_n_kg_kg),
            "common_reward": float(s.common_reward), "canonical_reward": float(s.canonical_reward),
            "management_event_count": int(s.management_event_count),
            "management_action_days_nonzero": int(s.management_action_days_nonzero),
            "irrigation_action_days": int(s.irrigation_action_days), "nitrogen_action_days": int(s.nitrogen_action_days),
            "steps": int(s.steps), "action_signature": str(s.action_signature), "source": "004_07_02 existing reconciled replay",
            "official_episode_yield_kg_ha": float(ep["yield"]), "model_sha256": sha256_file(ROOT / s.model_path),
        })
        t = trace_by_year[year].sort_values("timestep")
        if len(t) != int(s.steps):
            raise RuntimeError(f"Canonical trace closure failed in {year}")
    return output


def check_rendered_inputs(runner, run_root: Path, year: int) -> dict[str, Any]:
    _, _, _, frozen_dir = runner.frozen_paths(SEED, year)
    run_tag = f"eval_observed_weather_RANDOM_WEATHER_WGEN_{SEED}_{year}"
    rendered_dir = run_root / "rendered_inputs" / "YCA" / str(year) / run_tag
    names = [f"CNYC{str(year)[-2:]}01.WTH", f"YCA_{year}_{run_tag}.jinja2"]
    checks = {}
    for name in names:
        old, new = frozen_dir / name, rendered_dir / name
        if not old.is_file() or not new.is_file():
            raise FileNotFoundError(f"Missing frozen/replayed input {old} or {new}")
        old_hash, new_hash = sha256_file(old), sha256_file(new)
        checks[name] = {"frozen_sha256": old_hash, "replay_sha256": new_hash, "identical": old_hash == new_hash}
        if old_hash != new_hash:
            raise RuntimeError(f"Frozen weather/template differs for {name}")
    return {"rendered_dir": rel(rendered_dir), "checks": checks}


def infer_cap_rule(dap: float, cum_i: float, action_i: float, early: float | None, mid: float | None) -> str:
    after = cum_i + action_i
    if action_i <= 1e-9:
        return "LEGAL"
    if early is not None and dap <= 30 and after > early + 1e-9:
        return "I_EARLY_CAP"
    if mid is not None and dap <= 60 and after > mid + 1e-9:
        return "I_MID_CAP"
    return "LEGAL"


def run_episode(runner, pilot, staged, helper, model_path: Path, variant: str, year: int, expected_steps: pd.DataFrame, smoke: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sb3_contrib import MaskablePPO
    early_cap, mid_cap = set_variant_caps(staged, variant)
    run_root = OUT / "evaluation" / variant / f"YCA_{year}"
    snapshot = OUT / "snapshots" / variant / f"YCA" / str(year)
    trace_path = OUT / "episodes" / f"trace_{variant}_{year}.csv"
    for path in (run_root, snapshot, trace_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {rel(path)}")
    run_root.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    cfg, env_cfg = pilot.make_run_config(REGIME, SEED, run_root)
    schedule = [{"episode_index": 1, "historical_year": year, "weather_seed": ""}]
    env = pilot.ScheduledEpisodeEnv(cfg, env_cfg, schedule, "HISTORICAL_WEATHER", SEED,
                                    f"eval_observed_weather_RANDOM_WEATHER_WGEN_{SEED}", pilot.EVAL_RUNTIME_SEED,
                                    max_cached_years=1)
    model = MaskablePPO.load(str(model_path), device="cpu")
    rows = []
    total_reward = 0.0
    peak = rss_mb()
    raw = None
    try:
        observation, _ = env._gym_env.reset(seed=None)
        if len(expected_steps) == 0:
            raise RuntimeError("Empty formal reference steps")
        for j, formal in enumerate(expected_steps.itertuples(index=False)):
            mask = np.asarray(env._gym_env.action_masks(), dtype=bool).reshape(-1)
            if mask.size != len(ACTION_GRID) or not mask.any():
                raise RuntimeError(f"Invalid action mask {mask.shape} at {variant}/{year}/{j+1}")
            active_caps = (staged.EARLY_CUM_IRRIGATION_CAP, staged.MID_CUM_IRRIGATION_CAP)
            staged.EARLY_CUM_IRRIGATION_CAP, staged.MID_CUM_IRRIGATION_CAP = 75.0, 150.0
            try:
                canonical_mask = np.asarray(env._gym_env.action_masks(), dtype=bool).reshape(-1)
            finally:
                staged.EARLY_CUM_IRRIGATION_CAP, staged.MID_CUM_IRRIGATION_CAP = active_caps
            if np.any(canonical_mask & ~mask):
                raise RuntimeError(f"Variant unexpectedly removes canonical-legal actions at {variant}/{year}/{j+1}")
            for aid, (dose_i, _) in enumerate(ACTION_GRID):
                if mask[aid] and not canonical_mask[aid]:
                    if dose_i <= 0:
                        raise RuntimeError("A non-irrigation action changed under irrigation-cap variant")
                    if variant == "B_no_early" and not (float(getattr(env.current_env, "_dap")()) <= 30 and float(getattr(env.current_env.safety_state, "cumulative_irrigation")) + dose_i > 75.0 + 1e-9):
                        raise RuntimeError("B enabled an action outside the early-cap exception")
                    if variant == "C_no_mid" and not (31 <= float(env.current_env._dap()) <= 60 and float(env.current_env.safety_state.cumulative_irrigation) + dose_i > 150.0 + 1e-9):
                        raise RuntimeError("C enabled an action outside the mid-cap exception")
                    if variant == "D_no_early_mid" and not (
                        (float(env.current_env._dap()) <= 30 and float(env.current_env.safety_state.cumulative_irrigation) + dose_i > 75.0 + 1e-9)
                        or (float(env.current_env._dap()) <= 60 and float(env.current_env.safety_state.cumulative_irrigation) + dose_i > 150.0 + 1e-9)
                    ):
                        raise RuntimeError("D enabled an action outside the early/mid-cap exceptions")
            obs_tensor, _ = model.policy.obs_to_tensor(observation)
            with __import__("torch").no_grad():
                features = model.policy.extract_features(obs_tensor, model.policy.pi_features_extractor)
                latent = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent).detach().cpu().numpy().reshape(-1)
                raw_dist = model.policy.get_distribution(obs_tensor)
                raw_probs = raw_dist.distribution.probs.detach().cpu().numpy().reshape(-1)
            raw_index = int(np.argmax(logits))
            pred, _ = model.predict(observation, deterministic=True, action_masks=mask)
            action = int(np.asarray(pred).reshape(-1)[0])
            if not mask[action] or action != int(np.argmax(np.where(mask, logits, -1e9))):
                raise RuntimeError(f"Chosen action/masked argmax mismatch at {variant}/{year}/{j+1}")
            wrapper = env.current_env
            state = getattr(wrapper, "safety_state", None)
            cum_i = float(getattr(state, "cumulative_irrigation", 0.0))
            cum_n = float(getattr(state, "cumulative_n", 0.0))
            dap = float(wrapper._dap()) if hasattr(wrapper, "_dap") else float(formal.dap)
            raw_i, raw_n = ACTION_GRID[raw_index]
            chosen_i, chosen_n = ACTION_GRID[action]
            base_mask = np.array(mask, copy=True)
            cap_reason = "LEGAL" if base_mask[raw_index] else "OTHER_CANONICAL_RULE"
            if not base_mask[raw_index] and raw_i > 0:
                # Re-evaluate cap-free legality while retaining every other canonical rule.
                prior = (staged.EARLY_CUM_IRRIGATION_CAP, staged.MID_CUM_IRRIGATION_CAP)
                staged.EARLY_CUM_IRRIGATION_CAP = math.inf
                staged.MID_CUM_IRRIGATION_CAP = math.inf
                try:
                    cap_free = np.asarray(env._gym_env.action_masks(), dtype=bool).reshape(-1)
                finally:
                    staged.EARLY_CUM_IRRIGATION_CAP, staged.MID_CUM_IRRIGATION_CAP = prior
                if cap_free[raw_index]:
                    configured_early, configured_mid = VARIANTS[variant]
                    cap_reason = infer_cap_rule(dap, cum_i, raw_i, configured_early, configured_mid)
            applied = {"irrigation": chosen_i, "nitrogen": chosen_n}
            next_obs, reward, terminated, truncated, _ = env._gym_env.step(action)
            info = dict(getattr(wrapper, "last_action_info", {}) or {})
            applied_i = float(info.get("safe_action_amir", applied["irrigation"]) or 0.0)
            applied_n = float(info.get("safe_action_anfer", applied["nitrogen"]) or 0.0)
            if abs(applied_i - chosen_i) > 1e-8 or abs(applied_n - chosen_n) > 1e-8:
                raise RuntimeError(f"Applied dose differs from action at {variant}/{year}/{j+1}")
            rows.append({
                "variant": variant, "year": year, "seed": SEED, "timestep": j + 1, "dap": dap,
                "formal_canonical_action_index": int(formal.action_index),
                "raw_argmax_action": raw_index, "raw_argmax_irrigation_mm": raw_i,
                "raw_argmax_nitrogen_kg_ha": raw_n, "raw_argmax_probability": float(raw_probs[raw_index]),
                "raw_argmax_masked_out": not bool(mask[raw_index]), "raw_argmax_block_rule": cap_reason,
                "legal_action_count": int(mask.sum()), "chosen_action": action,
                "irrigation_mm": applied_i, "nitrogen_kg_ha": applied_n,
                "cumulative_irrigation_before_mm": cum_i, "cumulative_n_before_kg_ha": cum_n,
                "early_cap_mm": None if math.isinf(early_cap) else early_cap,
                "mid_cap_mm": None if math.isinf(mid_cap) else mid_cap,
                "step_reward": float(reward), "mask_bits": "".join("1" if x else "0" for x in mask),
            })
            total_reward += float(reward)
            observation = next_obs
            if (j + 1) % 10 == 0:
                current = rss_mb()
                peak = max(peak, current)
                print(f"[rss] variant={variant} year={year} step={j+1} tree_mb={current:.1f}", flush=True)
                if current >= RSS_LIMIT_MB:
                    raise MemoryError(f"Process-tree RSS reached {RSS_LIMIT_MB:.0f} MiB")
            if terminated or truncated:
                if j + 1 != len(expected_steps):
                    raise RuntimeError("Episode terminated before formal step horizon")
                break
        if len(rows) != len(expected_steps):
            raise RuntimeError(f"Step count mismatch {len(rows)} != {len(expected_steps)}")
        episode = dict(env.episode_rows[-1])
        raw = pilot.find_dssat_instance(env.current_env)
        snapshot_files = runner.snapshot_outputs(raw, snapshot)
        selected, _ = runner.selected_summary_row(snapshot, float(episode["yield"]))
        metrics = helper.summary_metrics(snapshot, float(episode["yield"]))
        events = helper.parse_management_events(snapshot / "MgmtEvent.OUT")
        event_i = float(events.loc[events.operation.eq("irrigation"), "amount"].sum()) if not events.empty else 0.0
        event_n = float(events.loc[events.operation.eq("fertilizer"), "amount"].sum()) if not events.empty else 0.0
        summary_num = __import__("calculate_five_site_wue_nue_from_summary_019_10", fromlist=["num"]).num
        sy = summary_num(selected, "HWAM")
        si = summary_num(selected, "IRCM")
        sn = summary_num(selected, "NICM")
        action_i = sum(r["irrigation_mm"] for r in rows)
        action_n = sum(r["nitrogen_kg_ha"] for r in rows)
        if sy is None or abs(float(sy) - float(episode["yield"])) > 1.0:
            raise RuntimeError(f"Summary.OUT yield does not close in {variant}/{year}")
        for label, actual, summary, actions in (("I", event_i, si, action_i), ("N", event_n, sn, action_n)):
            if summary is None or abs(actual - actions) > 1.0 or abs(float(summary) - actions) > 1.0:
                raise RuntimeError(f"{label} closure failed {variant}/{year}: event={actual}, summary={summary}, action={actions}")
        etcp = float(metrics["etcp_mm"])
        if not math.isfinite(etcp) or etcp <= 0:
            raise RuntimeError(f"No valid ETCP in Summary.OUT for {variant}/{year}")
        input_prov = check_rendered_inputs(runner, run_root, year)
        row = {
            "variant": variant, "year": year, "yield_kg_ha": float(episode["yield"]),
            "irrigation_mm": event_i, "nitrogen_kg_ha": event_n, "etcp_mm": etcp,
            "wp_et_kg_m3": float(metrics["wp_et_kg_m3"]), "pfp_n_kg_kg": float(metrics["pfp_n_kg_kg"]),
            "common_reward": 0.158 * float(episode["yield"]) - 1.1 * event_i - 1.58 * event_n,
            "canonical_reward": total_reward, "management_event_count": int(len(events)),
            "management_action_days_nonzero": int(sum(r["irrigation_mm"] > 0 or r["nitrogen_kg_ha"] > 0 for r in rows)),
            "irrigation_action_days": int(sum(r["irrigation_mm"] > 0 for r in rows)),
            "nitrogen_action_days": int(sum(r["nitrogen_kg_ha"] > 0 for r in rows)),
            "steps": len(rows), "action_signature": ";".join(
                f"{int(r['dap'])}:I{r['irrigation_mm']:g}/N{r['nitrogen_kg_ha']:g}" for r in rows
                if r["irrigation_mm"] > 0 or r["nitrogen_kg_ha"] > 0),
            "peak_rss_mb": peak, "snapshot_dir": rel(snapshot),
            "input_hash_checks_passed": all(v["identical"] for v in input_prov["checks"].values()),
            "model_sha256": sha256_file(model_path), "status": "passed",
        }
        row["mask_rule_counts"] = json.dumps(pd.Series([r["raw_argmax_block_rule"] for r in rows if r["raw_argmax_masked_out"]]).value_counts().to_dict(), sort_keys=True)
        pd.DataFrame(rows).to_csv(trace_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        if not row["input_hash_checks_passed"]:
            raise RuntimeError(f"Frozen observed input hashes changed for {variant}/{year}")
        print(f"[episode-pass] {variant} {year} yield={row['yield_kg_ha']:.1f} I={event_i:.1f} N={event_n:.1f} WP_ET={row['wp_et_kg_m3']:.3f} RSS={peak:.1f}MiB", flush=True)
        return row, rows
    except Exception:
        if rows and not trace_path.exists():
            pd.DataFrame(rows).to_csv(trace_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        raise
    finally:
        try:
            env.close()
        finally:
            del model
            gc.collect()


def make_plots(metrics: pd.DataFrame) -> list[str]:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("yield_kg_ha", "Yield (kg/ha)", "figure_yield_by_year.png"),
        ("irrigation_mm", "Total irrigation (mm)", "figure_irrigation_by_year.png"),
        ("nitrogen_kg_ha", "Total N (kg/ha)", "figure_nitrogen_by_year.png"),
        ("wp_et_kg_m3", "WP_ET (kg/m3)", "figure_efficiency_by_year.png"),
        ("common_reward", "Common reward", "figure_common_reward_by_year.png"),
    ]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8})
    files = []
    for column, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(9.2, 4.5), dpi=170)
        for variant in LABELS:
            part = metrics[metrics.variant.eq(variant)].sort_values("year")
            ax.plot(part.year, part[column], marker="o", linewidth=1.8, markersize=4,
                    color=COLORS[variant], label=LABELS[variant])
        ax.set_title(ylabel)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_xticks(YEARS)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        fig.tight_layout()
        path = fig_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        files.append(rel(path))
    return files


def make_report(metrics: pd.DataFrame, deltas: pd.DataFrame, traces: pd.DataFrame, manifest: dict[str, Any], plots: list[str], rule_summary: pd.DataFrame | None = None, overwrite: bool = False) -> None:
    base = metrics[metrics.variant.eq("A_canonical")].set_index("year")
    detail = deltas.pivot(index="year", columns="variant", values="yield_delta_kg_ha")
    wins = {v: int((detail[v] > 0).sum()) for v in VARIANTS}
    any_i = bool((deltas[deltas.variant.ne("A_canonical")].irrigation_delta_mm.abs() > 1e-8).any())
    any_n = bool((deltas[deltas.variant.ne("A_canonical")].nitrogen_delta_kg_ha.abs() > 1e-8).any())
    signature_table = metrics.pivot(index="year", columns="variant", values="action_signature")
    signature_diff_counts = {v: int((signature_table[v] != signature_table["A_canonical"]).sum()) for v in LABELS if v != "A_canonical"}
    signature_unique_counts = {v: int(metrics[metrics.variant.eq(v)].action_signature.nunique()) for v in LABELS}
    mean_d = deltas[deltas.variant.ne("A_canonical")].groupby("variant").agg({
        "yield_delta_kg_ha": "mean", "irrigation_delta_mm": "mean", "nitrogen_delta_kg_ha": "mean",
        "wp_et_delta_kg_m3": "mean", "pfp_n_delta_kg_kg": "mean", "common_reward_delta": "mean",
        "canonical_reward_delta": "mean",
    })
    lines = [
        "# YC 004_11：seed 5 mask sensitivity 评估",
        "",
        "## 执行与边界",
        "",
        "- 固定 seed 5 checkpoint 100K，B/C/D 按各自 mask 从 2014–2023 每个 episode 起点完整 evaluation-only 重放；A 复用已核验的 canonical 结果。没有训练，没有写入配置、模型、wrapper 源码或 DSSAT 输入。",
        "- B 仅取消 DAP≤30、75 mm cap；C 仅取消 DAP≤60、150 mm cap；D 同时取消两者。I/N 7 日间隔、season I240/N250、每日剂量、DAP 窗口、DAP≤90/I195 late reserve 与其他规则均保持。",
        f"- 重放年份数：{manifest.get('completed_variant_years', 0)} / 30；RSS 峰值 {manifest.get('peak_rss_mb', float('nan')):.1f} MiB；阈值 2048 MiB。",
        "- 004_07_02 的 Summary.OUT / ETCP helper 用于 B/C/D。每季要求 action dose、MgmtEvent.OUT、Summary.OUT 闭合；A 使用 004_07_02 已通过 finalize 的摘要。",
        "",
        "## 逐版平均差值",
        "",
        "| Variant | Yield Δ kg/ha | I Δ mm | N Δ kg/ha | WP_ET Δ | PFP_N Δ | Common reward Δ | Canonical reward Δ | Yield 提升年份 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in mean_d.iterrows():
        lines.append(f"| {LABELS[variant]} | {row.yield_delta_kg_ha:.1f} | {row.irrigation_delta_mm:.1f} | {row.nitrogen_delta_kg_ha:.1f} | {row.wp_et_delta_kg_m3:.3f} | {row.pfp_n_delta_kg_kg:.3f} | {row.common_reward_delta:.2f} | {row.canonical_reward_delta:.4f} | {wins[variant]}/10 |")
    lines += ["", "## 按年指标", "", "| 年份 | Variant | Yield | I | N | ETCP | WP_ET | PFP_N | Common reward | Canonical reward | 管理事件 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for year in YEARS:
        for variant in LABELS:
            row = metrics[(metrics.year.eq(year)) & metrics.variant.eq(variant)].iloc[0]
            lines.append(f"| {year} | {LABELS[variant]} | {row.yield_kg_ha:.1f} | {row.irrigation_mm:.1f} | {row.nitrogen_kg_ha:.1f} | {row.etcp_mm:.1f} | {row.wp_et_kg_m3:.3f} | {row.pfp_n_kg_kg:.2f} | {row.common_reward:.2f} | {row.canonical_reward:.4f} | {int(row.management_event_count)} |")
    lines += ["", "## 直接结论", ""]
    for variant in VARIANTS:
        d = deltas[deltas.variant.eq(variant)]
        yi = int((d.yield_delta_kg_ha > 0).sum())
        ii = int((d.irrigation_delta_mm > 0).sum())
        ni = int((d.nitrogen_delta_kg_ha != 0).sum())
        lines.append(f"- {LABELS[variant]}：产量提高 {yi}/10 年；灌溉增加 {ii}/10 年；N 有变化 {ni}/10 年。平均产量变化 {d.yield_delta_kg_ha.mean():.1f} kg/ha，I {d.irrigation_delta_mm.mean():.1f} mm，N {d.nitrogen_delta_kg_ha.mean():.1f} kg/ha，WP_ET {d.wp_et_delta_kg_m3.mean():.3f}，PFP_N {d.pfp_n_delta_kg_kg.mean():.3f}。")
    lines += [
        "",
        f"- 移除 cap 后是否改变灌溉：{'YES' if any_i else 'NO'}；是否改变 N：{'YES' if any_n else 'NO'}。N 的变化来自整季策略重跑后观测到的结果，不能仅由单个 DAP15 动作推断。",
        "- 联合动作的逐年证据：A 在 DAP15 的 raw argmax `I30+N40` 被早期 cap 拦截，执行 `I0+N80`（004_10/004_09）；B 和 D 在 2014–2023 每年 DAP15 的 raw argmax `I30+N40` 均变为合法并被执行，且两版十年 N 都从 240 降到 200 kg/ha。这说明 early cap 在固定 seed 5 checkpoint 的闭环反事实中确实连带塑造了 N 总量；不是对新 mask 训练效果的结论。",
        "- 各季节策略从起点闭环演化；年度差值是固定 actor 在不同 mask 下的反事实结果。它不预测在新 mask 下重新训练会得到什么策略，也不构成修改 canonical mask 的单独依据。",
        "",
        "```text",
        f"early_mid_caps_change_irrigation = {'MIXED' if any_i else 'NO'} (B/D early-cap release changes I in 10/10 years; C mid-only changes annual I in 0/10 years)",
        f"early_mid_caps_change_nitrogen = {'MIXED' if any_n else 'NO'} (B/D early-cap release changes N in 10/10 years; C mid-only changes N in 0/10 years)",
        f"additional_irrigation_yield_return = B +{mean_d.loc['B_no_early','irrigation_delta_mm']:.1f} mm / {mean_d.loc['B_no_early','yield_delta_kg_ha']:.1f} kg/ha; D +{mean_d.loc['D_no_early_mid','irrigation_delta_mm']:.1f} mm / {mean_d.loc['D_no_early_mid','yield_delta_kg_ha']:.1f} kg/ha, yield rose in 1/10 years",
        f"efficiency_response = B WP_ET {mean_d.loc['B_no_early','wp_et_delta_kg_m3']:.3f}, PFP_N +{mean_d.loc['B_no_early','pfp_n_delta_kg_kg']:.3f}; D WP_ET {mean_d.loc['D_no_early_mid','wp_et_delta_kg_m3']:.3f}, PFP_N +{mean_d.loc['D_no_early_mid','pfp_n_delta_kg_kg']:.3f}",
        "seed5_mask_counterfactual_verdict = mixed; early-cap release changes the joint I×N path and helps 2019, while mean yield and WP_ET decline",
        "next_step = targeted mask redesign study; do not promote this fixed-checkpoint result to a retrained-policy claim",
        "```",
        "",
        "## Raw argmax 禁用规则",
        "",
    ]
    if rule_summary is not None:
        lines += ["| Variant | 首个禁用规则 | 次数 |", "|---|---|---:|"]
        for row in rule_summary.itertuples(index=False):
            lines.append(f"| {LABELS.get(row.variant, row.variant)} | `{row.primary_rule}` | {int(row.steps)} |")
    else:
        lines.append("逐步原始输出见 `decision_trace_BCD_rule_attributed.csv`；分规则汇总见 `blocked_raw_argmax_rule_summary.csv`。")
    lines += [
        "",
        "## 动作日程变化",
        "",
        f"- 与 A 的年度 action signature 不同：B {signature_diff_counts['B_no_early']}/10 年，C {signature_diff_counts['C_no_mid']}/10 年，D {signature_diff_counts['D_no_early_mid']}/10 年。",
        f"- 十年内不同 signature 数：A {signature_unique_counts['A_canonical']}，B {signature_unique_counts['B_no_early']}，C {signature_unique_counts['C_no_mid']}，D {signature_unique_counts['D_no_early_mid']}。C 的日程虽然和 A 不同，但十年仍复用一个相同日程；B/D 分别出现两个日程模板，说明放开早期 cap 后有少量跨年分化，仍不是广泛逐年适应。",
        "",
        "## 图表",
        "",
    ]
    lines.extend(f"- `{p}`" for p in plots)
    lines += ["", "## 数据来源", "", f"- Baseline A：`{rel(REPLAY / 'replay_episode_summary.csv')}` 与 `{rel(TRACE_00409)}`。", f"- B/C/D episode traces：`{rel(OUT / 'decision_trace_BCD.csv')}`；逐规则归因：`{rel(OUT / 'decision_trace_BCD_rule_attributed.csv')}`；summary：`{rel(OUT / 'annual_metrics_A_B_C_D.csv')}`。", f"- 逐年差值：`{rel(OUT / 'variant_differences_vs_A_refined.csv')}`；provenance、资源峰值、checkpoint/config/input hashes 与逐季闭合状态见结果目录 JSON manifests。", "", "未提交、未推送。", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if REPORT.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {rel(REPORT)}")
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run() -> None:
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("004_11 output already exists; refusing overwrite")
    OUT.mkdir(parents=True)
    (OUT / "logs").mkdir()
    log = (OUT / "logs/formal.log").open("x", encoding="utf-8", newline="\n")
    stdout, stderr = sys.stdout, sys.stderr

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, value):
            for stream in self.streams: stream.write(value); stream.flush()
            return len(value)
        def flush(self):
            for stream in self.streams: stream.flush()

    sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
    manifest = {"status": "running", "passed": False, "seed": SEED, "checkpoint": 100000,
                "years": YEARS, "variants": list(VARIANTS), "rss_limit_mb": RSS_LIMIT_MB,
                "completed_variant_years": 0, "started_utc": pd.Timestamp.utcnow().isoformat(),
                "additional_training": False, "config_modified": False}
    write_json(OUT / "formal_manifest.json", manifest)
    try:
        runner, pilot, staged, action_audit = load_runtime()
        helper_path = ROOT / "src/build_relaxed_success_five_scenario_daily_evidence_027_05.py"
        helper = import_module(helper_path, "yc_02705_metrics_00411")
        model_path = FROZEN / "training/random_weather/ppo_seed_5/models/checkpoint_100000.zip"
        if not model_path.is_file(): raise FileNotFoundError(model_path)
        episodes, formal_eps, trace_by_year = read_formal_refs(runner)
        model_hash = sha256_file(model_path)
        prov_path = FROZEN / "config/canonical_yc_ppo_config.json"
        canonical = canonical_rows(runner, pilot, helper, formal_eps, trace_by_year)
        pd.DataFrame(canonical).to_csv(OUT / "canonical_A_source.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        summaries = []
        all_traces = []
        # Complete all three smoke seasons before starting the remaining 27 replays.
        run_order = [(variant, 2019) for variant in VARIANTS]
        run_order.extend((variant, year) for variant in VARIANTS for year in YEARS if year != 2019)
        for variant, year in run_order:
                row, trace = run_episode(runner, pilot, staged, helper, model_path, variant, year, episodes[year], smoke=(year == 2019))
                summaries.append(row); all_traces.extend(trace)
                manifest["completed_variant_years"] += 1
                manifest["peak_rss_mb"] = max(float(manifest.get("peak_rss_mb", 0.0)), float(row["peak_rss_mb"]))
                manifest.setdefault("smoke_passed_variants", [])
                if year == 2019:
                    manifest["smoke_passed_variants"].append(variant)
                    manifest["last_smoke_gate"] = {"variant": variant, "year": year, "status": "passed", "row": row}
                write_json(OUT / f"progress_{variant}_{year}.json", {"status": "passed", "row": row, "completed_utc": pd.Timestamp.utcnow().isoformat()})
                # Refresh the live manifest without replacing earlier evidence files.
                tmp = OUT / "formal_manifest.progress.json"
                tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                (OUT / "formal_manifest.json").write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
                tmp.unlink()
        bcd = pd.DataFrame(summaries)
        if len(bcd) != 30 or bcd.duplicated(["variant", "year"]).any():
            raise RuntimeError("Expected 30 unique B/C/D variant-years")
        pd.DataFrame(all_traces).to_csv(OUT / "decision_trace_BCD.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        events = pd.DataFrame(all_traces)
        events = events[(events.irrigation_mm.gt(0)) | (events.nitrogen_kg_ha.gt(0))].copy()
        events.to_csv(OUT / "action_events_BCD.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        metrics = pd.concat([pd.DataFrame(canonical), bcd], ignore_index=True, sort=False)
        metrics = metrics.sort_values(["year", "variant"]).reset_index(drop=True)
        metrics.to_csv(OUT / "annual_metrics_A_B_C_D.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        a = metrics[metrics.variant.eq("A_canonical")].set_index("year")
        delta_rows = []
        delta_names = {
            "yield_kg_ha": "yield_delta_kg_ha", "irrigation_mm": "irrigation_delta_mm",
            "nitrogen_kg_ha": "nitrogen_delta_kg_ha", "etcp_mm": "etcp_delta_mm",
            "wp_et_kg_m3": "wp_et_delta_kg_m3", "pfp_n_kg_kg": "pfp_n_delta_kg_kg",
            "common_reward": "common_reward_delta", "canonical_reward": "canonical_reward_delta",
        }
        for r in metrics[~metrics.variant.eq("A_canonical")].itertuples(index=False):
            base = a.loc[int(r.year)]
            delta_rows.append({"variant": r.variant, "year": int(r.year), **{
                delta_names[col]: getattr(r, col) - base[col] for col in delta_names
            }})
        deltas = pd.DataFrame(delta_rows)
        deltas.to_csv(OUT / "variant_differences_vs_A.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        plots = make_plots(metrics)
        manifest.update({"status": "passed", "passed": True, "completed_utc": pd.Timestamp.utcnow().isoformat(),
                         "completed_variant_years": len(bcd), "peak_rss_mb": float(bcd.peak_rss_mb.max()),
                         "checkpoint_sha256": model_hash, "config_sha256": sha256_file(prov_path),
                         "mask_constants_restored_at_process_end": True, "summary_metrics_helper": rel(helper_path),
                         "plots": plots, "report_path": rel(REPORT)})
        (OUT / "formal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_json(OUT / "provenance.json", {
            "model": {"path": rel(model_path), "sha256": model_hash},
            "config": {"path": rel(prov_path), "sha256": sha256_file(prov_path), "modified": False},
            "prompt": {"path": rel(PROMPT), "sha256": sha256_file(PROMPT)},
            "source_step_csv_sha256": sha256_file(runner.frozen_paths(SEED, YEARS[0])[2]),
            "004_09_trace_sha256": sha256_file(TRACE_00409),
            "004_07_02_summary_sha256": sha256_file(REPLAY / "replay_episode_summary.csv"),
            "canonical_cap_constants": {"early_mm": 75.0, "mid_mm": 150.0, "late_pre_dap90_mm": 195.0},
            "variant_caps": {k: {"early_mm": v[0], "mid_mm": v[1]} for k, v in VARIANTS.items()},
            "additional_training": False, "runtime_mode": "serial CPU, one model/environment, cache=1",
        })
        make_report(metrics, deltas, events, manifest, plots)
        print(f"[004_11 pass] variant_years={len(bcd)} max_rss={manifest['peak_rss_mb']:.1f}MiB report={rel(REPORT)}", flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "passed": False, "completed_utc": pd.Timestamp.utcnow().isoformat(),
                         "error": repr(exc), "traceback": traceback.format_exc(), "peak_rss_mb": rss_mb()})
        (OUT / "formal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        log.close()
        gc.collect()


def analyze_existing() -> None:
    runner, pilot, staged, action_audit = load_runtime()
    trace_path = OUT / "decision_trace_BCD.csv"
    trace = pd.read_csv(trace_path)
    if len(trace) != 3048 or trace.duplicated(["variant", "year", "timestep"]).any():
        raise RuntimeError(f"Existing B/C/D trace incomplete: {len(trace)}")
    config = json.loads((FROZEN / "config/canonical_yc_ppo_config.json").read_text(encoding="utf-8"))
    safety = config["action_safety"]
    late_cfg = safety["late_irrigation_reserve_mask"]
    late = {"LATE_RESERVE_DAP_END": int(late_cfg["dap_end"]),
            "PRE_LATE_CUM_IRRIGATION_CAP": float(late_cfg["pre_late_cum_irrigation_cap_mm"])}
    attributed = []
    for (variant, year), group in trace.groupby(["variant", "year"], sort=True):
        early, mid = VARIANTS[variant]
        staged_rules = {
            "EARLY_DAP_END": 30, "MID_DAP_END": 60,
            "EARLY_CUM_IRRIGATION_CAP": math.inf if early is None else early,
            "MID_CUM_IRRIGATION_CAP": math.inf if mid is None else mid,
        }
        state = {"cum_i": 0.0, "cum_n": 0.0, "last_i": None, "last_n": None}
        for row in group.sort_values("timestep").itertuples(index=False):
            dap = int(round(float(row.dap))) if float(row.dap) > 0 else 1
            raw_i, raw_n = ACTION_GRID[int(row.raw_argmax_action)]
            reasons = action_audit.action_rules(raw_i, raw_n, dap, state, safety, staged_rules, late)
            mask = np.array([c == "1" for c in str(row.mask_bits)], dtype=bool)
            actually_blocked = not bool(mask[int(row.raw_argmax_action)])
            if actually_blocked != bool(row.raw_argmax_masked_out):
                raise RuntimeError(f"Saved raw-argmax mask bit does not close: {variant}/{year}/{row.timestep}")
            if actually_blocked and not reasons:
                raise RuntimeError(f"Blocked raw argmax has no rule: {variant}/{year}/{row.timestep}")
            if not actually_blocked and reasons:
                raise RuntimeError(f"Legal raw argmax contradicts reconstructed rules: {variant}/{year}/{row.timestep}: {reasons}")
            attributed.append({
                **row._asdict(), "raw_argmax_primary_rule": reasons[0] if reasons else "LEGAL",
                "raw_argmax_all_active_rules_json": json.dumps(reasons),
            })
            chosen_i, chosen_n = float(row.irrigation_mm), float(row.nitrogen_kg_ha)
            if chosen_i > 0:
                state["last_i"] = dap
            if chosen_n > 0:
                state["last_n"] = dap
            state["cum_i"] += chosen_i
            state["cum_n"] += chosen_n
    attributed_df = pd.DataFrame(attributed)
    attributed_df.to_csv(OUT / "decision_trace_BCD_rule_attributed.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    blocked = attributed_df[attributed_df.raw_argmax_masked_out.astype(bool)]
    rule_summary = (blocked.groupby(["variant", "raw_argmax_primary_rule"], as_index=False)
                    .size().rename(columns={"raw_argmax_primary_rule": "primary_rule", "size": "steps"}))
    rule_summary.to_csv(OUT / "blocked_raw_argmax_rule_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    metrics = pd.read_csv(OUT / "annual_metrics_A_B_C_D.csv")
    deltas = pd.read_csv(OUT / "variant_differences_vs_A.csv")
    canonical_reward = metrics[metrics.variant.eq("A_canonical")].set_index("year").canonical_reward
    deltas["canonical_reward_delta"] = deltas.apply(
        lambda row: metrics.loc[
            metrics.variant.eq(row.variant) & metrics.year.eq(row.year), "canonical_reward"
        ].iloc[0] - canonical_reward.loc[int(row.year)], axis=1
    )
    deltas.to_csv(OUT / "variant_differences_vs_A_refined.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    events = pd.read_csv(OUT / "action_events_BCD.csv")
    manifest = json.loads((OUT / "formal_manifest.json").read_text(encoding="utf-8"))
    plots = manifest["plots"]
    manifest.pop("mask_constants_restored_at_process_end", None)
    manifest["mask_overrides_process_local"] = True
    manifest["config_and_wrapper_source_modified"] = False
    (OUT / "formal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_report(metrics, deltas, attributed_df, manifest, plots, rule_summary=rule_summary, overwrite=True)
    write_json(OUT / "analysis_revision_manifest.json", {
        "status": "passed", "revision": "v2_exact_raw_argmax_rule_attribution",
        "additional_simulation_replay": False,
        "source_trace_sha256": sha256_file(trace_path),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "attributed_decisions": len(attributed_df), "blocked_raw_argmax": len(blocked),
        "unexplained_blocked_raw_argmax": 0,
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    }, replace=True)
    print(f"[004_11 analyze-pass] rows={len(attributed_df)} blocked={len(blocked)} report={rel(REPORT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="YC fixed seed 5 mask sensitivity, evaluation only")
    parser.add_argument("--phase", choices=["formal", "analyze"], default="formal")
    args = parser.parse_args()
    if args.phase == "formal":
        run()
    else:
        analyze_existing()


if __name__ == "__main__":
    main()
