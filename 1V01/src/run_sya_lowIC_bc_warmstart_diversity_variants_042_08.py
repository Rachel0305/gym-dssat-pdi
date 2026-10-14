"""042_08: Compare BC warm-start variants for teacher diversity retention.

Offline/lightweight relative to PPO training:
- Builds MaskablePPO policy networks.
- Applies BC-only pretraining variants.
- Evaluates the deterministic BC-init policy on SYA 2014--2023.
- Does NOT call model.learn().

The purpose is to test whether the template-like policy appears already at
BC initialization and whether weaker/natural BC variants preserve diversity.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_042_02 as base04202
import run_sya_lowIC_teacher_warmstart_maskableppo_041_03 as base04103
import build_sya_lowIC_teacher_candidate_search_041_00 as base04100
import ppo_safe_rendering


TASK_ID = "042_08"
TASK_NAME = "sya_lowIC_bc_warmstart_diversity_variants"
STATION = "SYA"
SITE = "SY"
SEED = 0
YEARS = list(range(2014, 2024))

OUTPUT_ROOT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC_PATH = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT_PATH = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"


@dataclass(frozen=True)
class BCVariant:
    name: str
    epochs: int
    batch_size: int
    lr: float
    mode: str
    nonzero_fraction: float | None = None


VARIANTS = [
    BCVariant("baseline_balanced_20ep_f50", epochs=20, batch_size=256, lr=1e-4, mode="balanced", nonzero_fraction=0.5),
    BCVariant("weak_balanced_5ep_f30", epochs=5, batch_size=256, lr=1e-4, mode="balanced", nonzero_fraction=0.3),
    BCVariant("natural_weighted_20ep", epochs=20, batch_size=256, lr=1e-4, mode="natural", nonzero_fraction=None),
]


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs", "bc_dataset"]:
        (OUTPUT_ROOT / rel).mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def masked_argmax_from_distribution(distribution: Any, masks: torch.Tensor) -> torch.Tensor:
    probs = distribution.distribution.probs
    masked_probs = probs.masked_fill(~masks, -1.0)
    return torch.argmax(masked_probs, dim=1)


def pretrain_bc_variant(model: Any, dataset: dict[str, np.ndarray], variant: BCVariant, out: Path) -> pd.DataFrame:
    device = model.device
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.long, device=device)
    masks = torch.as_tensor(dataset["masks"], dtype=torch.bool, device=device)
    weights = torch.as_tensor(dataset["weights"], dtype=torch.float32, device=device)
    action_np = np.asarray(dataset["actions"], dtype=int)
    zero_idx = np.where(action_np == 0)[0]
    nonzero_idx = np.where(action_np != 0)[0]

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=variant.lr)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    n = int(obs.shape[0])
    n_batches = max(1, int(np.ceil(n / variant.batch_size)))

    for epoch in range(1, variant.epochs + 1):
        losses: list[float] = []
        acc_all: list[float] = []
        acc_zero: list[float] = []
        acc_nonzero: list[float] = []
        pred_nonzero_rates: list[float] = []

        if variant.mode == "natural":
            batches = []
            order = rng.permutation(n)
            for start in range(0, n, variant.batch_size):
                batches.append(order[start : start + variant.batch_size])
        elif variant.mode == "balanced":
            if len(nonzero_idx) == 0:
                raise RuntimeError("No nonzero teacher actions available for balanced BC.")
            nonzero_per_batch = max(1, int(round(variant.batch_size * float(variant.nonzero_fraction))))
            zero_per_batch = max(1, variant.batch_size - nonzero_per_batch)
            batches = []
            for _ in range(n_batches):
                nz = rng.choice(nonzero_idx, size=nonzero_per_batch, replace=len(nonzero_idx) < nonzero_per_batch)
                zz = rng.choice(zero_idx, size=zero_per_batch, replace=len(zero_idx) < zero_per_batch)
                idx_np = np.concatenate([nz, zz])
                rng.shuffle(idx_np)
                batches.append(idx_np)
        else:
            raise ValueError(f"Unknown BC mode: {variant.mode}")

        for idx_np in batches:
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_masks = masks[idx]
            batch_weights = weights[idx]
            try:
                distribution = model.policy.get_distribution(batch_obs, action_masks=batch_masks)
            except TypeError:
                distribution = model.policy.get_distribution(batch_obs)
            log_prob = distribution.log_prob(batch_actions)
            loss = -((log_prob * batch_weights).sum() / batch_weights.sum().clamp_min(1e-9))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 10.0)
            optimizer.step()

            with torch.no_grad():
                pred = masked_argmax_from_distribution(distribution, batch_masks)
                zero_mask = batch_actions == 0
                nonzero_mask = batch_actions != 0
                acc_all.append(float((pred == batch_actions).float().mean().detach().cpu()))
                if zero_mask.any():
                    acc_zero.append(float((pred[zero_mask] == batch_actions[zero_mask]).float().mean().detach().cpu()))
                if nonzero_mask.any():
                    acc_nonzero.append(float((pred[nonzero_mask] == batch_actions[nonzero_mask]).float().mean().detach().cpu()))
                pred_nonzero_rates.append(float((pred != 0).float().mean().detach().cpu()))
            losses.append(float(loss.detach().cpu()))

        row = {
            "variant": variant.name,
            "bc_epoch": epoch,
            "bc_loss": float(np.mean(losses)),
            "bc_action_accuracy": float(np.mean(acc_all)),
            "bc_zero_action_accuracy": float(np.mean(acc_zero)) if acc_zero else np.nan,
            "bc_nonzero_action_accuracy": float(np.mean(acc_nonzero)) if acc_nonzero else np.nan,
            "bc_pred_nonzero_rate": float(np.mean(pred_nonzero_rates)),
            "mode": variant.mode,
            "nonzero_fraction": variant.nonzero_fraction,
            "epochs": variant.epochs,
            "batch_size": variant.batch_size,
            "lr": variant.lr,
            "nonzero_teacher_samples": int(len(nonzero_idx)),
            "zero_teacher_samples": int(len(zero_idx)),
        }
        if not np.isfinite(row["bc_loss"]):
            raise RuntimeError(f"BC loss is not finite at epoch {epoch} for {variant.name}")
        rows.append(row)

    bc_log = pd.DataFrame(rows)
    bc_log.to_csv(out / "logs" / f"{TASK_ID}_{variant.name}_bc_log.csv", index=False, encoding="utf-8-sig")
    return bc_log


def action_signature_from_daily(daily: pd.DataFrame) -> str:
    if daily.empty:
        return "NO_NONZERO_ACTION"
    parts = []
    for row in daily.sort_values("dap").itertuples(index=False):
        parts.append(f"DAP{int(round(float(row.dap)))}:I{float(row.irrigation):g}/N{float(row.nitrogen):g}")
    return "; ".join(parts)


def summarize_action_diversity(eval_summary: pd.DataFrame, daily: pd.DataFrame, variant_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    event_rows = []
    for year, g in daily.groupby("year"):
        nonzero = g[(g["irrigation"] > 0) | (g["nitrogen"] > 0)].copy()
        irr = nonzero[nonzero["irrigation"] > 0]
        nit = nonzero[nonzero["nitrogen"] > 0]
        rows.append(
            {
                "variant": variant_name,
                "year": int(year),
                "action_signature": action_signature_from_daily(nonzero),
                "irrigation_signature": action_signature_from_daily(irr),
                "nitrogen_signature": action_signature_from_daily(nit),
                "irrigation_events": int(len(irr)),
                "nitrogen_events": int(len(nit)),
                "total_irrigation": float(g["irrigation"].sum()),
                "total_nitrogen": float(g["nitrogen"].sum()),
            }
        )
        for action_type, sub, amount_col in [
            ("irrigation", irr, "irrigation"),
            ("nitrogen", nit, "nitrogen"),
        ]:
            sub = sub.sort_values("dap").copy()
            for order, erow in enumerate(sub.itertuples(index=False), start=1):
                event_rows.append(
                    {
                        "variant": variant_name,
                        "year": int(year),
                        "action_type": action_type,
                        "event_order": int(order),
                        "dap": float(erow.dap),
                        "amount": float(getattr(erow, amount_col)),
                    }
                )
    by_year = pd.DataFrame(rows)
    event_df = pd.DataFrame(event_rows)
    pos_rows = []
    if not event_df.empty:
        for (action_type, order), g in event_df.groupby(["action_type", "event_order"]):
            pos_rows.append(
                {
                    "variant": variant_name,
                    "action_type": action_type,
                    "event_order": int(order),
                    "n_years_with_event": int(g["year"].nunique()),
                    "mean_dap": float(g["dap"].mean()),
                    "sd_dap": float(g["dap"].std(ddof=0)),
                    "min_dap": float(g["dap"].min()),
                    "max_dap": float(g["dap"].max()),
                    "mean_amount": float(g["amount"].mean()),
                    "sd_amount": float(g["amount"].std(ddof=0)),
                }
            )
    pos = pd.DataFrame(pos_rows)
    return by_year, event_df, pos


def evaluate_bc_model(model_path: Path, variant_name: str, config: dict[str, Any], env_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = base04100.load_baseline_thresholds()
    thresholds = thresholds[thresholds["year"].isin(YEARS)].copy()
    eval_df, daily = base04103.evaluate_checkpoint(OUTPUT_ROOT, model_path, 0, config, env_config, thresholds)
    eval_df["variant"] = variant_name
    daily["variant"] = variant_name
    # normalize daily action column names used by current code base
    rename_map = {}
    for cand in ["irrigation", "irrigation_mm", "irrigation_mm_action", "amir", "requested_irrigation_mm"]:
        if cand in daily.columns:
            rename_map[cand] = "irrigation"
            break
    for cand in ["nitrogen", "nitrogen_kg_ha", "nitrogen_kg_ha_action", "anfer", "requested_nitrogen_kg_ha"]:
        if cand in daily.columns:
            rename_map[cand] = "nitrogen"
            break
    for cand in ["dap", "DAP", "dap_before_action"]:
        if cand in daily.columns:
            rename_map[cand] = "dap"
            break
    daily = daily.rename(columns=rename_map)
    if not {"year", "dap", "irrigation", "nitrogen"}.issubset(daily.columns):
        raise RuntimeError(f"Daily evaluation table lacks required action columns: {list(daily.columns)}")
    return eval_df, daily


def pass_diversity(row: pd.Series) -> bool:
    return bool(
        row["unique_nonzero_action_signatures"] >= 3
        and row["sd_total_irrigation"] > 0
        and row["sd_total_nitrogen"] > 0
        and row["mean_irrigation_event_dap_sd"] > 0
    )


def write_record(result: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        f"# {TASK_ID} SYA lowIC BC warm-start 多样性保留审计",
        "",
        "## 结论",
        "",
        f"- 分支：`{result['branch']}`",
        f"- 通过多样性最低标准的 BC 变体数：{result['passing_variant_count']}",
        f"- next_step_allowed：`{str(result['next_step_allowed']).lower()}`",
        "",
        "## 审计口径",
        "",
        "- 只做 BC init，不做 PPO finetune。",
        "- 所有变体沿用同一份 041_02 teacher 数据、042_02 观测归一化/天气特征/奖励配置。",
        "- 通过标准：unique nonzero signature >= 3，且总水/总氮/灌溉事件 DAP 都存在跨年差异。",
        "",
        "## 变体结果",
        "",
        summary.to_markdown(index=False),
        "",
        "## 文件",
        "",
    ]
    for k, v in result["outputs"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(dry_run: bool = False, variants_filter: list[str] | None = None) -> None:
    ensure_dirs()
    base04202.configure_base_module()
    try:
        selected = base04103.load_selected_teachers()
        config, env_config = base04103.build_config_and_env_config(YEARS)
        config_path = OUTPUT_ROOT / "configs" / f"{TASK_ID}_config.json"
        env_config_path = OUTPUT_ROOT / "configs" / f"{TASK_ID}_env_config.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        env_config_path.write_text(json.dumps(env_config, indent=2, ensure_ascii=False), encoding="utf-8")
        active_variants = [v for v in VARIANTS if variants_filter is None or v.name in variants_filter]
        if dry_run:
            result = {
                "task": f"{TASK_ID}_{TASK_NAME}",
                "mode": "dry_run",
                "prompt_exists": PROMPT_PATH.exists(),
                "teacher_years": selected["year"].astype(int).tolist(),
                "variants": [asdict(v) for v in active_variants],
                "input_root": str(base04202.LOWIC_INPUT_ROOT.relative_to(ROOT)),
                "input_root_exists": base04202.LOWIC_INPUT_ROOT.exists(),
                "next_step_allowed": bool(PROMPT_PATH.exists() and base04202.LOWIC_INPUT_ROOT.exists()),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        dataset, bc_meta = base04103.collect_bc_dataset(OUTPUT_ROOT, selected, config, env_config)
        bc_meta.to_csv(OUTPUT_ROOT / "bc_dataset" / f"{TASK_ID}_bc_dataset_meta.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(OUTPUT_ROOT / "bc_dataset" / f"{TASK_ID}_bc_dataset.npz", **dataset)

        from sb3_contrib import MaskablePPO

        all_bc_logs = []
        all_eval = []
        all_daily = []
        all_by_year = []
        all_events = []
        all_pos = []
        old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04202.LOWIC_INPUT_ROOT
        try:
            for variant in active_variants:
                train_env = base04103.RandomYearEnv(config, env_config, YEARS, SEED)
                try:
                    model = MaskablePPO(
                        "MlpPolicy",
                        train_env,
                        verbose=0,
                        seed=SEED,
                        tensorboard_log=str(OUTPUT_ROOT / "tensorboard" / STATION / variant.name),
                        **base04103.ppo04040.base03222.base.ppo_kwargs(config),
                    )
                    bc_log = pretrain_bc_variant(model, dataset, variant, OUTPUT_ROOT)
                    all_bc_logs.append(bc_log)
                    model_path = OUTPUT_ROOT / "models" / STATION / f"{STATION}_{TASK_ID}_{variant.name}_bc_init.zip"
                    model.save(str(model_path.with_suffix("")))
                finally:
                    train_env.close()

                eval_df, daily = evaluate_bc_model(model_path, variant.name, config, env_config)
                all_eval.append(eval_df)
                all_daily.append(daily)
                by_year, events, pos = summarize_action_diversity(eval_df, daily, variant.name)
                all_by_year.append(by_year)
                all_events.append(events)
                all_pos.append(pos)
        finally:
            ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

        bc_logs = pd.concat(all_bc_logs, ignore_index=True)
        eval_summary = pd.concat(all_eval, ignore_index=True)
        daily_summary = pd.concat(all_daily, ignore_index=True)
        by_year = pd.concat(all_by_year, ignore_index=True)
        events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        pos = pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()

        bc_logs_path = OUTPUT_ROOT / "logs" / f"{TASK_ID}_all_bc_logs.csv"
        eval_path = OUTPUT_ROOT / "evaluation" / f"{TASK_ID}_bc_init_validation_summary.csv"
        daily_path = OUTPUT_ROOT / "evaluation" / f"{TASK_ID}_bc_init_daily_outputs.csv"
        by_year_path = OUTPUT_ROOT / "evaluation" / f"{TASK_ID}_bc_init_action_signature_by_year.csv"
        events_path = OUTPUT_ROOT / "evaluation" / f"{TASK_ID}_bc_init_action_events.csv"
        pos_path = OUTPUT_ROOT / "evaluation" / f"{TASK_ID}_bc_init_event_position_variability.csv"
        summary_path = OUTPUT_ROOT / "evaluation" / f"{TASK_ID}_bc_variant_diversity_summary.csv"
        result_path = OUTPUT_ROOT / f"{TASK_ID}_result.json"

        bc_logs.to_csv(bc_logs_path, index=False, encoding="utf-8-sig")
        eval_summary.to_csv(eval_path, index=False, encoding="utf-8-sig")
        daily_summary.to_csv(daily_path, index=False, encoding="utf-8-sig")
        by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")
        events.to_csv(events_path, index=False, encoding="utf-8-sig")
        pos.to_csv(pos_path, index=False, encoding="utf-8-sig")

        metric_rows = []
        for variant_name, g in by_year.groupby("variant"):
            p = pos[pos["variant"] == variant_name]
            irr_pos = p[p["action_type"] == "irrigation"]
            nit_pos = p[p["action_type"] == "nitrogen"]
            ev = eval_summary[eval_summary["variant"] == variant_name]
            row = {
                "variant": variant_name,
                "unique_nonzero_action_signatures": int(g["action_signature"].nunique()),
                "unique_irrigation_signatures": int(g["irrigation_signature"].nunique()),
                "unique_nitrogen_signatures": int(g["nitrogen_signature"].nunique()),
                "sd_total_irrigation": float(g["total_irrigation"].std(ddof=0)),
                "sd_total_nitrogen": float(g["total_nitrogen"].std(ddof=0)),
                "mean_irrigation_event_dap_sd": float(irr_pos["sd_dap"].mean()) if not irr_pos.empty else 0.0,
                "mean_nitrogen_event_dap_sd": float(nit_pos["sd_dap"].mean()) if not nit_pos.empty else 0.0,
                "mean_yield": float(ev["grain_yield_kg_ha"].mean()),
                "mean_wp_et": float(ev["WP_ET_kg_m3"].mean()),
                "mean_pfp_n": float(ev["PFP_N_kg_kg"].mean()),
                "any_metric_win_years": int(ev["any_metric_win_vs_four_max"].sum()),
                "all3_win_years": int(ev["all3_win_vs_four_max"].sum()),
            }
            metric_rows.append(row)
        summary = pd.DataFrame(metric_rows)
        summary["diversity_guardrail_pass"] = summary.apply(pass_diversity, axis=1)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

        passing = summary[summary["diversity_guardrail_pass"]].copy()
        branch = "A_bc_variant_preserves_diversity" if len(passing) else "C_no_bc_variant_preserves_diversity"
        result = {
            "task": f"{TASK_ID}_{TASK_NAME}",
            "branch": branch,
            "next_step_allowed": bool(len(passing)),
            "passing_variant_count": int(len(passing)),
            "passing_variants": passing["variant"].tolist(),
            "outputs": {
                "record_md": str(DOC_PATH.relative_to(ROOT)),
                "summary": str(summary_path.relative_to(ROOT)),
                "by_year": str(by_year_path.relative_to(ROOT)),
                "event_positions": str(pos_path.relative_to(ROOT)),
                "eval_summary": str(eval_path.relative_to(ROOT)),
                "bc_logs": str(bc_logs_path.relative_to(ROOT)),
                "result_json": str(result_path.relative_to(ROOT)),
            },
        }
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        write_record(result, summary)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        base04202.restore_base_module()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variants", type=str, default="")
    args = parser.parse_args()
    variants_filter = [x.strip() for x in args.variants.split(",") if x.strip()] or None
    run(dry_run=args.dry_run, variants_filter=variants_filter)


if __name__ == "__main__":
    main()
