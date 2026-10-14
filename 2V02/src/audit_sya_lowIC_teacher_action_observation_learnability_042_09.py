"""042_09: Can current observations explain teacher actions?

This is a lightweight offline audit. It does not train PPO and does not call
DSSAT. It uses the 042_08 BC dataset, which already contains the 042_02
normalized observation plus weather/forecast features.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASK_ID = "042_09"
TASK_NAME = "sya_lowIC_teacher_action_observation_learnability_audit"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TABLE_DIR = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"

DATASET_NPZ = ROOT / "benchmark_results/042_08_sya_lowIC_bc_warmstart_diversity_variants/bc_dataset/042_08_bc_dataset.npz"
DATASET_META = ROOT / "benchmark_results/042_08_sya_lowIC_bc_warmstart_diversity_variants/bc_dataset/042_08_bc_dataset_meta.csv"

FEATURE_NAMES = [
    "dap",
    "swfac",
    "nstres",
    "grnwt",
    "topwt",
    "rootd",
    "lai",
    "cum_irrigation",
    "cum_nitrogen",
    "days_since_irrigation",
    "days_since_fertilization",
    "remaining_irrigation",
    "remaining_nitrogen",
    "dap_norm",
    "stage_veg",
    "stage_repro",
    "stage_late",
    "soil_water_top",
    "soil_water_root",
    "soil_n_top",
    "soil_n_root",
    "tmax",
    "tmin",
    "srad",
    "rain",
    "rain_today",
    "tmin_today",
    "rain_past7",
    "rain_future7",
    "future7_tmean",
]


@dataclass
class Result:
    task: str
    branch: str
    next_step_allowed: bool
    rows: int
    years: int
    nonzero_rows: int
    no_op_rows: int
    nonzero_rate: float
    binary_loso_mean_balanced_accuracy: float | None
    binary_loso_dummy_mean_balanced_accuracy: float | None
    nonzero_multiclass_loso_mean_macro_f1: float | None
    nonzero_multiclass_dummy_mean_macro_f1: float | None
    nonzero_multiclass_macro_f1_gain: float | None
    dap_conflict_count: int
    dap_conflict_share: float
    record_md: str
    outputs: dict[str, str]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not DATASET_NPZ.exists() or not DATASET_META.exists():
        raise FileNotFoundError(f"Missing BC dataset: {DATASET_NPZ} or {DATASET_META}")
    data = np.load(DATASET_NPZ)
    meta = pd.read_csv(DATASET_META)
    obs = data["obs"].astype(float)
    actions = data["actions"].astype(int)
    masks = data["masks"].astype(bool)
    weights = data["weights"].astype(float)
    if len(meta) != obs.shape[0]:
        raise RuntimeError(f"Meta rows {len(meta)} != obs rows {obs.shape[0]}")
    return meta, obs, actions, masks, weights


def safe_feature_frame(obs: np.ndarray) -> pd.DataFrame:
    cols = FEATURE_NAMES[: obs.shape[1]]
    if len(cols) < obs.shape[1]:
        cols += [f"feature_{i}" for i in range(len(cols), obs.shape[1])]
    return pd.DataFrame(obs, columns=cols)


def profile_tables(meta: pd.DataFrame, obs: np.ndarray, actions: np.ndarray, masks: np.ndarray) -> dict[str, pd.DataFrame]:
    action_counts = (
        pd.Series(actions, name="teacher_action_index")
        .value_counts()
        .sort_index()
        .rename_axis("teacher_action_index")
        .reset_index(name="samples")
    )
    action_counts["share"] = action_counts["samples"] / len(actions)

    by_year = (
        meta.assign(is_nonzero=actions != 0)
        .groupby("year", as_index=False)
        .agg(
            rows=("teacher_action_index", "size"),
            nonzero_rows=("is_nonzero", "sum"),
            unique_actions=("teacher_action_index", "nunique"),
            unique_nonzero_actions=("teacher_action_index", lambda s: int(pd.Series(s)[pd.Series(s) != 0].nunique())),
        )
    )
    by_year["nonzero_rate"] = by_year["nonzero_rows"] / by_year["rows"]

    mask_patterns = pd.Series(["".join("1" if v else "0" for v in row) for row in masks], name="mask_pattern")
    mask_counts = mask_patterns.value_counts().rename_axis("mask_pattern").reset_index(name="samples")

    feature = safe_feature_frame(obs)
    feature_summary = feature.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "feature"})

    return {
        "action_counts": action_counts,
        "by_year_profile": by_year,
        "mask_counts": mask_counts,
        "feature_summary": feature_summary,
    }


def loso_binary(meta: pd.DataFrame, obs: np.ndarray, actions: np.ndarray) -> pd.DataFrame:
    y = (actions != 0).astype(int)
    rows: list[dict[str, Any]] = []
    for year in sorted(meta["year"].unique()):
        train = meta["year"].to_numpy() != year
        test = ~train
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            # balanced accuracy is less meaningful if held-out year has one class only;
            # keep accuracy and set balanced accuracy as nan.
            can_bal = False
        else:
            can_bal = True
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0),
        )
        dummy = DummyClassifier(strategy="most_frequent")
        model.fit(obs[train], y[train])
        dummy.fit(obs[train], y[train])
        pred = model.predict(obs[test])
        dpred = dummy.predict(obs[test])
        rows.append(
            {
                "heldout_year": int(year),
                "test_rows": int(test.sum()),
                "test_nonzero_rows": int(y[test].sum()),
                "model_accuracy": float(accuracy_score(y[test], pred)),
                "dummy_accuracy": float(accuracy_score(y[test], dpred)),
                "model_balanced_accuracy": float(balanced_accuracy_score(y[test], pred)) if can_bal else np.nan,
                "dummy_balanced_accuracy": float(balanced_accuracy_score(y[test], dpred)) if can_bal else np.nan,
                "model_f1_nonzero": float(f1_score(y[test], pred, zero_division=0)),
                "dummy_f1_nonzero": float(f1_score(y[test], dpred, zero_division=0)),
                "pred_nonzero_rate": float(np.mean(pred)),
                "true_nonzero_rate": float(np.mean(y[test])),
            }
        )
    return pd.DataFrame(rows)


def loso_nonzero_multiclass(meta: pd.DataFrame, obs: np.ndarray, actions: np.ndarray) -> pd.DataFrame:
    nz = actions != 0
    meta_nz = meta[nz].reset_index(drop=True)
    obs_nz = obs[nz]
    y = actions[nz]
    rows: list[dict[str, Any]] = []
    for year in sorted(meta_nz["year"].unique()):
        train = meta_nz["year"].to_numpy() != year
        test = ~train
        train_classes = set(y[train])
        test_classes = set(y[test])
        if len(train_classes) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=0),
        )
        dummy = DummyClassifier(strategy="most_frequent")
        model.fit(obs_nz[train], y[train])
        dummy.fit(obs_nz[train], y[train])
        pred = model.predict(obs_nz[test])
        dpred = dummy.predict(obs_nz[test])
        labels = sorted(train_classes | test_classes)
        rows.append(
            {
                "heldout_year": int(year),
                "test_nonzero_rows": int(test.sum()),
                "test_actions": ",".join(map(str, sorted(test_classes))),
                "unseen_test_actions": ",".join(map(str, sorted(test_classes - train_classes))),
                "model_accuracy": float(accuracy_score(y[test], pred)),
                "dummy_accuracy": float(accuracy_score(y[test], dpred)),
                "model_macro_f1": float(f1_score(y[test], pred, labels=labels, average="macro", zero_division=0)),
                "dummy_macro_f1": float(f1_score(y[test], dpred, labels=labels, average="macro", zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def dap_conflict_audit(meta: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    df = meta.copy()
    df["teacher_action_index"] = actions
    df = df[df["teacher_action_index"] != 0].copy()
    rows = []
    for dap, g in df.groupby("dap"):
        rows.append(
            {
                "dap": int(dap),
                "nonzero_samples": int(len(g)),
                "years": ",".join(map(str, sorted(g["year"].astype(int).unique()))),
                "unique_actions": int(g["teacher_action_index"].nunique()),
                "actions": ",".join(map(str, sorted(g["teacher_action_index"].astype(int).unique()))),
                "irrigation_values": ",".join(map(lambda x: f"{x:g}", sorted(g["teacher_irrigation_mm"].astype(float).unique()))),
                "nitrogen_values": ",".join(map(lambda x: f"{x:g}", sorted(g["teacher_nitrogen_kg_ha"].astype(float).unique()))),
            }
        )
    out = pd.DataFrame(rows).sort_values(["unique_actions", "nonzero_samples"], ascending=[False, False])
    out["has_conflict"] = out["unique_actions"] > 1
    return out


def feature_action_association(meta: pd.DataFrame, obs: np.ndarray, actions: np.ndarray) -> pd.DataFrame:
    features = safe_feature_frame(obs)
    is_nonzero = actions != 0
    rows = []
    for feature in features.columns:
        a = features.loc[is_nonzero, feature].astype(float)
        b = features.loc[~is_nonzero, feature].astype(float)
        pooled = float(np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2.0)) if len(a) and len(b) else np.nan
        diff = float((a.mean() - b.mean()) / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "mean_nonzero_days": float(a.mean()) if len(a) else np.nan,
                "mean_noop_days": float(b.mean()) if len(b) else np.nan,
                "standardized_difference": diff,
                "abs_standardized_difference": abs(diff) if np.isfinite(diff) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_standardized_difference", ascending=False)


def decide(result_parts: dict[str, Any]) -> tuple[str, bool]:
    binary_bal = result_parts.get("binary_bal")
    multi_f1 = result_parts.get("multi_f1")
    dummy_f1 = result_parts.get("dummy_f1")
    if binary_bal is not None and multi_f1 is not None and dummy_f1 is not None:
        gain = multi_f1 - dummy_f1
        if binary_bal >= 0.70 and multi_f1 >= 0.40 and gain >= 0.15:
            return "A_observation_supports_teacher_action_learning", True
        if binary_bal >= 0.70:
            return "B_nonzero_timing_learnable_but_action_identity_weak", False
    return "C_current_observation_teacher_labels_not_reliably_learnable", False


def write_record(result: Result, tables: dict[str, pd.DataFrame]) -> None:
    branch_text = {
        "A_observation_supports_teacher_action_learning": "当前 observation 对 teacher 动作有可学习信号，可以继续设计响应性 PPO/BC 训练目标。",
        "B_nonzero_timing_learnable_but_action_identity_weak": "是否操作可能可学，但具体动作/剂量身份弱，建议考虑分层策略或重构动作标签。",
        "C_current_observation_teacher_labels_not_reliably_learnable": "当前 observation + teacher 标签不足以可靠学习跨年动作差异，不建议继续单纯调 PPO/BC。",
    }
    lines = [
        f"# {TASK_ID} SYA lowIC teacher action observation learnability audit",
        "",
        "## 结论",
        "",
        f"- 分支：`{result.branch}`",
        f"- 解释：{branch_text.get(result.branch, '')}",
        f"- next_step_allowed：`{str(result.next_step_allowed).lower()}`",
        "",
        "## 数据概况",
        "",
        f"- 总样本：{result.rows}",
        f"- 年份数：{result.years}",
        f"- no-op 样本：{result.no_op_rows}",
        f"- 非零动作样本：{result.nonzero_rows}",
        f"- 非零动作比例：{result.nonzero_rate:.4f}",
        "",
        "## 可学习性指标",
        "",
        f"- 二分类 leave-one-year-out 平均 balanced accuracy：{result.binary_loso_mean_balanced_accuracy}",
        f"- 二分类 dummy baseline 平均 balanced accuracy：{result.binary_loso_dummy_mean_balanced_accuracy}",
        f"- 非零动作多分类 leave-one-year-out 平均 macro-F1：{result.nonzero_multiclass_loso_mean_macro_f1}",
        f"- 非零动作多分类 dummy baseline 平均 macro-F1：{result.nonzero_multiclass_dummy_mean_macro_f1}",
        f"- 多分类 macro-F1 增益：{result.nonzero_multiclass_macro_f1_gain}",
        f"- 非零动作 DAP 冲突数：{result.dap_conflict_count}",
        f"- 非零动作 DAP 冲突比例：{result.dap_conflict_share:.4f}",
        "",
        "## 动作分布",
        "",
        tables["action_counts"].to_markdown(index=False),
        "",
        "## 年份剖面",
        "",
        tables["by_year_profile"].to_markdown(index=False),
        "",
        "## 输出文件",
        "",
    ]
    for key, value in result.outputs.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    meta, obs, actions, masks, weights = load_data()
    tables = profile_tables(meta, obs, actions, masks)
    binary = loso_binary(meta, obs, actions)
    multi = loso_nonzero_multiclass(meta, obs, actions)
    conflicts = dap_conflict_audit(meta, actions)
    assoc = feature_action_association(meta, obs, actions)

    # Save tables first
    outputs = {
        "action_counts": TABLE_DIR / f"{TASK_ID}_action_counts.csv",
        "by_year_profile": TABLE_DIR / f"{TASK_ID}_by_year_profile.csv",
        "mask_counts": TABLE_DIR / f"{TASK_ID}_mask_counts.csv",
        "feature_summary": TABLE_DIR / f"{TASK_ID}_feature_summary.csv",
        "binary_loso": TABLE_DIR / f"{TASK_ID}_binary_nonzero_loso.csv",
        "nonzero_multiclass_loso": TABLE_DIR / f"{TASK_ID}_nonzero_action_multiclass_loso.csv",
        "dap_conflicts": TABLE_DIR / f"{TASK_ID}_nonzero_action_dap_conflicts.csv",
        "feature_action_association": TABLE_DIR / f"{TASK_ID}_feature_nonzero_action_association.csv",
    }
    for name, df in tables.items():
        if name in outputs:
            df.to_csv(outputs[name], index=False, encoding="utf-8-sig")
    binary.to_csv(outputs["binary_loso"], index=False, encoding="utf-8-sig")
    multi.to_csv(outputs["nonzero_multiclass_loso"], index=False, encoding="utf-8-sig")
    conflicts.to_csv(outputs["dap_conflicts"], index=False, encoding="utf-8-sig")
    assoc.to_csv(outputs["feature_action_association"], index=False, encoding="utf-8-sig")

    binary_bal = float(binary["model_balanced_accuracy"].dropna().mean()) if not binary.empty else None
    binary_dummy = float(binary["dummy_balanced_accuracy"].dropna().mean()) if not binary.empty else None
    multi_f1 = float(multi["model_macro_f1"].dropna().mean()) if not multi.empty else None
    dummy_f1 = float(multi["dummy_macro_f1"].dropna().mean()) if not multi.empty else None
    gain = (multi_f1 - dummy_f1) if multi_f1 is not None and dummy_f1 is not None else None
    conflict_count = int(conflicts["has_conflict"].sum()) if not conflicts.empty else 0
    conflict_share = float(conflict_count / len(conflicts)) if len(conflicts) else 0.0

    branch, next_allowed = decide({"binary_bal": binary_bal, "multi_f1": multi_f1, "dummy_f1": dummy_f1})
    result_path = OUT / f"{TASK_ID}_result.json"
    outputs_str = {k: str(v.relative_to(ROOT)) for k, v in outputs.items()}
    outputs_str["record_md"] = str(DOC.relative_to(ROOT))
    outputs_str["result_json"] = str(result_path.relative_to(ROOT))
    result = Result(
        task=f"{TASK_ID}_{TASK_NAME}",
        branch=branch,
        next_step_allowed=next_allowed,
        rows=int(len(meta)),
        years=int(meta["year"].nunique()),
        nonzero_rows=int(np.sum(actions != 0)),
        no_op_rows=int(np.sum(actions == 0)),
        nonzero_rate=float(np.mean(actions != 0)),
        binary_loso_mean_balanced_accuracy=binary_bal,
        binary_loso_dummy_mean_balanced_accuracy=binary_dummy,
        nonzero_multiclass_loso_mean_macro_f1=multi_f1,
        nonzero_multiclass_dummy_mean_macro_f1=dummy_f1,
        nonzero_multiclass_macro_f1_gain=gain,
        dap_conflict_count=conflict_count,
        dap_conflict_share=conflict_share,
        record_md=str(DOC.relative_to(ROOT)),
        outputs=outputs_str,
    )
    write_record(result, tables)
    result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
