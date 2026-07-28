from __future__ import annotations

import json
import shutil
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as lc_multi


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
PROMPT = ROOT / "prompts" / "032_22_five_site_half_split_stress_aware_maskableppo_batch.md"
OUT = ROOT / "benchmark_results" / "032_22_five_site_half_split_stress_aware_maskableppo_batch"
DOC = ROOT / "docs" / "032_22_five_site_half_split_stress_aware_maskableppo_batch_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
SPLIT_CSV = ROOT / "benchmark_results" / "032_21_five_site_half_split_free_timing_ppo_readiness_audit" / "tables" / "032_21_half_split_years.csv"

SITES = ["FQA", "HLA", "LCA", "SYA", "YCA"]
SITE_NAMES = {"FQA": "FQ", "HLA": "HLA", "LCA": "LC", "SYA": "SY", "YCA": "YC"}
SEED = 0
TOTAL_TIMESTEPS = 100_000
CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "logs", "models", "daily_outputs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_config() -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = SEED
    cfg["total_timesteps"] = TOTAL_TIMESTEPS
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["smoke_station"] = SITES[0]
    return cfg


def load_split() -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(SPLIT_CSV)
    split = pd.read_csv(SPLIT_CSV, keep_default_na=False)
    split = split[split["station_code"].isin(SITES)].copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    return split.sort_values(["station_code", "year"]).reset_index(drop=True)


def build_selection(split: pd.DataFrame) -> pd.DataFrame:
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selected = pool.merge(split[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selected["selected_for_train"] = selected["split"].eq("train")
    selected["selected_for_eval"] = True
    selected["selection_reason"] = "032_22_half_split_batch"
    return selected.sort_values(["station_code", "year"]).reset_index(drop=True)


def model_path(station: str, step: int) -> Path:
    return OUT / "models" / station / f"{station}_half_split_stress_aware_maskableppo_seed{SEED}_ckpt{int(step)}.zip"


class FixedStepCheckpointCallback:
    def __init__(self, station: str, checkpoint_steps: list[int]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                step_now = int(self.model.num_timesteps)
                pending = [x for x in outer.checkpoint_steps if x <= step_now and x not in outer.saved_steps]
                for target in pending:
                    path = model_path(outer.station, target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    outer.saved_steps.append(int(target))
                return True

        self.station = station
        self.checkpoint_steps = [int(x) for x in checkpoint_steps]
        self.saved_steps: list[int] = []
        self.callback = _Callback()


class RandomYearEnv(gym.Env):
    def __init__(self, config: dict[str, Any], env_config: dict[str, Any], station: str, years: list[int], seed: int) -> None:
        super().__init__()
        self.config = config
        self.env_config = env_config
        self.station = station
        self.years = [int(y) for y in years]
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.envs: dict[int, gym.Env] = {}
        self.current_year: int | None = None
        self.reset_counts: Counter[int] = Counter()
        self.switch_log: list[dict[str, Any]] = []
        first = self._env_for(self.years[0])
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        self.metadata = getattr(first, "metadata", {})

    def _env_for(self, year: int):
        year = int(year)
        if year not in self.envs:
            self.envs[year] = base.make_env(
                self.config,
                self.env_config,
                self.station,
                year,
                self.seed,
                f"{self.station}_{year}_032_22_train",
                evaluation=False,
            )
        return self.envs[year]

    @property
    def current_env(self):
        if self.current_year is None:
            self.current_year = self.years[0]
        return self._env_for(self.current_year)

    def reset(self, *args, **kwargs):
        year = int(self.rng.choice(self.years))
        self.current_year = year
        self.reset_counts[year] += 1
        self.switch_log.append({"episode_index": int(sum(self.reset_counts.values())), "station_code": self.station, "year": year})
        obs, info = self.current_env.reset(*args, **kwargs)
        info = dict(info or {})
        info["active_year"] = year
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = dict(info or {})
        info["active_year"] = int(self.current_year)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.current_env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        return dict(getattr(self.current_env, "last_action_info", {}))

    def close(self):
        for env in self.envs.values():
            try:
                env.close()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.current_env, name)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train_station(config: dict[str, Any], env_config: dict[str, Any], station: str, train_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sb3_contrib import MaskablePPO

    rows: list[dict[str, Any]] = []
    expected = [model_path(station, step) for step in CHECKPOINT_STEPS]
    reset_df = pd.DataFrame()
    if all(path.exists() for path in expected):
        for step, path in zip(CHECKPOINT_STEPS, expected):
            rows.append(
                {
                    "station_code": station,
                    "site": SITE_NAMES[station],
                    "train_years": ",".join(map(str, train_years)),
                    "seed": SEED,
                    "checkpoint_step": step,
                    "run_status": "ok_existing",
                    "model_path": path.relative_to(ROOT).as_posix(),
                    "model_sha256": sha256_file(path),
                }
            )
        return pd.DataFrame(rows), reset_df

    env: RandomYearEnv | None = None
    status = "ok"
    notes = ""
    try:
        env = RandomYearEnv(config, env_config, station, train_years, SEED)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED,
            tensorboard_log=str(OUT / "tensorboard" / station),
            **base.ppo_kwargs(config),
        )
        callback = FixedStepCheckpointCallback(station, CHECKPOINT_STEPS)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=True, progress_bar=False, callback=callback.callback)
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            reset_df = pd.DataFrame(
                [
                    {"station_code": station, "year": int(year), "episode_count": int(env.reset_counts.get(year, 0))}
                    for year in train_years
                ]
            )
            pd.DataFrame(env.switch_log).to_csv(OUT / "logs" / f"032_22_{station}_training_year_switch_log.csv", index=False, encoding="utf-8-sig")
            env.close()

    for step, path in zip(CHECKPOINT_STEPS, expected):
        rows.append(
            {
                "station_code": station,
                "site": SITE_NAMES[station],
                "train_years": ",".join(map(str, train_years)),
                "seed": SEED,
                "checkpoint_step": step,
                "run_status": status if status != "ok" else ("ok" if path.exists() else "missing"),
                "model_path": path.relative_to(ROOT).as_posix() if path.exists() else "",
                "model_sha256": sha256_file(path) if path.exists() else "",
                "notes": notes[-4000:] if notes else "",
            }
        )
    return pd.DataFrame(rows), reset_df


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], row: pd.Series, year: int) -> dict[str, Any]:
    original_out = lc_multi.OUT
    original_station = lc_multi.STATION
    original_seed = lc_multi.SEED
    try:
        lc_multi.OUT = OUT
        lc_multi.STATION = str(row["station_code"])
        lc_multi.SEED = SEED
        (OUT / "daily_outputs" / lc_multi.STATION).mkdir(parents=True, exist_ok=True)
        (OUT / "evaluation" / lc_multi.STATION).mkdir(parents=True, exist_ok=True)
        return lc_multi.evaluate_checkpoint(config, env_config, row, int(year))
    finally:
        lc_multi.OUT = original_out
        lc_multi.STATION = original_station
        lc_multi.SEED = original_seed


def add_comparison_flags(eval_df: pd.DataFrame) -> pd.DataFrame:
    baseline_path = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_completed_template_aware_unified_baseline_summary.csv"
    if not baseline_path.exists() or eval_df.empty:
        return eval_df
    base_df = pd.read_csv(baseline_path, keep_default_na=False)
    if "station_code" not in base_df.columns:
        return eval_df
    base_df["year"] = pd.to_numeric(base_df["year"], errors="coerce").astype("Int64")
    rows: list[dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        station = str(row["station_code"])
        year = int(row["year"])
        subset = base_df[base_df["station_code"].eq(station) & base_df["year"].astype("Int64").eq(year)]
        out = row.to_dict()
        if len(subset) >= 4:
            max_y = pd.to_numeric(subset["grain_yield_kg_ha"], errors="coerce").max()
            max_wp = pd.to_numeric(subset["WP_ET_kg_m3"], errors="coerce").max()
            max_pfp = pd.to_numeric(subset["PFP_N_kg_kg"], errors="coerce").max()
            out["baseline_rows"] = int(len(subset))
            out["gap_yield_vs_four_max"] = float(row.get("final_grnwt", np.nan)) - float(max_y)
            out["gap_wp_et_vs_four_max"] = float(row.get("WP_ET_kg_m3", np.nan)) - float(max_wp) if "WP_ET_kg_m3" in row else np.nan
            out["gap_pfp_n_vs_four_max"] = float(row.get("PFP_N", np.nan)) - float(max_pfp) if "PFP_N" in row else np.nan
            out["any_metric_win_four"] = bool(
                (out["gap_yield_vs_four_max"] > 0)
                or (pd.notna(out["gap_wp_et_vs_four_max"]) and out["gap_wp_et_vs_four_max"] > 0)
                or (pd.notna(out["gap_pfp_n_vs_four_max"]) and out["gap_pfp_n_vs_four_max"] > 0)
            )
        else:
            out["baseline_rows"] = int(len(subset))
            out["any_metric_win_four"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_by_station(eval_df: pd.DataFrame) -> pd.DataFrame:
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy() if not eval_df.empty else pd.DataFrame()
    if ok.empty:
        return pd.DataFrame()
    if "site" not in ok.columns and "station_code" in ok.columns:
        ok["site"] = ok["station_code"].map(SITE_NAMES)
    return (
        ok.groupby(["station_code", "site", "checkpoint_step"], as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
            max_nstres=("max_nstres", "max"),
            max_swfac=("max_swfac", "max"),
            any_metric_win_four_count=("any_metric_win_four", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_gap_yield_vs_four_max=("gap_yield_vs_four_max", "mean"),
            mean_gap_pfp_n_vs_four_max=("gap_pfp_n_vs_four_max", "mean"),
        )
        .sort_values(["station_code", "checkpoint_step"])
        .reset_index(drop=True)
    )


def write_record(split: pd.DataFrame, train_df: pd.DataFrame, reset_df: pd.DataFrame, eval_df: pd.DataFrame, by_station: pd.DataFrame) -> None:
    failed = train_df[~train_df["run_status"].astype(str).str.startswith("ok")].copy() if not train_df.empty else pd.DataFrame()
    lines = [
        "# 032_22 五站点前半训练/后半验证自由时序 stress-aware MaskablePPO 批量实验记录",
        "",
        "## 结论先说",
        "",
        f"- 状态：`{'completed' if failed.empty else 'partial'}`。",
        f"- 算法：MaskablePPO；seed={SEED}；每站点训练 {TOTAL_TIMESTEPS} timesteps。",
        f"- checkpoint：{', '.join(map(str, CHECKPOINT_STEPS))}。",
        "- 本轮不调参、不改奖励、不跨站点迁移。",
        "",
        "## 固定年份切分",
        "",
        md_table(split[["station_code", "site", "year", "split"]], max_rows=120),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train_df, max_rows=80),
        "",
        "## 训练年份采样次数",
        "",
        md_table(reset_df, max_rows=120),
        "",
        "## 验证集按站点和 checkpoint 汇总",
        "",
        md_table(by_station, max_rows=80),
        "",
        "## 逐年验证结果",
        "",
        md_table(eval_df, max_rows=160),
        "",
        "## 解释边界",
        "",
        "- 本轮只生成五站点批量数值结果，尚未重建五情景日过程图。",
        "- 若某站点多年动作序列高度相似，这需要后续用日值图和输入敏感性审计解释，不能只凭本轮表格下结论。",
        "- SY 的 recorded_farmer 基线在 032_21 中仍有缺口，因此 SY 与完整四情景比较需要后续补齐 recorded_farmer 后再冻结。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copy2(CONFIG, OUT / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    config = load_config()
    split = load_split()
    selection = build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    selection.to_csv(OUT / "configs" / "032_22_half_split_selection.csv", index=False, encoding="utf-8-sig")

    train_frames: list[pd.DataFrame] = []
    reset_frames: list[pd.DataFrame] = []
    eval_rows: list[dict[str, Any]] = []
    for station in SITES:
        station_split = split[split["station_code"].eq(station)]
        train_years = station_split[station_split["split"].eq("train")]["year"].astype(int).tolist()
        validation_years = station_split[station_split["split"].eq("validation")]["year"].astype(int).tolist()
        train_df, reset_df = train_station(config, env_config, station, train_years)
        train_frames.append(train_df)
        if not reset_df.empty:
            reset_frames.append(reset_df)
        train_ok = train_df[train_df["run_status"].astype(str).str.startswith("ok")].copy()
        for _, row in train_ok.iterrows():
            for year in validation_years:
                try:
                    eval_rows.append(evaluate_checkpoint(config, env_config, row, int(year)))
                except Exception:
                    eval_rows.append(
                        {
                            "station_code": station,
                            "site": SITE_NAMES[station],
                            "year": int(year),
                            "seed": SEED,
                            "checkpoint_step": int(row["checkpoint_step"]),
                            "run_status": "failed",
                            "notes": traceback.format_exc()[-4000:],
                        }
                    )
        pd.concat(train_frames, ignore_index=True).to_csv(OUT / "evaluation" / "032_22_training_checkpoint_inventory_partial.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(eval_rows).to_csv(OUT / "evaluation" / "032_22_checkpoint_validation_summary_partial.csv", index=False, encoding="utf-8-sig")

    train_all = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    reset_all = pd.concat(reset_frames, ignore_index=True) if reset_frames else pd.DataFrame()
    eval_all = pd.DataFrame(eval_rows)
    eval_all = add_comparison_flags(eval_all)
    by_station = summarize_by_station(eval_all)
    train_all.to_csv(OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    reset_all.to_csv(OUT / "logs" / "032_22_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
    eval_all.to_csv(OUT / "evaluation" / "032_22_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    by_station.to_csv(OUT / "evaluation" / "032_22_validation_summary_by_station_checkpoint.csv", index=False, encoding="utf-8-sig")
    write_record(split, train_all, reset_all, eval_all, by_station)
    result = {
        "task": "032_22_five_site_half_split_stress_aware_maskableppo_batch",
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "train_inventory": (OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
        "validation_summary": (OUT / "evaluation" / "032_22_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_station": (OUT / "evaluation" / "032_22_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
        "stations": SITES,
        "total_timesteps_per_station": TOTAL_TIMESTEPS,
    }
    (OUT / "032_22_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
