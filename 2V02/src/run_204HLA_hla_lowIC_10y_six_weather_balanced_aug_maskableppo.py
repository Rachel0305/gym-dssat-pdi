"""204HLA balanced-coverage continuation of the 203HLA augmentation smoke.

The only change from 203HLA is episode sampling: each shuffled 60-episode
cycle visits every source-year x weather-variant pair exactly once.  PPO,
reward, masks, observations, input files and validation remain unchanged.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

import run_203HLA_hla_lowIC_10y_six_weather_aug_maskableppo as base


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "204HLA"
TASK_NAME = "hla_lowIC_10y_six_weather_balanced_aug_maskableppo"


class BalancedAuditedRandomYearEnv(base.BoundedAuditedRandomYearEnv):
    """Uniform-per-cycle scenario sampler with deterministic seed provenance."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._balanced_queue: deque[int] = deque()
        self.sampling_cycle_index = 0

    def _refill_balanced_queue(self) -> None:
        metadata = base.SCENARIO_META
        source_years = sorted({int(item["source_year"]) for item in metadata.values()})
        variants = sorted({str(item["weather_variant"]) for item in metadata.values()})
        lookup = {
            (int(item["source_year"]), str(item["weather_variant"])): int(pseudo_year)
            for pseudo_year, item in metadata.items()
        }
        source_order = [int(value) for value in self.rng.permutation(source_years)]
        variant_order = [str(value) for value in self.rng.permutation(variants)]
        planned: list[int] = []
        for round_index in range(len(variant_order)):
            for source_index, source_year in enumerate(source_order):
                variant = variant_order[(source_index + round_index) % len(variant_order)]
                planned.append(lookup[(source_year, variant)])
        if len(planned) != len(self.years) or len(set(planned)) != len(self.years):
            raise RuntimeError("balanced sampler did not produce one visit per scenario")
        self._balanced_queue.extend(planned)
        self.sampling_cycle_index += 1

    def reset(self, *args, **kwargs):
        if not self._balanced_queue:
            self._refill_balanced_queue()
        year = int(self._balanced_queue.popleft())
        self.current_year = year
        self.reset_counts[year] += 1
        self.switch_log.append(
            {
                "episode_index": int(sum(self.reset_counts.values())),
                "station_code": self.station,
                "year": year,
                "sampling_cycle_index": self.sampling_cycle_index,
                "sampling_mode": "balanced_shuffled_cycle_without_replacement",
            }
        )
        obs, info = self.current_env.reset(*args, **kwargs)
        info = dict(info or {})
        info["active_year"] = year
        info["sampling_cycle_index"] = self.sampling_cycle_index
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_irrigation = 0.0
        self._episode_n = 0.0
        return obs, info


def patched_preflight(cfg):
    result, manifest, split = ORIGINAL_PREFLIGHT(cfg)
    result["scenario_sampler"] = cfg["runtime_safety"]["scenario_sampler"]
    result["sampler_change_from_203HLA"] = "with_replacement -> balanced shuffled cycle; reward/action evaluation unchanged"
    return result, manifest, split


def patched_code_provenance():
    result = ORIGINAL_CODE_PROVENANCE()
    path = Path(__file__).resolve()
    result[base.rel(path)] = base.sha256(path)
    return result


def patched_write_record(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error):
    ORIGINAL_WRITE_RECORD(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error)
    path = base.record_path(smoke)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "- 唯一实验性变化：训练集由 10 个原始天气 episode 扩展为 10×6=60 个天气 episode。",
        "- 相对 054_00 的变化：训练集扩展为 60 个天气 episode；相对 203HLA 的最小修正是分层无放回循环采样。",
    )
    text = text.replace(
        "- 为防 OOM，RandomYearEnv 的采样分布保持 uniform random，但环境对象缓存上限为 2；这是资源管理措施，不改变奖励或动作评价。",
        "- 为防 OOM，环境对象缓存上限为 2；采样器采用 60 情景分层无放回循环，每个完整循环中各情景恰好访问一次；奖励与动作评价保持不变。",
    )
    text += (
        "\n## 204HLA sampler 决策\n\n"
        "203HLA 的 uniform random with replacement 在 2K 仅覆盖 13/60 情景、6/10 源年、5/6 变体，因此未进入长训练。"
        "204HLA 不改奖励，先修正数据使用率：每个 60-episode cycle 对每个 source_year×variant 恰好访问一次，cycle 内顺序由 seed 打乱。\n"
    )
    path.write_text(text, encoding="utf-8")


ORIGINAL_PREFLIGHT = base.preflight
ORIGINAL_CODE_PROVENANCE = base.code_provenance
ORIGINAL_WRITE_RECORD = base.write_record

base.TASK_ID = TASK_ID
base.TASK_NAME = TASK_NAME
base.DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
base.BoundedAuditedRandomYearEnv = BalancedAuditedRandomYearEnv
base.preflight = patched_preflight
base.code_provenance = patched_code_provenance
base.write_record = patched_write_record


if __name__ == "__main__":
    base.main()
