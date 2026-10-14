"""Small, shared artifact resolver for configuration-driven SYA experiments.

Old experiment engines leave compatibility filenames such as 032_22 or 042_10.
Reporting code must resolve files by their columns and configured checkpoint,
not by an experiment-specific filename baked into each plot script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_REQUIRED = {"station_code", "year", "checkpoint_step", "final_grnwt", "total_irrigation", "total_n"}
INVENTORY_REQUIRED = {"station_code", "checkpoint_step", "model_path"}


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return None


def _candidates(run_root: Path, suffix: str) -> list[Path]:
    return sorted((run_root / "evaluation").glob(f"*{suffix}.csv"))


def _choose(matches: list[Path], preferred_prefix: str | None) -> Path:
    if preferred_prefix:
        preferred = [path for path in matches if path.name.startswith(preferred_prefix)]
        if len(preferred) == 1:
            return preferred[0]
    # The 042_10 files are the established compatibility layer above 032_22.
    compat = [path for path in matches if path.name.startswith("042_10_")]
    if len(compat) == 1:
        return compat[0]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"Ambiguous compatible artifacts: {matches}")


def resolve_validation_summary(run_root: Path, checkpoint: int, years: list[int], preferred_prefix: str | None = None) -> Path:
    """Find the sole evaluation CSV covering configured years/checkpoint."""
    valid: list[Path] = []
    wanted = sorted(map(int, years))
    for path in _candidates(run_root, "checkpoint_validation_summary"):
        frame = _read(path)
        if frame is None or not EVAL_REQUIRED.issubset(frame.columns):
            continue
        sub = frame[pd.to_numeric(frame["checkpoint_step"], errors="coerce").eq(int(checkpoint))]
        got = sorted(pd.to_numeric(sub["year"], errors="coerce").dropna().astype(int).tolist())
        if got == wanted:
            valid.append(path)
    if not valid:
        raise RuntimeError(f"Cannot resolve PPO validation summary under {run_root}")
    return _choose(valid, preferred_prefix)


def resolve_training_inventory(run_root: Path, checkpoint: int, preferred_prefix: str | None = None) -> Path:
    """Find the sole model inventory CSV containing the selected checkpoint."""
    valid: list[Path] = []
    for path in _candidates(run_root, "training_checkpoint_inventory"):
        frame = _read(path)
        if frame is None or not INVENTORY_REQUIRED.issubset(frame.columns):
            continue
        if pd.to_numeric(frame["checkpoint_step"], errors="coerce").eq(int(checkpoint)).any():
            valid.append(path)
    if not valid:
        raise RuntimeError(f"Cannot resolve PPO training inventory under {run_root}")
    return _choose(valid, preferred_prefix)
