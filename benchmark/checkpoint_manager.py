"""Checkpoint discovery and explicit full/partial resume planning."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

_MODEL_SUFFIXES = {".zip", ".pt", ".pth", ".ckpt"}


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata inferred from one persisted model checkpoint."""

    path: Path
    timesteps: int | None
    replay_buffer_path: Path | None
    modified_time_ns: int

    @property
    def has_replay_buffer(self) -> bool:
        return self.replay_buffer_path is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "timesteps": self.timesteps,
            "replay_buffer_path": (
                self.replay_buffer_path.as_posix() if self.replay_buffer_path else None
            ),
            "has_replay_buffer": self.has_replay_buffer,
            "modified_time_ns": self.modified_time_ns,
        }


@dataclass(frozen=True)
class ResumePlan:
    """Safe description of how a runner may continue an interrupted run."""

    action: str
    resume_kind: str
    checkpoint: Path | None
    replay_buffer: Path | None
    completed_timesteps: int | None
    remaining_timesteps: int | None
    can_resume: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "resume_kind": self.resume_kind,
            "checkpoint": self.checkpoint.as_posix() if self.checkpoint else None,
            "replay_buffer": self.replay_buffer.as_posix() if self.replay_buffer else None,
            "completed_timesteps": self.completed_timesteps,
            "remaining_timesteps": self.remaining_timesteps,
            "can_resume": self.can_resume,
            "reason": self.reason,
        }


def _extract_timesteps(path: Path) -> int | None:
    stems = [path.stem.lower(), path.parent.name.lower()]
    patterns = (
        r"checkpoint[_-]?(\d+)(?:[_-]?steps?)?",
        r"(\d+)[_-]?steps?",
        r"steps?[_-]?(\d+)",
        r"timesteps?[_-]?(\d+)",
    )
    for stem in stems:
        for pattern in patterns:
            matches = re.findall(pattern, stem)
            if matches:
                return int(matches[-1])
    return None


class CheckpointManager:
    """Discover Stable-Baselines3-style checkpoints below a run directory."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve(strict=False)

    def _candidate_paths(self) -> Iterable[Path]:
        searched: set[Path] = set()
        directories = (
            self.run_dir / "checkpoints",
            self.run_dir / "models",
            self.run_dir,
        )
        for directory in directories:
            if not directory.is_dir():
                continue
            iterator = directory.glob("**/*") if directory != self.run_dir else directory.glob("*")
            for path in iterator:
                is_model_file = path.suffix.lower() == ".zip" or any(
                    token in path.stem.lower() for token in ("model", "checkpoint")
                )
                if path.is_file() and path.suffix.lower() in _MODEL_SUFFIXES and is_model_file:
                    resolved = path.resolve()
                    if resolved not in searched:
                        searched.add(resolved)
                        yield resolved

    def _find_replay_buffer(self, checkpoint: Path, timesteps: int | None) -> Path | None:
        names = [
            f"{checkpoint.stem}_replay_buffer.pkl",
            f"replay_buffer_{timesteps}.pkl" if timesteps is not None else "",
            f"replay_buffer_{timesteps}_steps.pkl" if timesteps is not None else "",
            "replay_buffer.pkl",
        ]
        directories = (
            checkpoint.parent,
            self.run_dir / "checkpoints",
            self.run_dir / "models",
            self.run_dir,
        )
        for directory in directories:
            for name in names:
                if not name:
                    continue
                candidate = directory / name
                if candidate.is_file():
                    return candidate.resolve()
        return None

    def discover(self) -> list[CheckpointInfo]:
        """Return all checkpoints ordered from least to most resumable/latest."""

        checkpoints: list[CheckpointInfo] = []
        for path in self._candidate_paths():
            timesteps = _extract_timesteps(path)
            checkpoints.append(
                CheckpointInfo(
                    path=path,
                    timesteps=timesteps,
                    replay_buffer_path=self._find_replay_buffer(path, timesteps),
                    modified_time_ns=path.stat().st_mtime_ns,
                )
            )
        return sorted(
            checkpoints,
            key=lambda item: (
                item.timesteps is not None,
                item.timesteps if item.timesteps is not None else -1,
                item.modified_time_ns,
                item.path.as_posix(),
            ),
        )

    def latest(self) -> CheckpointInfo | None:
        """Return the checkpoint with the greatest known step, then mtime."""

        checkpoints = self.discover()
        return checkpoints[-1] if checkpoints else None

    def resume_plan(self, total_timesteps: int | None = None) -> ResumePlan:
        """Describe whether training is fresh, complete, or safely resumable.

        A model without its replay buffer is explicitly labelled
        ``partial_resume``.  Loading its network weights may still be useful,
        but it is *not* equivalent to continuing the original off-policy DQN
        training state.
        """

        if total_timesteps is not None and total_timesteps < 1:
            raise ValueError("total_timesteps must be positive when supplied")
        checkpoint = self.latest()
        if checkpoint is None:
            return ResumePlan(
                action="train",
                resume_kind="fresh",
                checkpoint=None,
                replay_buffer=None,
                completed_timesteps=0,
                remaining_timesteps=total_timesteps,
                can_resume=False,
                reason="No checkpoint was found; start a fresh run",
            )

        completed = checkpoint.timesteps
        remaining = None
        if total_timesteps is not None and completed is not None:
            remaining = max(0, total_timesteps - completed)
            if remaining == 0:
                return ResumePlan(
                    action="skip_training",
                    resume_kind="already_complete",
                    checkpoint=checkpoint.path,
                    replay_buffer=checkpoint.replay_buffer_path,
                    completed_timesteps=completed,
                    remaining_timesteps=0,
                    can_resume=False,
                    reason=(
                        f"Checkpoint at {completed} steps meets target {total_timesteps}"
                    ),
                )

        if checkpoint.has_replay_buffer:
            kind = "checkpoint_replay_resume"
            reason = (
                "Checkpoint and replay buffer are available; the environment is rebuilt "
                "at an episode boundary, so bitwise DSSAT process continuation is not claimed"
            )
        else:
            kind = "partial_resume"
            reason = (
                "Checkpoint is available but no replay buffer was found; network weights "
                "can be loaded, but replay history and exact DQN continuation are unavailable"
            )
        return ResumePlan(
            action="resume_training",
            resume_kind=kind,
            checkpoint=checkpoint.path,
            replay_buffer=checkpoint.replay_buffer_path,
            completed_timesteps=completed,
            remaining_timesteps=remaining,
            can_resume=True,
            reason=reason,
        )
