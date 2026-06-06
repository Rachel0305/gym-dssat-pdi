from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ScheduleEvent:
    name: str
    dap: int
    irrigation: float = 0.0
    nitrogen: float = 0.0


class DeterministicSchedulePolicy:
    """Fixed water/N event policy used for offline DSSAT scenario search."""

    def __init__(self, events: Iterable[ScheduleEvent]):
        by_dap: dict[int, dict[str, float | str]] = {}
        names: dict[int, list[str]] = {}
        for event in events:
            item = by_dap.setdefault(int(event.dap), {"amir": 0.0, "anfer": 0.0})
            item["amir"] = float(item.get("amir", 0.0)) + float(event.irrigation)
            item["anfer"] = float(item.get("anfer", 0.0)) + float(event.nitrogen)
            names.setdefault(int(event.dap), []).append(event.name)
        self.by_dap = by_dap
        self.names = {dap: "+".join(values) for dap, values in names.items()}

    def action_for_dap(self, dap: int) -> dict[str, float]:
        event = self.by_dap.get(int(dap), {})
        return {
            "amir": float(event.get("amir", 0.0)),
            "anfer": float(event.get("anfer", 0.0)),
        }

    def event_name_for_dap(self, dap: int) -> str:
        return self.names.get(int(dap), "")

    def is_event_day(self, dap: int) -> bool:
        return int(dap) in self.by_dap
