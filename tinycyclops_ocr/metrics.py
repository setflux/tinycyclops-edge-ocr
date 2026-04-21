from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class StageMetric:
    count: int = 0
    item_count: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0

    def observe(self, seconds: float, item_count: int = 1) -> None:
        self.count += 1
        self.item_count += item_count
        self.seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "item_count": self.item_count,
            "seconds": self.seconds,
            "avg_seconds": self.seconds / self.count if self.count else 0.0,
            "avg_seconds_per_item": self.seconds / self.item_count if self.item_count else 0.0,
            "max_seconds": self.max_seconds,
        }


class PipelineMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, StageMetric] = {}
        self._counters: dict[str, int] = {}

    def observe(self, stage: str, seconds: float, item_count: int = 1) -> None:
        with self._lock:
            self._stages.setdefault(stage, StageMetric()).observe(seconds, item_count)

    def increment(self, counter: str, value: int = 1) -> None:
        with self._lock:
            self._counters[counter] = self._counters.get(counter, 0) + value

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "stages": {name: metric.snapshot() for name, metric in sorted(self._stages.items())},
                "counters": dict(sorted(self._counters.items())),
            }
