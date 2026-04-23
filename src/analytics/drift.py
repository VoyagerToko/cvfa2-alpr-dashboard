from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DriftReport:
    baseline_accuracy: float
    current_accuracy: float
    absolute_drop: float
    relative_drop_pct: float
    drift_detected: bool


class DriftMonitor:
    def __init__(self, threshold_absolute_drop: float = 0.05) -> None:
        self.threshold_absolute_drop = threshold_absolute_drop

    def evaluate(self, baseline_accuracy: float, current_accuracy: float) -> DriftReport:
        absolute_drop = max(0.0, baseline_accuracy - current_accuracy)
        relative_drop = (absolute_drop / baseline_accuracy * 100.0) if baseline_accuracy > 0 else 0.0

        return DriftReport(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            absolute_drop=absolute_drop,
            relative_drop_pct=relative_drop,
            drift_detected=absolute_drop >= self.threshold_absolute_drop,
        )
