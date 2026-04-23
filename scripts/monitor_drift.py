from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analytics.drift import DriftMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor model drift using evaluation metrics")
    parser.add_argument("--baseline", type=float, required=True, help="Baseline full-plate accuracy")
    parser.add_argument("--current", type=float, required=True, help="Current full-plate accuracy")
    parser.add_argument("--threshold", type=float, default=0.05, help="Absolute drop threshold")
    parser.add_argument("--output", type=str, default="artifacts/drift_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor = DriftMonitor(threshold_absolute_drop=args.threshold)
    report = monitor.evaluate(baseline_accuracy=args.baseline, current_accuracy=args.current)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")

    print(json.dumps(report.__dict__, indent=2))
    if report.drift_detected:
        print("[monitor_drift] Drift detected: schedule retraining")
    else:
        print("[monitor_drift] No significant drift detected")


if __name__ == "__main__":
    main()
