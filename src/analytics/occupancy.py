from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.analytics.storage import EventStore


class ParkingOccupancyEstimator:
    def __init__(self, store: EventStore, timeout_seconds: int = 300) -> None:
        self.store = store
        self.timeout_seconds = timeout_seconds

    def estimate_current_occupancy(self) -> dict[str, int]:
        events = self.store.get_recent_events(limit=5000)
        if not events:
            return {}

        df = pd.DataFrame(events)
        if df.empty:
            return {}

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = datetime.utcnow() - timedelta(seconds=self.timeout_seconds)
        active = df[df["timestamp"] >= cutoff]

        if active.empty:
            return {}

        grouped = active.groupby("zone")["plate_text"].nunique()
        return {str(zone): int(count) for zone, count in grouped.items() if pd.notna(zone)}

    def total_active_vehicles(self) -> int:
        return sum(self.estimate_current_occupancy().values())
