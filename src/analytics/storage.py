from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class EventStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plate_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    plate_text TEXT NOT NULL,
                    camera_id TEXT,
                    zone TEXT,
                    track_id INTEGER,
                    confidence REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    is_duplicate INTEGER DEFAULT 0
                )
                """
            )

    def insert_event(self, event: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plate_events (
                    timestamp, plate_text, camera_id, zone, track_id, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, is_duplicate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["timestamp"],
                    event["plate_text"],
                    event.get("camera_id"),
                    event.get("zone"),
                    event.get("track_id"),
                    event.get("confidence"),
                    event.get("bbox", (None, None, None, None))[0],
                    event.get("bbox", (None, None, None, None))[1],
                    event.get("bbox", (None, None, None, None))[2],
                    event.get("bbox", (None, None, None, None))[3],
                    int(bool(event.get("is_duplicate", False))),
                ),
            )

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM plate_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_zone_counts(self, since_timestamp: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if since_timestamp:
                rows = conn.execute(
                    """
                    SELECT zone, COUNT(*) as count
                    FROM plate_events
                    WHERE timestamp >= ?
                    GROUP BY zone
                    """,
                    (since_timestamp,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT zone, COUNT(*) as count FROM plate_events GROUP BY zone"
                ).fetchall()
        return [dict(r) for r in rows]
