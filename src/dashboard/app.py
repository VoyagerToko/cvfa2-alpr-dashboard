from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.analytics.occupancy import ParkingOccupancyEstimator
from src.analytics.storage import EventStore


st.set_page_config(page_title="ALPR Parking Dashboard", layout="wide")
st.title("Hybrid ALPR Parking Analytics")


def load_store() -> EventStore:
    db_path = os.getenv("SQLITE_DB_PATH", "artifacts/parking_events.sqlite3")
    return EventStore(db_path)


store = load_store()
estimator = ParkingOccupancyEstimator(store=store, timeout_seconds=300)

events = store.get_recent_events(limit=1000)
if not events:
    st.info("No events found yet. Start inference to populate analytics.")
    st.stop()


df = pd.DataFrame(events)
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Events", int(len(df)))
with col2:
    st.metric("Unique Plates", int(df["plate_text"].nunique()))
with col3:
    duplicate_rate = float(df["is_duplicate"].mean() * 100.0) if "is_duplicate" in df else 0.0
    st.metric("Duplicate Rate", f"{duplicate_rate:.2f}%")

st.subheader("Current Occupancy")
occupancy = estimator.estimate_current_occupancy()
if occupancy:
    occ_df = pd.DataFrame(list(occupancy.items()), columns=["zone", "active_vehicles"])
    st.bar_chart(occ_df.set_index("zone"))
else:
    st.write("No active occupancy detected in the configured window.")

st.subheader("Zone Event Volume")
zone_counts = store.get_zone_counts()
if zone_counts:
    zone_df = pd.DataFrame(zone_counts)
    st.bar_chart(zone_df.set_index("zone"))

st.subheader("Recent Recognitions")
show_cols = [
    c
    for c in ["timestamp", "plate_text", "camera_id", "zone", "track_id", "confidence", "is_duplicate"]
    if c in df.columns
]
st.dataframe(df[show_cols].sort_values("timestamp", ascending=False).head(200), use_container_width=True)
