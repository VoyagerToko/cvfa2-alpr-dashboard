from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

from src.analytics.occupancy import ParkingOccupancyEstimator
from src.analytics.storage import EventStore


st.set_page_config(page_title="ALPR Parking Dashboard", layout="wide")
st.title("Hybrid ALPR Studio")
st.caption("Upload an image to detect a number plate, and review parking analytics.")

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


def load_store() -> EventStore:
    db_path = os.getenv("SQLITE_DB_PATH", "artifacts/parking_events.sqlite3")
    return EventStore(db_path)


def draw_bbox(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    annotated = image.copy()
    drawer = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = bbox
    drawer.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=4)
    return annotated


def detect_from_api(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    api_url: str,
    camera_id: str,
) -> dict[str, object]:
    endpoint = f"{api_url.rstrip('/')}/predict/image"
    response = requests.post(
        endpoint,
        files={"file": (filename, image_bytes, content_type)},
        data={"camera_id": camera_id},
        timeout=45,
    )
    response.raise_for_status()
    return response.json()


def render_detection_tab() -> None:
    st.subheader("Image Detection")

    api_default = os.getenv("ALPR_API_URL", "http://127.0.0.1:8000")
    col_cfg_1, col_cfg_2 = st.columns([2, 1])
    with col_cfg_1:
        api_url = st.text_input("ALPR API URL", value=api_default)
    with col_cfg_2:
        camera_id = st.text_input("Camera ID", value="streamlit_cam_1")

    uploaded_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an image and click Detect Number Plate.")
        return

    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    col_img_1, col_img_2 = st.columns(2)
    with col_img_1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Number Plate", type="primary"):
        with st.spinner("Running detection..."):
            try:
                payload = detect_from_api(
                    image_bytes=image_bytes,
                    filename=uploaded_file.name,
                    content_type=uploaded_file.type or "image/jpeg",
                    api_url=api_url,
                    camera_id=camera_id,
                )
                st.session_state["last_prediction"] = payload
            except requests.RequestException as exc:
                st.error(
                    "Detection failed. Ensure FastAPI is running at the configured URL and try again."
                )
                st.exception(exc)

    payload = st.session_state.get("last_prediction")
    if not payload:
        return

    bbox_raw = payload.get("bbox", [0, 0, 0, 0])
    bbox = tuple(int(v) for v in bbox_raw)
    annotated = draw_bbox(image, bbox)

    with col_img_2:
        st.image(annotated, caption="Detection Result", use_container_width=True)

    st.success(
        f"Plate: {payload.get('plate_text', 'N/A')} | Confidence: {float(payload.get('confidence', 0.0)):.3f}"
    )

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Plate", str(payload.get("plate_text", "N/A")))
    metric_col_2.metric("Zone", str(payload.get("zone", "N/A")))
    metric_col_3.metric("Track ID", int(payload.get("track_id", -1)))
    metric_col_4.metric("Duplicate", "Yes" if bool(payload.get("is_duplicate", False)) else "No")

    with st.expander("Raw Prediction Payload"):
        st.json(payload)


def render_analytics_tab() -> None:
    st.subheader("Parking Analytics")
    store = load_store()
    estimator = ParkingOccupancyEstimator(store=store, timeout_seconds=300)

    events = store.get_recent_events(limit=1000)
    if not events:
        st.info("No events found yet. Run image detection or video inference to populate analytics.")
        return

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
        for c in [
            "timestamp",
            "plate_text",
            "camera_id",
            "zone",
            "track_id",
            "confidence",
            "is_duplicate",
        ]
        if c in df.columns
    ]
    st.dataframe(df[show_cols].sort_values("timestamp", ascending=False).head(200), use_container_width=True)

detection_tab, analytics_tab = st.tabs(["Detect Plate", "Analytics"])

with detection_tab:
    render_detection_tab()

with analytics_tab:
    render_analytics_tab()
