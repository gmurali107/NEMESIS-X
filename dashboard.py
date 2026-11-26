
import streamlit as st
import pandas as pd
import networkx as nx
import os
import csv
import threading
import av
import plotly.graph_objects as go
from pathlib import Path
import cv2
import subprocess
import imageio_ffmpeg

from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from video_analyzer import analyze_video
from anomaly_detection import detect_anomalies

# ==========================================
# NEMESIS-X COMPLETE DASHBOARD
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
EVENT_FILE = os.path.join(BASE_DIR, "event_history.csv")
ALERT_FILE = os.path.join(BASE_DIR, "anomaly_alerts.csv")
EXPLANATION_FILE = os.path.join(BASE_DIR, "event_explanations.txt")
GRAPH_FILE = os.path.join(BASE_DIR, "event_graph.gexf")
GRAPH_IMAGE = os.path.join(BASE_DIR, "event_graph.png")

st.set_page_config(
    page_title="NEMESIS-X Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e2e8f0;
}
h1, h2, h3 {
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD YOLO
# ==========================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ==========================================
# CAMERA PROCESSOR
# ==========================================


class NemesisProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_count = 0
        self.object_count = 0

        self.object_memory = {}
        self.event_file = EVENT_FILE
        self.debug_printed = False

    def log_event(self, track_id, name, event, details=""):

        file_exists = (
            os.path.exists(self.event_file)
            and os.path.getsize(self.event_file) > 0
        )

        with open(
            self.event_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "Timestamp",
                    "Track_ID",
                    "Object",
                    "Event",
                    "Details"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                track_id,
                name,
                event,
                details
            ])

        try:
            detect_anomalies()
        except Exception as error:
            print("Anomaly detection error:", error)

    def recv(self, frame):

        print("NEMESIS: Camera frame received")

        image = frame.to_ndarray(format="bgr24")

        with self.lock:

            results = model.track(
                image,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.40,
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            boxes = result.boxes
            if not self.debug_printed:
                print("NEMESIS DEBUG - Boxes:", boxes is not None)
                print(
                    "NEMESIS DEBUG - Tracking IDs:",
                    boxes.id is not None if boxes is not None else False
                )
                print("NEMESIS DEBUG - Event file:", self.event_file)
                self.debug_printed = True
                self.object_count = (
                len(boxes) if boxes is not None else 0
            )

                self.frame_count += 1

            now = datetime.now()
            active_ids = set()

            height, width = image.shape[:2]

            # Monitored zone
            zone_x1 = int(width * 0.25)
            zone_x2 = int(width * 0.75)
            zone_y1 = int(height * 0.20)
            zone_y2 = int(height * 0.90)

            cv2.rectangle(
                annotated,
                (zone_x1, zone_y1),
                (zone_x2, zone_y2),
                (0, 255, 255),
                2
            )

            if boxes is not None and boxes.id is not None:

                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()

                for index, track_id in enumerate(track_ids):

                    active_ids.add(track_id)

                    name = model.names[class_ids[index]]

                    x1, y1, x2, y2 = (
                        boxes.xyxy[index].int().cpu().tolist()
                    )

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    in_zone = (
                        zone_x1 <= center_x <= zone_x2
                        and zone_y1 <= center_y <= zone_y2
                    )

                    if track_id not in self.object_memory:

                        self.object_memory[track_id] = {
                            "name": name,
                            "first_seen": now,
                            "last_seen": now,
                            "in_zone": False,
                            "zone_enter_time": None,
                            "dwell_alerted": False,
                            "lost_logged": False
                        }

                        self.log_event(
                            track_id,
                            name,
                            "FIRST SEEN",
                            "Object detected by live camera."
                        )

                    memory = self.object_memory[track_id]
                    memory["last_seen"] = now
                    memory["lost_logged"] = False

                    if in_zone and not memory["in_zone"]:

                        memory["in_zone"] = True
                        memory["zone_enter_time"] = now
                        memory["dwell_alerted"] = False

                        self.log_event(
                            track_id,
                            name,
                            "ENTERED ZONE",
                            "Object entered monitored zone."
                        )

                    elif not in_zone and memory["in_zone"]:

                        duration = (
                            now - memory["zone_enter_time"]
                        ).total_seconds()

                        memory["in_zone"] = False
                        memory["zone_enter_time"] = None
                        memory["dwell_alerted"] = False

                        self.log_event(
                            track_id,
                            name,
                            "EXITED ZONE",
                            f"Time in zone: {duration:.1f} seconds."
                        )

                    if (
                        memory["in_zone"]
                        and not memory["dwell_alerted"]
                    ):

                        duration = (
                            now - memory["zone_enter_time"]
                        ).total_seconds()

                        if duration >= 15:

                            memory["dwell_alerted"] = True

                            self.log_event(
                                track_id,
                                name,
                                "LONG ZONE STAY",
                                f"Object remained in zone for {duration:.1f} seconds."
                            )

            # Detect objects that disappear from camera view
            for track_id, memory in self.object_memory.items():

                if (
                    track_id not in active_ids
                    and not memory["lost_logged"]
                    and (now - memory["last_seen"]).total_seconds() >= 3
                ):

                    memory["lost_logged"] = True

                    self.log_event(
                        track_id,
                        memory["name"],
                        "LOST FROM VIEW",
                        "Object disappeared from camera view."
                    )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ==========================================
# LOAD CSV DATA
# ==========================================

@st.cache_data(ttl=3)
def load_csv(path):

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return pd.DataFrame()

    return pd.DataFrame()

# ==========================================
# HEADER
# ==========================================

st.title("🛡️ NEMESIS-X")
st.subheader("Real-Time Open-World Video Intelligence System")

st.caption(
    "YOLO • ByteTrack • Temporal Analysis • Spatial Reasoning • Anomaly Detection"
)

st.divider()



# ==========================================
# PHASE 9 - VIDEO UPLOAD AND ANALYSIS
# ==========================================

st.markdown("## 🎬 Video Intelligence")

st.write("Upload a recorded video for NEMESIS-X analysis.")

VIDEO_DIR = Path(BASE_DIR) / "input_videos"
OUTPUT_DIR = Path(BASE_DIR) / "video_outputs"

if VIDEO_DIR.exists() and not VIDEO_DIR.is_dir():
    st.error("input_videos exists as a file. Rename it before continuing.")

else:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis_mode = st.radio(
        "Choose video analysis mode:",
        [
            "Both (Detection + Full Intelligence)",
            "Object Detection Only",
            "Full Intelligence Analysis"
        ],
        index=0
    )

    uploaded_video = st.file_uploader(
        "Upload your video",
        type=["mp4", "avi", "mov", "mkv"],
        key="nemesis_video_upload"
    )

    if uploaded_video is not None:

        st.success("Video selected: " + uploaded_video.name)
        st.video(uploaded_video)

        if st.button("🚀 ANALYZE VIDEO", key="analyze_uploaded_video"):

            video_name = Path(uploaded_video.name).name
            video_path = VIDEO_DIR / video_name

            video_path.write_bytes(uploaded_video.getvalue())

            video_output_dir = (
                OUTPUT_DIR / Path(video_name).stem
            )

            progress = st.progress(0)
            status = st.empty()

            try:
                status.info("NEMESIS-X is processing your video...")

                results = analyze_video(
                    video_path=str(video_path),
                    output_dir=str(video_output_dir),
                    model=model,
                    analysis_mode=analysis_mode,
                    progress_callback=lambda value: progress.progress(value)
                )

                status.success("Video analysis completed!")

                st.markdown("### 📊 Analysis Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Frames Processed",
                    results["frames_processed"]
                )

                col2.metric(
                    "Unique Tracked IDs",
                    results["unique_tracks"]
                )

                col3.metric(
                    "Detected Object Classes",
                    len(results["class_counts"])
                )

                st.markdown("### Object Detection Counts")

                st.json(results["class_counts"])

                st.markdown("### 🎥 Annotated Video")

                video_path = results["output_video"]

                if os.path.exists(video_path):

                    with open(video_path, "rb") as video_file:
                        video_bytes = video_file.read()

                    st.video(
                        video_bytes,
                        format="video/mp4"
                    )

                    st.download_button(
                        "⬇️ Download Annotated Video",
                        data=video_bytes,
                        file_name="NEMESIS_X_annotated.mp4",
                        mime="video/mp4"
                    )

                st.markdown("### 📄 Detection Report")

                st.dataframe(
                    results["report"].head(100),
                    width="stretch",
                    hide_index=True
                )

                with open(results["output_csv"], "rb") as file:
                    st.download_button(
                        "⬇️ Download Detection CSV",
                        data=file.read(),
                        file_name="NEMESIS_X_detection_report.csv",
                        mime="text/csv"
                    )

            except Exception as error:
                status.error("Video analysis failed: " + str(error))

st.warning("DEBUG: Streamlit reached the camera section")


## ==========================================
# LIVE CAMERA
# ==========================================

st.divider()

st.markdown("## 📹 Live AI Camera")

st.info("Click START to activate your webcam.")

webrtc_ctx = webrtc_streamer(
    key="nemesis-live-camera-final",
    video_processor_factory=NemesisProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
    video_html_attrs={
        "autoPlay": True,
        "controls": True,
        "muted": True,
        "playsInline": True
    }
)

st.write("WebRTC State:", webrtc_ctx.state)
st.write("Dashboard diagnostic:", datetime.now().strftime("%H:%M:%S"))

if webrtc_ctx.state.playing:

    st.success("Camera is running.")

if webrtc_ctx.video_processor:

    processor = webrtc_ctx.video_processor

    with processor.lock:
        current_objects = processor.object_count
        total_frames = processor.frame_count

    col1, col2 = st.columns(2)

    col1.metric("Objects Detected", current_objects)
    col2.metric("Frames Processed", total_frames)

st.divider()


# ==========================================
# LOAD INTELLIGENCE DATA
# ==========================================

events = load_csv(EVENT_FILE)
alerts = load_csv(ALERT_FILE)

# ==========================================
# SYSTEM STATUS
# ==========================================

st.markdown("## 🟢 System Status")

c1, c2 = st.columns(2)

c1.success("Dashboard online")

c2.info(
    "Last refresh: " +
    datetime.now().strftime("%H:%M:%S")
)

# ==========================================
# METRICS
# ==========================================

st.markdown("## 📊 Intelligence Overview")

total_events = len(events)

total_tracks = (
    events["Track_ID"].nunique()
    if not events.empty and "Track_ID" in events.columns
    else 0
)

total_alerts = len(alerts)

zone_entries = (
    (events["Event"] == "ENTERED ZONE").sum()
    if not events.empty and "Event" in events.columns
    else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Events", total_events)
c2.metric("Tracked IDs", total_tracks)
c3.metric("Anomaly Alerts", total_alerts)
c4.metric("Zone Entries", int(zone_entries))

st.divider()

# ==========================================
# EVENT ANALYTICS
# ==========================================

st.markdown("## 📈 Event Analytics")

if not events.empty and "Event" in events.columns:

    st.bar_chart(events["Event"].value_counts())

else:

    st.info("No event history available yet.")

st.divider()

# ==========================================
# ANOMALY ALERTS
# ==========================================

st.markdown("## 🚨 Anomaly Detection")

if not alerts.empty:

    st.warning(f"{total_alerts} anomaly alert(s) recorded.")

    st.dataframe(
        alerts,
        width="stretch",
        hide_index=True
    )

else:

    st.success("No anomaly alerts recorded.")

st.divider()

# ==========================================
# EVENT HISTORY
# ==========================================

st.markdown("## 📋 Event History")

if not events.empty:

    st.dataframe(
        events.tail(100).iloc[::-1],
        width="stretch",
        hide_index=True
    )

else:

    st.info("Waiting for event history...")

st.divider()



# ==========================================
# PHASE 8.7 - ADVANCED EVENT INTELLIGENCE
# ==========================================

st.markdown("## 🧠 Advanced Event Intelligence Graph")

# ------------------------------------------
# MANUAL REFRESH
# ------------------------------------------

if st.button("🔄 Refresh Intelligence Data"):

    st.cache_data.clear()
    st.rerun()

# ------------------------------------------
# PREPARE EVENT DATA
# ------------------------------------------

if not events.empty and "Track_ID" in events.columns:

    graph_events = events.copy()

    graph_events["Track_ID"] = (
        graph_events["Track_ID"].astype(str)
    )

    graph_events["Parsed_Time"] = pd.to_datetime(
        graph_events["Timestamp"],
        errors="coerce"
    )

    graph_events = graph_events.dropna(
        subset=["Parsed_Time"]
    )

    # Normalize integer-style IDs such as 3.0 to 3
    def normalize_track_id(value):

        value = str(value)

        try:
            number = float(value)

            if number.is_integer():
                return str(int(number))

        except ValueError:
            pass

        return value

    graph_events["Track_ID"] = (
        graph_events["Track_ID"].apply(normalize_track_id)
    )

    # --------------------------------------
    # FILTER CONTROLS
    # --------------------------------------

    st.markdown("### 🔎 Intelligence Filters")

    col1, col2 = st.columns(2)

    track_options = sorted(
        graph_events["Track_ID"].unique().tolist()
    )

    with col1:

        selected_track = st.selectbox(
            "Select Tracked Object",
            ["All Objects"] + track_options
        )

    with col2:

        event_options = sorted(
            graph_events["Event"].dropna().unique().tolist()
        )

        selected_event_types = st.multiselect(
            "Filter Event Types",
            event_options,
            default=event_options
        )

    # --------------------------------------
    # DATE FILTER
    # --------------------------------------

    min_date = graph_events["Parsed_Time"].min().date()
    max_date = graph_events["Parsed_Time"].max().date()

    selected_dates = st.date_input(
        "Filter Event Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # --------------------------------------
    # APPLY FILTERS
    # --------------------------------------

    filtered_events = graph_events.copy()

    if selected_track != "All Objects":

        filtered_events = filtered_events[
            filtered_events["Track_ID"] == selected_track
        ]

    filtered_events = filtered_events[
        filtered_events["Event"].isin(selected_event_types)
    ]

    if isinstance(selected_dates, (tuple, list)):

        if len(selected_dates) == 2:

            start_date, end_date = selected_dates

            filtered_events = filtered_events[
                (filtered_events["Parsed_Time"].dt.date >= start_date)
                &
                (filtered_events["Parsed_Time"].dt.date <= end_date)
            ]

    # --------------------------------------
    # PREPARE ANOMALY LOOKUP
    # --------------------------------------

    anomaly_lookup = {}

    if (
        not alerts.empty
        and "Track_ID" in alerts.columns
        and "Anomaly" in alerts.columns
        and "Timestamp" in alerts.columns
    ):

        for _, alert in alerts.iterrows():

            alert_id = normalize_track_id(
                alert["Track_ID"]
            )

            alert_time = pd.to_datetime(
                alert["Timestamp"],
                errors="coerce"
            )

            if pd.notna(alert_time):

                alert_key = (
                    alert_id,
                    alert_time.floor("s")
                )

                anomaly_lookup[alert_key] = str(
                    alert["Anomaly"]
                )

    # --------------------------------------
    # MARK ANOMALIES IN EVENT DATA
    # --------------------------------------

    filtered_events["Anomaly"] = "NORMAL"

    for index, event in filtered_events.iterrows():

        event_key = (
            event["Track_ID"],
            event["Parsed_Time"].floor("s")
        )

        anomaly = anomaly_lookup.get(
            event_key,
            "NORMAL"
        )

        filtered_events.at[index, "Anomaly"] = anomaly

    # --------------------------------------
    # INTELLIGENCE STATISTICS
    # --------------------------------------

    total_filtered = len(filtered_events)

    long_stay_count = (
        filtered_events["Anomaly"] == "LONG ZONE STAY"
    ).sum()

    rapid_count = (
        filtered_events["Anomaly"]
        == "RAPID ZONE ENTRY AND EXIT"
    ).sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("Filtered Events", total_filtered)

    c2.metric(
        "Long Stay Anomalies",
        int(long_stay_count)
    )

    c3.metric(
        "Rapid Entry/Exit Anomalies",
        int(rapid_count)
    )

    st.divider()

    # --------------------------------------
    # SELECTED OBJECT EVENT HISTORY
    # --------------------------------------

    st.markdown("### 🕒 Chronological Event History")

    if not filtered_events.empty:

        history = filtered_events.sort_values(
            "Parsed_Time"
        )

        display_columns = [
            column
            for column in [
                "Timestamp",
                "Track_ID",
                "Object",
                "Event",
                "Details",
                "Anomaly"
            ]
            if column in history.columns
        ]

        st.dataframe(
            history[display_columns].iloc[::-1],
            width="stretch",
            hide_index=True
        )

    else:

        st.info("No events match the selected filters.")

    st.divider()

    # --------------------------------------
    # BUILD FILTERED KNOWLEDGE GRAPH
    # --------------------------------------

    st.markdown("### 🌐 Interactive Intelligence Network")

    graph = nx.DiGraph()

    track_event_history = {}

    for index, event in filtered_events.iterrows():

        track_id = event["Track_ID"]

        event_node = f"Event_{index}"

        event_type = str(event["Event"])

        anomaly = str(event["Anomaly"])

        timestamp = str(event["Timestamp"])

        object_name = str(event.get("Object", "Object"))

        details = str(event.get("Details", ""))

        track_node = f"Track_{track_id}"

        # Tracked object node
        graph.add_node(
            track_node,
            node_type="Entity",
            label=f"{object_name} {track_id}"
        )

        # Event node
        if anomaly == "LONG ZONE STAY":

            node_type = "LongStay"

        elif anomaly == "RAPID ZONE ENTRY AND EXIT":

            node_type = "RapidEntryExit"

        else:

            node_type = "Event"

        graph.add_node(
            event_node,
            node_type=node_type,
            label=event_type,
            timestamp=timestamp,
            details=details,
            anomaly=anomaly
        )

        # Object participates in event
        graph.add_edge(
            track_node,
            event_node,
            relation="PARTICIPATED_IN"
        )

        # Zone relationship
        if event_type in [
            "ENTERED ZONE",
            "EXITED ZONE"
        ]:

            graph.add_node(
                "Monitored_Zone",
                node_type="Location",
                label="MONITORED ZONE"
            )

            graph.add_edge(
                event_node,
                "Monitored_Zone",
                relation=event_type
            )

        # Store chronological event
        track_event_history.setdefault(
            track_id,
            []
        ).append({
            "node": event_node,
            "time": event["Parsed_Time"]
        })

    # --------------------------------------
    # CHRONOLOGICAL CONNECTIONS
    # --------------------------------------

    for track_id, history in track_event_history.items():

        history.sort(
            key=lambda item: item["time"]
        )

        for previous, current in zip(
            history,
            history[1:]
        ):

            graph.add_edge(
                previous["node"],
                current["node"],
                relation="NEXT_EVENT"
            )

    # --------------------------------------
    # DRAW INTERACTIVE GRAPH
    # --------------------------------------

    if graph.number_of_nodes() > 0:

        positions = nx.spring_layout(
            graph,
            seed=42,
            k=1.8,
            iterations=100
        )

        # Draw connections
        edge_x = []
        edge_y = []

        for source, target in graph.edges():

            x0, y0 = positions[source]
            x1, y1 = positions[target]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    width=1.2,
                    color="#52647d"
                ),
                hoverinfo="none",
                name="Relationships"
            )
        )

        # Node color categories
        categories = {
            "Entity": "#38bdf8",
            "Event": "#22c55e",
            "LongStay": "#ef4444",
            "RapidEntryExit": "#a855f7",
            "Location": "#f97316"
        }

        for node_type, color in categories.items():

            nodes = [
                node
                for node, data in graph.nodes(data=True)
                if data.get("node_type") == node_type
            ]

            if not nodes:
                continue

            node_x = []
            node_y = []
            labels = []
            hover_text = []

            for node in nodes:

                data = graph.nodes[node]

                x, y = positions[node]

                node_x.append(x)
                node_y.append(y)

                labels.append(
                    str(data.get("label", node))
                )

                hover_text.append(
                    f"Node: {node}<br>"
                    f"Type: {node_type}<br>"
                    f"Label: {data.get('label', '')}<br>"
                    f"Timestamp: {data.get('timestamp', 'N/A')}<br>"
                    f"Anomaly: {data.get('anomaly', 'NORMAL')}<br>"
                    f"Details: {data.get('details', 'N/A')}"
                )

            figure.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont=dict(
                        size=10,
                        color="white"
                    ),
                    hovertext=hover_text,
                    hoverinfo="text",
                    marker=dict(
                        size=19,
                        color=color,
                        line=dict(
                            width=1.5,
                            color="white"
                        )
                    ),
                    name=node_type
                )
            )

        figure.update_layout(
            title="NEMESIS-X | Advanced Event Intelligence",
            template="plotly_dark",
            height=750,
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120",
            hovermode="closest",
            showlegend=True,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            )
        )

        st.plotly_chart(
            figure,
            width="stretch",
            key="nemesis_phase87_graph"
        )

        st.caption(
            "Drag to explore | Scroll to zoom | "
            "Hover over nodes to inspect events"
        )

    else:

        st.info(
            "No graph nodes available for these filters."
        )

    # --------------------------------------
    # GRAPH LEGEND
    # --------------------------------------

    st.markdown("### 🎨 Graph Legend")

    st.markdown("""
    - 🔵 Blue: Tracked objects
    - 🟢 Green: Normal events
    - 🔴 Red: LONG ZONE STAY anomaly
    - 🟣 Purple: RAPID ZONE ENTRY AND EXIT anomaly
    - 🟠 Orange: Monitored zone
    """)

else:

    st.info(
        "No event history available. "
        "Run the NEMESIS-X detection pipeline first."
    )

# ==========================================
# EVENT EXPLANATIONS
# ==========================================

st.markdown("## 📝 Event Intelligence Explanations")

if os.path.exists(EXPLANATION_FILE):

    with open(
        EXPLANATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        explanation_text = file.read()

    st.text_area(
        "Generated Explanations",
        explanation_text,
        height=300
    )

else:

    st.info("No event explanations available.")

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "NEMESIS-X | Phase 8.5 | Integrated Intelligence Dashboard"
)
