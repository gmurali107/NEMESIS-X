
import csv
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from video_analyzer import analyze_video


# =====================================
# NEMESIS-X API
# =====================================

app = FastAPI(
    title="NEMESIS-X API",
    description="NEMESIS-X Intelligence Backend",
    version="2.0.0"
)


# =====================================
# DIRECTORIES AND CSV FILES
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "video_outputs"

CSV_FILE = BASE_DIR / "event_history.csv"
ALERT_FILE = BASE_DIR / "anomaly_alerts.csv"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/video-outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="video-outputs"
)


# =====================================
# CONFIGURATION
# =====================================

LONG_STAY_SECONDS = 15
RAPID_EVENT_SECONDS = 5
LOST_TRACK_SECONDS = 3

CONFIDENCE_THRESHOLD = 0.40

# Monitored zone, expressed as frame percentages.
ZONE_X1_RATIO = 0.25
ZONE_Y1_RATIO = 0.20
ZONE_X2_RATIO = 0.75
ZONE_Y2_RATIO = 0.90


# =====================================
# THREAD SAFETY
# =====================================

data_lock = threading.RLock()


# =====================================
# YOLO MODEL INITIALIZATION
# =====================================

MODEL_PATH = BASE_DIR / "yolo11n.pt"

model = None
model_status = "offline"

try:
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        model_status = "online"
        print("NEMESIS-X YOLO11 model loaded successfully")
    else:
        print("YOLO model file not found:", MODEL_PATH)

except Exception as error:
    print("YOLO loading error:", error)


# =====================================
# CORS CONFIGURATION
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# CREATE CSV FILES IF REQUIRED
# =====================================

EVENT_FIELDS = [
    "Timestamp",
    "Track_ID",
    "Object",
    "Event",
    "Details"
]

ALERT_FIELDS = [
    "Track_ID",
    "Anomaly",
    "Timestamp",
    "Explanation"
]


def initialize_csv_files():

    if not CSV_FILE.exists():
        with open(
            CSV_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(EVENT_FIELDS)

    if not ALERT_FILE.exists():
        with open(
            ALERT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(ALERT_FIELDS)


initialize_csv_files()


# =====================================
# TEMPORAL TRACKING MEMORY
# =====================================

object_memory = {}


# =====================================
# TIMESTAMP PARSER
# =====================================

def parse_timestamp(value):

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            pass

    return None


# =====================================
# ANOMALY DETECTION
# =====================================

def detect_anomalies():

    events = []

    try:
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            events = list(reader)

    except FileNotFoundError:
        return []

    alerts = []
    track_events = defaultdict(list)

    for event in events:

        track_id = event.get("Track_ID", "")
        event_name = event.get("Event", "")
        object_name = event.get("Object", "")

        timestamp = parse_timestamp(event.get("Timestamp", ""))

        if timestamp is None:
            continue

        track_events[track_id].append({
            "event": event_name,
            "time": timestamp,
            "details": event.get("Details", "")
        })

        # Immediate alert when a person enters the monitored zone
        if event_name == "ENTERED ZONE" and object_name == "person":

            alerts.append({
                "Track_ID": track_id,
                "Anomaly": "PERSON ENTERED MONITORED ZONE",
                "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Explanation": "A person has entered the monitored zone."
            })

        # Long stay alert
        if event_name == "LONG ZONE STAY":

            alerts.append({
                "Track_ID": track_id,
                "Anomaly": "LONG ZONE STAY",
                "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Explanation": event.get("Details") or
                    "Person remained in the monitored zone too long."
            })

    # Rapid entry and exit alert
    for track_id, history in track_events.items():

        history.sort(key=lambda item: item["time"])

        for previous, current in zip(history, history[1:]):

            elapsed = (
                current["time"] - previous["time"]
            ).total_seconds()

            if (
                previous["event"] == "ENTERED ZONE"
                and current["event"] == "EXITED ZONE"
                and 0 <= elapsed <= RAPID_EVENT_SECONDS
            ):

                alerts.append({
                    "Track_ID": track_id,
                    "Anomaly": "RAPID ZONE ENTRY AND EXIT",
                    "Timestamp": current["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "Explanation": f"Zone entry and exit occurred within {elapsed:.1f} seconds."
                })

    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=ALERT_FIELDS)
        writer.writeheader()
        writer.writerows(alerts)

    return alerts



# =====================================
# EVENT LOGGING
# =====================================

def log_event(
    track_id,
    object_name,
    event_name,
    details=""
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    event = {
        "Timestamp": timestamp,
        "Track_ID": str(track_id),
        "Object": object_name,
        "Event": event_name,
        "Details": details
    }

    with data_lock:

        with open(
            CSV_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=EVENT_FIELDS
            )

            writer.writerow(event)

        # Refresh anomaly report after each new event
        detect_anomalies()

    print(
        f"[{timestamp}] ID {track_id} | "
        f"{object_name} | {event_name} | {details}"
    )

    return event


# =====================================
# MONITORED ZONE COORDINATES
# =====================================

def get_monitored_zone(frame):

    height, width = frame.shape[:2]

    return [
        int(width * ZONE_X1_RATIO),
        int(height * ZONE_Y1_RATIO),
        int(width * ZONE_X2_RATIO),
        int(height * ZONE_Y2_RATIO)
    ]


# =====================================
# TEMPORAL-SPATIAL EVENT PROCESSING
# =====================================

def process_temporal_events(frame, detections):

    current_time = datetime.now().timestamp()

    zone = get_monitored_zone(frame)

    zone_x1, zone_y1, zone_x2, zone_y2 = zone

    active_ids = set()
    generated_events = []

    person_count = 0
    inside_count = 0

    for detection in detections:

        track_id = detection.get("track_id")

        # Tracking IDs are required for temporal analysis
        if track_id is None:
            continue

        track_id = int(track_id)

        object_name = detection["object"]

        x1, y1, x2, y2 = detection["coordinates"]

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        active_ids.add(track_id)

        inside_zone = (
            zone_x1 <= center_x <= zone_x2
            and zone_y1 <= center_y <= zone_y2
        )

        # Create memory for new objects
        if track_id not in object_memory:

            object_memory[track_id] = {
                "name": object_name,
                "first_seen": current_time,
                "last_seen": current_time,
                "in_zone": False,
                "zone_enter_time": None,
                "dwell_alerted": False,
                "lost_logged": False
            }

            event = log_event(
                track_id,
                object_name,
                "FIRST SEEN"
            )

            generated_events.append(event)

        memory = object_memory[track_id]

        memory["last_seen"] = current_time
        memory["lost_logged"] = False

        # Zone entry
        if inside_zone and not memory["in_zone"]:

            memory["in_zone"] = True
            memory["zone_enter_time"] = current_time
            memory["dwell_alerted"] = False

            event = log_event(
                track_id,
                object_name,
                "ENTERED ZONE"
            )

            generated_events.append(event)

        # Zone exit
        elif not inside_zone and memory["in_zone"]:

            memory["in_zone"] = False

            duration = (
                current_time - memory["zone_enter_time"]
            )

            event = log_event(
                track_id,
                object_name,
                "EXITED ZONE",
                f"Duration: {duration:.1f} seconds"
            )

            generated_events.append(event)

            memory["zone_enter_time"] = None
            memory["dwell_alerted"] = False

        # Long zone stay: persons only
        if (
            object_name == "person"
            and memory["in_zone"]
            and memory["zone_enter_time"] is not None
        ):

            zone_duration = (
                current_time - memory["zone_enter_time"]
            )

            if (
                zone_duration >= LONG_STAY_SECONDS
                and not memory["dwell_alerted"]
            ):

                event = log_event(
                    track_id,
                    object_name,
                    "LONG ZONE STAY",
                    f"Stayed {zone_duration:.1f} seconds"
                )

                generated_events.append(event)

                memory["dwell_alerted"] = True

        if object_name == "person":

            person_count += 1

            if inside_zone:
                inside_count += 1

    # Detect objects that have disappeared from the camera
    for track_id, memory in list(object_memory.items()):

        if (
            track_id not in active_ids
            and not memory["lost_logged"]
            and current_time - memory["last_seen"] > LOST_TRACK_SECONDS
        ):

            event = log_event(
                track_id,
                memory["name"],
                "LOST FROM VIEW"
            )

            generated_events.append(event)

            memory["lost_logged"] = True

    return {
        "zone": {
            "coordinates": zone,
            "description": "Monitored Zone"
        },
        "person_count": person_count,
        "people_inside_zone": inside_count,
        "events": generated_events
    }


# =====================================
# SHARED IMAGE DETECTION FUNCTION
# =====================================

def run_detection(image, tracking=False):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO model is not available"
        )

    try:

        if tracking:

            results = model.track(
                source=image,
                conf=CONFIDENCE_THRESHOLD,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

        else:

            results = model.predict(
                source=image,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

        detections = []
        class_counts = {}

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                class_id = int(box.cls.item())

                confidence = float(box.conf.item())

                class_name = model.names[class_id]

                coordinates = [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]

                detection = {
                    "object": class_name,
                    "confidence": round(confidence, 3),
                    "coordinates": coordinates
                }

                if tracking and box.id is not None:

                    detection["track_id"] = int(
                        box.id.item()
                    )

                detections.append(detection)

                class_counts[class_name] = (
                    class_counts.get(class_name, 0) + 1
                )

        return {
            "total_objects": len(detections),
            "class_counts": class_counts,
            "detections": detections
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(error)}"
        )


# =====================================
# HOME AND STATUS ENDPOINTS
# =====================================

@app.get("/")
def home():

    return {
        "message": "NEMESIS-X Backend is running",
        "status": "online"
    }


@app.get("/api/health")
def health():

    return {
        "system": "NEMESIS-X",
        "backend": "online",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/system-status")
def system_status():

    return {
        "system": "NEMESIS-X",
        "backend": "online",
        "yolo11": model_status,
        "model_file": MODEL_PATH.name,
        "model_exists": MODEL_PATH.exists(),
        "temporal_analysis": "online",
        "anomaly_detection": "online",
        "timestamp": datetime.now().isoformat()
    }


# =====================================
# IMAGE DETECTION
# =====================================

@app.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...)):

    try:

        image_bytes = await file.read()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        result = run_detection(image)

        return {
            "system": "NEMESIS-X",
            "status": "success",
            **result
        }

    finally:
        await file.close()


# =====================================
# LIVE CAMERA DETECTION + TRACKING
# =====================================

@app.post("/api/detect-frame")
async def detect_frame(file: UploadFile = File(...)):

    try:

        image_bytes = await file.read()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid camera frame"
            )

        # YOLO + ByteTrack
        result = run_detection(
            frame,
            tracking=True
        )

        # Temporal-spatial processing
        with data_lock:

            temporal_result = process_temporal_events(
                frame,
                result["detections"]
            )

        return {
            "system": "NEMESIS-X",
            "status": "success",
            "tracking": True,
            "timestamp": datetime.now().isoformat(),
            **result,
            **temporal_result
        }

    finally:
        await file.close()


# =====================================
# EVENT HISTORY API
# =====================================

@app.get("/api/events")
def get_events(limit: int = 100):

    limit = max(1, min(limit, 1000))

    try:

        with data_lock:

            with open(
                CSV_FILE,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)
                events = list(reader)

        events.reverse()

        return {
            "system": "NEMESIS-X",
            "total_events": len(events),
            "events": events[:limit]
        }

    except FileNotFoundError:

        return {
            "system": "NEMESIS-X",
            "total_events": 0,
            "events": []
        }


# =====================================
# ANOMALY ALERTS API
# =====================================

@app.get("/api/alerts")
def get_alerts(limit: int = 100):

    limit = max(1, min(limit, 1000))

    with data_lock:

        alerts = detect_anomalies()

    alerts.reverse()

    return {
        "system": "NEMESIS-X",
        "total_alerts": len(alerts),
        "alerts": alerts[:limit]
    }


# =====================================
# TEMPORAL ANALYSIS STATUS API
# =====================================

@app.get("/api/temporal-status")
def temporal_status():

    with data_lock:

        tracked_objects = len(object_memory)

        active_objects = sum(
            1
            for memory in object_memory.values()
            if not memory["lost_logged"]
        )

        objects_inside_zone = sum(
            1
            for memory in object_memory.values()
            if memory["in_zone"]
        )

    return {
        "system": "NEMESIS-X",
        "temporal_analysis": "online",
        "tracked_objects": tracked_objects,
        "active_objects": active_objects,
        "objects_inside_zone": objects_inside_zone,
        "long_stay_threshold_seconds": LONG_STAY_SECONDS,
        "rapid_event_threshold_seconds": RAPID_EVENT_SECONDS,
        "lost_track_threshold_seconds": LOST_TRACK_SECONDS
    }


# =====================================
# VIDEO ANALYSIS
# =====================================

@app.post("/api/analyze-video")
async def analyze_uploaded_video(
    file: UploadFile = File(...),
    analysis_mode: str = Form(
        "Object Detection and Tracking"
    )
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="YOLO model is not available"
        )

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    if analysis_mode not in {
        "Object Detection Only",
        "Object Detection and Tracking"
    }:

        raise HTTPException(
            status_code=400,
            detail="Invalid analysis mode"
        )

    job_id = uuid4().hex

    input_path = INPUT_DIR / f"{job_id}{extension}"

    job_output_dir = OUTPUT_DIR / job_id

    try:

        video_bytes = await file.read()

        if not video_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty"
            )

        with open(input_path, "wb") as video_file:
            video_file.write(video_bytes)

        result = analyze_video(
            str(input_path),
            str(job_output_dir),
            model,
            analysis_mode
        )

        output_video = Path(
            result["output_video"]
        )

        output_csv = Path(
            result["output_csv"]
        )

        if (
            not output_video.exists()
            or not output_csv.exists()
        ):

            raise RuntimeError(
                "Analysis output files were not created"
            )

        return {
            "system": "NEMESIS-X",
            "status": "success",
            "job_id": job_id,
            "frames_processed": result["frames_processed"],
            "unique_tracks": result["unique_tracks"],
            "class_counts": result["class_counts"],
            "annotated_video_url": (
                f"/video-outputs/{job_id}/{output_video.name}"
            ),
            "report_csv_url": (
                f"/video-outputs/{job_id}/{output_csv.name}"
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(error)}"
        )

    finally:
        await file.close()
