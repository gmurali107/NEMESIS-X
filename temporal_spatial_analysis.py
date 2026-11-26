
import cv2
import time
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------
# 1. INITIALIZE YOLO AND WEBCAM
# ---------------------------------

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Webcam not detected!")
    exit()

# Store information for every tracked object
object_memory = {}

# Event history file
CSV_FILE = "event_history.csv"

# Create CSV if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "Track_ID",
            "Object",
            "Event",
            "Details"
        ])

# ---------------------------------
# 2. EVENT LOGGING FUNCTION
# ---------------------------------

def log_event(track_id, name, event, details=""):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{timestamp}] ID {track_id} | "
        f"{name} | {event} | {details}"
    )

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            track_id,
            name,
            event,
            details
        ])

# ---------------------------------
# 3. START VIDEO PROCESSING
# ---------------------------------

print("NEMESIS-X Temporal-Spatial Analysis Started!")

while True:

    success, frame = camera.read()

    if not success:
        print("Cannot read webcam frame.")
        break

    height, width = frame.shape[:2]

    current_time = time.time()

    # Define monitored zone
    zone_x1 = int(width * 0.25)
    zone_y1 = int(height * 0.20)
    zone_x2 = int(width * 0.75)
    zone_y2 = int(height * 0.90)

    # Run YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.40,
        verbose=False
    )

    result = results[0]

    active_ids = set()
    person_count = 0
    inside_count = 0

    # ---------------------------------
    # 4. PROCESS DETECTED OBJECTS
    # ---------------------------------

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            # Object center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            object_name = model.names[class_id]

            active_ids.add(track_id)

            # Check whether object is inside monitored zone
            inside_zone = (
                zone_x1 <= center_x <= zone_x2
                and zone_y1 <= center_y <= zone_y2
            )

            # ---------------------------------
            # 5. CREATE OBJECT MEMORY
            # ---------------------------------

            if track_id not in object_memory:

                object_memory[track_id] = {
                    "name": object_name,
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "history": [],
                    "in_zone": False,
                    "zone_enter_time": None,
                    "dwell_alerted": False,
                    "lost_logged": False
                }

                log_event(
                    track_id,
                    object_name,
                    "FIRST SEEN"
                )

            memory = object_memory[track_id]

            memory["last_seen"] = current_time
            memory["lost_logged"] = False

            # ---------------------------------
            # 6. UPDATE MOVEMENT HISTORY
            # ---------------------------------

            memory["history"].append(
                (center_x, center_y)
            )

            # Keep only recent 30 positions
            memory["history"] = memory["history"][-30:]

            # Draw movement trail
            points = memory["history"]

            for i in range(1, len(points)):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (255, 0, 0),
                    2
                )

            # ---------------------------------
            # 7. ZONE ENTRY AND EXIT DETECTION
            # ---------------------------------

            if inside_zone and not memory["in_zone"]:

                memory["in_zone"] = True
                memory["zone_enter_time"] = current_time
                memory["dwell_alerted"] = False

                log_event(
                    track_id,
                    object_name,
                    "ENTERED ZONE"
                )

            elif not inside_zone and memory["in_zone"]:

                memory["in_zone"] = False

                duration = (
                    current_time -
                    memory["zone_enter_time"]
                )

                log_event(
                    track_id,
                    object_name,
                    "EXITED ZONE",
                    f"Duration: {duration:.1f} seconds"
                )

                memory["zone_enter_time"] = None
                memory["dwell_alerted"] = False

            # ---------------------------------
            # 8. TEMPORAL ANALYSIS
            # ---------------------------------

            total_time = (
                current_time -
                memory["first_seen"]
            )

            zone_duration = 0

            if memory["in_zone"]:

                zone_duration = (
                    current_time -
                    memory["zone_enter_time"]
                )

            # Alert after 15 seconds in zone
            if (
                object_name == "person"
                and memory["in_zone"]
                and zone_duration >= 15
                and not memory["dwell_alerted"]
            ):

                log_event(
                    track_id,
                    object_name,
                    "LONG ZONE STAY",
                    f"Stayed {zone_duration:.1f} seconds"
                )

                memory["dwell_alerted"] = True

            if object_name == "person":
                person_count += 1

                if inside_zone:
                    inside_count += 1

            # ---------------------------------
            # 9. DRAW INFORMATION
            # ---------------------------------

            color = (0, 255, 0)

            if memory["in_zone"]:
                color = (0, 165, 255)

            label = f"{object_name} | ID: {track_id}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (255, 0, 0),
                -1
            )

    # ---------------------------------
    # 10. DETECT LOST TRACKS
    # ---------------------------------

    for track_id, memory in object_memory.items():

        if (
            track_id not in active_ids
            and not memory["lost_logged"]
            and current_time - memory["last_seen"] > 3
        ):

            log_event(
                track_id,
                memory["name"],
                "LOST FROM VIEW"
            )

            memory["lost_logged"] = True

    # ---------------------------------
    # 11. DRAW MONITORED ZONE
    # ---------------------------------

    cv2.rectangle(
        frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "MONITORED ZONE",
        (zone_x1, zone_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # ---------------------------------
    # 12. DISPLAY STATISTICS
    # ---------------------------------

    cv2.putText(
        frame,
        f"People Detected: {person_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"People Inside Zone: {inside_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "NEMESIS-X | TEMPORAL SPATIAL ANALYSIS",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "NEMESIS-X Phase 4",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------------
# 13. RELEASE RESOURCES
# ---------------------------------

camera.release()
cv2.destroyAllWindows()

print("Temporal-Spatial Analysis Completed!")
print("Event history saved in:", CSV_FILE)
