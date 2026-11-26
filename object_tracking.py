
import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Webcam not detected!")
    exit()

print("NEMESIS-X Object Tracking Started!")
print("Press Q to exit.")

while True:

    # Read webcam frame
    success, frame = camera.read()

    if not success:
        print("Unable to read frame.")
        break

    # Run YOLO with ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.40,
        verbose=False
    )

    # Get detection results
    result = results[0]

    # Draw bounding boxes and tracking IDs
    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            object_name = model.names[class_id]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display object name and ID
            label = f"{object_name} | ID: {track_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display project title
    cv2.putText(
        frame,
        "NEMESIS-X | OBJECT TRACKING",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Show tracking output
    cv2.imshow("NEMESIS-X Tracking", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Tracking stopped successfully!")
