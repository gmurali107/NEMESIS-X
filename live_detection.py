
import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Connect webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Webcam not detected!")
    exit()

print("NEMESIS-X AI detection started!")
print("Click the video and Press Q or q to exit.")

while True:

    # Capture webcam frame
    success, frame = camera.read()

    if not success:
        print("Cannot read webcam frame.")
        break

    # Run YOLO detection
    results = model(frame, conf=0.40, verbose=False)

    # Draw detection boxes and labels
    annotated_frame = results[0].plot()

    # Display project title
    cv2.putText(
        annotated_frame,
        "NEMESIS-X | AI LIVE DETECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Display video
    cv2.imshow("NEMESIS-X Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("NEMESIS-X detection stopped.")
