
import cv2

# Connect to the default webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check camera connection
if not camera.isOpened():
    print("Error: Webcam not detected!")
    exit()

print("NEMESIS-X webcam started!")
print("Click the video and Press Q or q to close the camera.")

while True:

    # Read a frame from webcam
    success, frame = camera.read()

    if not success:
        print("Unable to read webcam frame.")
        break

    # Add project name to video
    cv2.putText(
        frame,
        "NEMESIS-X | LIVE MONITORING",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Display live video
    cv2.imshow("NEMESIS-X Webcam", frame)

    # Press Q or q to exit(also exit if user clicks the window's X button)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

    if cv2.getWindowProperty("NEMESIS-X Webcam", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release webcam
camera.release()

# Close all windows
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Webcam stopped successfully.")
