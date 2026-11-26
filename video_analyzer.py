
import cv2
import pandas as pd
from pathlib import Path
from datetime import timedelta
import subprocess
import imageio_ffmpeg



def analyze_video(
    video_path,
    output_dir,
    model,
    analysis_mode,
    progress_callback=None
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = output_dir / "annotated_video.mp4"
    temp_video = output_dir / "annotated_video_temp.mp4"
    output_csv = output_dir / "detection_report.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(temp_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create output video.")

    records = []
    frame_number = 0

    unique_tracks = set()
    class_counts = {}
    unique_object_ids = {}

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            if analysis_mode == "Object Detection Only":
                results = model.predict(
                    frame,
                    conf=0.40,
                    verbose=False
                )
            else:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.40,
                    verbose=False
                )

            result = results[0]
            annotated = result.plot()
            boxes = result.boxes

            if boxes is not None:

                for index, box in enumerate(boxes):

                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = model.names[class_id]

                    # Count all detections across frames
                    class_counts[class_name] = (
                        class_counts.get(class_name, 0) + 1
                    )

                    track_id = None

                    # Get ByteTrack ID when tracking is enabled
                    if (
                        analysis_mode != "Object Detection Only"
                        and boxes.id is not None
                    ):
                        track_id = int(boxes.id[index].item())
                        unique_tracks.add(track_id)

                        # Count each tracked object only once per class
                        unique_object_ids.setdefault(
                            class_name, set()
                        ).add(track_id)

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    records.append({
                        "Frame": frame_number,
                        "Timestamp_Seconds": round(
                            frame_number / fps, 3
                        ),
                        "Track_ID": track_id,
                        "Object": class_name,
                        "Confidence": round(confidence, 4),
                        "X1": round(x1, 2),
                        "Y1": round(y1, 2),
                        "X2": round(x2, 2),
                        "Y2": round(y2, 2)
                    })

            writer.write(annotated)

            frame_number += 1

            if progress_callback and total_frames > 0:
                progress_callback(
                    min(frame_number / total_frames, 1.0)
                )

    finally:
        cap.release()
        writer.release()

    # Convert temporary video to browser-compatible H.264 MP4
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(temp_video),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_video)
    ]

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True
    )

    if temp_video.exists():
        temp_video.unlink()

    report = pd.DataFrame(records)
    report.to_csv(output_csv, index=False)

    # Use unique tracked IDs for individual object counts
    unique_counts = {
        name: len(ids)
        for name, ids in unique_object_ids.items()
    }

    return {
        "output_video": str(output_video),
        "output_csv": str(output_csv),
        "frames_processed": frame_number,
        "unique_tracks": len(unique_tracks),
        "class_counts": class_counts,
        "unique_object_counts": unique_counts,
        "report": report
    }
