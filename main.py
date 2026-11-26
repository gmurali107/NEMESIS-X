
import sys
import subprocess
from pathlib import Path

# ==========================================
# NEMESIS-X PHASE 8.3
# REAL-TIME APPLICATION CONTROLLER
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

print("\n" + "=" * 55)
print("       NEMESIS-X REAL-TIME INTELLIGENCE")
print("=" * 55)

required_files = [
    "yolo11n.pt",
    "temporal_spatial_analysis.py",
    "realtime_monitor.py"
]

for filename in required_files:

    if not (PROJECT_DIR / filename).exists():

        print("Missing file:", filename)
        sys.exit(1)


monitor = None

try:

    # --------------------------------------
    # START REAL-TIME EVENT MONITOR
    # --------------------------------------

    print("\nStarting real-time event monitor...")

    monitor = subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_DIR / "realtime_monitor.py")
        ],
        cwd=str(PROJECT_DIR)
    )

    # --------------------------------------
    # START WEBCAM ANALYSIS
    # --------------------------------------

    print("\nStarting webcam intelligence pipeline...")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "temporal_spatial_analysis.py")
        ],
        cwd=str(PROJECT_DIR),
        check=False
    )

    if result.returncode != 0:

        print("\nWebcam analysis failed.")

    else:

        print("\nWebcam analysis completed.")

except KeyboardInterrupt:

    print("\nNEMESIS-X interrupted.")

finally:

    # --------------------------------------
    # STOP BACKGROUND MONITOR
    # --------------------------------------

    if monitor is not None:

        print("\nStopping background event monitor...")

        monitor.terminate()

        try:
            monitor.wait(timeout=5)

        except subprocess.TimeoutExpired:

            monitor.kill()
            monitor.wait()

# ------------------------------------------
# FINAL UPDATE
# ------------------------------------------

print("\nGenerating final intelligence report...")

final_result = subprocess.run(
    [
        sys.executable,
        str(PROJECT_DIR / "realtime_monitor.py"),
        "--once"
    ],
    cwd=str(PROJECT_DIR),
    check=False
)

if final_result.returncode != 0:

    print("Final report generation failed.")
    sys.exit(final_result.returncode)

print("\n" + "=" * 55)
print("       NEMESIS-X PROCESSING COMPLETED")
print("=" * 55)

outputs = [
    "event_history.csv",
    "anomaly_alerts.csv",
    "event_explanations.txt",
    "event_graph.png",
    "event_graph.gexf"
]

for filename in outputs:

    if (PROJECT_DIR / filename).exists():
        print("[SUCCESS]", filename)
    else:
        print("[MISSING]", filename)

print("\nNEMESIS-X Phase 8.3 completed!")
