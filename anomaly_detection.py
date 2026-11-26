
import csv
import os
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, "event_history.csv")
ALERT_FILE = os.path.join(BASE_DIR, "anomaly_alerts.csv")

# Detection rules
LONG_STAY_SECONDS = 15
RAPID_EVENT_SECONDS = 5

def parse_timestamp(value):
    """Convert a CSV timestamp into a datetime."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    return None


def detect_anomalies():
    events = []

    # Read event history
    try:
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                events.append(row)

    except FileNotFoundError:
        print("ERROR: event_history.csv not found.")
        print("Run Phase 4 first to generate the event history.")
        return

    if not events:
        print("No events found. Run Phase 4 and collect some events.")
        return

    alerts = []
    track_events = defaultdict(list)

    for event in events:
        track_id = event.get("Track_ID", "")
        event_name = event.get("Event", "")
        timestamp = parse_timestamp(event.get("Timestamp", ""))

        if timestamp is None:
            print("Skipping event with unrecognized timestamp:", event)
            continue

        track_events[track_id].append({
            "event": event_name,
            "time": timestamp,
            "details": event.get("Details", "")
        })

        # Rule 1: Long zone stay
        if event_name == "LONG ZONE STAY":
            alerts.append({
                "Track_ID": track_id,
                "Anomaly": "LONG ZONE STAY",
                "Timestamp": timestamp.isoformat(sep=" "),
                "Explanation": "The tracked object remained in the zone longer than the configured threshold."
            })

    # Rule 2: Rapid entry/exit event sequence
    for track_id, history in track_events.items():
        history.sort(key=lambda item: item["time"])

        for previous, current in zip(history, history[1:]):
            rapid_events = (
                previous["event"] == "ENTERED ZONE"
                and current["event"] == "EXITED ZONE"
            )

            elapsed = (current["time"] - previous["time"]).total_seconds()

            if rapid_events and 0 <= elapsed <= RAPID_EVENT_SECONDS:
                alerts.append({
                    "Track_ID": track_id,
                    "Anomaly": "RAPID ZONE ENTRY AND EXIT",
                    "Timestamp": current["time"].isoformat(sep=" "),
                    "Explanation": (
                        f"Zone entry and exit occurred within {elapsed:.1f} seconds."
                    )
                })

    # Display alerts
    print("\n========== NEMESIS-X ANOMALY REPORT ==========")

    if not alerts:
        print("No anomalies detected in the available event history.")

    for number, alert in enumerate(alerts, start=1):
        print(f"\nALERT {number}")
        print("Track ID:", alert["Track_ID"])
        print("Anomaly:", alert["Anomaly"])
        print("Timestamp:", alert["Timestamp"])
        print("Explanation:", alert["Explanation"])

    # Save alerts
    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "Track_ID", "Anomaly", "Timestamp", "Explanation"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(alerts)

    print("\nAnomaly report saved to:", ALERT_FILE)

    return alerts

if __name__ == "__main__":
    detect_anomalies()
