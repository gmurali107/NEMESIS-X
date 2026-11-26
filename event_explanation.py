
import csv

INPUT_FILE = "anomaly_alerts.csv"
OUTPUT_FILE = "event_explanations.txt"


def explain_event(alert):
    anomaly = alert["Anomaly"]
    track_id = alert["Track_ID"]
    timestamp = alert["Timestamp"]
    evidence = alert["Explanation"]

    if anomaly == "LONG ZONE STAY":
        explanation = (
            f"Track ID {track_id} remained in the monitored zone "
            "longer than the configured time threshold."
        )

        recommendation = (
            "Review the recorded video to understand the context "
            "of this prolonged stay."
        )

    elif anomaly == "RAPID ZONE ENTRY AND EXIT":
        explanation = (
            f"Track ID {track_id} entered and exited the monitored "
            "zone within a short time interval."
        )

        recommendation = (
            "Review the video around the recorded entry and exit times."
        )

    else:
        explanation = (
            f"Track ID {track_id} triggered an event requiring review."
        )

        recommendation = (
            "Review the associated video and event history."
        )

    return {
        "track_id": track_id,
        "anomaly": anomaly,
        "timestamp": timestamp,
        "evidence": evidence,
        "explanation": explanation,
        "recommendation": recommendation
    }


def generate_report():
    try:
        with open(INPUT_FILE, "r", newline="", encoding="utf-8") as file:
            alerts = list(csv.DictReader(file))

    except FileNotFoundError:
        print("ERROR: anomaly_alerts.csv not found.")
        print("Run Phase 6 first.")
        return

    if not alerts:
        print("No anomaly alerts available to explain.")
        return

    reports = []

    for alert in alerts:
        reports.append(explain_event(alert))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("========== NEMESIS-X EVENT EXPLANATIONS ==========\n\n")

        for number, report in enumerate(reports, start=1):
            text = (
                f"ALERT {number}\n"
                f"Track ID: {report['track_id']}\n"
                f"Anomaly: {report['anomaly']}\n"
                f"Timestamp: {report['timestamp']}\n"
                f"Recorded evidence: {report['evidence']}\n"
                f"Explanation: {report['explanation']}\n"
                f"Recommended review: {report['recommendation']}\n"
                "-----------------------------------------------\n"
            )

            file.write(text)
            print(text)

    print("Explanation report saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    generate_report()
