
import csv
import os
import sys
import time
import argparse
from datetime import datetime
from collections import defaultdict

import networkx as nx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==========================================
# NEMESIS-X REAL-TIME EVENT MONITOR
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, "event_history.csv")
ALERT_FILE = os.path.join(BASE_DIR, "anomaly_alerts.csv")
EXPLANATION_FILE = os.path.join(BASE_DIR, "event_explanations.txt")
GRAPH_IMAGE = os.path.join(BASE_DIR, "event_graph.png")
GRAPH_FILE = os.path.join(BASE_DIR, "event_graph.gexf")

LONG_STAY_SECONDS = 15
RAPID_EVENT_SECONDS = 5
UPDATE_INTERVAL = 3


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


# ==========================================
# 1. READ EVENT HISTORY
# ==========================================

def read_events():

    if not os.path.exists(CSV_FILE):
        return []

    with open(
        CSV_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "Timestamp",
            "Track_ID",
            "Object",
            "Event",
            "Details"
        }

        if not reader.fieldnames or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "event_history.csv has unexpected column names."
            )

        return list(reader)


# ==========================================
# 2. DETECT ANOMALIES
# ==========================================

def detect_anomalies(events):

    alerts = []
    track_events = defaultdict(list)

    for event in events:

        track_id = event.get("Track_ID", "")
        event_name = event.get("Event", "")

        timestamp = parse_timestamp(
            event.get("Timestamp", "")
        )

        if timestamp is None:
            continue

        track_events[track_id].append({
            "event": event_name,
            "time": timestamp,
            "details": event.get("Details", "")
        })

        # Long zone stay
        if event_name == "LONG ZONE STAY":

            alerts.append({
                "Track_ID": track_id,
                "Anomaly": "LONG ZONE STAY",
                "Timestamp": timestamp.isoformat(sep=" "),
                "Explanation": (
                    "The tracked object remained in the zone "
                    "longer than the configured threshold."
                )
            })

    # Rapid entry and exit
    for track_id, history in track_events.items():

        history.sort(key=lambda item: item["time"])

        for previous, current in zip(
            history,
            history[1:]
        ):

            if (
                previous["event"] == "ENTERED ZONE"
                and current["event"] == "EXITED ZONE"
            ):

                elapsed = (
                    current["time"] -
                    previous["time"]
                ).total_seconds()

                if 0 <= elapsed <= RAPID_EVENT_SECONDS:

                    alerts.append({
                        "Track_ID": track_id,
                        "Anomaly": "RAPID ZONE ENTRY AND EXIT",
                        "Timestamp": current["time"].isoformat(
                            sep=" "
                        ),
                        "Explanation": (
                            f"Zone entry and exit occurred "
                            f"within {elapsed:.1f} seconds."
                        )
                    })

    return alerts


# ==========================================
# 3. GENERATE EVENT EXPLANATIONS
# ==========================================

def generate_explanations(alerts):

    with open(
        EXPLANATION_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "========== NEMESIS-X EVENT EXPLANATIONS ==========\n\n"
        )

        if not alerts:
            file.write(
                "No anomaly alerts are currently available.\n"
            )

        for number, alert in enumerate(alerts, start=1):

            anomaly = alert["Anomaly"]
            track_id = alert["Track_ID"]

            if anomaly == "LONG ZONE STAY":

                explanation = (
                    f"Track ID {track_id} remained in the "
                    "monitored zone longer than the configured "
                    "time threshold."
                )

                recommendation = (
                    "Review the recorded video to understand "
                    "the context of this prolonged stay."
                )

            elif anomaly == "RAPID ZONE ENTRY AND EXIT":

                explanation = (
                    f"Track ID {track_id} entered and exited "
                    "the monitored zone within a short interval."
                )

                recommendation = (
                    "Review the video around the recorded "
                    "entry and exit times."
                )

            else:

                explanation = (
                    f"Track ID {track_id} triggered an event "
                    "requiring review."
                )

                recommendation = (
                    "Review the associated video and event history."
                )

            file.write(
                f"ALERT {number}\n"
                f"Track ID: {track_id}\n"
                f"Anomaly: {anomaly}\n"
                f"Timestamp: {alert['Timestamp']}\n"
                f"Evidence: {alert['Explanation']}\n"
                f"Explanation: {explanation}\n"
                f"Recommended review: {recommendation}\n"
                "---------------------------------------------\n\n"
            )


# ==========================================
# 4. SAVE ANOMALY ALERTS
# ==========================================

def save_alerts(alerts):

    fields = [
        "Track_ID",
        "Anomaly",
        "Timestamp",
        "Explanation"
    ]

    with open(
        ALERT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(alerts)


# ==========================================
# 5. BUILD EVENT GRAPH
# ==========================================

def build_event_graph(events):

    graph = nx.DiGraph()
    track_events = defaultdict(list)

    for index, event in enumerate(events):

        track_id = event.get("Track_ID", "")
        object_name = event.get("Object", "")
        event_type = event.get("Event", "")
        timestamp = event.get("Timestamp", "")
        details = event.get("Details", "")

        track_node = f"Track_{track_id}"
        event_node = f"Event_{index}"
        time_node = f"Time_{index}"

        if not graph.has_node(track_node):

            graph.add_node(
                track_node,
                node_type="Entity",
                label=f"{object_name.upper()} {track_id}"
            )

        graph.add_node(
            event_node,
            node_type="Event",
            label=event_type,
            timestamp=timestamp,
            details=details
        )

        graph.add_edge(
            track_node,
            event_node,
            relation="PARTICIPATED_IN"
        )

        graph.add_node(
            time_node,
            node_type="Time",
            label=timestamp
        )

        graph.add_edge(
            event_node,
            time_node,
            relation="OCCURRED_AT"
        )

        if event_type in ["ENTERED ZONE", "EXITED ZONE"]:

            graph.add_node(
                "Zone_A",
                node_type="Location",
                label="MONITORED ZONE"
            )

            relation = (
                "ENTERED"
                if event_type == "ENTERED ZONE"
                else "EXITED"
            )

            graph.add_edge(
                event_node,
                "Zone_A",
                relation=relation
            )

        parsed_time = parse_timestamp(timestamp)

        if parsed_time is not None:

            track_events[track_id].append({
                "node": event_node,
                "time": parsed_time,
                "index": index
            })

    # Connect events in chronological order
    for track_id, history in track_events.items():

        history.sort(
            key=lambda item: (item["time"], item["index"])
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

    return graph


# ==========================================
# 6. SAVE GRAPH IMAGE AND GRAPH DATA
# ==========================================

def save_graph(graph):

    nx.write_gexf(graph, GRAPH_FILE)

    if graph.number_of_nodes() == 0:

        plt.figure(figsize=(10, 5))

        plt.text(
            0.5,
            0.5,
            "No events recorded yet",
            ha="center",
            va="center"
        )

        plt.axis("off")

    else:

        plt.figure(figsize=(16, 10))

        positions = nx.spring_layout(
            graph,
            seed=42,
            k=1.5
        )

        colors = []

        for node, data in graph.nodes(data=True):

            node_type = data.get("node_type")

            if node_type == "Entity":
                colors.append("skyblue")

            elif node_type == "Event":
                colors.append("lightgreen")

            elif node_type == "Time":
                colors.append("lightyellow")

            elif node_type == "Location":
                colors.append("orange")

            else:
                colors.append("gray")

        labels = {
            node: data.get("label", node)
            for node, data in graph.nodes(data=True)
        }

        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=colors,
            node_size=1500
        )

        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=7
        )

        nx.draw_networkx_edges(
            graph,
            positions,
            arrows=True,
            arrowsize=15
        )

        edge_labels = nx.get_edge_attributes(
            graph,
            "relation"
        )

        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            font_size=6
        )

        plt.title("NEMESIS-X | LIVE EVENT GRAPH")
        plt.axis("off")
        plt.tight_layout()

    plt.savefig(
        GRAPH_IMAGE,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# 7. UPDATE ALL INTELLIGENCE OUTPUTS
# ==========================================

def update_system():

    events = read_events()

    alerts = detect_anomalies(events)

    graph = build_event_graph(events)

    save_alerts(alerts)

    generate_explanations(alerts)

    save_graph(graph)

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"Events: {len(events)} | "
        f"Alerts: {len(alerts)} | "
        f"Graph nodes: {graph.number_of_nodes()}"
    )


# ==========================================
# 8. MONITORING LOOP
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one update and exit."
    )

    args = parser.parse_args()

    print("\nNEMESIS-X REAL-TIME EVENT MONITOR")
    print("--------------------------------")

    try:

        if args.once:

            update_system()

            print("Final event update completed.")
            return

        print("Monitoring event_history.csv...")
        print("Update interval:", UPDATE_INTERVAL, "seconds")

        while True:

            try:

                update_system()

            except (OSError, csv.Error, ValueError) as error:

                print("Monitoring warning:", error)

            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:

        print("\nEvent monitor stopped.")


if __name__ == "__main__":
    main()
