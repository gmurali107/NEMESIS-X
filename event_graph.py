
import csv
import os
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. INITIALIZE EVENT GRAPH
# -----------------------------------------

CSV_FILE = "event_history.csv"

G = nx.DiGraph()

if not os.path.exists(CSV_FILE):
    print("Error: event_history.csv not found!")
    print("Run the Phase 4 program first.")
    exit()

# Store events by tracking ID
track_events = {}

# -----------------------------------------
# 2. READ EVENT HISTORY
# -----------------------------------------

with open(CSV_FILE, "r", newline="") as file:

    reader = csv.DictReader(file)

    for index, row in enumerate(reader):

        track_id = row["Track_ID"]
        object_name = row["Object"]
        event_type = row["Event"]
        timestamp = row["Timestamp"]
        details = row["Details"]

        track_node = f"Track_{track_id}"
        event_node = f"Event_{index}"

        # Create entity node
        if not G.has_node(track_node):

            G.add_node(
                track_node,
                node_type="Entity",
                label=f"{object_name.upper()} {track_id}"
            )

        # Create event node
        G.add_node(
            event_node,
            node_type="Event",
            label=event_type,
            timestamp=timestamp,
            details=details
        )

        # Connect entity to event
        G.add_edge(
            track_node,
            event_node,
            relation="PARTICIPATED_IN"
        )

        # Create timestamp node
        time_node = f"Time_{index}"

        G.add_node(
            time_node,
            node_type="Time",
            label=timestamp
        )

        G.add_edge(
            event_node,
            time_node,
            relation="OCCURRED_AT"
        )

        # Connect zone events to monitored zone
        if event_type in ["ENTERED ZONE", "EXITED ZONE"]:

            zone_node = "Zone_A"

            G.add_node(
                zone_node,
                node_type="Location",
                label="MONITORED ZONE"
            )

            relation = (
                "ENTERED"
                if event_type == "ENTERED ZONE"
                else "EXITED"
            )

            G.add_edge(
                event_node,
                zone_node,
                relation=relation
            )

        # Remember event sequence
        track_events.setdefault(track_id, []).append({
            "node": event_node,
            "timestamp": timestamp,
            "index": index
        })

# -----------------------------------------
# 3. CONNECT TEMPORAL EVENT SEQUENCES
# -----------------------------------------

for track_id, events in track_events.items():

    events.sort(
        key=lambda item: (item["timestamp"], item["index"])
    )

    for i in range(len(events) - 1):

        current_event = events[i]["node"]
        next_event = events[i + 1]["node"]

        G.add_edge(
            current_event,
            next_event,
            relation="NEXT_EVENT"
        )

# -----------------------------------------
# 4. DISPLAY GRAPH INFORMATION
# -----------------------------------------

print("\nNEMESIS-X EVENT GRAPH")
print("----------------------")

print("Total nodes:", G.number_of_nodes())
print("Total relationships:", G.number_of_edges())

print("\nNode information:")

for node, data in G.nodes(data=True):

    print(
        node,
        "|",
        data.get("node_type"),
        "|",
        data.get("label")
    )

print("\nRelationship information:")

for source, target, data in G.edges(data=True):

    print(
        source,
        "--",
        data["relation"],
        "-->",
        target
    )

# -----------------------------------------
# 5. VISUALIZE EVENT GRAPH
# -----------------------------------------

node_colors = []

for node, data in G.nodes(data=True):

    node_type = data.get("node_type")

    if node_type == "Entity":
        node_colors.append("skyblue")

    elif node_type == "Event":
        node_colors.append("lightgreen")

    elif node_type == "Time":
        node_colors.append("lightyellow")

    elif node_type == "Location":
        node_colors.append("orange")

    else:
        node_colors.append("gray")

labels = {
    node: data.get("label", node)
    for node, data in G.nodes(data=True)
}

plt.figure(figsize=(16, 10))

positions = nx.spring_layout(
    G,
    seed=42,
    k=1.5
)

nx.draw_networkx_nodes(
    G,
    positions,
    node_color=node_colors,
    node_size=1800
)

nx.draw_networkx_labels(
    G,
    positions,
    labels=labels,
    font_size=7
)

nx.draw_networkx_edges(
    G,
    positions,
    arrows=True,
    arrowstyle="->",
    arrowsize=15,
    connectionstyle="arc3,rad=0.08"
)

edge_labels = nx.get_edge_attributes(
    G,
    "relation"
)

nx.draw_networkx_edge_labels(
    G,
    positions,
    edge_labels=edge_labels,
    font_size=6
)

plt.title("NEMESIS-X | Unknown Event Graph")

plt.axis("off")
plt.tight_layout()

# Save graph image
plt.savefig(
    "event_graph.png",
    dpi=200,
    bbox_inches="tight"
)

print("\nGraph image saved: event_graph.png")

# -----------------------------------------
# 6. EXPORT GRAPH DATA
# -----------------------------------------

nx.write_gexf(
    G,
    "event_graph.gexf"
)

print("Graph data saved: event_graph.gexf")

plt.show()

print("\nNEMESIS-X Event Graph completed!")
