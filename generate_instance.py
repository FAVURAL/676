"""
generate_instance.py
=====================
Generates a fully reproducible 15-node transit benchmark instance for the
Bus Transit Route Network Design Problem (BTRNDP).

The instance is *deterministic* (no random seed needed): node coordinates,
link travel times, and the origin-destination (O-D) demand matrix are all
produced by closed-form rules, so re-running this script always yields the
identical data files. It is built at the same scale as Mandl's classic Swiss
benchmark (15 nodes, ~21 undirected links) so results are comparable in
spirit, while remaining a transparently self-defined instance.

Outputs (written to ./data/):
    nodes.csv   : node_id, x_km, y_km, weight
    links.csv   : u, v, length_km, time_min
    demand.csv  : 15x15 symmetric O-D demand matrix (passengers/hour)

Author : Fatih Alper Vural  (N25123161)
Course : EMU676 - Optimization Models and Algorithms in Transportation
"""

import csv
import math
import os

# --------------------------------------------------------------------------
# 1. Node layout (km). A compact, city-like planar layout with 4 high-weight
#    "hub" zones (1, 4, 8, 12 in 1-indexed terms) carrying more trip ends.
# --------------------------------------------------------------------------
# (x, y, demand_weight)
NODES = [
    (1.0, 6.0, 3),   # 1
    (3.0, 8.0, 1),   # 2
    (3.0, 4.0, 2),   # 3
    (5.0, 6.0, 4),   # 4  hub
    (5.0, 9.0, 1),   # 5
    (5.0, 3.0, 2),   # 6
    (7.0, 7.5, 1),   # 7
    (8.0, 5.0, 4),   # 8  hub
    (7.0, 2.5, 2),   # 9
    (9.0, 8.0, 1),   # 10
    (10.0, 3.5, 2),  # 11
    (11.0, 6.0, 3),  # 12 hub
    (12.5, 8.0, 1),  # 13
    (12.5, 4.0, 1),  # 14
    (13.5, 6.0, 2),  # 15
]

# Undirected links (1-indexed endpoints). Travel time = 2 min per km
# (i.e. 30 km/h cruising speed), rounded to the nearest minute.
LINK_PAIRS = [
    (1, 2), (1, 3), (2, 4), (2, 5), (3, 4), (3, 6),
    (4, 5), (4, 6), (4, 7), (6, 9), (7, 8), (5, 7),
    (8, 9), (8, 10), (8, 11), (7, 10), (9, 11), (10, 12),
    (11, 12), (12, 13), (12, 14), (13, 15), (14, 15),
]

SPEED_MIN_PER_KM = 2.0   # 30 km/h


def euclid(a, b):
    return math.hypot(NODES[a][0] - NODES[b][0], NODES[a][1] - NODES[b][1])


def build_links():
    rows = []
    for (u, v) in LINK_PAIRS:
        length = euclid(u - 1, v - 1)
        time = round(length * SPEED_MIN_PER_KM)
        rows.append((u, v, round(length, 2), int(time)))
    return rows


def build_demand():
    """Gravity model: d_ij = round( K * w_i * w_j / dist_ij ), symmetric,
    zero diagonal. Small flows (<5) are dropped to keep the matrix realistic."""
    n = len(NODES)
    K = 38.0
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = NODES[i][2], NODES[j][2]
            dist = euclid(i, j)
            val = K * wi * wj / max(dist, 1.0)
            val = int(round(val))
            if val < 5:
                val = 0
            D[i][j] = val
            D[j][i] = val
    return D


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    # nodes.csv
    with open(os.path.join(data_dir, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x_km", "y_km", "weight"])
        for idx, (x, y, wt) in enumerate(NODES, start=1):
            w.writerow([idx, x, y, wt])

    # links.csv
    links = build_links()
    with open(os.path.join(data_dir, "links.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "length_km", "time_min"])
        for row in links:
            w.writerow(row)

    # demand.csv
    D = build_demand()
    with open(os.path.join(data_dir, "demand.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = [""] + [str(i + 1) for i in range(len(NODES))]
        w.writerow(header)
        for i in range(len(NODES)):
            w.writerow([str(i + 1)] + D[i])

    total = sum(sum(r) for r in D) // 2
    print(f"Instance written to {data_dir}/")
    print(f"  nodes : {len(NODES)}")
    print(f"  links : {len(links)} undirected ({2*len(links)} directed)")
    print(f"  O-D pairs with demand: "
          f"{sum(1 for i in range(len(NODES)) for j in range(i+1, len(NODES)) if D[i][j] > 0)}")
    print(f"  total demand (one direction sum): {total} pass/h")


if __name__ == "__main__":
    main()
