"""
btrndp_sa.py
============
Bus Transit Route Network Design Problem (BTRNDP) solved with a
Simulated Annealing (SA) metaheuristic, following the framework of

    Fan, W. & Machemehl, R. B. (2006). "Using a Simulated Annealing Algorithm
    to Solve the Transit Route Network Design Problem."
    Journal of Transportation Engineering, 132(2), 122-132.

Three integrated components are implemented:
    1. ICRSGP - Initial Candidate Route Set Generation Procedure
               (Dijkstra shortest paths + Yen's k-shortest paths, filtered).
    2. NAP    - Network Analysis Procedure (lexicographic transit assignment
               with frequency / headway determination by load factor).
    3. SA     - Simulated Annealing optimisation loop over the candidate pool,
               plus a Genetic Algorithm (GA) baseline for comparison.

Standard library + numpy/scipy only. Run `python btrndp_sa.py --help`.

Author : Fatih Alper Vural  (N25123161)
Course : EMU676 - Optimization Models and Algorithms in Transportation
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ==========================================================================
# 1. PROBLEM DATA
# ==========================================================================

@dataclass
class Params:
    """Problem and algorithm parameters (Section 3.2 / 4.5 of the report)."""
    R_max: int = 8          # max number of routes
    U_max: float = 0.15     # max fraction of unsatisfied demand
    D_min: float = 3.0      # min route length (km)
    D_max: float = 30.0     # max route length (km)
    h_min: int = 5          # min headway (min)
    h_max: int = 30         # max headway (min)
    L_max: float = 1.0      # max load factor
    P: int = 90             # bus seating capacity (pass/veh)
    W: int = 58             # fleet size (veh)
    Cv: float = 50.0        # operating cost ($/veh/h)
    Cm: float = 0.20        # value of time ($/min)
    Ov: float = 1.0         # operating-hours scaling
    Cd: float = 2.0         # unit unsatisfied-demand cost ($/person)
    C1: float = 0.4         # weight: user cost
    C2: float = 0.4         # weight: operator cost
    C3: float = 0.2         # weight: unsatisfied-demand cost
    transfer_penalty: float = 5.0   # equivalent minutes per transfer
    k_paths: int = 6        # k for Yen's k-shortest paths


class Instance:
    """Loads a network instance from the data/ CSV files."""

    def __init__(self, data_dir: str):
        self.n, self.coords, self.weight = self._read_nodes(
            os.path.join(data_dir, "nodes.csv"))
        self.adj, self.link_len, self.link_time = self._read_links(
            os.path.join(data_dir, "links.csv"))
        self.demand = self._read_demand(
            os.path.join(data_dir, "demand.csv"))
        self.total_demand = sum(self.demand[i][j]
                                for i in range(self.n)
                                for j in range(self.n)) / 2.0

    @staticmethod
    def _read_nodes(path):
        coords, weight = {}, {}
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                i = int(row["node_id"]) - 1
                coords[i] = (float(row["x_km"]), float(row["y_km"]))
                weight[i] = float(row["weight"])
        return len(coords), coords, weight

    def _read_links(self, path):
        adj: Dict[int, List[int]] = {}
        link_len: Dict[Tuple[int, int], float] = {}
        link_time: Dict[Tuple[int, int], float] = {}
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                u, v = int(row["u"]) - 1, int(row["v"]) - 1
                L, t = float(row["length_km"]), float(row["time_min"])
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
                link_len[(u, v)] = link_len[(v, u)] = L
                link_time[(u, v)] = link_time[(v, u)] = t
        return adj, link_len, link_time

    def _read_demand(self, path):
        D = []
        with open(path) as f:
            r = csv.reader(f)
            next(r)  # header
            for row in r:
                D.append([float(x) for x in row[1:]])
        return D


# ==========================================================================
# 2. ICRSGP  - candidate route generation
# ==========================================================================

def dijkstra(inst: Instance, src: int) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Label-setting shortest paths by travel time from src to all nodes."""
    dist = {src: 0.0}
    prev: Dict[int, int] = {}
    pq = [(0.0, src)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v in inst.adj.get(u, []):
            nd = d + inst.link_time[(u, v)]
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def _path_from_prev(prev, src, dst):
    if dst != src and dst not in prev:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def shortest_path_constrained(inst, src, dst, removed_edges, removed_nodes):
    """Dijkstra honouring removed edges/nodes (used inside Yen's algorithm)."""
    dist = {src: 0.0}
    prev: Dict[int, int] = {}
    pq = [(0.0, src)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v in inst.adj.get(u, []):
            if v in removed_nodes:
                continue
            if (u, v) in removed_edges:
                continue
            nd = d + inst.link_time[(u, v)]
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    p = _path_from_prev(prev, src, dst)
    if p is None:
        return None, math.inf
    return p, dist[dst]


def yen_k_shortest(inst, src, dst, K):
    """Yen's algorithm: K loopless shortest paths between src and dst."""
    p0, c0 = shortest_path_constrained(inst, src, dst, set(), set())
    if p0 is None:
        return []
    A = [(c0, p0)]
    B: List[Tuple[float, List[int]]] = []
    for k in range(1, K):
        prev_path = A[k - 1][1]
        for i in range(len(prev_path) - 1):
            spur = prev_path[i]
            root = prev_path[:i + 1]
            removed_edges = set()
            for (cost, p) in A:
                if len(p) > i and p[:i + 1] == root:
                    removed_edges.add((p[i], p[i + 1]))
                    removed_edges.add((p[i + 1], p[i]))
            removed_nodes = set(root[:-1])
            spur_path, spur_cost = shortest_path_constrained(
                inst, spur, dst, removed_edges, removed_nodes)
            if spur_path is not None:
                total = root[:-1] + spur_path
                # recompute true cost
                cost = sum(inst.link_time[(total[t], total[t + 1])]
                           for t in range(len(total) - 1))
                cand = (cost, total)
                if cand not in B and cand not in A:
                    B.append(cand)
        if not B:
            break
        B.sort(key=lambda x: x[0])
        A.append(B.pop(0))
    return A


def route_length(inst, path):
    return sum(inst.link_len[(path[t], path[t + 1])] for t in range(len(path) - 1))


def route_time(inst, path):
    return sum(inst.link_time[(path[t], path[t + 1])] for t in range(len(path) - 1))


def icrsgp(inst: Instance, prm: Params) -> List[List[int]]:
    """Build the candidate route pool (Algorithm 2 in the report)."""
    pool = {}

    def canon(path):
        # canonical key so a route and its reverse are identical
        return tuple(path) if path[0] <= path[-1] else tuple(reversed(path))

    def consider(path):
        if len(path) < 2:
            return
        L = route_length(inst, path)
        if prm.D_min <= L <= prm.D_max:
            pool[canon(path)] = list(path)

    for i in range(inst.n):
        for j in range(i + 1, inst.n):
            if inst.demand[i][j] <= 0:
                continue
            for cost, path in yen_k_shortest(inst, i, j, prm.k_paths):
                consider(path)
    return list(pool.values())


# ==========================================================================
# 3. NAP  - Network Analysis Procedure (assignment + frequencies)
# ==========================================================================

class RouteInfo:
    """Pre-computed per-route structures for fast assignment."""
    __slots__ = ("nodes", "pos", "seg_time", "cum_time", "length", "headway")

    def __init__(self, inst, nodes):
        self.nodes = nodes
        self.pos = {nd: k for k, nd in enumerate(nodes)}
        self.seg_time = [inst.link_time[(nodes[t], nodes[t + 1])]
                         for t in range(len(nodes) - 1)]
        self.cum_time = [0.0]
        for s in self.seg_time:
            self.cum_time.append(self.cum_time[-1] + s)
        self.length = route_length(inst, nodes)
        self.headway = float('nan')

    def in_vehicle_time(self, a, b):
        ia, ib = self.pos[a], self.pos[b]
        return abs(self.cum_time[ia] - self.cum_time[ib])

    def covers(self, a, b):
        return a in self.pos and b in self.pos


@dataclass
class NAPResult:
    z: float
    user_cost: float
    operator_cost: float
    unsatisfied_cost: float
    unsatisfied_frac: float
    fleet_used: float
    headways: List[float]
    load_factors: List[float]
    feasible: bool


def network_analysis(inst: Instance, prm: Params, route_set: List[List[int]],
                     freq_iters: int = 3) -> NAPResult:
    """Assign demand (0-transfer then 1-transfer), iterate headways to satisfy
    the load-factor constraint, and return the objective decomposition."""
    routes = [RouteInfo(inst, r) for r in route_set]
    M = len(routes)
    for r in routes:
        r.headway = prm.h_max  # initialise at maximum headway

    # transfer-node index: node -> list of route indices covering it
    node_routes: Dict[int, List[int]] = {}
    for ri, r in enumerate(routes):
        for nd in r.nodes:
            node_routes.setdefault(nd, []).append(ri)

    def assign():
        """One transit-assignment pass at current headways.
        Lexicographic priority: direct (0-transfer) then 1-transfer.
        Returns (flow_per_route, user_cost, served_demand)."""
        flow = [[0.0] * len(r.seg_time) for r in routes]
        served = 0.0
        user_cost = 0.0
        for i in range(inst.n):
            Di = inst.demand[i]
            for j in range(inst.n):
                if i == j:
                    continue
                d = Di[j]
                if d <= 0:
                    continue
                # priority 1: direct (0-transfer)
                direct = [(ri, routes[ri].headway / 2.0 + routes[ri].in_vehicle_time(i, j))
                          for ri in range(M) if i in routes[ri].pos and j in routes[ri].pos]
                if direct:
                    inv = sum(1.0 / t for _, t in direct)
                    for ri, t in direct:
                        share = d * (1.0 / t) / inv
                        _add_flow(routes[ri], flow[ri], i, j, share)
                        user_cost += share * t
                    served += d
                    continue
                # priority 2: one transfer
                best = None
                for x, ris in node_routes.items():
                    if x == i or x == j:
                        continue
                    r1s = [ri for ri in ris if i in routes[ri].pos]
                    if not r1s:
                        continue
                    r2s = [ri for ri in ris if j in routes[ri].pos]
                    if not r2s:
                        continue
                    for r1 in r1s:
                        ht1 = routes[r1].headway / 2.0 + routes[r1].in_vehicle_time(i, x)
                        for r2 in r2s:
                            if r1 == r2:
                                continue
                            t = (ht1 + prm.transfer_penalty
                                 + routes[r2].headway / 2.0
                                 + routes[r2].in_vehicle_time(x, j))
                            if best is None or t < best[0]:
                                best = (t, r1, r2, x)
                if best is not None:
                    t, r1, r2, x = best
                    _add_flow(routes[r1], flow[r1], i, x, d)
                    _add_flow(routes[r2], flow[r2], x, j, d)
                    user_cost += d * t
                    served += d
        return flow, user_cost, served

    # iterate: assign -> update headways from critical-link flow & load factor
    flow, user_cost, served = assign()
    for _ in range(freq_iters - 1):
        for ri, r in enumerate(routes):
            qmax = max(flow[ri]) if flow[ri] else 0.0
            if qmax <= 1e-9:
                r.headway = prm.h_max
            else:
                # pass/bus = qmax[pass/h]*h[min]/60 ; L = pass/bus/P <= L_max
                #   ->  h <= 60*L_max*P/qmax
                h = 60.0 * prm.L_max * prm.P / qmax
                r.headway = max(prm.h_min, min(prm.h_max, h))
        flow, user_cost, served = assign()

    # operator cost & fleet
    fleet_used = 0.0
    load_factors = []
    headways = []
    for ri, r in enumerate(routes):
        round_trip = 2.0 * sum(r.seg_time)
        buses = round_trip / r.headway
        fleet_used += buses
        qmax = max(flow[ri]) if flow[ri] else 0.0
        load_factors.append(qmax * r.headway / (60.0 * prm.P))
        headways.append(r.headway)
    operator_term = (prm.Cv / prm.Cm) * prm.Ov * sum(
        2.0 * sum(r.seg_time) / r.headway for r in routes)

    unsatisfied = inst.total_demand * 2.0 - served  # demand stored both ways
    # demand matrix is symmetric & both directions assigned above -> use full sum
    full_demand = sum(inst.demand[i][j] for i in range(inst.n) for j in range(inst.n))
    unsatisfied = full_demand - served
    unsat_frac = unsatisfied / full_demand if full_demand > 0 else 0.0
    unsatisfied_term = (prm.Cd / prm.Cm) * unsatisfied

    z = prm.C1 * user_cost + prm.C2 * operator_term + prm.C3 * unsatisfied_term

    # ---- constraints as penalties (soft) ----
    feasible = True
    penalty = 0.0
    if fleet_used > prm.W:
        feasible = False
        penalty += 5000.0 + 800.0 * (fleet_used - prm.W)
    if unsat_frac > prm.U_max:
        feasible = False
        penalty += 5000.0 + 80.0 * (unsat_frac - prm.U_max) * full_demand
    for lf in load_factors:
        if lf > prm.L_max + 1e-6:
            feasible = False
            penalty += 5000.0 + 20000.0 * (lf - prm.L_max)

    return NAPResult(z=z + penalty, user_cost=user_cost,
                     operator_cost=operator_term,
                     unsatisfied_cost=unsatisfied_term,
                     unsatisfied_frac=unsat_frac, fleet_used=fleet_used,
                     headways=headways, load_factors=load_factors,
                     feasible=feasible)


def _add_flow(route: RouteInfo, flow_vec, a, b, amount):
    ia, ib = route.pos[a], route.pos[b]
    lo, hi = (ia, ib) if ia < ib else (ib, ia)
    for s in range(lo, hi):
        flow_vec[s] += amount


# ==========================================================================
# 4. SIMULATED ANNEALING
# ==========================================================================

def random_solution(pool_size, n, rng):
    return rng.sample(range(pool_size), n)


def neighbour(sol, pool_size, rng):
    s = list(sol)
    out_idx = rng.randrange(len(s))
    in_set = set(s)
    choices = [r for r in range(pool_size) if r not in in_set]
    if not choices:
        return s
    s[out_idx] = rng.choice(choices)
    return s


def evaluate(inst, prm, pool, sol):
    rs = [pool[i] for i in sol]
    return network_analysis(inst, prm, rs).z


def simulated_annealing(inst, prm, pool, T0=2000.0, alpha=0.6,
                        G_max=20, K=10, seed=0, trace=False):
    """SA framework (Algorithm 1). Outer loop grows the route-set size n;
    the inner SA searches fixed-size subsets of the candidate pool."""
    rng = random.Random(seed)
    best_sol, best_z = None, math.inf
    conv = []  # convergence trace (incumbent best z per accepted improvement)

    for n in range(1, prm.R_max + 1):
        if n > len(pool):
            break
        S = random_solution(len(pool), n, rng)
        zS = evaluate(inst, prm, pool, S)
        T = T0
        rep = 0
        gen = 0
        while gen < G_max:
            Sp = neighbour(S, len(pool), rng)
            zSp = evaluate(inst, prm, pool, Sp)
            dz = zSp - zS
            if dz < 0:
                S, zS = Sp, zSp
                if zS < best_z:
                    best_z, best_sol = zS, list(S)
            elif rng.random() < math.exp(-dz / max(T, 1e-9)):
                S, zS = Sp, zSp
            rep += 1
            if rep >= K:
                T *= alpha
                rep = 0
                gen += 1
                if trace:
                    conv.append(best_z)
    return best_sol, best_z, conv


# ==========================================================================
# 5. GENETIC ALGORITHM (baseline for comparison)
# ==========================================================================

def genetic_algorithm(inst, prm, pool, pop_size=30, generations=60,
                       mut_rate=0.2, seed=0):
    """Simple GA baseline. Chromosome = set of route indices (size <= R_max).
    Selection: tournament. Crossover: union-then-trim. Mutation: route swap."""
    rng = random.Random(seed)
    psz = len(pool)

    def rand_indiv():
        n = rng.randint(2, prm.R_max)
        return rng.sample(range(psz), min(n, psz))

    def fitness(ind):
        return evaluate(inst, prm, pool, ind)

    pop = [rand_indiv() for _ in range(pop_size)]
    fit = [fitness(p) for p in pop]
    best_z = min(fit)
    best_sol = list(pop[fit.index(best_z)])

    def tournament():
        a, b = rng.randrange(pop_size), rng.randrange(pop_size)
        return pop[a] if fit[a] < fit[b] else pop[b]

    for _ in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = tournament(), tournament()
            union = list(set(p1) | set(p2))
            rng.shuffle(union)
            size = min(rng.randint(2, prm.R_max), len(union))
            child = union[:size]
            if rng.random() < mut_rate:  # mutation: swap one route
                out = rng.randrange(len(child))
                pool_choices = [r for r in range(psz) if r not in set(child)]
                if pool_choices:
                    child[out] = rng.choice(pool_choices)
            new_pop.append(child)
        pop = new_pop
        fit = [fitness(p) for p in pop]
        gen_best = min(fit)
        if gen_best < best_z:
            best_z = gen_best
            best_sol = list(pop[fit.index(gen_best)])
    return best_sol, best_z


# ==========================================================================
# 6. EXPERIMENT DRIVERS
# ==========================================================================

def describe_solution(inst, prm, pool, sol):
    rs = [pool[i] for i in sol]
    res = network_analysis(inst, prm, rs)
    lines = []
    lines.append(f"  Routes ({len(rs)}):")
    for k, (idx, r) in enumerate(zip(sol, rs), start=1):
        disp = "-".join(str(x + 1) for x in r)
        lines.append(f"    R{k}: {disp}  "
                     f"(len={route_length(inst, r):.1f} km, "
                     f"h={res.headways[k-1]:.1f} min, "
                     f"L={res.load_factors[k-1]:.2f})")
    lines.append(f"  Objective z          : {res.z:10.1f}")
    lines.append(f"    user cost          : {res.user_cost:10.1f}")
    lines.append(f"    operator cost      : {res.operator_cost:10.1f}")
    lines.append(f"    unsatisfied cost   : {res.unsatisfied_cost:10.1f}")
    lines.append(f"  Unsatisfied demand   : {100*res.unsatisfied_frac:6.2f} %")
    lines.append(f"  Fleet used / avail.  : {res.fleet_used:.1f} / {prm.W}")
    lines.append(f"  Feasible             : {res.feasible}")
    return "\n".join(lines), res


def main():
    ap = argparse.ArgumentParser(description="BTRNDP via Simulated Annealing")
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--mode", choices=["demo", "experiment", "tune",
                                       "sa", "ga", "combine"],
                    default="demo")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prm = Params()
    inst = Instance(args.data)
    print(f"Instance: {inst.n} nodes, total demand "
          f"{sum(inst.demand[i][j] for i in range(inst.n) for j in range(inst.n)):.0f} pass/h")

    t0 = time.time()
    pool = icrsgp(inst, prm)
    print(f"ICRSGP: {len(pool)} candidate routes generated "
          f"in {time.time()-t0:.2f}s\n")

    if args.mode == "demo":
        t0 = time.time()
        sol, z, conv = simulated_annealing(inst, prm, pool, seed=args.seed,
                                           trace=True)
        cpu = time.time() - t0
        txt, _ = describe_solution(inst, prm, pool, sol)
        print("=== Best solution found by SA ===")
        print(txt)
        print(f"  CPU time             : {cpu:.2f} s")
        print(f"\nConvergence (first 15 logged points): "
              f"{[round(c,1) for c in conv[:15]]}")

    elif args.mode == "tune":
        run_parameter_tuning(inst, prm, pool)

    elif args.mode == "experiment":
        run_full_experiment(inst, prm, pool, runs=args.runs)

    elif args.mode == "sa":
        _run_sa_batch(inst, prm, pool, args.runs)
    elif args.mode == "ga":
        _run_ga_batch(inst, prm, pool, args.runs)
    elif args.mode == "combine":
        _combine_results(inst, prm, pool)


# ---- parameter tuning ----------------------------------------------------

def run_parameter_tuning(inst, prm, pool, n_runs=3):
    import numpy as np
    os.makedirs("results", exist_ok=True)
    T0_vals = [500, 1000, 2000, 5000]
    alpha_vals = [0.5, 0.6, 0.7, 0.9]
    print(f"Parameter tuning (mean z, std) over {n_runs} runs per cell")
    print("(tuning uses a shorter SA: G_max=15, K=8)\n")
    results = {}
    header = "T0\\alpha |" + "".join(f"{a:>14}" for a in alpha_vals)
    print(header); print("-" * len(header))
    best_combo, best_mean = None, math.inf
    rows = []
    for T0 in T0_vals:
        row = f"{T0:>7} |"
        for alpha in alpha_vals:
            zs = []
            for s in range(n_runs):
                _, z, _ = simulated_annealing(inst, prm, pool, T0=T0,
                                              alpha=alpha, G_max=15, K=8, seed=s)
                zs.append(z)
            m, sd = float(np.mean(zs)), float(np.std(zs))
            results[(T0, alpha)] = (m, sd)
            rows.append((T0, alpha, m, sd))
            row += f"  ({m:6.0f},{sd:4.0f})"
            if m < best_mean:
                best_mean, best_combo = m, (T0, alpha)
        print(row)
    print(f"\nBest combination: T0={best_combo[0]}, alpha={best_combo[1]} "
          f"(mean z = {best_mean:.1f})")
    with open("results/tuning.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["T0", "alpha", "mean_z", "std_z"])
        for r in rows: w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}"])
    return results, best_combo


# ---- full experiment + statistics ---------------------------------------

def _run_sa_batch(inst, prm, pool, runs, out="results"):
    import numpy as np
    os.makedirs(out, exist_ok=True)
    rows = []
    best_z, best_sol, conv_trace = math.inf, None, None
    for s in range(runs):
        t0 = time.time()
        sol, z, conv = simulated_annealing(inst, prm, pool, T0=2000, alpha=0.6,
                                            G_max=20, K=10, seed=s, trace=True)
        cpu = time.time() - t0
        rows.append((s, z, cpu))
        if z < best_z:
            best_z, best_sol, conv_trace = z, sol, conv
        print(f"  SA run {s:2d}: z={z:9.1f}  cpu={cpu:4.2f}s")
    with open(os.path.join(out, "sa_runs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "z", "cpu_s"])
        for r in rows: w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}"])
    with open(os.path.join(out, "convergence.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["logged_step", "incumbent_best_z"])
        for k, c in enumerate(conv_trace): w.writerow([k, f"{c:.4f}"])
    with open(os.path.join(out, "sa_best_sol.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["route_index"]); 
        for ri in best_sol: w.writerow([ri])
    print(f"\nSA: mean z={np.mean([r[1] for r in rows]):.1f}, best z={best_z:.1f}")
    return rows, best_sol


def _run_ga_batch(inst, prm, pool, runs, out="results"):
    import numpy as np
    os.makedirs(out, exist_ok=True)
    rows = []
    for s in range(runs):
        t0 = time.time()
        _, zg = genetic_algorithm(inst, prm, pool, pop_size=24,
                                  generations=40, seed=s)
        cpu = time.time() - t0
        rows.append((s, zg, cpu))
        print(f"  GA run {s:2d}: z={zg:9.1f}  cpu={cpu:4.2f}s")
    with open(os.path.join(out, "ga_runs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "z", "cpu_s"])
        for r in rows: w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}"])
    print(f"\nGA: mean z={np.mean([r[1] for r in rows]):.1f}, "
          f"best z={min(r[1] for r in rows):.1f}")
    return rows


def _combine_results(inst, prm, pool, out="results"):
    import numpy as np
    from scipy import stats

    def read(name):
        rows = []
        with open(os.path.join(out, name)) as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append((int(row["run"]), float(row["z"]), float(row["cpu_s"])))
        return rows

    sa = read("sa_runs.csv"); ga = read("ga_runs.csv")
    sa_z = [r[1] for r in sa]; sa_cpu = [r[2] for r in sa]
    ga_z = [r[1] for r in ga]; ga_cpu = [r[2] for r in ga]
    bks = min(min(sa_z), min(ga_z))
    sa_gap = [100 * (z - bks) / bks for z in sa_z]
    ga_gap = [100 * (z - bks) / bks for z in ga_z]

    def line(name, zs, cpus, gaps):
        return (f"{name:>4} | mean={np.mean(zs):8.1f}  std={np.std(zs):6.1f}  "
                f"min={np.min(zs):8.1f}  max={np.max(zs):8.1f}  "
                f"gap={np.mean(gaps):5.2f}%  cpu={np.mean(cpus):5.2f}s")

    print("=== Aggregate results ===")
    print(line("SA", sa_z, sa_cpu, sa_gap))
    print(line("GA", ga_z, ga_cpu, ga_gap))
    print(f"Best-known (best found): z = {bks:.1f}")

    t_stat, t_p = stats.ttest_ind(sa_z, ga_z, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(sa_z, ga_z, alternative="two-sided")
    w_stat, w_p = stats.wilcoxon(sa_z, ga_z) if len(sa_z) == len(ga_z) else (float('nan'), float('nan'))
    print("\n=== SA vs GA statistical tests ===")
    print(f"Welch t-test   : t={t_stat:7.3f}  p={t_p:.4g}")
    print(f"Mann-Whitney U : U={u_stat:7.1f}  p={u_p:.4g}")
    print(f"Wilcoxon       : W={w_stat:7.1f}  p={w_p:.4g}")

    # write combined CSVs
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "SA_z", "SA_cpu_s", "SA_gap_pct",
                    "GA_z", "GA_cpu_s", "GA_gap_pct"])
        for i in range(min(len(sa), len(ga))):
            w.writerow([i, f"{sa_z[i]:.2f}", f"{sa_cpu[i]:.3f}", f"{sa_gap[i]:.3f}",
                        f"{ga_z[i]:.2f}", f"{ga_cpu[i]:.3f}", f"{ga_gap[i]:.3f}"])
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_z", "std_z", "min_z", "max_z",
                    "mean_gap_pct", "mean_cpu_s"])
        w.writerow(["SA", f"{np.mean(sa_z):.2f}", f"{np.std(sa_z):.2f}",
                    f"{np.min(sa_z):.2f}", f"{np.max(sa_z):.2f}",
                    f"{np.mean(sa_gap):.3f}", f"{np.mean(sa_cpu):.3f}"])
        w.writerow(["GA", f"{np.mean(ga_z):.2f}", f"{np.std(ga_z):.2f}",
                    f"{np.min(ga_z):.2f}", f"{np.max(ga_z):.2f}",
                    f"{np.mean(ga_gap):.3f}", f"{np.mean(ga_cpu):.3f}"])
    with open(os.path.join(out, "stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "statistic", "p_value"])
        w.writerow(["welch_t", f"{t_stat:.4f}", f"{t_p:.6f}"])
        w.writerow(["mann_whitney_u", f"{u_stat:.4f}", f"{u_p:.6f}"])
        w.writerow(["wilcoxon", f"{w_stat:.4f}", f"{w_p:.6f}"])

    # best solution (from SA best)
    with open(os.path.join(out, "sa_best_sol.csv")) as f:
        r = csv.DictReader(f)
        best_sol = [int(row["route_index"]) for row in r]
    txt, res = describe_solution(inst, prm, pool, best_sol)
    print("\n=== Best network configuration (SA) ===")
    print(txt)
    with open(os.path.join(out, "best_solution.txt"), "w") as f:
        f.write(txt + "\n")
    print("\nAll result files written to results/")


def run_full_experiment(inst, prm, pool, runs=20):
    _run_sa_batch(inst, prm, pool, runs)
    _run_ga_batch(inst, prm, pool, runs)
    _combine_results(inst, prm, pool)


if __name__ == "__main__":
    main()
