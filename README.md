# BTRNDP-SA: Bus Transit Route Network Design via Simulated Annealing

A complete Python implementation of the Simulated Annealing (SA) framework for
the **Bus Transit Route Network Design Problem (BTRNDP)**, following
Fan & Machemehl (2006), *Using a Simulated Annealing Algorithm to Solve the
Transit Route Network Design Problem*, Journal of Transportation Engineering
132(2), 122–132.

The solver minimises a weighted sum of passenger user cost, operator cost, and
unsatisfied-demand cost, subject to constraints on headway, load factor, fleet
size, route length, number of routes, and maximum unsatisfied demand. A Genetic
Algorithm (GA) baseline is included for comparison.

EMU676 — Optimization Models and Algorithms in Transportation and Distribution
Hacettepe University · Fatih Alper Vural (N25123161)

---

## Method overview

The framework has three integrated components:

1. **ICRSGP** — Initial Candidate Route Set Generation Procedure. Uses
   Dijkstra's shortest path + Yen's *k*-shortest paths to enumerate all
   length-feasible candidate routes once, before optimisation.
2. **NAP** — Network Analysis Procedure. Performs a lexicographic transit
   assignment (direct service, then one-transfer, else unsatisfied) and sets
   each route's headway endogenously from the critical-link flow, iterated to
   convergence. Returns the objective value and feasibility.
3. **SA loop** — Searches the candidate pool. A solution is a *set of route
   indices*; the neighbourhood operator swaps one route. Constraints on fleet,
   load factor, and unsatisfied demand are enforced with a fixed-plus-marginal
   penalty so that every feasible solution dominates every infeasible one.

---

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/FAVURAL/676.git
cd 676
pip install -r requirements.txt
```

`numpy` and `scipy` are needed for the experiment/statistics modes;
`matplotlib` is needed only if you regenerate the figures. The `demo` mode runs
with the standard library alone.

---

## Usage

### 1. Generate the benchmark instance (deterministic — no random seed)

```bash
python3 generate_instance.py
```

Writes `data/nodes.csv`, `data/links.csv`, `data/demand.csv`.

### 2. Run a single SA solve (quick demo)

```bash
python3 btrndp_sa.py --mode demo
```

### 3. Parameter tuning (4×4 grid over T0 and alpha)

```bash
python3 btrndp_sa.py --mode tune
```

Writes `results/tuning.csv`.

### 4. Full experiment (20 runs each, resumable phases)

```bash
python3 btrndp_sa.py --mode sa  --runs 20   # SA batch  -> results/sa_runs.csv, convergence.csv, sa_best_sol.csv
python3 btrndp_sa.py --mode ga  --runs 20   # GA batch  -> results/ga_runs.csv
python3 btrndp_sa.py --mode combine          # gaps + statistics -> results/runs.csv, summary.csv, stats.csv, best_solution.txt
```

`--mode experiment` runs SA + GA + combine in one call if your environment does
not reap long-running jobs.

---

## Sample data (`data/`)

A fully-specified 15-node benchmark of Mandl scale, generated deterministically
so that all results are exactly reproducible.

| File | Contents |
|------|----------|
| `nodes.csv` | 15 nodes: `node_id, x_km, y_km, weight` |
| `links.csv` | 23 undirected links: `u, v, length_km, time_min` (2 min/km ≈ 30 km/h) |
| `demand.csv` | 15×15 O–D demand matrix (gravity model); 102 O–D pairs with positive demand, total 7172 pass./h |

Operational parameters (defaults in the `Params` dataclass): `R_max=8`,
`U_max=0.15`, route length `[3, 30]` km, headway `[5, 30]` min, `L_max=1.0`,
bus capacity `P=90`, fleet `W=58`, weights `(C1,C2,C3)=(0.4,0.4,0.2)`.

---

## Results (`results/`)

Produced by the experiment modes:

- `sa_runs.csv`, `ga_runs.csv` — per-run objective and CPU time
- `runs.csv` — merged per-run objectives with gaps to the best-known solution
- `summary.csv` — aggregate mean / std / min / max / gap / CPU per method
- `stats.csv` — Welch *t*, Mann–Whitney *U*, Wilcoxon signed-rank tests
- `convergence.csv`, `convergence.pdf/png` — incumbent trajectory (best run)
- `sa_vs_ga.pdf/png` — objective distribution boxplot
- `best_solution.txt` — best feasible network (routes, headways, load factors)
- `tuning.csv` — parameter-tuning grid

### Headline result

| Method | Mean z | Std | Min z | Mean gap | Mean CPU |
|--------|-------:|----:|------:|---------:|---------:|
| SA | 48841.7 | 2405.2 | 45361.1 | 7.67% | 3.37 s |
| GA | 52940.7 | 2996.2 | 45855.7 | 16.71% | 2.19 s |

SA significantly outperforms GA (Welch *t* = −4.65, *p* < 10⁻⁴; Mann–Whitney and
Wilcoxon both *p* < 10⁻³), consistent with Fan & Machemehl (2006).

---

## Notes on modelling choices

- The 15-node instance is **self-generated** at Mandl scale (the published
  network's exact data is not reproduced verbatim); it ships with the code so
  every number in the report is reproducible.
- The optimality **gap** is measured against the **best objective found** across
  all runs, as no published optimum exists for this instance.
- Bus capacity `P=90` and the listed operational parameters are explicit
  modelling choices documented here and in the report.

---

## File structure

```
676/
├── btrndp_sa.py          # solver: ICRSGP, NAP, SA, GA, experiments, statistics
├── generate_instance.py  # deterministic 15-node instance generator
├── requirements.txt
├── README.md
├── data/                 # CSV inputs (generated)
└── results/              # CSV outputs + figures (generated)
```

## Acknowledgements
"In making of this project AI tools are used to convert cluttered report parts into a given latex format as well as cleaning and engineering the python code."

## License

MIT
