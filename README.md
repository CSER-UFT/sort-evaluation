# Sorting Algorithms Benchmark with `perf`

A reproducible benchmarking pipeline for **twelve sorting algorithms** written in C
with **OpenMP** parallelization, measured with **Linux `perf`** hardware counters and
**RAPL** energy domains. It collects performance, microarchitecture and energy data,
parses it into CSV, and produces a full set of analysis figures.

---

## 1. What the experiment does

Each algorithm is compiled in two flavors (serial and parallel) and run many times
under tight reproducibility controls while `perf` records hardware events. The pipeline
then aggregates the runs and generates plots covering empirical complexity, parallel
scalability (speedup, efficiency, Amdahl serial fraction), microarchitectural behavior
(IPC, branch/cache misses), memory hierarchy pressure, run-to-run distributions, and
energy (package + DRAM, EDP/ED²P, energy–time Pareto).

**Reproducibility controls** (built into the runner):

- `:u` modifier on every PMU event — counts user space only, removing kernel-attributed
  noise so `instructions:u` becomes exactly constant across runs.
- `setarch -R` — disables ASLR, fixing the memory layout.
- `taskset` core pinning + OpenMP placement (`OMP_PROC_BIND`, `OMP_PLACES`).
- NMI watchdog disabled — frees one hardware counter the kernel would otherwise hold.
- Events split into **multiplexing-free passes** (≤ 4 general-purpose events each), so no
  counter is time-sliced and every count is exact, not estimated.

**The twelve algorithms** (two per family):

| Family       | Algorithms                          |
|--------------|-------------------------------------|
| Insertion    | `insertionsort`, `shellsort`        |
| Exchange     | `combsort`, `quicksort` (dual-pivot)|
| Selection    | `heapsort`, `smoothsort`            |
| Merge        | `mergesort` (top-down), `bottomupsort` |
| Distribution | `countingsort`, `radixsort` (LSD)   |
| Network      | `bitonicsort`, `oddevensort`        |

> Network sorts require a power-of-two input size; other sizes are padded, which roughly
> doubles the work just above a power of two. Prefer power-of-two `N` for those two.

---

## 2. Repository contents

```
*.c                 # algorithm sources (shared template: same main, distributions, flags)
run_perf.sh         # benchmark runner: 3 passes per run (P1, P2 counters; P3 energy)
parse_perf.py       # parses results/*.p1/.p2/.p3 -> CSV (raw + summary)
plot_perf.py        # per-metric plots (mean +- 95% CI, std, speedup)
analyze_perf.py     # analysis figures (complexity, scalability, microarch, memory, energy, ...)
README.md           # this file
```

---

## 3. Requirements

This pipeline is **Linux-only** (`perf` and RAPL are Linux features) and needs
**`sudo`** (perf accesses RAPL and toggles the NMI watchdog).

### System packages (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential \
                    linux-tools-common linux-tools-generic linux-tools-$(uname -r) \
                    util-linux python3 python3-pip
```

- `build-essential` provides `gcc` (with OpenMP via `-fopenmp`).
- `linux-tools-$(uname -r)` provides `perf`.
- `util-linux` provides `setarch` and `taskset` (usually preinstalled).

### Python packages

```bash
pip3 install pandas numpy matplotlib scipy
```

`scipy` is optional: `plot_perf.py` uses it for the t-distribution confidence interval and
falls back to 1.96 if it is missing; `analyze_perf.py` does not require it.

### Energy support (optional but recommended)

Check which RAPL domains your machine exposes:

```bash
sudo perf list | grep energy
```

You should see `power/energy-pkg/` and ideally `power/energy-ram/`. The DRAM (`energy-ram`)
domain is often **absent on consumer desktops** (typically present on Xeon/server, varies on
AMD). If it is missing, the runner measures package only and the energy-breakdown figures are
skipped automatically — everything else still works.

---

## 4. How to run it

### Step 1 — Compile the binaries

The parser reads `algo`, `mode` and `N` from file names, so binaries **must** be named
`<algo>_serial` and `<algo>_parallel`. Build both flavors of every source:

```bash
for src in *.c; do
    algo="${src%.c}"
    gcc -O2 -fopenmp            -o "${algo}_serial"   "$src" -lm
    gcc -O2 -fopenmp -DPARALLEL -o "${algo}_parallel" "$src" -lm
done
```

> The serial build also needs `-fopenmp` because `main` calls `omp_get_wtime()`.
> `-DPARALLEL` switches the parallel code path on.

### Step 2 — Create input files

Each program reads two numbers from its input file: **a random seed** (line 1) and **the
size `N`** (line 2). It generates the array internally (uniform by default). Name each input
file by its size, so the complexity plot gets the real `N` on its x-axis:

```bash
printf '42\n1000000\n'  > 1000000.in
printf '42\n10000000\n' > 10000000.in
```

> To change the input distribution, edit `main` in the `.c` files: `fill_uniform` is active by
> default; `fill_normal` and `fill_exponential` are provided just below it, commented out.

### Step 3 — Run the benchmark

```
./run_perf.sh <binary> <input> <threads> [nruns]
```

Run the **serial** binary at 1 thread (the speedup baseline) and the **parallel** binary at
2+ threads. Results go to `results/` as `.p1`, `.p2` and (if RAPL is available) `.p3` files.
A full sweep:

```bash
for inp in 1000000.in 10000000.in; do
    ./run_perf.sh combsort_serial   "$inp" 1 100      # serial baseline
    for t in 2 3 4; do
        ./run_perf.sh combsort_parallel "$inp" "$t" 100
    done
done
```

Replace `combsort` with each algorithm (or wrap the block in an outer loop over algorithms).
`nruns` defaults to 100; use a small number (e.g. 10) for a quick smoke test.

> **Run final measurements on an idle, dedicated machine.** RAPL is per-socket, not
> per-process, so any background load contaminates the energy numbers. Energy is only
> meaningful for **large `N`** (millions); for tiny inputs it is dominated by start-up noise.

### Step 4 — Parse the results

```bash
python3 parse_perf.py
```

Produces:

- `results/results_raw.csv` — one row per run.
- `results/results_summary.csv` — mean and std per (algo, input, mode, threads).

### Step 5 — Generate figures

```bash
python3 plot_perf.py      # per-metric plots: results/{by_algo,by_threads,std_*,speedup}/
python3 analyze_perf.py   # analysis figures: results/analysis/<topic>/
```

`analyze_perf.py` needs the **summary** CSV; the **raw** CSV is optional and unlocks the
distribution, correlation and reproducibility figures. Energy figures appear only if the
`.p3` files were collected.

---

## 5. Output structure

```
results/
├── *.p1 / *.p2 / *.p3            # raw perf output per run
├── results_raw.csv
├── results_summary.csv
└── analysis/
    ├── complexity/               # log-log time & instructions vs N (fitted exponent k)
    ├── scalability/              # efficiency heatmap, speedup vs threads, Amdahl fraction
    ├── microarch/                # IPC vs branch/cache miss, normalized counter portrait
    ├── memory/                   # L1/LLC/dTLB pressure
    ├── distributions/            # boxplots over runs            (needs raw)
    ├── correlation/              # counter-vs-time correlation   (needs raw)
    ├── reproducibility/          # coefficient of variation      (needs raw)
    └── energy/                   # energy & power vs threads, energy-time Pareto,
                                  # EDP/ED2P heatmaps, package-vs-DRAM breakdown
```

---

## 6. Configuration knobs

| File              | Variable                  | Purpose                                              |
|-------------------|---------------------------|------------------------------------------------------|
| `run_perf.sh`     | `nruns` (4th argument)    | Repetitions per configuration (default 100).         |
| `parse_perf.py`   | —                         | Auto-detects numeric vs `inputN` file names.         |
| `plot_perf.py`    | `BASELINE_MODE`           | Speedup baseline: `"serial"` or `"self"`.            |
| `plot_perf.py`    | `CONF_LEVEL`              | Confidence level for error bars (default 0.95).      |
| `analyze_perf.py` | `FIGFMT`                  | Figure format, `"pdf"` (default) or `"png"`.         |
| `analyze_perf.py` | `BASE_MODE`, `BASE_THREADS` | Baseline used for the per-algorithm fingerprints.  |

---

## 7. Verifying a clean run

After collecting data, the runner prints two checks. No `.p1`/`.p2` file should contain a
`%` in parentheses (that would mean multiplexing), and `instructions:u` / `branches:u` must
be identical across all runs of the same configuration:

```bash
grep -H '%)' results/*.p1 results/*.p2 || echo 'OK: no multiplexed events'
grep -h 'instructions:u' results/<algo>_<mode>_<input>_t<threads>_run*.p1
```

---

## 8. Troubleshooting

- **`perf` permission denied** — the runner uses `sudo perf`; make sure your user can run
  `sudo`. If you prefer not to use sudo, lower `kernel.perf_event_paranoid`, but RAPL and the
  NMI-watchdog toggle still need privilege.
- **Energy columns are all zero / breakdown skipped** — your machine does not expose
  `power/energy-ram/` (and maybe not `power/energy-pkg/`). Confirm with
  `sudo perf list | grep energy`. The pipeline degrades gracefully to package-only or no energy.
- **Complexity exponent looks wrong** — make sure input files are named by their numeric size
  (`10000000.in`), not `input1`/`input2`. The exponent is fitted against the `N` parsed from
  the file name.
- **Noisy parallel speedup** — increase `nruns`, run on a dedicated idle machine, and prefer
  the median over the mean for volatile metrics (long-tailed time distributions). 
