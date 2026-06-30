#!/bin/bash
set -euo pipefail

RESULTS_DIR="results"
LOGS_DIR="condor/logs"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

echo "[INFO] Limpando e compilando..."
make clean
make

INPUTS=(
    inputs/100000.in
    inputs/500000.in
    inputs/1000000.in
    inputs/5000000.in
    inputs/10000000.in
)

ALGORITHMS=(
    bottomupsort
    combsort
    insertionsort
    mergesort
    quicksort
    smoothsort
    shellsort 
    heapsort 
    countingsort 
    radixsort 
    bitonicsort 
    oddevensort
)

for alg in "${ALGORITHMS[@]}"; do

    # Serial
    for input in "${INPUTS[@]}"; do
        ./run_perf_job.sh "./${alg}_serial" "$input" 1 1
    done

    # Paralelo
    for threads in 2 3 4; do
        for input in "${INPUTS[@]}"; do
            ./run_perf_job.sh "./${alg}_parallel" "$input" "$threads" 1
        done
    done

done


echo "[INFO] Completed locally."
