#!/bin/bash
set -euo pipefail

RESULTS_DIR="results"
LOGS_DIR="condor/logs"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

echo "[INFO] Limpando e compilando..."
make clean
make

echo "[INFO] Submitting Condor jobs..."
SUBMIT_OUTPUT=$(condor_submit condor/perf.sub)
echo "$SUBMIT_OUTPUT"

# Captura ClusterID corretamente e limpa caracteres extras
CLUSTER_ID=$(echo "$SUBMIT_OUTPUT" | grep -i "submitted to cluster" | awk '{print $NF}' | tr -d '.')
if [[ -z "$CLUSTER_ID" ]]; then
    echo "[ERROR] Não foi possível capturar o ClusterID!"
    exit 1
fi
echo "[INFO] Cluster ID: $CLUSTER_ID"

LOG_FILE="$LOGS_DIR/${CLUSTER_ID}.log"
if [[ ! -f "$LOG_FILE" ]]; then
    echo "[ERROR] Log file $LOG_FILE não encontrado. Esperando alguns segundos..."
    sleep 5
fi

#echo "[INFO] Waiting for all jobs in cluster $CLUSTER_ID to finish..."
#condor_wait "$LOG_FILE"

#echo "[INFO] Running post-processing..."
#./post_process.sh

echo "[INFO] All done!"
