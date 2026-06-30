#!/bin/bash
set -e

echo "[INFO] Post-processing results"

python3 scripts/collect_csv.py
python3 scripts/plot_metrics.py
python3 scripts/analyze_perf.py

echo "[INFO] Post-processing done"
