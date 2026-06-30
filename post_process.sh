#!/bin/bash
set -e

echo "[INFO] Post-processing results"

rm -rf results/*.csv
rm -rf results/plots/*
rm -rf results/analysis/*

python3 scripts/parse_perf.py
python3 scripts/plot_metrics.py
python3 scripts/analyze_perf.py

echo "[INFO] Post-processing done"
