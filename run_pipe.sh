#!/usr/bin/env bash

# stop at excution fail on pipe

set -euo pipefail

uv run python pipeline.py

echo
cat reports/hit_scoreboard.csv

echo
cat reports/ragas_evaluation_result.csv
