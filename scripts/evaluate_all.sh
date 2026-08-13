#!/bin/bash

set -e

echo "Evaluating SASRec Untied..."
uv run python scripts/eval.py --config configs/ml-1m/sasrec_untied.yaml

echo ""
echo "Evaluating SASRecNext Untied..."
uv run python scripts/eval.py --config configs/ml-1m/sasrecnext_untied.yaml

echo ""
echo "Evaluating SASRec Tied..."
uv run python scripts/eval.py --config configs/ml-1m/sasrec_tied.yaml

echo ""
echo "Evaluating SASRecNext Tied..."
uv run python scripts/eval.py --config configs/ml-1m/sasrecnext_tied.yaml

echo ""
echo "All ML-1M evaluations completed!"
