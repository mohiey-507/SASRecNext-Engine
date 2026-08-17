#!/bin/bash

set -e

echo "=========================================="
echo "Evaluating ML-1M Models"
echo "=========================================="

echo ""
echo "Evaluating SASRec Untied (ML-1M)..."
uv run python scripts/eval.py --config configs/ml-1m/sasrec_untied.yaml

echo ""
echo "Evaluating SASRecNext Untied (ML-1M)..."
uv run python scripts/eval.py --config configs/ml-1m/sasrecnext_untied.yaml

echo ""
echo "Evaluating SASRec Tied (ML-1M)..."
uv run python scripts/eval.py --config configs/ml-1m/sasrec_tied.yaml

echo ""
echo "Evaluating SASRecNext Tied (ML-1M)..."
uv run python scripts/eval.py --config configs/ml-1m/sasrecnext_tied.yaml

echo ""
echo "=========================================="
echo "Evaluating ML-10M Models"
echo "=========================================="

echo ""
echo "Evaluating SASRec Untied (ML-10M)..."
uv run python scripts/eval.py --config configs/ml-10m/sasrec_untied.yaml

echo ""
echo "Evaluating SASRecNext Untied (ML-10M)..."
uv run python scripts/eval.py --config configs/ml-10m/sasrecnext_untied.yaml

echo ""
echo "Evaluating SASRec Tied (ML-10M)..."
uv run python scripts/eval.py --config configs/ml-10m/sasrec_tied.yaml

echo ""
echo "Evaluating SASRecNext Tied (ML-10M)..."
uv run python scripts/eval.py --config configs/ml-10m/sasrecnext_tied.yaml

echo ""
echo "=========================================="
echo "Evaluating Sequence Extrapolation"
echo "=========================================="

echo ""
uv run python scripts/eval_sequences.py --config configs/ml-10m/sasrecnext_tied.yaml

echo ""
uv run python scripts/eval_sequences.py --config configs/ml-10m/sasrec_tied.yaml --min_history 301 --max_seq_lens "2,4,8,16,32,48,64,80,96,112,128,144,160,176,192,200"

echo ""
echo "All evaluations completed!"
