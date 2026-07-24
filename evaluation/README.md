# Evaluation Pipeline

This folder contains the evaluation workflow for the Ghana Constitution RAG system.

## Files

- `ground_truth_dataset.json` – Benchmark question/answer pairs.
- `generate_predictions.py` – Generates RAG responses and retrieved contexts.
- `rag_predictions.json` – Predictions produced by the RAG system.
- `evaluate.py` – Computes Ragas metrics.
- `ragas_results.csv` – Evaluation metrics and scores.

## Workflow

ground_truth_dataset.json
↓
generate_predictions.py
↓
rag_predictions.json
↓
evaluate.py
↓
ragas_results.csv
