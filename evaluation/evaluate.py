import json
from datetime import datetime

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas import EvaluationDataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import  OPENROUTER_API_KEY, DATA_DIR, EVAL_DIR
from evaluation.run_inference import run_inference

GROUND_TRUTH_DATASET_PATH = DATA_DIR / "ground_truth_dataset.json"
PREDICTION_DATASET_PATH = DATA_DIR / "rag_predictions.json"
LATEST_METRICS_PATH = EVAL_DIR / "reports" / "latest_metrics.json"
LATEST_RESULTS_PATH = EVAL_DIR / "reports" / "latest_results.csv"
FAILED_CASES_PATH = EVAL_DIR / "reports" / "failed_cases.csv"
METRICS_HISTORY_PATH = EVAL_DIR / "reports" / "metrics_history"

MIN_FAITHFULNESS = 0.80
MIN_ANSWER_CORRECTNESS = 0.80
MIN_CONTEXT_PRECISION = 0.80
MIN_CONTEXT_RECALL = 0.80

# LOAD EVALUATION DATASET---------------------------------
with open(GROUND_TRUTH_DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)
len(evaluation_dataset)


# RUN RAG PIPELINE FOR EACH QUESTION----------------------
run_inference(evaluation_dataset)


# RUN RAG EVALUATION --------------------------------------
llm = ChatOpenAI(
    model='openai/gpt-4.1-mini',
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=120,
    max_retries=5,
)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

with open(PREDICTION_DATASET_PATH, "r") as f:
    dataset = json.load(f)

print(dataset[0])

dataset = EvaluationDataset.from_list(dataset)
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_correctness,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
    batch_size=1,
)
print(results)


# SAVE LATEST METRICS -------------------------------
metrics = results._repr_dict
with open(LATEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


# SAVE METRICS HISTORY -------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
history_path = METRICS_HISTORY_PATH / f"{timestamp}.json"

with open(history_path, "w") as f:
    json.dump(metrics, f, indent=4)

# SAVE LATEST RESULTS -------------------------------
df = results.to_pandas()
df.to_csv(LATEST_RESULTS_PATH, index=False)


# SAVE FAILED CASES --------------------------------
failed_cases = df[
    (df["faithfulness"] < MIN_FAITHFULNESS) |
    (df["answer_correctness"] < MIN_ANSWER_CORRECTNESS) |
    (df["context_precision"] < MIN_CONTEXT_PRECISION) |
    (df["context_recall"] < MIN_CONTEXT_RECALL)
]
failed_cases.to_csv(FAILED_CASES_PATH, index=False)


# # ADD QUALITY GATES --------------------------------
# if metrics['faithfulness'] < MIN_FAITHFULNESS:
#      raise RuntimeError(
#           f"Faithfulness dropped to {metrics['faithfulness']:.3f}"
#      )

# if metrics['answer_correctness'] < MIN_ANSWER_CORRECTNESS:
#      raise RuntimeError(
#           f"Answer correctness dropped to {metrics['answer_correctness']:.3f}"
#      )

# if metrics['context_precision'] < MIN_CONTEXT_PRECISION:
#      raise RuntimeError(
#           f"Context precision dropped to {metrics['context_precision']:.3f}"
#      )

# if metrics['context_recall'] < MIN_CONTEXT_RECALL:
#      raise RuntimeError(
#           f"Context recall dropped to {metrics['context_recall']:.3f}"
#      )