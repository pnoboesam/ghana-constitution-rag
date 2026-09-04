import json
from datetime import datetime
import pandas as pd

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall
)
from ragas import EvaluationDataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from evaluation.markdown_report import generate_markdown_report
from src.config import  OPENROUTER_API_KEY, DATA_DIR, EVAL_DIR
from evaluation.run_inference import run_inference

from evaluation.no_answer_evaluator import evaluate_no_answer_response

GROUND_TRUTH_DATASET_PATH = DATA_DIR / "ground_truth_dataset.json"
PREDICTION_DATASET_PATH = DATA_DIR / "rag_predictions.json"
LATEST_METRICS_PATH = EVAL_DIR / "reports" / "latest_metrics.json"
LATEST_ANSWERABLE_RESULTS_PATH = EVAL_DIR / "reports" / "latest_answerable_results.csv"
LATEST_UNANSWERABLE_RESULTS_PATH = EVAL_DIR / "reports" / "latest_unanswerable_results.csv"
FAILED_ANSWERABLE_CASES_PATH = EVAL_DIR / "reports" / "failed_answerable_cases.csv"
FAILED_UNANSWERABLE_CASES_PATH = EVAL_DIR / "reports" / "failed_unanswerable_cases.csv"
METRICS_HISTORY_PATH = EVAL_DIR / "reports" / "metrics_history"
LATEST_REPORT_PATH = EVAL_DIR / "reports" / "latest_report.md"

MIN_FAITHFULNESS = 0.80
MIN_ANSWER_CORRECTNESS = 0.80
MIN_CONTEXT_PRECISION = 0.80
MIN_CONTEXT_RECALL = 0.80

# LOAD EVALUATION DATASET---------------------------------
with open(GROUND_TRUTH_DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)

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

answerable_cases_dataset = []
no_answer_cases_dataset = []

for item in dataset:
    if item["answerable"]:
          answerable_cases_dataset.append(item)
    else:
        no_answer_cases_dataset.append(item)

answerable_count = len(answerable_cases_dataset)
unanswerable_count = len(no_answer_cases_dataset)

print("Length of answerable questions", answerable_count)
print("Length of non-answerable questions", unanswerable_count)

answerable_cases_dataset = EvaluationDataset.from_list(answerable_cases_dataset)
answerable_eval_results = evaluate(
    answerable_cases_dataset,
    metrics=[
        faithfulness,
        answer_correctness,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
    batch_size=5,
)
print(answerable_eval_results)

[no_answer_eval_results, no_answer_summary] = evaluate_no_answer_response(no_answer_cases_dataset)


# SAVE LATEST RESULTS -------------------------------
df_answerable = answerable_eval_results.to_pandas()
df_answerable.to_csv(LATEST_ANSWERABLE_RESULTS_PATH, index=False)

df_unanswerable = pd.DataFrame(no_answer_eval_results)
df_unanswerable.to_csv(LATEST_UNANSWERABLE_RESULTS_PATH, index=False)


# SAVE FAILED CASES --------------------------------
failed_answerable_cases = df_answerable[
    (df_answerable["faithfulness"] < MIN_FAITHFULNESS) |
    (df_answerable["answer_correctness"] < MIN_ANSWER_CORRECTNESS) |
    (df_answerable["context_precision"] < MIN_CONTEXT_PRECISION) |
    (df_answerable["context_recall"] < MIN_CONTEXT_RECALL) |
    (
        df_answerable[
            [
                "faithfulness",
                "answer_correctness",
                "context_precision",
                "context_recall"
            ]
        ].isna().any(axis=1)
    )
]
failed_answerable_cases.to_csv(FAILED_ANSWERABLE_CASES_PATH, index=False)

failed_unanswerable_cases = df_unanswerable[
    (df_unanswerable["outcome"] == "unsafe_answer")
]
failed_unanswerable_cases.to_csv(FAILED_UNANSWERABLE_CASES_PATH, index=False)



# SAVE LATEST METRICS -------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# answerable_metrics = answerable_eval_results._repr_dict

metrics = {
     "evaluation_run": timestamp,

     "dataset": {
          "total_questions": len(dataset),
          "answerable_questions": answerable_count,
          "unanswerable_questions": unanswerable_count
     },

     "answerable_evaluation": {
          "framework": "ragas",
          "metrics": {
            "faithfulness": round(df_answerable["faithfulness"].mean(), 4),
            "answer_correctness": round(df_answerable["answer_correctness"].mean(), 4),
            "context_precision": round(df_answerable["context_precision"].mean(), 4),
            "context_recall": round(df_answerable["context_recall"].mean(), 4),
        }
     },

     "unanswerable_evaluation": no_answer_summary
}

with open(LATEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


# SAVE METRICS HISTORY -------------------------------
history_path = METRICS_HISTORY_PATH / f"{timestamp}.json"

with open(history_path, "w") as f:
    json.dump(metrics, f, indent=4)


# GENERATE MARKDOWN REPORT -------------------------------
thresholds = {
    "min_faithfulness": MIN_FAITHFULNESS,
    "min_answer_correctness": MIN_ANSWER_CORRECTNESS,
    "min_context_precision": MIN_CONTEXT_PRECISION,
    "min_context_recall": MIN_CONTEXT_RECALL
}
report = generate_markdown_report(metrics, df_answerable, df_unanswerable, thresholds)

with open(LATEST_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

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