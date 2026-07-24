import json
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
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from src.config import OPENAI_API_KEY, DATA_DIR, EVAL_DIR
from evaluation.run_inference import run_inference

GROUND_TRUTH_DATASET_PATH = DATA_DIR / "ground_truth_dataset.json"
PREDICTION_DATASET_PATH = DATA_DIR / "predictions_dataset.json"
LATEST_METRICS_PATH = EVAL_DIR / "reports" / "latest_metrics.json"
LATEST_RESULTS_PATH = EVAL_DIR / "reports" / "latest_results.csv"
FAILED_CASES_PATH = EVAL_DIR / "reports" / "failed_cases.csv"

MIN_FAITHFULNESS = 0.80
MIN_ANSWER_CORRECTNESS = 0.80
MIN_CONTEXT_PRECISION = 0.80
MIN_CONTEXT_RECALL = 0.80

# LOAD EVALUATION DATASET---------------------------------
with open(GROUND_TRUTH_DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)
len(evaluation_dataset)


# RUN RAG PIPELINE FOR EACH QUESTION----------------------
run_inference(evaluation_dataset[:5])


# RUN RAG EVALUATION --------------------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=OPENAI_API_KEY,
    timeout=120,
    max_retries=5,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
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
with open(LATEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


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


# ADD QUALITY GATES --------------------------------
if results['faithfulness'] < MIN_FAITHFULNESS:
     raise RuntimeError(
          f"Faithfulness dropped to {results['faithfulness']:.3f}"
     )

if results['answer_correctness'] < MIN_ANSWER_CORRECTNESS:
     raise RuntimeError(
          f"Answer correctness dropped to {results['answer_correctness']:.3f}"
     )

if results['context_precision'] < MIN_CONTEXT_PRECISION:
     raise RuntimeError(
          f"Context precision dropped to {results['context_precision']:.3f}"
     )

if results['context_recall'] < MIN_CONTEXT_RECALL:
     raise RuntimeError(
          f"Context recall dropped to {results['context_recall']:.3f}"
     )