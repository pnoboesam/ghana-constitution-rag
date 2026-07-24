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
from src.config import OPENAI_API_KEY, DATA_DIR
from evaluation.run_inference import run_inference

GROUND_TRUTH_DATASET_PATH = DATA_DIR / "ground_truth_dataset.json"
PREDICTION_DATASET_PATH = DATA_DIR / "predictions_dataset.json"

# LOAD EVALUATION DATASET---------------------------------
with open(GROUND_TRUTH_DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)
len(evaluation_dataset)


# RUN RAG PIPELINE FOR EACH QUESTION----------------------
run_inference(evaluation_dataset[:5])


# RUN RAG EVALUATION --------------------------------------
llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

with open(PREDICTION_DATASET_PATH, "r") as f:
    dataset = json.load(f)

dataset = EvaluationDataset.from_list(dataset)
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
)

print(results)