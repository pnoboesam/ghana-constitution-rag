import json
import time
from src.rag import answer_question
from src.config import DATA_DIR


PREDICTION_DATASET_PATH = DATA_DIR / "rag_predictions.json"

def run_inference(evaluation_dataset):
    # RUN RAG PIPELINE FOR EACH QUESTION----------------------
    dataset = []
    total = len(evaluation_dataset)
    start_time = time.perf_counter()

    for data in evaluation_dataset:
        question = data["question"]
        results = answer_question(question)
        answer = results['answer']
        retrieved_chunks = results['retrieved_chunks']

        eval_data = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts":  [doc.page_content for doc in retrieved_chunks],
            "reference": data["ground_truth"]
        }
        
        elapsed = time.perf_counter()-start_time
        avg_time = elapsed / data['id']
        remaining = avg_time * (total - data['id'])

        print(f"[{data['id']}/{total}] "
            f"Ans: {answer[:50]}... | "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {remaining:.1f}s"
            )

        dataset.append(eval_data)


    # SAVE RAG PREDICTIONS-------------------------------------
    with open(PREDICTION_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
