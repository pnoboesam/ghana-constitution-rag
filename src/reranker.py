from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def filter_relevant_context(reranked_docs, MIN_RERANK_SCORE=0.35):
    relevant_docs = []

    for doc in reranked_docs[:5]:
        if doc['score'] > MIN_RERANK_SCORE:
            relevant_docs.append(doc['doc'])

    return relevant_docs


def rerank(question, retrieved_docs):
    pairs = [ (question, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(pairs)

    reranked_docs = []
    for doc, score in zip(retrieved_docs, scores):
        reranked_docs.append({
            "doc": doc,
            "score": score
        })
    reranked_docs.sort(key=lambda x:x["score"], reverse=True)

    reranked_docs = filter_relevant_context(reranked_docs)
    print(f"number of final context docs: {len(reranked_docs)}")

    return reranked_docs



