from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from parser import get_documents
from chunker import chunk_documents
from embeddings import embedding_model
from config import CHROMA_DIR

# Reciprocal Rank Fusion (RRF) Implementation
def reciprocal_rank_fusion(retriever_results: list[list], weights: list[float] | None = None, k=60):
    
    if weights is None:
        weights = [1.0] * len(retriever_results)

    if len(weights) != len(retriever_results):
        raise ValueError(
            "The number of weights must match the number of retrievers."
        )

    fused_scores = {}

    for docs, weight in zip(retriever_results, weights):
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.page_content
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                "doc": doc,
                "fused_score": 0
                }
            
            fused_scores[doc_id]["fused_score"] += weight/(k+rank)

    ranked_docs = sorted(
        fused_scores.items(),
        key=lambda x: x[1]["fused_score"],
        reverse=True
        )
    
    ranked_docs = [ ranked_doc[1]["doc"] for ranked_doc in ranked_docs]

    return ranked_docs


def get_retrievers(chunks, embedding_model, k=10):
    # Keyword search retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # Vector search retriever
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k":k})

    return [bm25_retriever, vector_retriever]



def retrieve(question):
    documents = get_documents()
    chunks = chunk_documents(documents)
    retrievers = get_retrievers(chunks, embedding_model)

    bm25_docs = retrievers[0].invoke(question)
    vector_docs = retrievers[1].invoke(question)

    ranked_docs = reciprocal_rank_fusion([bm25_docs, vector_docs])

    return ranked_docs


