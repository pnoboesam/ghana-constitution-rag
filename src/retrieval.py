from langchain_community.retrievers import BM25Retriever
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langsmith import traceable

from .chunker import chunks
from .embeddings import embeddings
from .config import CHROMA_DIR_OPENAI, CHROMA_DIR_HF, CHROMA_DIR_NOMIC
from .prompt import load_prompt, get_llm
from .metadata_search import metadata_search

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


def get_retrievers(chunks, embedding_model, model_name, k=10):
    # Keyword search retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    if model_name == 'nomic':
        CHROMA_DIR = CHROMA_DIR_NOMIC
    elif model_name == 'hf':
        CHROMA_DIR = CHROMA_DIR_HF
    elif model_name == 'openai':
        CHROMA_DIR = CHROMA_DIR_OPENAI    

    # Vector search retriever
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k":k})

    return [bm25_retriever, vector_retriever]

[embedding_model, model_name] = embeddings

[bm25_retriever, vector_retriever] = get_retrievers(
    chunks,
    embedding_model,
    model_name
    )

llm = get_llm()

@traceable(name='Query Router')
def query_router(question):
    class RouteQuery(BaseModel):
        """Route user query to the most relevant document search"""

        docsearch: Literal['metadata_search', 'hybrid_search'] = Field(
            ...,
            description='Given a user question choose which document search would be most relevant for retrieving documents to answer their question'
        )

    structured_llm = llm.with_structured_output((RouteQuery))

    system = load_prompt("router_system_promptv1")

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system),
            ('human', '{question}')
        ]
    )

    router = prompt | structured_llm

    relevant_document_search = router.invoke({'question': question}).docsearch.lower()

    return relevant_document_search


@traceable(name="Hybrid Retrieval")
def hybrid_search(question):
    bm25_docs = bm25_retriever.invoke(question)
    vector_docs = vector_retriever.invoke(question)

    return reciprocal_rank_fusion([bm25_docs, vector_docs])



def retrieve(question):

    def choose_route(result):
        if 'metadata_search' in result:
            retrieved_docs = metadata_search(question)
            return retrieved_docs
        else:
            retrieved_docs = hybrid_search(question)
            return retrieved_docs


    result = query_router(question)
    retrieved_docs = choose_route(result)

    return retrieved_docs



