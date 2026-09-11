from langchain_core.output_parsers import StrOutputParser
from .retrieval import retrieve
from .utils import format_docs
from .reranker import rerank
from .prompt import build_generation_prompt, get_llm
    

prompt = build_generation_prompt("generation_promptv2")
llm = get_llm(provider='openai')


def generate_answer(question, context):
    if not context:
        return "No context provided for a grounded answer."
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"context": context, "question":question})


def answer_question(question: str) -> str:
    retrieved_chunks = retrieve(question)
    context = format_docs(retrieved_chunks)
    answer = {
        "answer": generate_answer(question, context),
        "retrieved_chunks": retrieved_chunks
    }
    return answer