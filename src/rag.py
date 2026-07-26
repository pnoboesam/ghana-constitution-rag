from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .retrieval import retrieve
from .utils import format_docs
from .reranker import rerank
from .prompt import load_prompt, get_llm
    

answer_prompt_template = load_prompt("generation_promptv1")
prompt = ChatPromptTemplate.from_template(answer_prompt_template)
llm = get_llm(provider='openai')


def generate_answer(question, context):
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"context": context, "question":question})


def answer_question(question: str) -> str:
    docs = retrieve(question)
    reranked = rerank(question, docs)
    context = format_docs(reranked)
    answer = {
        "answer": generate_answer(question, context),
        "retrieved_chunks": reranked
    }
    return answer