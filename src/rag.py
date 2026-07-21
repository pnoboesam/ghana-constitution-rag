from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .retrieval import retrieve
from .utils import format_docs
from .reranker import rerank


def get_llm(provider="anthropic", temperature=0, max_tokens=1024):
    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "anthropic":
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    else:
        raise ValueError(f"Unknown provide: {provider}. Avaliable provider: 'openai', 'anthropic'")
    

## PROMPT LOADING
def load_prompt(name):
    PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

    path = PROMPTS_DIR / f"{name}.txt"

    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    

answer_prompt_template = load_prompt("generation_promptv1")
prompt = ChatPromptTemplate.from_template(answer_prompt_template)
llm = get_llm()


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