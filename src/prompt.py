from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)

from .config import OPENROUTER_API_KEY, BASE_DIR
from prompts.examples import examples

def load_prompt(name):
    PROMPTS_DIR = BASE_DIR / "prompts"

    path = PROMPTS_DIR / f"{name}.txt"

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_generation_prompt(name):
    answer_prompt_template = load_prompt(name)

    example_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        ),
        (
            "ai",
            "{answer}"
        ),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", answer_prompt_template),

        few_shot_prompt,

        ("human", """
            Context:
            {context}

            Question:
            {question}
            """),
    ])

    return prompt


def get_llm(provider="openai", temperature=0, max_tokens=1024):
    if provider == "openai":
        return ChatOpenAI(
            model='openai/gpt-4.1-mini',
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
    
    elif provider == "anthropic":
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "ollama":
        return OllamaLLM(
            model="llama3.1:8b",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    else:
        raise ValueError(f"Unknown provide: {provider}. Avaliable provider: 'openai', 'anthropic'")

if __name__ == "__main__":
    from icecream import ic as print
    prompt = build_generation_prompt("generation_promptv2")
    prompt = prompt.invoke({"context": "THIS IS MY CONTEXT HERE", "question": "THIS IS MY QUESTION HERE"})
    print(prompt)