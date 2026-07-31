from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from .config import OPENROUTER_API_KEY


def load_prompt(name):
    PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

    path = PROMPTS_DIR / f"{name}.txt"

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_llm(provider="openai", temperature=0, max_tokens=1024):
    if provider == "openai":
        return ChatOpenAI(
            model='openai/gpt-oss-120b',
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