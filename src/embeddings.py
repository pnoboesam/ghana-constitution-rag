# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from .config import OPENAI_API_KEY


def get_embedding_model(model_name: str ='nomic'):
    if model_name == 'nomic':
        embedding_model = OllamaEmbeddings(
            model='nomic-embed-text-v2-moe'
        )

    elif model_name == 'openai':
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(model_name, 'was used for embeddings')

    return [embedding_model, model_name]


embeddings = get_embedding_model(model_name='nomic')
