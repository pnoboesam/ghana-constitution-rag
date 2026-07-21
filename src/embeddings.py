# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from .config import OPENAI_API_KEY


def get_embedding_model(model_name: str ='nomic'):
    if model_name == 'nomic':
        embedding_model = OllamaEmbeddings(
            model='nomic-embed-text-v2-moe'
        )

    elif model_name == 'hf':
        # embedding_model = HuggingFaceEmbeddings(
        #     model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # )
        pass

    elif model_name == 'openai':
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )

    return [embedding_model, model_name]

