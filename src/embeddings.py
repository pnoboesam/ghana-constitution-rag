from langchain_openai import OpenAIEmbeddings
from .config import OPENROUTER_API_KEY


def get_embedding_model(model_name: str ="openai/text-embedding-3-small"):

    embedding_model = OpenAIEmbeddings(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    print(model_name, 'was used for embeddings')

    return [embedding_model, model_name]


embeddings = get_embedding_model()
