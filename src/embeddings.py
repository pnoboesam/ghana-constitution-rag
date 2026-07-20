from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=OPENAI_API_KEY
# )