from chunker import chunks
from embeddings import embeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from .config import CHROMA_DIR_OPENAI, CHROMA_DIR_HF, CHROMA_DIR_NOMIC

[embedding_model, model_name] = embeddings

if model_name == 'nomic':
    CHROMA_DIR = CHROMA_DIR_NOMIC
elif model_name == 'hf':
    CHROMA_DIR = CHROMA_DIR_HF
elif model_name == 'openai':
    CHROMA_DIR = CHROMA_DIR_OPENAI    

Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model,
    persist_directory=str(CHROMA_DIR)
)

