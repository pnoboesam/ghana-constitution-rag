import shutil
from pathlib import Path
from .chunker import chunks
from .embeddings import embeddings
from langchain_chroma import Chroma
from .config import CHROMA_DIR_OPENAI, CHROMA_DIR_HF, CHROMA_DIR_NOMIC

[embedding_model, model_name] = embeddings

if model_name == 'nomic':
    CHROMA_DIR = CHROMA_DIR_NOMIC
elif model_name == 'hf':
    CHROMA_DIR = CHROMA_DIR_HF
elif model_name == 'openai':
    CHROMA_DIR = CHROMA_DIR_OPENAI    

if Path(CHROMA_DIR).exists():
    print("Removing existing vector database...")
    shutil.rmtree(CHROMA_DIR)

print("Building new vector database...")
Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model,
    persist_directory=str(CHROMA_DIR)
)

print(f"Indexed {len(chunks)} chunks successfully.")
