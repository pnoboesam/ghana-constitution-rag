import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from .chunker import chunks
from .embeddings import embeddings

[embedding_model, model_name] = embeddings

# --------------------------------------------------
# Pinecone configuration
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "ghana-constitution"
NAMESPACE = "ghana-legal_docs"

pc = Pinecone(api_key=PINECONE_API_KEY)
# --------------------------------------------------

test_embedding = embedding_model.embed_query("test")
dimension = len(test_embedding)

if not pc.has_index(INDEX_NAME):
    print(f"Creating Pinecone index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        vector_type="dense",
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
        deletion_protection="disabled",
    )

    print("Pinecone index created.")


# --------------------------------------------------
# Connect to the Pinecone index
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding_model,
    namespace=NAMESPACE,
)
# --------------------------------------------------


# --------------------------------------------------
# Replace existing corpus
print("Removing existing documents...")

vectorstore.delete(delete_all=True)

print("Uploading chunks to Pinecone...")

vectorstore.add_documents(chunks)

print(f"Indexed {len(chunks)} chunks successfully.")
# --------------------------------------------------
