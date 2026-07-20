from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 500,
        chunk_overlap = 100
    )

    chunks = text_splitter.split_documents(documents)

    print(f"number of chunks: {len(chunks)}")

    return chunks

