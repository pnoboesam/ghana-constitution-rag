from .embeddings import embeddings
from src.config import CHROMA_DIR_OPENAI, CHROMA_DIR_HF, CHROMA_DIR_NOMIC
from .prompt import load_prompt, get_llm
from typing import Optional
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


llm = get_llm()

class MetadataSearch(BaseModel):
    """Search over legal documents by their structure number"""

    article: Optional[list[int]] = Field(
        None,
        description=(
            "The article number explicitly referenced in the user's query. "
            "Populate this field only if the user mentions a specific article "
            "(e.g., 'Article 6', 'Article 40'). Otherwise, leave it as None."
        )
    )

    chapter: Optional[list[int]] = Field(
        None,
        description=(
            "The chapter number explicitly referenced in the user's query. "
            "Populate this field only if the user mentions a specific chapter "
            "(e.g., 'Chapter 5'). Otherwise, leave it as None."
        )
    )

    section: Optional[list[int]] = Field(
        None,
        description=(
            "The section number explicitly referenced in the user's query. "
            "Populate this field only if the user mentions a specific section "
            "(e.g., 'Section 12'). Otherwise, leave it as None."
        )
    )

    schedule: Optional[list[int]] = Field(
        None,
        description=(
            "The schedule number explicitly referenced in the user's query. "
            "Populate this field only if the user mentions a specific schedule "
            "(e.g., 'Schedule 2', 'Second Schedule'). Convert ordinal references "
            "such as 'First Schedule' or 'Second Schedule' to their corresponding "
            "numeric values. Otherwise, leave it as None."
        )
    )

    part: Optional[list[int]] = Field(
        None,
        description=(
            "The part number explicitly referenced in the user's query. "
            "Populate this field only if the user mentions a specific part "
            "(e.g., 'Part 3' or 'Part III'). Convert Roman numerals to integers "
            "when possible. Otherwise, leave it as None."
        )
    )


structured_llm = llm.with_structured_output((MetadataSearch))


system = load_prompt("metadata_search_system_prompt")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


[embedding_model, model_name] = embeddings

if model_name == 'nomic':
    CHROMA_DIR = CHROMA_DIR_NOMIC
elif model_name == 'hf':
    CHROMA_DIR = CHROMA_DIR_HF
elif model_name == 'openai':
    CHROMA_DIR = CHROMA_DIR_OPENAI

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)
collection = vectorstore._collection


def metadata_to_where(metadata_results):
    conditions = []

    for field, value in metadata_results:
        if value is None:
            continue

        if isinstance(value, list):
            if len(value) == 1:
                conditions.append({
                    field: value[0]
                })
            else:
                conditions.append({
                    field: {
                        "$in": [v for v in value]
                    }
                })
        else:
            conditions.append({
                field: value
            })

    if not conditions:
        return None

    return conditions[0] if len(conditions) == 1 else {"$and": conditions}



def metadata_search(question):

    query_analyzer = prompt | structured_llm
    metadata_results = query_analyzer.invoke({"question": question})

    results = collection.get(
        where=metadata_to_where(metadata_results)
    )

    docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]

    return docs


    



