import os

from .prompt import load_prompt, get_llm
from .embeddings import embeddings

from typing import Optional
from pinecone import Pinecone
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langsmith import traceable


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


# --------------------------------------------------
# Pinecone configuration

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

INDEX_HOST = "https://ghana-constitution-4i7bb15.svc.aped-4627-b74a.pinecone.io"
INDEX_NAME = "ghana-constitution"
NAMESPACE = "ghana-legal_docs"

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)
# -------------------------------------------------

embedding_model = embeddings[0]

@traceable(name="Metadata Filter Builder")
def metadata_to_filter(metadata_results):

    conditions = []

    for field, value in metadata_results:

        if value is None:
            continue

        if isinstance(value, list):

            if len(value) == 1:
                conditions.append({
                    field: {
                        "$eq": value[0]
                    }
                })
            else:
                conditions.append({
                    field: {
                        "$in": value
                    }
                })

        else:
            conditions.append({
                field: {
                    "$eq": value
                }
            })

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$or": conditions
    }


@traceable(name="Metadata Search")
def metadata_search(question):

    query_analyzer = prompt | structured_llm

    metadata_results = query_analyzer.invoke(
        {"question": question}
    )

    metadata_filter = metadata_to_filter(
        metadata_results.model_dump().items()
    )

    if metadata_filter is None:
        return []

    # Embed the user's actual question
    query_vector = embedding_model.embed_query(question)

    # Semantic search constrained by metadata
    results = index.query(
        vector=query_vector,
        top_k=10,
        namespace=NAMESPACE,
        filter=metadata_filter,
        include_metadata=True,
    )

    docs = []

    for match in results.matches:
        metadata = dict(match.metadata or {})
        page_content = metadata.pop("text", "")

        docs.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )


    return docs


    



