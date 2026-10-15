
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "nexus-resume-index"

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Google API key is missing.")

# Must match the embedding model and dimension used during ingestion.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key,
    output_dimensionality=768,
)

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)


def retrieve_resume_context(question: str, k: int = 4) -> str:
    """Retrieve the most relevant resume chunks from Pinecone."""

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    if not results:
        return "No relevant resume information was found."

    context_parts = []

    for i, (document, score) in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source {i}: {source}, page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)
