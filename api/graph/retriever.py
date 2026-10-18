import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "nexus-resume-index"


# Safely isolate initialization inside a helper function to avoid Vercel cold-start errors
def get_vector_store():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("Google API key is missing from environment variables.")
    if not pinecone_key:
        raise ValueError("Pinecone API key is missing from environment variables.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
        output_dimensionality=768,
    )

    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_key
    )


def retrieve_resume_context(question: str, k: int = 4) -> str:
    """Retrieve the most relevant resume chunks from Pinecone securely at runtime."""

    # Get vector store instance at the time of invocation
    vector_store = get_vector_store()

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
