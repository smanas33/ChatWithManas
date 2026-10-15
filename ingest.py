
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "nexus-resume-index"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768


def ingest_resume(file_path: str) -> None:
    """Load, chunk, embed, and store a resume in Pinecone."""

    # 1. Validate configuration and input file
    google_api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not google_api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
        )

    if not pinecone_api_key:
        raise ValueError(
            "Set PINECONE_API_KEY in your .env file."
        )

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Resume file not found: {path.resolve()}"
        )

    # 2. Load document
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            "Unsupported file type. Please use PDF or TXT."
        )

    docs = loader.load()

    if not docs:
        raise ValueError(
            "No content was extracted from the resume."
        )

    print(f"Loaded {len(docs)} document page(s).")

    # 3. Split document and preserve source metadata

    # 3. Split document and preserve essential metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )

    raw_chunks = text_splitter.split_documents(docs)

    chunks = []

    for chunk in raw_chunks:
        # Keep only small, useful metadata fields.
        metadata = {
            key: value
            for key, value in chunk.metadata.items()
            if key in {"source", "page", "start_index"}
               and isinstance(value, (str, int, float, bool))
        }

        # Create a clean document with the original chunk text.
        from langchain_core.documents import Document

        clean_chunk = Document(
            page_content=chunk.page_content,
            metadata=metadata,
        )

        chunks.append(clean_chunk)

    print(f"Created {len(chunks)} chunks with cleaned metadata.")

    if not chunks:
        raise ValueError(
            "No text chunks were generated from the resume."
        )

    print(f"Created {len(chunks)} text chunks.")

    # 4. Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key,
        output_dimensionality=EMBEDDING_DIMENSION,
    )

    # 5. Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    existing_indexes = [
        index.name for index in pc.list_indexes()
    ]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        print(f"Created Pinecone index: {INDEX_NAME}")

    # 6. Validate the existing index dimension
    index_description = pc.describe_index(INDEX_NAME)
    actual_dimension = index_description.dimension

    if actual_dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Pinecone index dimension is {actual_dimension}, "
            f"but the configured embedding dimension is "
            f"{EMBEDDING_DIMENSION}. Use a compatible index "
            "or create a separate index."
        )

    # 7. Generate embeddings and upsert chunks
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    print(
        f"Successfully ingested {len(chunks)} chunks "
        f"into Pinecone index '{INDEX_NAME}'."
    )


if __name__ == "__main__":
    ingest_resume("Manas_resume.pdf")
