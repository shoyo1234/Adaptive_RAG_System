from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PERSIST_DIRECTORY = BASE_DIR / "db" / "chroma_minilm"

COLLECTION_NAME = "langchain"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Initialize Embedding Model
# --------------------------------------------------

def get_embedding_model():
    """Load the same embedding model used during ingestion."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# --------------------------------------------------
# Load Existing Vector Store
# --------------------------------------------------

def load_vector_store():
    """Open the existing ChromaDB vector store."""

    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Vector database not found at: {PERSIST_DIRECTORY}\n"
            "Run ingestion_pipeline.py first."
        )

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    # Check whether documents have been stored.
    stored_data = vectorstore.get(include=[])

    if not stored_data["ids"]:
        raise ValueError(
            f"No documents found in the '{COLLECTION_NAME}' collection. "
            "Check your ingestion pipeline and collection name."
        )

    print(f"Vector database loaded: {PERSIST_DIRECTORY}")
    print(f"Stored chunks available: {len(stored_data['ids'])}")

    return vectorstore


# --------------------------------------------------
# Retrieve Relevant Documents
# --------------------------------------------------

def retrieve_documents(vectorstore, query, k=3):
    """
    Retrieve the top-k most relevant document chunks
    for a user query.
    """

    if not query or not query.strip():
        raise ValueError("Please enter a valid question.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    print(f"Query: {query}")

    # Retrieve documents with similarity scores.
    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
    )

    if not results:
        print("No relevant documents found.")
        return []

    retrieved_documents = []

    for rank, (document, distance) in enumerate(results, start=1):
        retrieved_documents.append(document)

        print(f"\n--- Result {rank} ---")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(f"Distance: {distance:.4f}")
        print("Content:")
        print(document.page_content)

    print("\n" + "=" * 60)

    return retrieved_documents


# --------------------------------------------------
# Main Function
# --------------------------------------------------

def main():
    """Run the retrieval pipeline interactively."""

    vectorstore = load_vector_store()

    print("\nRAG Retrieval System")
    print("Enter 'exit' to stop.")

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting retrieval system.")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            retrieve_documents(
                vectorstore=vectorstore,
                query=query,
                k=3,
            )
        except Exception as error:
            print(f"Retrieval failed: {error}")


if __name__ == "__main__":
    main()
