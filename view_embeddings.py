from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Locate the existing ChromaDB database
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "db" / "chroma_minilm"

# Initialize the same embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Open the existing vector store
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=str(PERSIST_DIRECTORY),
)

# Retrieve stored documents and embeddings
data = vectorstore.get(
    include=["documents", "metadatas", "embeddings"]
)

ids = data["ids"]
documents = data["documents"]
metadatas = data["metadatas"]
embeddings = data["embeddings"]

print("=" * 60)
print("STORED CHROMADB EMBEDDINGS")
print("=" * 60)

print(f"Total stored chunks: {len(ids)}")

if embeddings is not None and len(embeddings) > 0:
    for i, embedding in enumerate(embeddings):
        print(f"\n{'=' * 60}")
        print(f"Chunk {i + 1}")
        print(f"ID: {ids[i]}")
        print(f"Source: {metadatas[i].get('source', 'Unknown')}")
        print(f"Text: {documents[i][:150]}...")
        print(f"Dimensions: {len(embedding)}")
        print(f"Embedding:\n{embedding}")
else:
    print("No embeddings found in this collection.")

print("\nFinished displaying stored embeddings.")
