import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your company files."
        )

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. Please add your company documents."
        )

    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks with overlap."""
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError(
            "No chunks were created. Check whether your documents contain text."
        )

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Preview: {chunk.page_content[:150]}...")

    if len(chunks) > 3:
        print(f"\n... and {len(chunks) - 3} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_minilm"):
    """Create and persist a ChromaDB vector store."""

    if not chunks:
        raise ValueError("Cannot create a vector store with no chunks.")

    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Finished creating vector store.")
    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


def main():
    print("Main Function")

    documents = load_documents(docs_path="docs")
    print(f"\nDocuments loaded: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Chunks generated: {len(chunks)}")

    vectorstore = create_vector_store(chunks)

    print("\nIngestion completed successfully!")

if __name__ == "__main__":
    main()