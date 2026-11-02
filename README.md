# Adaptive RAG System

A lightweight retrieval-augmented generation (RAG) project built with Python, LangChain, ChromaDB, and sentence-transformer embeddings. The system ingests local text documents, splits them into searchable chunks, stores them in a vector database, and retrieves the most relevant content for user queries.

This project is designed for small document corpora and local experimentation, making it a simple starting point for building knowledge-based Q&A systems without needing a full cloud stack.

## Overview

The workflow is intentionally simple:

1. Load text files from the `docs/` directory.
2. Split long documents into overlapping chunks.
3. Generate embeddings using a lightweight Hugging Face model.
4. Store the embeddings in a local Chroma vector database.
5. Query the database for the most relevant chunks and return them.

This makes it easy to test how document retrieval works for domain-specific content, such as product docs, policies, or article collections.

## Features

- Local document ingestion from plain text files
- Recursive chunking with overlap for better retrieval
- Embedding generation with `sentence-transformers/all-MiniLM-L6-v2`
- Persistent vector storage using ChromaDB
- Simple interactive retrieval loop for asking questions
- Easy to extend for chat-based or LLM-powered answer generation

## Project Structure

- `ingestion_pipeline.py` – loads text files, splits them into chunks, and creates the vector store.
- `retrieval_pipeline.py` – loads the saved vector database and allows interactive similarity search.
- `main.py` – minimal project entry point placeholder.
- `docs/` – source documents used for retrieval.
- `db/chroma_minilm/` – persisted Chroma database files.
- `pyproject.toml` – Python project metadata and dependencies.

## Tech Stack

- Python 3.12+
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Python dotenv

## Getting Started

### 1. Install dependencies

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

If you are not using `uv`, install the dependencies from `pyproject.toml` with your preferred package manager or via pip for the listed packages.

### 2. Add documents

Place your `.txt` files inside the `docs/` directory. The example project includes sample files such as:

- `docs/bike.txt`
- `docs/car.txt`

### 3. Run ingestion

```bash
python ingestion_pipeline.py
```

This will:

- load all text files in `docs/`
- split them into chunks
- embed each chunk
- save the resulting vector store to `db/chroma_minilm/`

### 4. Run retrieval

```bash
python retrieval_pipeline.py
```

Then enter questions at the prompt, such as:

```text
Enter your question: What are the main differences between bicycles and cars?
```

The system will return the closest matching document chunks based on embedding similarity.

## How It Works

The retrieval pipeline uses dense vector search:

- Each document chunk is converted into a vector embedding.
- User questions are also embedded.
- Chroma finds the chunks with the highest semantic similarity.
- Those chunks are returned as the most relevant context.

This simple approach is effective for knowledge lookup, grounded Q&A, and searchable document systems.

## Notes

- The project is intentionally lightweight and local-first.
- It is best suited for experimentation and small-scale document retrieval.
- The current design focuses on retrieval only; it does not yet include a full LLM-based answer generation layer.
- You can extend it by adding an LLM prompt that summarizes or answers from the retrieved context.

## Future Enhancements

Possible upgrades include:

- adding support for PDF and Markdown files
- implementing a conversational chat layer
- integrating an LLM for final answer generation
- adding metadata filtering and hybrid search
- improving UI or API interfaces for deployment

## License

This project is intended for learning and experimentation. Add a license file if you plan to distribute it publicly or in production.
