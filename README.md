# DingDong RAG: Hybrid Document Intelligence System

A comprehensive Retrieval-Augmented Generation (RAG) system with advanced document processing, vector storage, reranking, and conversational AI capabilities.

## Features

DingDong RAG offers a robust pipeline for building powerful RAG applications:

-   **Hybrid Document Parsing**: Extracts text from PDFs using PyMuPDF4LLM or Docling, with intelligent OCR fallback (Tesseract, Gemini).
-   **Advanced Chunking**: Supports various strategies including recursive, semantic, and Chonkie-based chunking for optimal retrieval.
-   **Flexible Embedding Models**: Integrates with a wide range of embedding models, including Sentence Transformers (multilingual) and OpenAI.
-   **Configurable Vector Stores**: Supports ChromaDB (default, local) and Pinecone (cloud) for efficient semantic search.
-   **Sophisticated Retrieval & Reranking**: Implements BM25, Cross-Encoder, and Hybrid reranking strategies, including precision-tuned modes.
-   **Corrective RAG (CRAG)**: Enhances retrieval quality through query reformulation and self-correction.
-   **Intelligent Context Management**: Features query enhancement, automatic domain detection, and context compression for optimized LLM interaction.
-   **Extensible Chat Interface**: Seamlessly integrates with various Large Language Models (LLMs) for conversational AI.

## Installation

**Prerequisites**:

-   Python 3.10 or newer.
-   For Tesseract OCR, ensure it's installed and available in your system's PATH.
-   For Docling parser, it may download additional models on first run (can be pre-downloaded).

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/dingdong-rag.git
    cd dingdong-rag
    ```

2.  **Install dependencies**: For best results, it's recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    pip install -e .
    ```

    **Note on Dependencies**: The `pip install -e .` command will install all core dependencies listed in `pyproject.toml`:
    `pymupdf4llm`, `pytesseract`, `pillow`, `pdf2image`, `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `openai`, `networkx`, `pyvis`, `pinecone-client`, `rank-bm25`, `psutil`, `pymupdf`, `docling`, `google-generativeai`, `chromadb`, `python-dotenv`, `gradio`, `cohere`, `chonkie`.

## Environment Variables

Depending on the models and services you use, you may need to set the following environment variables:

-   `OPENAI_API_KEY`: Required for OpenAI embedding models (`text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`) and OpenAI LLMs (`gpt-4o`, `gpt-3.5-turbo`, etc.).
-   `PINECONE_API_KEY`: Required if using Pinecone as your vector store.
-   `COHERE_API_KEY`: Required if using Cohere for reranking.
-   `GOOGLE_API_KEY`: Required if using Gemini for OCR or LLM.

You can set these in your shell or create a `.env` file in the project root (e.g., `OPENAI_API_KEY='sk-...'`).

## Usage: Command Line Interface (CLI)

The main CLI entry point is `dingdong-rag`. You can use it to run the pipeline in different modes and configure various components.

**Basic End-to-End Pipeline**:

This command processes documents from `./documents`, chunks them, creates embeddings, stores them in a ChromaDB vector store, sets up reranking, and starts an interactive chat session.

```bash
dingdong-rag --documents-dir ./documents --e2e
```

**Step-by-Step Pipeline with Custom Settings**:

This mode prompts you before each major step, allowing you to review and control the flow. Here, we also set a custom chunk size.

```bash
dingdong-rag --documents-dir ./documents --step-by-step --chunk-size 800
```

**Jump Directly to Chat (using existing data)**:

If you have already processed documents and built your vector store, you can skip the ingestion steps and go straight to chatting.

```bash
dingdong-rag --documents-dir ./documents --chat-only
```

**Using Docling Parser (instead of PyMuPDF)**:

Docling offers advanced table and figure extraction. You can also pre-download models for offline use.

```bash
# Use Docling (tables and figures enabled by default)
dingdong-rag --documents-dir ./documents --e2e --parser-method docling

# Pre-download Docling models (useful for network issues) and exit
dingdong-rag --pre-download-docling-models --docling-download-timeout 600

# Use Docling in offline mode (after pre-downloading)
dingdong-rag --documents-dir ./documents --e2e --parser-method docling --docling-offline-mode
```

**Recommended Setup (High Quality, requires OpenAI key)**:

This configuration uses a high-quality OpenAI embedding model and a hybrid reranking strategy for optimal results.

```bash
dingdong-rag --documents-dir ./documents --e2e \
  --embedding-model text-embedding-3-large \
  --reranking-strategy hybrid --reranking-top-k 20 --retrieval-top-k 75
```

**Multilingual Setup (Recommended for Indonesian + English)**:

Uses a multilingual Sentence Transformer model for better performance with mixed-language documents.

```bash
dingdong-rag --documents-dir ./documents --e2e --embedding-model paraphrase-multilingual-MiniLM-L12-v2
```

**Enabling CRAG (Corrective RAG)**:

Improve retrieval quality by enabling CRAG, which reformulates queries based on initial retrieval results.

```bash
dingdong-rag --documents-dir ./documents --e2e --enable-crag --crag-trigger-mode hybrid
```

**Precision-Tuned Reranking & Similarity Threshold**:

Reduce "topic bleed" (retrieving irrelevant chunks) by filtering results based on similarity and using a precision-focused reranking.

```bash
dingdong-rag --documents-dir ./documents --e2e --similarity-threshold 0.3 --precision-reranking
```

**Disabling Automatic Domain Detection**:

By default, the system tries to detect the document domain from your query. Disable it for broader, cross-domain questions.

```bash
dingdong-rag --documents-dir ./documents --e2e --disable-auto-domain-detection
```

**Other Important CLI Arguments**:

-   `--working-dir <path>`: Specify a custom directory for intermediate files (default: `./rag_working_dir`).
-   `--no-ocr-fallback`: Disable OCR if you expect all your PDFs to be text-searchable.
-   `--chunking-strategy [fixed|sentence|recursive|semantic|chonkie]`: Choose your chunking method.
-   `--vector-store [chroma|pinecone]`: Select your vector database.
-   `--llm-model <model_name>`: Specify the LLM for chat completion (e.g., `gpt-4o`, `gemini-pro`).
-   `--max-response-tokens <int>`: Adjust the length of LLM responses.
-   `--temperature <float>`: Control LLM creativity (0.0-1.0).
-   `--disable-query-enhancement`: Skip LLM-based query rephrasing.
-   `--disable-context-compression`: Turn off context summarization for long conversations.
-   `--no-interactive`: Run the pipeline without starting an interactive chat session.

For a full list of arguments and their descriptions, run:

```bash
dingdong-rag --help
```

## Usage: Python Class

You can also integrate DingDong RAG directly into your Python applications using the `DingDongRAG` class.

First, make sure your environment variables (like `OPENAI_API_KEY`) are set up if you plan to use proprietary models.

```python
import os
from dingdong_rag.dingdong import DingDongRAG

# --- 1. Initialize the RAG Pipeline ---
# You can pass arguments to configure the pipeline, similar to CLI options
rag_pipeline = DingDongRAG(
    documents_dir="./documents",
    working_dir="./my_rag_data",
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # Or 'text-embedding-3-large' with OpenAI API key
    vector_store_type="chroma",
    llm_model="gpt-4o",  # Or 'gemini-pro' with Google API key
    reranking_strategy="hybrid",
    reranking_top_k=20,
    retrieval_top_k=75,
    enable_crag=True, # Enable Corrective RAG
)

# Initialize the pipeline components (embeddings, vector store, reranker, chat engine)
# This step typically happens once at the start of your application
print("Initializing RAG pipeline...")
initialization_results = rag_pipeline.initialize()
print("Initialization complete:", initialization_results)

# --- 2. Ingest Documents (Build your knowledge base) ---
# This processes PDFs, chunks them, generates embeddings, and stores them
print("Ingesting documents...")
ingestion_results = rag_pipeline.ingest()
print("Ingestion complete:", ingestion_results)

# --- 3. Query the RAG Pipeline ---
# Ask questions and get answers augmented by your documents
question1 = "What is discrete mathematics?"
response1 = rag_pipeline.query(question1)
print(f"\nQuestion: {question1}")
print(f"Answer: {response1.message}")
if response1.sources_used:
    print(f"Sources: {response1.sources_used[:2]}...") # Show top 2 sources

question2 = "Explain the concept of process scheduling in operating systems."
response2 = rag_pipeline.query(question2)
print(f"\nQuestion: {question2}")
print(f"Answer: {response2.message}")
if response2.sources_used:
    print(f"Sources: {response2.sources_used[:2]}...")

# --- Get Pipeline Statistics ---
stats = rag_pipeline.stats()
print("\nPipeline Statistics:", stats)

# --- Save Pipeline State (Optional) ---
# Useful for reproducibility or auditing
rag_pipeline.save_state("pipeline_state.json")
print("Pipeline state saved to pipeline_state.json")
```
