from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...parsing.pdf_parser import process_documents_folder, ValidationConfig
from ...chunking.chunking import ChunkingConfig, get_chunking_strategy


@dataclass
class IngestionResult:
    documents_processed: int
    documents_failed: int
    chunks_created: int
    chunks_stored: int
    ingestion_time: float
    average_chunks_per_doc: float


def ingest_documents(config: Any, vector_store: Any) -> IngestionResult | Dict[str, Any]:
    start_time = time.time()

    if vector_store is None:
        return {"error": "Vector store not initialized"}

    # 1) Process PDFs
    validation_config = ValidationConfig()
    pdf_results = process_documents_folder(
        documents_dir=config.documents_dir,
        max_files=config.max_documents,
        fallback_to_ocr=True,
        validation_config=validation_config,
    )

    if not pdf_results:
        return {"error": "No documents found or processed"}

    processed_docs = [k for k, v in pdf_results.items() if v is not None]
    failed_docs = [k for k, v in pdf_results.items() if v is None]

    # 2) Chunking
    chunking_config = ChunkingConfig(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunking_strategy = get_chunking_strategy(config.chunking_strategy, chunking_config)

    all_chunks = []
    for doc_path, pdf_data in pdf_results.items():
        if pdf_data is None:
            continue
        if isinstance(pdf_data, dict):
            content = pdf_data.get("content", "")
            metadata = pdf_data.get("metadata", {})
        else:
            content = pdf_data
            metadata = {}
        if not content or not isinstance(content, str):
            continue
        doc_id = Path(doc_path).stem
        chunks = chunking_strategy.chunk_text(content, doc_id)
        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata.update(metadata)
        all_chunks.extend(chunks)

    # 3) Store
    if not vector_store.add_chunks(all_chunks):
        return {"error": "Failed to store chunks in vector database"}

    ingestion_time = time.time() - start_time
    return IngestionResult(
        documents_processed=len(processed_docs),
        documents_failed=len(failed_docs),
        chunks_created=len(all_chunks),
        chunks_stored=len(all_chunks),
        ingestion_time=ingestion_time,
        average_chunks_per_doc=(len(all_chunks) / len(processed_docs) if processed_docs else 0.0),
    )


