import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..parsing.pdf_parser import process_documents_folder, ValidationConfig
from ..chunking.chunking import (
    Chunk, ChunkingConfig, ChunkingStrategy, 
    get_chunking_strategy
)


@dataclass
class IngestionConfig:
    """Configuration for document ingestion pipeline."""
    chunking_strategy: str = "fixed"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    semantic_similarity_threshold: float = 0.7
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    cache_embeddings: bool = True
    
    def to_chunking_config(self) -> ChunkingConfig:
        """Convert to ChunkingConfig."""
        return ChunkingConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            semantic_similarity_threshold=self.semantic_similarity_threshold
        )


@dataclass
class Document:
    """Represents a processed document."""
    doc_id: str
    content: str
    chunks: List[Chunk]
    metadata: Dict[str, Any]
    
    def get_chunk_count(self) -> int:
        return len(self.chunks)
    
    def get_total_chars(self) -> int:
        return len(self.content)
    
    def get_avg_chunk_size(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(len(chunk.content) for chunk in self.chunks) / len(self.chunks)


class DocumentStore:
    """Vector store for documents and chunks with memory optimization."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        from ..embeddings.embedding_manager import get_embedding_model
        self.embedding_model = get_embedding_model(embedding_model)
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_cache_limit = 10000  # Limit in-memory embeddings
        
    def add_document(self, document: Document):
        """Add document to store."""
        self.documents[document.doc_id] = document
        
        # Add chunks
        for chunk in document.chunks:
            self.chunks[chunk.chunk_id] = chunk
    
    def compute_embeddings(self, batch_size: int = 32, show_progress: bool = True):
        """Compute embeddings for all chunks with memory optimization."""
        import gc
        chunk_ids = list(self.chunks.keys())
        total_chunks = len(chunk_ids)
        
        if show_progress:
            print(f"Computing embeddings for {total_chunks} chunks...")
        
        # Process in smaller batches to manage memory
        processing_batch_size = min(batch_size, 100)  # Limit memory usage
        
        for i in range(0, total_chunks, processing_batch_size):
            batch_chunk_ids = chunk_ids[i:i + processing_batch_size]
            batch_texts = [self.chunks[cid].content for cid in batch_chunk_ids]
            
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=min(batch_size, len(batch_texts)),
                show_progress_bar=False
            )
            
            for chunk_id, embedding in zip(batch_chunk_ids, batch_embeddings):
                if len(self.embeddings) >= self.embedding_cache_limit:
                    oldest_keys = list(self.embeddings.keys())[:len(self.embeddings) - self.embedding_cache_limit + 1]
                    for old_key in oldest_keys:
                        del self.embeddings[old_key]
                self.embeddings[chunk_id] = embedding
            
            if i % (processing_batch_size * 5) == 0:
                gc.collect()
                
            if show_progress and i % (processing_batch_size * 10) == 0:
                progress = (i + processing_batch_size) / total_chunks * 100
                print(f"  Progress: {progress:.1f}% ({i + processing_batch_size}/{total_chunks})")
    
    def compute_embeddings_streaming(self, batch_size: int = 32, persist_to_disk: bool = True):
        """Compute embeddings with disk persistence to save memory."""
        import gc
        import tempfile
        import pickle
        
        chunk_ids = list(self.chunks.keys())
        embedding_cache = {}
        temp_files = []
        
        print(f"Computing embeddings for {len(chunk_ids)} chunks (streaming mode)...")
        
        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_texts = [self.chunks[cid].content for cid in batch_ids]
            
            # Compute batch embeddings
            batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
            
            if persist_to_disk:
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
                batch_data = {cid: emb for cid, emb in zip(batch_ids, batch_embeddings)}
                pickle.dump(batch_data, temp_file)
                temp_file.close()
                temp_files.append(temp_file.name)
            else:
                # Keep in memory cache
                for cid, emb in zip(batch_ids, batch_embeddings):
                    embedding_cache[cid] = emb
            
            if i % (batch_size * 10) == 0:
                gc.collect()
                progress = (i + batch_size) / len(chunk_ids) * 100
                print(f"  Progress: {progress:.1f}%")
        
        if persist_to_disk:
            print(f"Embeddings computed and cached in {len(temp_files)} temporary files")
            return temp_files
        else:
            self.embeddings.update(embedding_cache)
            return None
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.embeddings:
            raise ValueError("No embeddings computed. Call compute_embeddings() first.")
        
        query_embedding = self.embedding_model.encode([query])
        chunk_ids = list(self.embeddings.keys())
        chunk_embeddings = np.array([self.embeddings[cid] for cid in chunk_ids])
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk_id = chunk_ids[idx]
            chunk = self.chunks[chunk_id]
            score = similarities[idx]
            results.append((chunk, score))
        return results
    
    def save(self, filepath: str):
        """Save document store to disk."""
        data = {
            'documents': {k: asdict(v) for k, v in self.documents.items()},
            'chunks': {k: asdict(v) for k, v in self.chunks.items()},
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
            'model_name': self.embedding_model.get_sentence_embedding_dimension()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for doc_id, doc_data in data['documents'].items():
            chunks = [Chunk(**chunk_data) for chunk_data in doc_data['chunks']]
            doc = Document(
                doc_id=doc_data['doc_id'],
                content=doc_data['content'],
                chunks=chunks,
                metadata=doc_data['metadata']
            )
            self.documents[doc_id] = doc
        
        for chunk_id, chunk_data in data['chunks'].items():
            self.chunks[chunk_id] = Chunk(**chunk_data)
        
        for chunk_id, embedding_list in data['embeddings'].items():
            self.embeddings[chunk_id] = np.array(embedding_list)


class IngestionPipeline:
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.chunking_strategy = get_chunking_strategy(
            config.chunking_strategy, 
            config.to_chunking_config()
        )
        self.document_store = DocumentStore(config.embedding_model)
    
    def ingest_documents(self, documents_dir: str, max_docs: Optional[int] = None, 
                        streaming_mode: bool = True, doc_batch_size: int = 10) -> DocumentStore:
        import gc
        validation_config = ValidationConfig()
        pdf_results = process_documents_folder(
            documents_dir=documents_dir,
            max_files=max_docs,
            fallback_to_ocr=True,
            validation_config=validation_config,
            extract_metadata=True,
            use_cache=True,  # Enable caching by default
            cache_dir=".cache/documents"
        )
        
        if streaming_mode:
            doc_items = list(pdf_results.items())
            total_docs = len(doc_items)
            
            for batch_start in range(0, total_docs, doc_batch_size):
                batch_end = min(batch_start + doc_batch_size, total_docs)
                doc_batch = doc_items[batch_start:batch_end]
                
                print(f"Processing batch {batch_start//doc_batch_size + 1}/{(total_docs-1)//doc_batch_size + 1} ({len(doc_batch)} docs)")
                
                # Process documents in this batch
                batch_documents = []
                for doc_path, doc_data in doc_batch:
                    document = self._process_single_document(doc_path, doc_data)
                    if document:
                        batch_documents.append(document)
                
                # Add batch to store
                for document in batch_documents:
                    self.document_store.add_document(document)
                
                # Compute embeddings for this batch if needed
                if len(self.document_store.chunks) > 1000:  # Process embeddings in chunks
                    print(f"  Computing embeddings for accumulated chunks...")
                    self.document_store.compute_embeddings(
                        batch_size=self.config.batch_size,
                        show_progress=False
                    )
                
                batch_documents.clear()
                gc.collect()
                print(f"  Batch completed. Total docs: {len(self.document_store.documents)}, chunks: {len(self.document_store.chunks)}")
        else:
            for doc_path, doc_data in tqdm(pdf_results.items(), desc="Processing documents"):
                document = self._process_single_document(doc_path, doc_data)
                if document:
                    self.document_store.add_document(document)
        
        # Final embedding computation for remaining chunks
        if self.document_store.chunks and not streaming_mode:
            self.document_store.compute_embeddings(
                batch_size=self.config.batch_size,
                show_progress=True
            )
        elif streaming_mode and self.document_store.chunks:
            print("Computing final embeddings for remaining chunks...")
            self.document_store.compute_embeddings(
                batch_size=self.config.batch_size,
                show_progress=True
            )
        
        print(f"Ingested {len(self.document_store.documents)} documents with {len(self.document_store.chunks)} chunks")
        
        return self.document_store
    
    def _process_single_document(self, doc_path: str, doc_data: Any) -> Optional[Document]:
        """Process a single document with memory management."""
        if doc_data is None:
            print(f"Skipping failed document: {doc_path}")
            return None
        
        # Handle both old format (string content) and new format (dict with metadata)
        if isinstance(doc_data, dict):
            content = doc_data.get('content', '')
            doc_metadata = doc_data.get('metadata', {})
            processing_method = doc_data.get('processing_method', 'unknown')
        else:
            # Fallback for old format
            content = doc_data
            doc_metadata = {'source_path': doc_path}
            processing_method = 'legacy'
        
        if not content or not content.strip():
            print(f"Skipping empty document: {doc_path}")
            return None
        
        doc_id = Path(doc_path).stem
        chunks = self.chunking_strategy.chunk_text(content, doc_id)
        for chunk in chunks:
            if hasattr(chunk, 'metadata') and chunk.metadata:
                chunk.metadata.update({
                    'document_metadata': doc_metadata,
                    'source_path': doc_path
                })
        
        merged_metadata = {
            'source_path': doc_path,
            'chunking_strategy': self.config.chunking_strategy,
            'chunk_count': len(chunks),
            'total_chars': len(content),
            'avg_chunk_size': sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
            'processing_method': processing_method
        }
        
        # Merge with extracted document metadata
        merged_metadata.update(doc_metadata)
        
        # Create document
        document = Document(
            doc_id=doc_id,
            content=content,
            chunks=chunks,
            metadata=merged_metadata
        )
        
        return document
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingestion process."""
        if not self.document_store.documents:
            return {"error": "No documents ingested"}
        
        documents = list(self.document_store.documents.values())
        
        stats = {
            'total_documents': len(documents),
            'total_chunks': len(self.document_store.chunks),
            'total_characters': sum(doc.get_total_chars() for doc in documents),
            'avg_chunks_per_doc': sum(doc.get_chunk_count() for doc in documents) / len(documents),
            'avg_chunk_size': sum(doc.get_avg_chunk_size() for doc in documents) / len(documents),
            'chunk_size_distribution': self._get_chunk_size_distribution(),
            'config': asdict(self.config)
        }
        
        return stats
    
    def _get_chunk_size_distribution(self) -> Dict[str, float]:
        """Get distribution of chunk sizes."""
        if not self.document_store.chunks:
            return {}
        
        chunk_sizes = [len(chunk.content) for chunk in self.document_store.chunks.values()]
        
        return {
            'min_size': min(chunk_sizes),
            'max_size': max(chunk_sizes),
            'mean_size': np.mean(chunk_sizes),
            'median_size': np.median(chunk_sizes),
            'std_size': np.std(chunk_sizes),
            'percentile_25': np.percentile(chunk_sizes, 25),
            'percentile_75': np.percentile(chunk_sizes, 75)
        }


def run_ingestion_experiment(
    documents_dir: str,
    configs: List[IngestionConfig],
    output_dir: str = "ingestion_results",
    max_docs: Optional[int] = None
) -> Dict[str, Any]:
    """Run ingestion experiment with multiple configurations."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\n--- Running Configuration {i+1}/{len(configs)} ---")
        print(f"Strategy: {config.chunking_strategy}, Size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
        
        # Create pipeline
        pipeline = IngestionPipeline(config)
        
        # Ingest documents
        document_store = pipeline.ingest_documents(documents_dir, max_docs)
        
        # Get stats
        stats = pipeline.get_ingestion_stats()
        
        # Save results
        config_name = f"{config.chunking_strategy}_{config.chunk_size}_{config.chunk_overlap}"
        results[config_name] = stats
        
        # Save document store
        store_path = output_path / f"{config_name}_store.json"
        document_store.save(str(store_path))
        
        print(f"✓ Processed {stats['total_documents']} docs into {stats['total_chunks']} chunks")
        print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
    
    # Save summary results
    summary_path = output_path / "ingestion_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    return results