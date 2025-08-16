try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime

from ..chunking.chunking import Chunk
from ..embeddings.embedding_config import EmbeddingFunction


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""
    store_type: str = "chroma"  # chroma, pinecone
    collection_name: str = "rag_documents"
    
    # ChromaDB specific
    persist_directory: str = "./chroma_db"
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp-free"
    index_name: str = "rag-index"
    
    # Common settings
    dimension: int = 384  # Will be updated based on embedding model
    metric: str = "cosine"
    batch_size: int = 1000


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk: Optional[Chunk] = None


@dataclass
class VectorStoreStats:
    """Statistics about vector store."""
    total_vectors: int
    total_collections: int
    storage_size_mb: float
    last_updated: str
    index_type: str


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, config: VectorStoreConfig, embedding_function: EmbeddingFunction):
        self.config = config
        self.embedding_function = embedding_function
        self.config.dimension = embedding_function.get_dimension()
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], batch_size: Optional[int] = None) -> bool:
        """Add chunks to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10, filter_metadata: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def clear_collection(self) -> bool:
        """Clear all data from the collection without deleting the collection."""
        pass
    
    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        pass
    
    @abstractmethod
    def get_stats(self) -> VectorStoreStats:
        """Get statistics about the vector store."""
        pass
    
    @abstractmethod
    def save_results(self, query: str, results: List[SearchResult], metadata: Optional[Dict] = None) -> str:
        """Save search results for later use (reranking, chat completion)."""
        pass
    
    @abstractmethod
    def load_results(self, result_id: str) -> Tuple[str, List[SearchResult], Dict]:
        """Load previously saved search results."""
        pass


class SimpleInMemoryVectorStore(VectorStore):
    """Simple in-memory vector store as fallback when ChromaDB fails."""
    
    def __init__(self, config: VectorStoreConfig, embedding_function: EmbeddingFunction):
        super().__init__(config, embedding_function)
        self.vectors = []  # List of (chunk_id, embedding, content, metadata)
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize in-memory store."""
        self.initialized = True
        print(f"✓ In-Memory Vector Store initialized: {self.config.collection_name}")
        return True
    
    def add_chunks(self, chunks: List[Chunk], batch_size: Optional[int] = None) -> bool:
        """Add chunks to in-memory store."""
        if not self.initialized:
            print("✗ In-Memory Vector Store not initialized")
            return False
        
        try:
            batch_size = batch_size or self.config.batch_size
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_function.embed_text(chunk.content)
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                # Store vector data
                metadata = chunk.metadata.copy() if chunk.metadata else {}
                metadata.update({
                    'source_doc': chunk.source_doc,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'chunk_size': len(chunk.content),
                    'created_at': datetime.now().isoformat()
                })
                
                self.vectors.append((chunk.chunk_id, embedding, chunk.content, metadata))
                
                if (i + 1) % batch_size == 0:
                    print(f"✓ Added batch {(i // batch_size) + 1}")
            
            print(f"✓ Successfully added {len(chunks)} chunks to In-Memory Vector Store")
            return True
            
        except Exception as e:
            print(f"✗ Failed to add chunks to In-Memory Vector Store: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, filter_metadata: Optional[Dict] = None) -> List[SearchResult]:
        """Search in-memory vectors using cosine similarity."""
        if not self.initialized or not self.vectors:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_function.embed_text(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Calculate similarities
            similarities = []
            for chunk_id, embedding, content, metadata in self.vectors:
                # Cosine similarity
                dot_product = np.dot(query_embedding, embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_doc = np.linalg.norm(embedding)
                similarity = dot_product / (norm_query * norm_doc)
                
                similarities.append((similarity, chunk_id, content, metadata))
            
            # Sort by similarity and take top_k
            similarities.sort(reverse=True, key=lambda score_and_result: score_and_result[0])
            top_results = similarities[:top_k]
            
            # Convert to SearchResult objects
            results = []
            for similarity, chunk_id, content, metadata in top_results:
                result = SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                    score=similarity
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return []
    
    def clear_collection(self) -> bool:
        """Clear all vectors."""
        count = len(self.vectors)
        self.vectors.clear()
        print(f"✓ Cleared {count} vectors from In-Memory Vector Store")
        return True
    
    def delete_collection(self) -> bool:
        """Delete all vectors."""
        self.vectors.clear()
        return True
    
    def get_stats(self) -> VectorStoreStats:
        """Get statistics."""
        return VectorStoreStats(
            total_vectors=len(self.vectors),
            total_collections=1,
            storage_size_mb=len(str(self.vectors)) / 1024 / 1024,  # Rough estimate
            last_updated=datetime.now().isoformat(),
            index_type="in_memory"
        )
    
    def save_results(self, query: str, results: List[SearchResult], metadata: Optional[Dict] = None) -> str:
        """Save search results (simple implementation)."""
        result_id = str(uuid.uuid4())
        # Simple in-memory storage - would use file in real implementation
        return result_id
    
    def load_results(self, result_id: str) -> Tuple[str, List[SearchResult], Dict]:
        """Load search results (simple implementation)."""
        return "", [], {}


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, config: VectorStoreConfig, embedding_function: EmbeddingFunction):
        super().__init__(config, embedding_function)
        self.client = None
        self.collection = None
        self.results_cache = {}  # In-memory cache for results
    
    def initialize(self) -> bool:
        """Initialize ChromaDB with improved error handling."""
        try:
            import chromadb
            from chromadb.config import Settings
            import os
            
            # Disable telemetry to avoid OpenTelemetry issues
            os.environ['ANONYMIZED_TELEMETRY'] = 'False'
            
            # Try different initialization approaches
            try:
                # First try: PersistentClient with minimal settings
                self.client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
            except Exception as e1:
                print(f"Warning: PersistentClient failed ({e1}), trying EphemeralClient...")
                try:
                    # Fallback: Use EphemeralClient (in-memory)
                    self.client = chromadb.EphemeralClient(
                        settings=Settings(anonymized_telemetry=False)
                    )
                    print("Note: Using in-memory ChromaDB (data won't persist)")
                except Exception as e2:
                    print(f"Warning: EphemeralClient failed ({e2}), trying basic Client...")
                    # Last resort: Basic client
                    self.client = chromadb.Client(
                        settings=Settings(anonymized_telemetry=False)
                    )
                    print("Note: Using basic ChromaDB client")
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.metric}
            )
            
            print(f"✓ ChromaDB initialized: {self.config.collection_name}")
            return True
            
        except Exception as e:
            print(f"✗ ChromaDB initialization failed: {e}")
            print("Suggestion: Try 'uv add chromadb==0.4.22' for version compatibility")
            return False
    
    def add_chunks(self, chunks: List[Chunk], batch_size: Optional[int] = None) -> bool:
        """Add chunks to ChromaDB with optimized duplicate handling."""
        if not self.collection:
            print("✗ ChromaDB not initialized")
            return False
        
        try:
            batch_size = batch_size or self.config.batch_size
            
            # Remove duplicates within the input chunks first
            unique_chunks = {}
            input_duplicates = 0
            for chunk in chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk
                else:
                    input_duplicates += 1
            
            chunks_to_process = list(unique_chunks.values())
            if input_duplicates > 0:
                print(f"Removed {input_duplicates} duplicates from input chunks")
            
            # Process in batches with optimized duplicate checking
            total_added = 0
            total_skipped = 0
            
            for i in range(0, len(chunks_to_process), batch_size):
                batch = chunks_to_process[i:i + batch_size]
                batch_ids = [chunk.chunk_id for chunk in batch]
                
                # Check which IDs already exist (batch query instead of full collection scan)
                try:
                    existing_check = self.collection.get(ids=batch_ids, include=[])
                    existing_in_batch = set(existing_check.get('ids', []))
                except Exception:
                    # If batch ID check fails, fall back to trying to add all
                    existing_in_batch = set()
                
                # Filter batch to only new chunks
                new_batch = []
                skipped_in_batch = 0
                for chunk in batch:
                    if chunk.chunk_id not in existing_in_batch:
                        new_batch.append(chunk)
                    else:
                        skipped_in_batch += 1
                
                total_skipped += skipped_in_batch
                
                if not new_batch:
                    continue
                
                # Prepare data for ChromaDB
                ids = [chunk.chunk_id for chunk in new_batch]
                documents = [chunk.content for chunk in new_batch]
                metadatas = []
                embeddings = []
                
                for chunk in new_batch:
                    # Prepare metadata
                    metadata = chunk.metadata.copy() if chunk.metadata else {}
                    metadata.update({
                        'source_doc': chunk.source_doc,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'chunk_size': len(chunk.content),
                        'created_at': datetime.now().isoformat()
                    })
                    sanitized_metadata = self._sanitize_metadata_for_chroma(metadata)
                    metadatas.append(sanitized_metadata)
                    
                    # Generate embedding
                    embedding = self.embedding_function.embed_text(chunk.content)
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    embeddings.append(embedding)
                
                # Add to collection with error handling for duplicates
                try:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                    total_added += len(new_batch)
                    print(f"Added batch {i//batch_size + 1}/{(len(chunks_to_process) + batch_size - 1)//batch_size} ({len(new_batch)} chunks)")
                
                except Exception as e:
                    # If batch add fails due to duplicates, try adding individually
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        individual_added = 0
                        for j, chunk in enumerate(new_batch):
                            try:
                                self.collection.add(
                                    ids=[ids[j]],
                                    documents=[documents[j]],
                                    embeddings=[embeddings[j]],
                                    metadatas=[metadatas[j]]
                                )
                                individual_added += 1
                            except Exception:
                                total_skipped += 1  # Count as duplicate
                        total_added += individual_added
                        if individual_added > 0:
                            print(f"Added {individual_added}/{len(new_batch)} chunks individually (batch {i//batch_size + 1})")
                    else:
                        print(f"Batch add failed: {e}")
                        return False
            
            if total_skipped > 0:
                print(f"Skipped {total_skipped} duplicate chunks")
            print(f"Successfully added {total_added} new chunks to ChromaDB")
            return True
            
        except Exception as e:
            print(f"Failed to add chunks to ChromaDB: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, filter_metadata: Optional[Dict] = None) -> List[SearchResult]:
        if not self.collection:
            print("ChromaDB not initialized")
            return []
        
        try:
            query_embedding = self.embedding_function.embed_text(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = SearchResult(
                        chunk_id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        score=1.0 - results['distances'][0][i]  # Convert distance to similarity
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"ChromaDB search failed: {e}")
            return []
    
    def clear_collection(self) -> bool:
        try:
            if not self.collection:
                print("Collection not initialized")
                return True
            
            existing_data = self.collection.get()
            if existing_data and existing_data.get('ids'):
                ids_to_delete = existing_data['ids']
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    print(f"Cleared {len(ids_to_delete)} chunks from ChromaDB collection")
                else:
                    print("Collection already empty")
            else:
                print("Collection already empty")
            return True
        except Exception as e:
            print(f"Failed to clear ChromaDB collection: {e}")
            return False

    def delete_collection(self) -> bool:
        """Delete ChromaDB collection."""
        try:
            if self.client and self.collection:
                self.client.delete_collection(self.config.collection_name)
                print(f"Deleted ChromaDB collection: {self.config.collection_name}")
                return True
            return False
        except Exception as e:
            print(f"Failed to delete ChromaDB collection: {e}")
            return False
    
    def get_stats(self) -> VectorStoreStats:
        """Get ChromaDB statistics."""
        try:
            if not self.collection:
                return VectorStoreStats(0, 0, 0.0, "Never", "ChromaDB")
            count = self.collection.count()
            storage_size_mb = count * self.config.dimension * 4 / (1024 * 1024)  # 4 bytes per float
            return VectorStoreStats(
                total_vectors=count,
                total_collections=1,
                storage_size_mb=storage_size_mb,
                last_updated=datetime.now().isoformat(),
                index_type="ChromaDB HNSW"
            )
            
        except Exception as e:
            print(f"Failed to get ChromaDB stats: {e}")
            return VectorStoreStats(0, 0, 0.0, "Error", "ChromaDB")
    
    def save_results(self, query: str, results: List[SearchResult], metadata: Optional[Dict] = None) -> str:
        """Save search results for later use."""
        result_id = str(uuid.uuid4())
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'chunk_id': result.chunk_id,
                'content': result.content,
                'metadata': result.metadata,
                'score': result.score
            })
        
        # Store in cache and optionally persist
        self.results_cache[result_id] = {
            'query': query,
            'results': serializable_results,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'result_count': len(results)
        }
        
        try:
            results_dir = f"{self.config.persist_directory}/results"
            os.makedirs(results_dir, exist_ok=True)
            
            with open(f"{results_dir}/{result_id}.json", 'w') as f:
                json.dump(self.results_cache[result_id], f, indent=2)
        except Exception as e:
            print(f"Failed to persist results to disk: {e}")
        
        return result_id
    
    def load_results(self, result_id: str) -> Tuple[str, List[SearchResult], Dict]:
        """Load previously saved search results."""
        if result_id in self.results_cache:
            data = self.results_cache[result_id]
        else:
            try:
                results_file = f"{self.config.persist_directory}/results/{result_id}.json"
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    self.results_cache[result_id] = data  # Cache it
            except Exception as e:
                raise ValueError(f"Result ID {result_id} not found: {e}")
        
        results = []
        for result_data in data['results']:
            result = SearchResult(
                chunk_id=result_data['chunk_id'],
                content=result_data['content'],
                metadata=result_data['metadata'],
                score=result_data['score']
            )
            results.append(result)
        
        return data['query'], results, data['metadata']
    
    def _sanitize_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata dictionary for ChromaDB compatibility.
        ChromaDB only accepts: bool, int, float, str values (no None, lists, dicts, etc.)
        """
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            str_key = str(key)
            if isinstance(value, (bool, int, float, str)):
                sanitized[str_key] = value
            elif isinstance(value, (list, tuple)):
                try:
                    if all(isinstance(item, (str, int, float, bool)) for item in value):
                        sanitized[str_key] = ",".join(str(item) for item in value)
                    else:
                        sanitized[str_key] = str(value)
                except Exception:
                    sanitized[str_key] = str(value)
            elif isinstance(value, dict):
                try:
                    import json
                    sanitized[str_key] = json.dumps(value)
                except Exception:
                    sanitized[str_key] = str(value)
            else:
                try:
                    sanitized[str_key] = str(value)
                except Exception:
                    continue
        
        return sanitized


class PineconeVectorStore(VectorStore):
    """Pinecone implementation of vector store."""
    
    def __init__(self, config: VectorStoreConfig, embedding_function: EmbeddingFunction):
        super().__init__(config, embedding_function)
        self.pc = None
        self.index = None
        self.results_storage = {}  # Simple in-memory storage for results
    
    def initialize(self) -> bool:
        try:
            from pinecone import Pinecone
            
            api_key = self.config.pinecone_api_key or os.getenv("PINECONE_API_KEY")
            if not api_key:
                print("Pinecone API key required")
                return False
            
            self.pc = Pinecone(api_key=api_key)
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.config.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                print(f"Created Pinecone index: {self.config.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.config.index_name)
            
            print(f"✓ Pinecone initialized: {self.config.index_name}")
            return True
            
        except Exception as e:
            print(f"✗ Pinecone initialization failed: {e}")
            return False
    
    def add_chunks(self, chunks: List[Chunk], batch_size: Optional[int] = None) -> bool:
        """Add chunks to Pinecone."""
        if not self.index:
            print("✗ Pinecone not initialized")
            return False
        
        try:
            batch_size = batch_size or self.config.batch_size
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare vectors for Pinecone
                vectors = []
                for chunk in batch:
                    # Generate embedding
                    embedding = self.embedding_function.embed_text(chunk.content)
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    # Prepare metadata
                    metadata = chunk.metadata.copy() if chunk.metadata else {}
                    metadata.update({
                        'content': chunk.content,  # Store content in metadata
                        'source_doc': chunk.source_doc,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'chunk_size': len(chunk.content),
                        'created_at': datetime.now().isoformat()
                    })
                    
                    vectors.append({
                        'id': chunk.chunk_id,
                        'values': embedding,
                        'metadata': metadata
                    })
                
                # Upsert to Pinecone
                self.index.upsert(vectors=vectors)
                
                print(f"✓ Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            print(f"✓ Successfully added {len(chunks)} chunks to Pinecone")
            return True
            
        except Exception as e:
            print(f"✗ Failed to add chunks to Pinecone: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, filter_metadata: Optional[Dict] = None) -> List[SearchResult]:
        """Search Pinecone for similar vectors."""
        if not self.index:
            print("✗ Pinecone not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_function.embed_text(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Perform search
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            for match in search_response.matches:
                result = SearchResult(
                    chunk_id=match.id,
                    content=match.metadata.get('content', ''),
                    metadata=match.metadata,
                    score=match.score
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"✗ Pinecone search failed: {e}")
            return []
    
    def clear_collection(self) -> bool:
        """Clear all vectors from Pinecone index."""
        try:
            if not self.index:
                print("⚠️ Pinecone index not initialized")
                return True
                
            # Delete all vectors (Pinecone doesn't have a clear all method)
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors > 0:
                # For now, we can't efficiently clear all vectors from Pinecone
                # without knowing all IDs. This would require a different approach.
                print(f"⚠️ Cannot clear Pinecone index efficiently ({total_vectors} vectors)")
                print("⚠️ Consider deleting and recreating the index for full clear")
                return False
            else:
                print("ℹ️ Pinecone index already empty")
                return True
        except Exception as e:
            print(f"✗ Failed to clear Pinecone index: {e}")
            return False

    def delete_collection(self) -> bool:
        """Delete Pinecone index."""
        try:
            if self.pc:
                self.pc.delete_index(self.config.index_name)
                print(f"✓ Deleted Pinecone index: {self.config.index_name}")
                return True
            return False
        except Exception as e:
            print(f"✗ Failed to delete Pinecone index: {e}")
            return False
    
    def get_stats(self) -> VectorStoreStats:
        """Get Pinecone statistics."""
        try:
            if not self.index:
                return VectorStoreStats(0, 0, 0.0, "Never", "Pinecone")
            
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            # Estimate storage size
            storage_size_mb = total_vectors * self.config.dimension * 4 / (1024 * 1024)
            
            return VectorStoreStats(
                total_vectors=total_vectors,
                total_collections=1,
                storage_size_mb=storage_size_mb,
                last_updated=datetime.now().isoformat(),
                index_type="Pinecone"
            )
            
        except Exception as e:
            print(f"✗ Failed to get Pinecone stats: {e}")
            return VectorStoreStats(0, 0, 0.0, "Error", "Pinecone")
    
    def save_results(self, query: str, results: List[SearchResult], metadata: Optional[Dict] = None) -> str:
        """Save search results for later use."""
        result_id = str(uuid.uuid4())
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'chunk_id': result.chunk_id,
                'content': result.content,
                'metadata': result.metadata,
                'score': result.score
            })
        
        # Store in memory (for production, use proper database)
        self.results_storage[result_id] = {
            'query': query,
            'results': serializable_results,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'result_count': len(results)
        }
        
        return result_id
    
    def load_results(self, result_id: str) -> Tuple[str, List[SearchResult], Dict]:
        """Load previously saved search results."""
        if result_id not in self.results_storage:
            raise ValueError(f"Result ID {result_id} not found")
        
        data = self.results_storage[result_id]
        
        # Convert back to SearchResult objects
        results = []
        for result_data in data['results']:
            result = SearchResult(
                chunk_id=result_data['chunk_id'],
                content=result_data['content'],
                metadata=result_data['metadata'],
                score=result_data['score']
            )
            results.append(result)
        
        return data['query'], results, data['metadata']


def create_vector_store(config: VectorStoreConfig, embedding_function: EmbeddingFunction) -> VectorStore:
    """Factory function to create vector store with automatic fallback."""
    
    if config.store_type.lower() == "chroma":
        try:
            import chromadb
            return ChromaVectorStore(config, embedding_function)
        except Exception as e:
            print(f"Warning: ChromaDB not available ({e}), using in-memory fallback")
            return SimpleInMemoryVectorStore(config, embedding_function)
    elif config.store_type.lower() == "pinecone":
        return PineconeVectorStore(config, embedding_function)
    elif config.store_type.lower() == "memory" or config.store_type.lower() == "in_memory":
        return SimpleInMemoryVectorStore(config, embedding_function)
    else:
        raise ValueError(f"Unsupported vector store type: {config.store_type}")
        
def create_vector_store_safe(config: VectorStoreConfig, embedding_function: EmbeddingFunction) -> VectorStore:
    """Create vector store but return pre-initialized instance."""
    store = create_vector_store(config, embedding_function)
    if not hasattr(store, 'initialized') or not store.initialized:
        store.initialize()
    return store


def create_chroma_store(
    collection_name: str = "rag_documents",
    persist_directory: str = "./chroma_db",
    embedding_function: Optional[EmbeddingFunction] = None
) -> ChromaVectorStore:
    """Create ChromaDB vector store with defaults."""
    
    if not embedding_function:
        from ..embeddings.embedding_config import get_fast_embedding_function
        embedding_function = get_fast_embedding_function()
    
    config = VectorStoreConfig(
        store_type="chroma",
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    return ChromaVectorStore(config, embedding_function)


def create_pinecone_store(
    index_name: str = "rag-index",
    api_key: Optional[str] = None,
    embedding_function: Optional[EmbeddingFunction] = None
) -> PineconeVectorStore:
    """Create Pinecone vector store with defaults."""
    
    if not embedding_function:
        from ..embeddings.embedding_config import get_fast_embedding_function
        embedding_function = get_fast_embedding_function()
    
    config = VectorStoreConfig(
        store_type="pinecone",
        index_name=index_name,
        pinecone_api_key=api_key
    )
    
    return PineconeVectorStore(config, embedding_function)